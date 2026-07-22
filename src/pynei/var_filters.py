from functools import partial
import itertools
from typing import Sequence

import numpy

from pynei.variants import Variants, VariantsChunk
from pynei.gt_counts import (
    _calc_gt_is_missing,
    _calc_maf_per_var,
    _calc_obs_het_per_var,
)
from .ld_calc import _calc_rogers_huff_r2


class _FilterChunkIterFactory:
    def __init__(self, in_vars, filter_funct):
        self.in_vars = in_vars
        self._chunks = in_vars.iter_vars_chunks()
        self.filter_funct = filter_funct
        self.num_vars_processed = 0
        self.num_vars_kept = 0
        self._metadata = None
        self._first_processed_chunk = None

    def _get_metadata(self):
        if self._metadata is not None:
            return self._metadata.copy()

        if self._first_processed_chunk is None:
            try:
                first_chunk = next(self.iter_vars_chunks())
            except StopIteration:
                raise RuntimeError("No variations to get the data from")
            self._first_processed_chunk = first_chunk
        else:
            first_chunk = self._first_processed_chunk

        self._metadata = {
            "samples": first_chunk.gts.samples,
            "num_samples": first_chunk.num_samples,
            "ploidy": first_chunk.gts.ploidy,
        }
        return self._metadata.copy()

    def iter_vars_chunks(self):
        if self._first_processed_chunk is not None:
            yield self._first_processed_chunk

        for chunk in self._chunks:
            if self._metadata is None:
                self._metadata = {
                    "samples": chunk.gts.samples,
                    "num_samples": chunk.num_samples,
                    "ploidy": chunk.gts.ploidy,
                }
            filtered_chunk, num_vars_kept = self.filter_funct(chunk)
            self.num_vars_processed += chunk.num_vars
            self.num_vars_kept += num_vars_kept
            yield filtered_chunk


class _MissingFilterIterFactory(_FilterChunkIterFactory):
    kind = "missing_data"


def gather_filtering_stats(vars: Variants, stats=None):
    if stats is None:
        stats = {}  # stats by filter kind
    chunk_factory = vars._vars_chunks_iter_factory
    if isinstance(chunk_factory, _FilterChunkIterFactory):
        filter_kind = chunk_factory.kind
        if filter_kind not in stats:
            stats[filter_kind] = {"vars_processed": 0, "vars_kept": 0}
        stats[filter_kind]["vars_processed"] += chunk_factory.num_vars_processed
        stats[filter_kind]["vars_kept"] += chunk_factory.num_vars_kept
    if hasattr(chunk_factory, "in_vars"):
        gather_filtering_stats(chunk_factory.in_vars, stats)

    for filtering_stats in stats.values():
        if "vars_kept" in filtering_stats and hasattr(
            filtering_stats["vars_kept"], "item"
        ):
            # this is to convert from np.int64 to native python int
            filtering_stats["vars_kept"] = filtering_stats["vars_kept"].item()
    return stats


def _filter_chunk_by_missing(chunk, max_missing_rate):
    num_missing_per_var = numpy.sum(_calc_gt_is_missing(chunk)["gt_is_missing"], axis=1)
    missing_rate_per_var = num_missing_per_var / chunk.num_samples
    mask = missing_rate_per_var <= max_missing_rate
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_missing_data(
    vars: Variants, max_allowed_missing_rate: float = 0.0
) -> Variants:
    filter_chunk_by_missing = partial(
        _filter_chunk_by_missing, max_missing_rate=max_allowed_missing_rate
    )
    chunk_factory = _MissingFilterIterFactory(vars, filter_chunk_by_missing)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


class _MafFilterIterFactory(_FilterChunkIterFactory):
    kind = "maf"


def _filter_chunk_by_maf(chunk, max_allowed_maf):
    mafs = _calc_maf_per_var(chunk, pops={"pop": slice(None, None)}, min_num_samples=0)[
        "major_allele_freqs_per_var"
    ]["pop"]
    mask = mafs <= max_allowed_maf
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_maf(vars: Variants, max_allowed_maf) -> Variants:
    filter_chunk = partial(_filter_chunk_by_maf, max_allowed_maf=max_allowed_maf)
    chunk_factory = _MafFilterIterFactory(vars, filter_chunk)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


class _ObsHetFilterIterFactory(_FilterChunkIterFactory):
    kind = "obs_het"


def _filter_chunk_by_obs_het(chunk, max_allowed_obs_het):
    obs_hets = _calc_obs_het_per_var(chunk, pops={"pop": slice(None, None)})[
        "obs_het_per_var"
    ]["pop"]
    mask = obs_hets <= max_allowed_obs_het
    num_vars_kept = mask.sum()
    chunk = chunk.get_vars(mask)
    return chunk, num_vars_kept


def filter_by_obs_het(vars: Variants, max_allowed_obs_het: float):
    filter_chunk = partial(
        _filter_chunk_by_obs_het, max_allowed_obs_het=max_allowed_obs_het
    )
    chunk_factory = _ObsHetFilterIterFactory(vars, filter_chunk)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


def _filter_samples(chunk, sample_idxs):
    gts = chunk.gts.filter_samples_with_idxs(sample_idxs)
    chunk = VariantsChunk(gts, vars_info=chunk.vars_info, alleles=chunk.alleles)
    return chunk, chunk.num_vars


class _SampleFilterIterFactory(_FilterChunkIterFactory):
    kind = "sample"


def filter_samples(vars, samples: Sequence[str] | Sequence[int] | slice) -> Variants:
    orig_samples = vars.samples
    if isinstance(samples, slice):
        samples = orig_samples[samples]
    sample_idxs = numpy.where(numpy.isin(orig_samples, samples))[0]

    filter_samples = partial(_filter_samples, sample_idxs=sample_idxs)
    chunk_factory = _SampleFilterIterFactory(vars, filter_samples)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )


def _filter_chunk_by_ld(chunk, ref_gt, filter_chunk_by_maf, min_allowed_r2):
    selected_vars = []
    filtered_chunk, _ = filter_chunk_by_maf(chunk)

    if not filtered_chunk.num_vars:
        return filtered_chunk, 0, ref_gt

    gts_012 = filtered_chunk.gts.to_012()
    var_offset = 0

    while True:
        if ref_gt is None:
            ref_gt = gts_012[0, :].reshape((1, gts_012.shape[1]))
            selected_vars.append(0)
            if gts_012.shape[0] == 1:
                break
            else:
                gts_012 = gts_012[1:, :]
                var_offset = 1

        r2 = _calc_rogers_huff_r2(ref_gt, gts_012, check_no_mafs_above=None).flat
        r2 = numpy.abs(r2)

        unlinked_vars = numpy.where(r2 < min_allowed_r2)[0]
        if not unlinked_vars.size:
            break
        first_non_linked_var_idx = unlinked_vars[0]

        selected_vars.append(first_non_linked_var_idx + var_offset)

        ref_gt = gts_012[first_non_linked_var_idx : first_non_linked_var_idx + 1, :]

        if first_non_linked_var_idx == gts_012.shape[0] - 1:
            # this is the last var of the chunk
            break

        gts_012 = gts_012[first_non_linked_var_idx + 1 :, :]

        var_offset += first_non_linked_var_idx + 1
    filtered_chunk = filtered_chunk.get_vars(selected_vars)
    return filtered_chunk, len(selected_vars), ref_gt


class _FilterLDChunkIterFactory(_FilterChunkIterFactory):
    kind = "ld_and_maf"

    def __init__(self, in_vars, filter_funct):
        self.in_vars = in_vars
        self._chunks = in_vars.iter_vars_chunks()
        self.filter_funct = filter_funct
        self.num_vars_processed = 0
        self.num_vars_kept = 0
        self._metadata = None
        self._first_processed_chunk = None

    def iter_vars_chunks(self):
        if self._first_processed_chunk is not None:
            yield self._first_processed_chunk

        ref_gt = None
        for chunk in self._chunks:
            if self._metadata is None:
                self._metadata = {
                    "samples": chunk.gts.samples,
                    "num_samples": chunk.num_samples,
                    "ploidy": chunk.gts.ploidy,
                }
            filtered_chunk, num_vars_kept, ref_gt = self.filter_funct(chunk, ref_gt)
            self.num_vars_processed += chunk.num_vars
            self.num_vars_kept += num_vars_kept
            yield filtered_chunk


def filter_by_ld_and_maf(vars, min_allowed_r2=0.1, max_allowed_maf=0.95) -> Variants:
    filter_chunk_by_maf = partial(_filter_chunk_by_maf, max_allowed_maf=max_allowed_maf)
    filter_chunk_by_ld = partial(
        _filter_chunk_by_ld,
        filter_chunk_by_maf=filter_chunk_by_maf,
        min_allowed_r2=min_allowed_r2,
    )

    chunk_factory = _FilterLDChunkIterFactory(vars, filter_chunk_by_ld)
    return Variants(
        vars_chunk_iter_factory=chunk_factory,
        desired_num_vars_per_chunk=vars.desired_num_vars_per_chunk,
    )
