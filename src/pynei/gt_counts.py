from functools import partial

import numpy
import pandas

from pynei.config import MIN_NUM_SAMPLES_FOR_POP_STAT, DEF_POP_NAME, MISSING_ALLELE
from pynei.utils_pop import _calc_pops_idxs
from pynei.utils_stats import _calc_stats_per_var


def _calc_gt_is_missing(chunk, partial_res=None):
    res = {} if partial_res is None else partial_res
    if "gt_is_missing" in res:
        return res

    allele_is_missing = chunk.gts.missing_mask
    res["gt_is_missing"] = numpy.any(allele_is_missing, axis=2)
    return res


def _calc_gt_is_het(chunk, partial_res=None):
    res = {} if partial_res is None else partial_res
    if "gt_is_het" in res:
        return res

    res = _calc_gt_is_missing(chunk, partial_res=res)
    gt_is_missing = res["gt_is_missing"]

    gt_array = chunk.gts.gt_values
    gt_is_het = numpy.logical_not(
        numpy.all(gt_array == gt_array[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    res["gt_is_het"] = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return res


def _calc_obs_het_per_var(chunk, pops):
    res = _calc_gt_is_het(chunk)
    gt_is_het = res["gt_is_het"]
    gt_is_missing = res["gt_is_missing"]

    obs_het_per_var = {}
    called_gts_per_var = {}
    for pop_name, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        gt_is_missing_for_pop = gt_is_missing[:, pop_slice]
        num_samples = gt_is_missing_for_pop.shape[1]
        num_non_missing_per_var = num_samples - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        with numpy.errstate(invalid="ignore"):
            obs_het_per_var[pop_name] = num_vars_het_per_var / num_non_missing_per_var
        called_gts_per_var[pop_name] = num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    called_gts_per_var = pandas.DataFrame(called_gts_per_var)
    return {
        "obs_het_per_var": obs_het_per_var,
        "called_gts_per_var": called_gts_per_var,
    }


def calc_obs_het_stats_per_var(
    variants,
    pops: list[str] | None = None,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    pops = _calc_pops_idxs(pops, variants.samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(_calc_obs_het_per_var, pops=pops),
        get_stats_for_chunk_result=lambda x: x["obs_het_per_var"],
        hist_kwargs=hist_kwargs,
    )


def _count_alleles_per_var(
    chunk,
    calc_freqs: bool,
    pops: dict[str, list[int]] | None = None,
    alleles=None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
):
    gts = chunk.gts.gt_values
    missing_mask = chunk.gts.missing_mask

    alleles_in_chunk = set(numpy.unique(gts).tolist()).difference([MISSING_ALLELE])
    alleles = sorted(alleles_in_chunk)
    ploidy = chunk.ploidy

    if pops is None:
        pops = {DEF_POP_NAME: slice(None, None)}

    if alleles is not None:
        if alleles_in_chunk.difference(alleles):
            raise RuntimeError(
                f"These gts have alleles ({alleles_in_chunk}) not present in the given ones ({alleles})"
            )

    result = {}
    for pop_id, pop_slice in pops.items():
        pop_gts = gts[:, pop_slice, :]
        pop_missing_mask = missing_mask[:, pop_slice, :]
        allele_counts = numpy.empty(
            shape=(pop_gts.shape[0], len(alleles)), dtype=numpy.int32
        )
        for idx, allele in enumerate(alleles):
            is_allele = numpy.logical_and(
                pop_gts == allele, numpy.logical_not(pop_missing_mask)
            )
            allele_counts_per_row = numpy.sum(is_allele, axis=(1, 2))
            allele_counts[:, idx] = allele_counts_per_row
        allele_counts = pandas.DataFrame(allele_counts, columns=alleles)
        missing_counts = numpy.sum(pop_missing_mask, axis=(1, 2))

        result[pop_id] = {
            "allele_counts": allele_counts,
            "missing_gts_per_var": missing_counts,
        }

        if calc_freqs:
            expected_num_allelic_gts_in_snp = pop_gts.shape[1] * pop_gts.shape[2]
            num_allelic_gts_per_snp = expected_num_allelic_gts_in_snp - missing_counts
            num_allelic_gts_per_snp = num_allelic_gts_per_snp.reshape(
                (num_allelic_gts_per_snp.shape[0], 1)
            )
            allelic_freqs_per_snp = allele_counts / num_allelic_gts_per_snp
            num_gts_per_snp = (
                num_allelic_gts_per_snp.reshape((num_allelic_gts_per_snp.size,))
                / ploidy
            )
            not_enough_data = num_gts_per_snp < min_num_samples
            allelic_freqs_per_snp[not_enough_data] = numpy.nan

            result[pop_id]["allelic_freqs"] = allelic_freqs_per_snp

    return {"counts": result, "alleles": alleles_in_chunk}


def _calc_maf_per_var(
    chunk,
    pops,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
):
    res = _count_alleles_per_var(
        chunk,
        pops=pops,
        alleles=None,
        calc_freqs=True,
        min_num_samples=min_num_samples,
    )
    major_allele_freqs = {}
    for pop, pop_res in res["counts"].items():
        pop_allelic_freqs = pop_res["allelic_freqs"]
        major_allele_freqs[pop] = pop_allelic_freqs.max(axis=1)
    major_allele_freqs = pandas.DataFrame(major_allele_freqs)
    return {"major_allele_freqs_per_var": major_allele_freqs}


def calc_major_allele_stats_per_var(
    variants,
    pops: list[str] | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples = variants.samples
    pops = _calc_pops_idxs(pops, samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(
            _calc_maf_per_var, pops=pops, min_num_samples=min_num_samples
        ),
        get_stats_for_chunk_result=lambda x: x["major_allele_freqs_per_var"],
        hist_kwargs=hist_kwargs,
    )
