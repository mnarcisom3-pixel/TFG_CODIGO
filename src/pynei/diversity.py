from functools import partial
from typing import Sequence

import numpy
import pandas

from pynei.config import (
    MIN_NUM_SAMPLES_FOR_POP_STAT,
    DEF_POLY_THRESHOLD,
)
from pynei.gt_counts import _count_alleles_per_var, _calc_maf_per_var
from pynei.utils_pop import _calc_pops_idxs
from pynei.utils_stats import _calc_stats_per_var
from pynei.pipeline import Pipeline


def _calc_exp_het_per_var(
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None
):
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _count_alleles_per_var(
        chunk,
        pops=pops,
        calc_freqs=True,
        min_num_samples=min_num_samples,
    )

    sorted_pops = sorted(pops.keys())

    missing_allelic_gts = {
        pop_id: res["counts"][pop_id]["missing_gts_per_var"] for pop_id in sorted_pops
    }
    missing_allelic_gts = pandas.DataFrame(missing_allelic_gts, columns=sorted_pops)

    exp_het = {}
    for pop_id in sorted_pops:
        allele_freqs = res["counts"][pop_id]["allelic_freqs"].values
        exp_het[pop_id] = 1 - numpy.sum(allele_freqs**ploidy, axis=1)
    exp_het = pandas.DataFrame(exp_het, columns=sorted_pops)

    return {"exp_het": exp_het, "missing_allelic_gts": missing_allelic_gts}


def _calc_unbiased_exp_het_per_var(
    chunk, pops, min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT, ploidy=None
):
    "Calculated using Unbiased Heterozygosity (Codom Data) Genalex formula"
    if ploidy is None:
        ploidy = chunk.ploidy

    res = _calc_exp_het_per_var(
        chunk,
        pops=pops,
        min_num_samples=min_num_samples,
        ploidy=ploidy,
    )
    exp_het = res["exp_het"]

    missing_allelic_gts = res["missing_allelic_gts"]

    num_allelic_gtss = []
    for pop in missing_allelic_gts.columns:
        pop_slice = pops[pop]
        if (
            isinstance(pop_slice, slice)
            and pop_slice.start is None
            and pop_slice.stop is None
            and pop_slice.step is None
        ):
            num_allelic_gts = chunk.num_samples * ploidy
        else:
            num_allelic_gts = len(pops[pop]) * ploidy
        num_allelic_gtss.append(num_allelic_gts)
    num_exp_allelic_gts_per_pop = numpy.array(num_allelic_gtss)
    num_called_allelic_gts_per_snp = (
        num_exp_allelic_gts_per_pop[numpy.newaxis, :] - missing_allelic_gts
    )
    num_samples = num_called_allelic_gts_per_snp / ploidy

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return {
        "exp_het": unbiased_exp_het,
        "missing_allelic_gts": missing_allelic_gts,
    }


def calc_exp_het_stats_per_var(
    variants,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
    ploidy=None,
    hist_kwargs=None,
    unbiased=True,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples = variants.samples
    pops = _calc_pops_idxs(pops, samples)

    if unbiased:
        calc_het_funct = _calc_unbiased_exp_het_per_var
    else:
        calc_het_funct = _calc_exp_het_per_var

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(
            calc_het_funct,
            pops=pops,
            min_num_samples=min_num_samples,
            ploidy=ploidy,
        ),
        get_stats_for_chunk_result=lambda x: x["exp_het"],
        hist_kwargs=hist_kwargs,
    )


def _calc_num_poly_vars(
    chunk,
    poly_threshold=DEF_POLY_THRESHOLD,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
):
    res = _calc_maf_per_var(
        chunk,
        pops=pops,
        min_num_samples=min_num_samples,
    )
    mafs = res["major_allele_freqs_per_var"]

    num_not_nas = mafs.notna().sum(axis=0)
    num_variable = (mafs < 1).sum(axis=0)
    num_poly = (mafs < poly_threshold).sum(axis=0)
    res = {
        "num_poly": num_poly,
        "num_variable": num_variable,
        "tot_num_variants_with_data": num_not_nas,
    }
    return res


def _accumulate_pop_sums(
    accumulated_result: pandas.DataFrame | None, next_result: pandas.DataFrame
):
    if accumulated_result is None:
        accumulated_result = next_result
    else:
        accumulated_result = {
            param: accumulated_result[param] + values
            for param, values in next_result.items()
        }
    return accumulated_result


def calc_poly_vars_ratio_per_var(
    variants,
    poly_threshold=DEF_POLY_THRESHOLD,
    pops: dict[str, Sequence[str] | Sequence[int]] | None = None,
    min_num_samples=MIN_NUM_SAMPLES_FOR_POP_STAT,
):
    samples = variants.samples
    pops = _calc_pops_idxs(pops, samples)

    calc_num_poly_vars = partial(
        _calc_num_poly_vars,
        poly_threshold=poly_threshold,
        pops=pops,
        min_num_samples=min_num_samples,
    )

    pipeline = Pipeline(
        map_functs=[calc_num_poly_vars],
        reduce_funct=_accumulate_pop_sums,
    )
    res = pipeline.map_and_reduce(variants)

    num_poly = res["num_poly"]
    num_variable = res["num_variable"]
    num_not_nas = res["tot_num_variants_with_data"]

    poly_ratio = num_poly / num_not_nas
    poly_ratio2 = num_poly / num_variable

    res = {
        "num_poly": num_poly,
        "poly_ratio": poly_ratio,
        "poly_ratio_over_variables": poly_ratio2,
        "num_variable": num_variable,
        "tot_num_variants_with_data": num_not_nas,
    }
    return res
