import pandas

from pynei.variants import Variants
from pynei.pipeline import Pipeline
from pynei.gt_counts import _calc_gt_is_het


def _calc_per_sample_stats_for_chunk(chunk):
    res = _calc_gt_is_het(chunk)
    num_vars = res["gt_is_missing"].shape[0]
    gt_is_missing = res["gt_is_missing"].sum(axis=0)
    gt_is_het = res["gt_is_het"].sum(axis=0)
    return {
        "num_vars": num_vars,
        "gt_is_missing": gt_is_missing,
        "gt_is_het": gt_is_het,
        "samples": chunk.gts.samples,
    }


def _reduce_per_sample_stats(accumulated, new_result):
    if accumulated is None:
        accumulated = {
            "num_vars": new_result["num_vars"],
            "gt_is_missing": new_result["gt_is_missing"],
            "gt_is_het": new_result["gt_is_het"],
            "samples": new_result["samples"],
        }
    else:
        accumulated = {
            "num_vars": accumulated["num_vars"] + new_result["num_vars"],
            "gt_is_missing": accumulated["gt_is_missing"] + new_result["gt_is_missing"],
            "gt_is_het": accumulated["gt_is_het"] + new_result["gt_is_het"],
            "samples": accumulated["samples"],
        }
    return accumulated


def _calc_final_result_per_sample_stats(accumulated_result):
    gt_is_missing = accumulated_result["gt_is_missing"] / accumulated_result["num_vars"]
    gt_is_het = accumulated_result["gt_is_het"] / accumulated_result["num_vars"]
    samples = accumulated_result["samples"]
    return pandas.DataFrame(
        {
            "missing_gt_rate": gt_is_missing,
            "obs_het_rate": gt_is_het,
        },
        index=samples,
    )


def calc_per_sample_stats(vars: Variants):
    pipeline = Pipeline(
        map_functs=[_calc_per_sample_stats_for_chunk],
        reduce_funct=_reduce_per_sample_stats,
        after_reduce_funct=_calc_final_result_per_sample_stats,
    )
    return pipeline.map_and_reduce(vars)
