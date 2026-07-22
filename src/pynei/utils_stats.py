from functools import partial

import numpy
import pandas

from pynei.config import (
    BinType,
    LINEAL,
    LOGARITHMIC,
)
from pynei.pipeline import Pipeline


def _prepare_bins(
    hist_kwargs: dict,
    range=tuple[int, int],
    default_num_bins=40,
    default_bin_type: BinType = LINEAL,
):
    num_bins = hist_kwargs.get("num_bins", default_num_bins)
    bin_type = hist_kwargs.get("bin_type", default_bin_type)

    if bin_type == LINEAL:
        bins = numpy.linspace(range[0], range[1], num_bins + 1)
    elif bin_type == LOGARITHMIC:
        if range[0] == 0:
            raise ValueError("range[0] cannot be zero for logarithmic bins")
        bins = numpy.logspace(range[0], range[1], num_bins + 1)
    return bins


def _collect_stats_from_pop_dframes(
    accumulated_result, next_result: pandas.DataFrame, hist_bins_edges: numpy.array
):
    if accumulated_result is None:
        accumulated_result = {
            "sum_per_pop": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "total_num_rows": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "hist_counts": None,
        }

    accumulated_result["sum_per_pop"] += next_result.sum(axis=0)
    accumulated_result["total_num_rows"] += next_result.shape[
        0
    ] - next_result.isna().sum(axis=0)

    this_counts = {}
    for pop, pop_stats in next_result.items():
        this_counts[pop] = numpy.histogram(pop_stats, bins=hist_bins_edges)[0]
    this_counts = pandas.DataFrame(this_counts)

    if accumulated_result["hist_counts"] is None:
        accumulated_result["hist_counts"] = this_counts
    else:
        accumulated_result["hist_counts"] += this_counts

    return accumulated_result


def _calc_stats_per_var(
    variants,
    calc_stats_for_chunk,
    get_stats_for_chunk_result,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_bins_edges = _prepare_bins(hist_kwargs, range=hist_kwargs["range"])

    collect_stats_from_pop_dframes = partial(
        _collect_stats_from_pop_dframes, hist_bins_edges=hist_bins_edges
    )

    pipeline = Pipeline(
        map_functs=[
            calc_stats_for_chunk,
            get_stats_for_chunk_result,
        ],
        reduce_funct=collect_stats_from_pop_dframes,
    )
    accumulated_result = pipeline.map_and_reduce(variants)

    mean = accumulated_result["sum_per_pop"] / accumulated_result["total_num_rows"]
    return {
        "mean": mean,
        "hist_bin_edges": hist_bins_edges,
        "hist_counts": accumulated_result["hist_counts"],
    }
