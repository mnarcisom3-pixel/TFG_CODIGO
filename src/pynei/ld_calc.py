import numpy

DDOF = 1


def _calc_maf_from_012_gts(gts: numpy.array):
    counts = {}
    for gt in [0, 1, 2]:
        counts[gt] = numpy.sum(gts == gt, axis=1)

    num_gts_per_var = counts[0] + counts[1] + counts[2]
    counts_major_allele = numpy.maximum(counts[0], counts[2])
    freqs_major_allele = counts_major_allele / num_gts_per_var
    freqs_het = counts[1] / num_gts_per_var
    maf_per_var = freqs_major_allele + 0.5 * freqs_het
    return maf_per_var


def _calc_rogers_huff_r2(
    gts1: numpy.ndarray,
    gts2: numpy.ndarray,
    check_no_mafs_above: float | None = 0.95,
    debug=False,
):
    if check_no_mafs_above is not None:
        maf_per_var = _calc_maf_from_012_gts(gts1)
        if numpy.any(maf_per_var > check_no_mafs_above):
            raise ValueError(
                f"There are variations with mafs above {check_no_mafs_above}, filter them out or modify this check"
            )

    covars = numpy.cov(gts1, gts2, ddof=DDOF)
    n_vars1 = gts1.shape[0]
    n_vars2 = gts2.shape[0]
    if debug:
        print("nvars", n_vars1, n_vars2)
    variances = numpy.diag(covars)
    vars1 = variances[:n_vars1]
    vars2 = variances[n_vars1:]
    if debug:
        print("vars1", vars1)
        print("vars2", vars2)

    covars = covars[:n_vars1, n_vars1:]
    if debug:
        print("covars", covars)

    vars1 = numpy.repeat(vars1, n_vars2).reshape((n_vars1, n_vars2))
    vars2 = numpy.tile(vars2, n_vars1).reshape((n_vars1, n_vars2))
    with numpy.errstate(divide="ignore", invalid="ignore"):
        rogers_huff_r = covars / numpy.sqrt(vars1 * vars2)
    # print(vars1)
    # print(vars2)
    return rogers_huff_r
