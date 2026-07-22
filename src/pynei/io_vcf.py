from pathlib import Path
import array
from enum import Enum
import gzip
import functools
import itertools

import numpy
import pandas

from pynei.variants import Variants, VariantsChunk, Genotypes
from pynei import config
from pynei.config import (
    MISSING_ALLELE,
    MAX_ALLELE_NUMBER,
    PYTHON_ARRAY_TYPE,
    BYTE_SIZE_OF_INT,
)

VCF_SAMPLE_LINE_ITEMS = [
    "#CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "QUAL",
    "FILTER",
    "INFO",
    "FORMAT",
]


class _VCFKind(Enum):
    VCF = "vcf"
    GzippedVCF = "GzippedVCF"


def _guess_vcf_file_kind(path: Path):
    is_gzipped = False
    with path.open("rb") as fhand:
        start = fhand.read(2)
        if start[:1] == b"#":
            return _VCFKind.VCF
        elif start[:2] == b"\x1f\x8b":
            is_gzipped = True

    if not is_gzipped:
        raise ValueError(
            "Invalid VCF file, it does not start with # and its not gzipped"
        )

    with gzip.open(path) as fhand:
        start = fhand.read(1)
        if start[:1] == b"#":
            return _VCFKind.GzippedVCF
    raise ValueError("Invalid VCF gzipped file, it does not start with #")


def _parse_metadata(fhand):
    metadata = {}
    for line in fhand:
        if line.startswith(b"##"):
            pass
        elif line.startswith(b"#CHROM"):
            items = line.decode().strip().split("\t")
            if items[:9] != VCF_SAMPLE_LINE_ITEMS:
                raise ValueError(
                    "Invalid VCF file, it has an invalid sample line: {line.decode()}"
                )
            metadata["samples"] = items[9:]
            break
        else:
            raise ValueError("Invalid VCF file, it has no header")

    metadata["samples"] = numpy.array(metadata["samples"])
    num_samples = metadata["samples"].size
    metadata["num_samples"] = num_samples
    try:
        var_line = next(fhand)
    except StopIteration:
        raise ValueError("Empty VCF file, it has no variants")
    var_ = _parse_var_line(var_line, num_samples, ploidy=None)
    metadata["ploidy"] = var_["gts"].shape[1]

    return metadata


def _open_vcf(fpath):
    kind = _guess_vcf_file_kind(fpath)
    if kind == _VCFKind.GzippedVCF:
        fhand = gzip.open(fpath, mode="rb")
    else:
        fhand = fpath.open("rb")
    return fhand


@functools.lru_cache
def _parse_allele(allele):
    if allele == b".":
        return True, MISSING_ALLELE
    else:
        allele = int(allele)
    if allele > MAX_ALLELE_NUMBER:
        raise NotImplementedError(
            f"Only alleles up to {MAX_ALLELE_NUMBER} are implemented: {allele}"
        )
    return False, allele


_EXPECT_PHASED = False


@functools.lru_cache
def _parse_gt(gt):
    if _EXPECT_PHASED:
        sep, other_sep = b"|", b"/"
        is_phased = True
    else:
        sep, other_sep = b"/", b"|"
        is_phased = False
    try:
        return is_phased, tuple(map(_parse_allele, gt.split(sep)))
    except ValueError:
        pass
    is_phased = not is_phased
    return is_phased, tuple(map(_parse_allele, gt.split(other_sep)))


@functools.lru_cache
def _decode_chrom(chrom):
    return chrom.decode()


@functools.lru_cache
def _get_gt_fmt_idx(gt_fmt):
    return gt_fmt.split(b":").index(b"GT")


@functools.lru_cache
def _parse_qual(qual):
    if qual == b".":
        return numpy.nan
    return float(qual)


@functools.lru_cache
def _parse_id(id_):
    if id_ == b".":
        return None
    return id_.decode()


def _parse_var_line(line, num_samples, ploidy=None):
    fields = line.split(b"\t")
    ref = fields[3].decode()
    alt = fields[4]
    if alt != b".":
        alt = alt.decode().split(",")
        alleles = [ref] + alt
    else:
        alleles = [ref]

    gt_fmt_idx = _get_gt_fmt_idx(fields[8])

    if ploidy is None:
        alleles = _parse_gt(fields[9].split(b":")[gt_fmt_idx])
        ploidy = len(alleles)

    ref_gt_str = b"/".join([b"0"] * ploidy)
    gts = array.array(
        PYTHON_ARRAY_TYPE, bytearray(num_samples * ploidy * BYTE_SIZE_OF_INT)
    )
    missing_mask = array.array("b", bytearray(num_samples * ploidy))
    sample_idx = 0
    for gt_str in fields[9:]:
        gt_str = gt_str.split(b":")[gt_fmt_idx]
        if gt_str == ref_gt_str:
            sample_idx += ploidy
            continue
        for allele_idx, (is_missing, allele) in enumerate(_parse_gt(gt_str)[1]):
            if is_missing:
                missing_mask[sample_idx + allele_idx] = 1
            if allele != 0:
                gts[sample_idx + allele_idx] = allele
        sample_idx += ploidy
    gts = numpy.frombuffer(gts, dtype=config.GT_NUMPY_DTYPE).reshape(
        num_samples, ploidy
    )
    missing_mask = (
        numpy.frombuffer(missing_mask, dtype=numpy.int8)
        .reshape(num_samples, ploidy)
        .astype(bool)
    )
    return {
        "chrom": _decode_chrom(fields[0]),
        "pos": int(fields[1]),
        "alleles": alleles,
        "id": _parse_id(fields[2]),
        "qual": _parse_qual(fields[5]),
        "gts": gts,
        "missing_mask": missing_mask,
    }


def _read_vars(fhand, metadata):
    for line in fhand:
        if line.startswith(b"#CHROM"):
            break

    num_samples = len(metadata["samples"])
    parse_var_line = functools.partial(
        _parse_var_line, num_samples=num_samples, ploidy=metadata["ploidy"]
    )
    vars = map(parse_var_line, fhand)
    return vars


def parse_vcf(vcf_path: Path):
    fpath = Path(vcf_path)
    fhand = _open_vcf(fpath)
    metadata = _parse_metadata(fhand)

    fhand = _open_vcf(fpath)
    vars = _read_vars(fhand, metadata)

    return {"metadata": metadata, "vars": vars, "fhand": fhand}


def _parse_vcf_vars_chunk(vars_chunk, samples):
    chroms = []
    poss = []
    ids = []
    quals = []
    alleles = []
    gts = []
    missing_masks = []
    max_num_alleles = 0
    for var in vars_chunk:
        chroms.append(var["chrom"])
        poss.append(var["pos"])
        ids.append(var["id"])
        quals.append(var["qual"])
        alleles.append(var["alleles"])
        max_num_alleles = max(max_num_alleles, len(var["alleles"]))
        gts.append(var["gts"])
        missing_masks.append(var["missing_mask"])
    vars_info = pandas.DataFrame(
        {
            config.VAR_TABLE_CHROM_COL: pandas.Series(
                chroms, dtype=config.PANDAS_STR_DTYPE()
            ),
            config.VAR_TABLE_POS_COL: pandas.Series(
                poss, dtype=config.PANDAS_INT_DTYPE()
            ),
            config.VAR_TABLE_ID_COL: pandas.Series(
                ids, dtype=config.PANDAS_STR_DTYPE()
            ),
            config.VAR_TABLE_QUAL_COL: pandas.Series(
                quals, dtype=config.PANDAS_FLOAT_DTYPE()
            ),
        },
    )
    alleles = pandas.DataFrame(alleles, dtype=config.PANDAS_STR_DTYPE())
    gts = numpy.ma.array(gts, mask=missing_masks, fill_value=config.MISSING_ALLELE)
    gts.flags.writeable = False
    gts = Genotypes(gts, samples=samples, skip_mask_check=True)
    chunk = VariantsChunk(gts=gts, vars_info=vars_info, alleles=alleles)
    return chunk


class _FromVCFChunkIterFactory:
    def __init__(
        self, vcf_path, desired_num_vars_per_chunk=config.DEF_NUM_VARS_PER_CHUNK
    ):
        self.desired_num_vars_per_chunk = desired_num_vars_per_chunk
        self.vcf_path = vcf_path
        res = parse_vcf(self.vcf_path)
        self.metadata = res["metadata"]
        res["fhand"].close()

    def iter_vars_chunks(self):
        res = parse_vcf(self.vcf_path)
        fhand = res["fhand"]
        samples = self.metadata["samples"]

        vars_chunks = itertools.batched(res["vars"], self.desired_num_vars_per_chunk)
        for vars_chunk in vars_chunks:
            yield _parse_vcf_vars_chunk(vars_chunk, samples)
        fhand.close()

    def _get_metadata(self):
        return self.metadata


def vars_from_vcf(
    vcf_path: Path, desired_num_vars_per_chunk: int = config.DEF_NUM_VARS_PER_CHUNK
) -> Variants:
    chunk_factory = _FromVCFChunkIterFactory(
        vcf_path, desired_num_vars_per_chunk=desired_num_vars_per_chunk
    )
    vars = Variants(
        chunk_factory, desired_num_vars_per_chunk=desired_num_vars_per_chunk
    )

    return vars
