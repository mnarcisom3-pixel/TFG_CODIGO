from pathlib import Path
import json
import gzip

import numpy
import pandas

from pynei.variants import Variants, Genotypes, VariantsChunk
import pynei.config as config


def _create_vars_info_path(chunk_dir):
    return chunk_dir / "vars_info.parquet"


def _create_gt_path(chunk_dir):
    return chunk_dir / "gts.npy.gz"


def _create_gt_mask_path(chunk_dir):
    return chunk_dir / "gt_mask.npy.gz"


def _create_metadata_path(output_dir):
    return output_dir / "var_dir_metadata.json"


def write_vars(
    vars: Variants,
    output_dir: Path,
    numpy_array_compression_level=config.DEF_NUMPY_GZIP_COMPRESSION_LEVEL,
):
    output_dir = Path(output_dir)

    metadata = {
        "var_dir_format_version": "1.0",
        "var_chunks_metadata": [],
        "num_samples": vars.num_samples,
        "ploidy": vars.ploidy,
    }

    samples = vars.samples
    if (
        samples is None
        or (isinstance(samples, (list, tuple)) and not samples)
        or not vars.samples.shape[0]
    ):
        samples = None
    if samples is not None:
        metadata["samples"] = list(samples)

    for chunk_idx, chunk in enumerate(vars.iter_vars_chunks()):
        chunk_dir = output_dir / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir()
        chunk_metadata = {"dir": str(chunk_dir.relative_to(output_dir))}

        vars_info = chunk.vars_info
        if vars_info is not None:
            fpath = str(_create_vars_info_path(chunk_dir))
            fhand = open(fpath, "wb")
            chunk.vars_info.to_parquet(fhand)
            if (
                config.VAR_TABLE_CHROM_COL in vars_info.columns
                and config.VAR_TABLE_POS_COL in vars_info.columns
            ):
                chunk_metadata["start_chrom"] = vars_info[
                    config.VAR_TABLE_CHROM_COL
                ].iloc[0]
                chunk_metadata["start_pos"] = int(
                    vars_info[config.VAR_TABLE_POS_COL].iloc[0]
                )
                chunk_metadata["end_chrom"] = vars_info[
                    config.VAR_TABLE_CHROM_COL
                ].iloc[-1]
                chunk_metadata["end_pos"] = int(
                    vars_info[config.VAR_TABLE_POS_COL].iloc[-1]
                )
            fhand.flush()

        array = chunk.gts.gt_ma_array
        fpath = str(_create_gt_path(chunk_dir))
        with gzip.open(
            fpath,
            mode="wb",
            compresslevel=numpy_array_compression_level,
        ) as fhand:
            numpy.save(fhand, array.data)
            fhand.flush()
        fpath = str(_create_gt_mask_path(chunk_dir))
        with gzip.open(
            fpath,
            mode="wb",
            compresslevel=numpy_array_compression_level,
        ) as fhand:
            numpy.save(fhand, array.mask)
            fhand.flush()

        metadata["var_chunks_metadata"].append(chunk_metadata)

    with open(_create_metadata_path(output_dir), "wt") as fhand:
        if "samples" in metadata:
            metadata["samples"] = list(metadata["samples"])
        json.dump(metadata, fhand)
        fhand.flush()


class VariantsDir:
    def __init__(self, dir):
        self.dir = Path(dir)
        self.metadata = json.load(open(_create_metadata_path(self.dir), "rt"))
        if "samples" in self.metadata:
            self.samples = numpy.array(self.metadata["samples"])
        self.num_samples = self.metadata["num_samples"]
        self.ploidy = int(self.metadata["ploidy"])
        self._chunks_metadata = self.metadata["var_chunks_metadata"]

    def _get_metadata(self):
        return self.metadata

    def iter_vars_chunks(self):
        samples = None
        with open(_create_metadata_path(self.dir), "rt") as fhand:
            metadata = json.load(fhand)
            if "samples" in metadata:
                samples = list(metadata["samples"])

        for chunk_metadata in self._chunks_metadata:
            chunk_kwargs = {}
            chunk_dir = self.dir / chunk_metadata["dir"]
            path = _create_vars_info_path(chunk_dir)
            if path.exists():
                chunk_kwargs["vars_info"] = pandas.read_parquet(path)

            path = _create_gt_path(chunk_dir)
            if path.exists():
                gts = numpy.load(gzip.open(path, "rb"))
                mask = numpy.load(gzip.open(_create_gt_mask_path(chunk_dir), "rb"))
                gts = numpy.ma.masked_array(gts, mask)
                gts = Genotypes(numpy.ma.array(gts), samples=samples)
                chunk_kwargs["gts"] = gts

            yield VariantsChunk(**chunk_kwargs)


def load_vars(vars_dir: Path, desired_num_vars_per_chunk=config.DEF_NUM_VARS_PER_CHUNK):
    return Variants(
        vars_chunk_iter_factory=VariantsDir(vars_dir),
        desired_num_vars_per_chunk=desired_num_vars_per_chunk,
    )
