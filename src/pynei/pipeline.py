from typing import Callable, Iterator
import functools

import threaded_map_reduce
from pynei.config import MAP_REDUCE_CHUNK_SIZE


class _ChunkProcessor:
    def __init__(self, map_functs):
        self.map_functs = map_functs

    def __call__(self, item):
        processed_item = item
        for one_funct in self.map_functs:
            processed_item = one_funct(processed_item)
        return processed_item


class Pipeline:
    def __init__(
        self,
        map_functs: list[Callable] | None = None,
        reduce_funct: Callable = None,
        reduce_initializer=None,
        after_reduce_funct: Callable = None,
    ):
        self.map_functs = map_functs
        self.reduce_funct = reduce_funct
        self.reduce_initializer = reduce_initializer
        self.after_reduce_funct = after_reduce_funct

    def append_map_funct(self, map_funct: Callable):
        self.map_functs.append(map_funct)

    def set_reduce_funct(self, reduce_funct: Callable, reduce_initializer=None):
        self.reduce_funct = reduce_funct
        self.reduce_initializer = reduce_initializer

    def _process_vars(
        self, vars, num_processes: int = 1, map_reduce_chunk_size=MAP_REDUCE_CHUNK_SIZE
    ):
        process_chunk = _ChunkProcessor(self.map_functs)

        use_multiprocessing = num_processes > 1

        if use_multiprocessing:
            if self.reduce_funct:
                result = threaded_map_reduce.map_reduce(
                    map_fn=process_chunk,
                    reduce_fn=self.reduce_funct,
                    iterable=vars.iter_vars_chunks(),
                    num_computing_threads=num_processes,
                    chunk_size=map_reduce_chunk_size,
                )
            else:
                result = threaded_map_reduce.map(
                    map_fn=process_chunk,
                    items=vars.iter_vars_chunks(),
                    num_computing_threads=num_processes,
                    chunk_size=map_reduce_chunk_size,
                )
        else:
            processed_chunks = map(process_chunk, vars.iter_vars_chunks())
            result = processed_chunks
            if self.reduce_funct is not None:
                result = functools.reduce(
                    self.reduce_funct, processed_chunks, self.reduce_initializer
                )

        if self.after_reduce_funct is not None:
            result = self.after_reduce_funct(result)

        return result

    def map_chunks(self, vars, num_processes: int = 1) -> Iterator:
        if self.reduce_funct is not None or self.reduce_initializer is not None:
            raise ValueError(
                "For mapping reduce_funct and reduce_initializer must be None"
            )
        return self._process_vars(vars, num_processes)

    def map_and_reduce(self, vars, num_processes: int = 1):
        if self.reduce_funct is None:
            raise ValueError("For mapping and reducing reduce_funct must be set")
        return self._process_vars(vars, num_processes)
