import struct
import numpy as np
 
import json
from typing import Iterator, List

from dataset_reader.base_reader import Query
from dataset_reader.json_reader import JSONReader
 
class BinCompoundReader(JSONReader):
    """
    A reader created specifically to read the format ftbin
    """

    VECTORS_FILE = "base.1B.fbin"
    QUERIES_FILE = "query.public.10K.fbin"
    GROUNDTRUTH_FILE = "groundtruth.public.10K.ibin"

    def read_vectors(self) -> Iterator[List[float]]:
        with open(self.path / self.VECTORS_FILE) as vectors_fp:
            nvecs, dim = np.fromfile(vectors_fp, count=2, dtype=np.int32)
            if self.dataset_offset > 0:
                vectors_fp.seek(self.dataset_offset * dim * 4 + 8)
            for i in range(nvecs):
                vector = np.fromfile(
                    vectors_fp, count=dim, dtype=np.float32)
                yield vector.tolist()
                

    def read_queries(self) -> Iterator[Query]:
        with open(self.path / self.QUERIES_FILE) as query_fp, \
             open(self.path / self.GROUNDTRUTH_FILE) as gt_fp:
                nvecs, dim = np.fromfile(query_fp, count=2, dtype=np.int32)
                nvecs_q, dim_q = np.fromfile(gt_fp, count=2, dtype=np.int32)

                if nvecs != nvecs_q:
                    raise ValueError("Queries and groundtruth files have different number of vectors")
                
                for i in range(nvecs):
                    vector = np.fromfile(
                        query_fp, count=dim, dtype=np.float32)
                    exp_vector_idxes = np.fromfile(
                        gt_fp, count=dim_q, dtype=np.int32)
                    
                    yield Query(
                        vector=vector.T.tolist(),
                        sparse_vector=None,
                        meta_conditions=None,
                        expected_result=exp_vector_idxes.tolist(),
                    )
 
