from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import torch


class FaissANN:
    def __init__(self, num_clusters: int = 100, device: str = "cpu"):
        self.num_clusters = num_clusters
        self.device = device
        self._cpu_device = "cpu"
        self.index = None

    def train(self, embeddings: torch.Tensor) -> None:
        _embeddings = np.asarray(embeddings.detach().to(self._cpu_device).numpy(), dtype=np.float32)
        # faiss.normalize_L2(_embeddings)

        d = _embeddings.shape[1]

        if self.num_clusters > 0:
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.num_clusters, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(d)

        if self.device.startswith("cuda"):
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        self.index.metric_type = faiss.METRIC_INNER_PRODUCT
        self.index.train(_embeddings)
        self.index.add(_embeddings)

    def search(
        self,
        query_embeddings: torch.Tensor,
        k: int = 10,
        indices_mapping: Optional[OrderedDict[int, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        assert self.index.is_trained

        _query_embeddings = np.asarray(query_embeddings.detach().to(self._cpu_device).numpy(), dtype=np.float32)
        # faiss.normalize_L2(_query_embeddings)

        distances, raw_indices = self.index.search(_query_embeddings, k)

        if indices_mapping is not None:
            indices = np.vectorize(indices_mapping.get)(raw_indices)
        else:
            indices = None
        return distances, raw_indices, indices

    def save(self, path: str | Path = "flat.index") -> None:
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path = "flat.index") -> "FaissANN":
        self.index = faiss.read_index(str(path))
        return self
