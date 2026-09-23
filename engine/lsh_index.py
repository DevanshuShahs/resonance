"""Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search.

Random-hyperplane LSH: for each of L independent hash tables, we generate K
random hyperplanes in the embedding space. For each vector, the bucket key is
the sign pattern (which side of each hyperplane). At query time, we retrieve
all vectors in the query's bucket(s), optionally widening via Hamming-neighbor
multiprobe if not enough candidates."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from engine import config


class LSHIndex:
    def __init__(self, hyperplanes: np.ndarray, buckets: list[dict]):
        """
        Args:
            hyperplanes: shape (L, K, D) - L tables, K hyperplanes each, D dimensions
            buckets: list of L dicts, each {bucket_key_as_str: [row_indices]}
        """
        self.hyperplanes = hyperplanes
        self.buckets = buckets
        self.L = len(hyperplanes)

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "LSHIndex":
        """Generate hyperplanes and hash all embeddings.

        Args:
            embeddings: shape (N, D) float array, already unit-normalized

        Returns: LSHIndex ready for querying"""
        N, D = embeddings.shape
        L, K = config.LSH_L, config.LSH_K

        rng = np.random.default_rng(config.LSH_SEED)
        hyperplanes = rng.normal(size=(L, K, D)).astype(np.float32)
        hyperplanes /= np.linalg.norm(hyperplanes, axis=2, keepdims=True)

        buckets = []
        for table_idx in range(L):
            table_buckets = {}
            for row_idx in range(N):
                projections = np.dot(hyperplanes[table_idx], embeddings[row_idx])
                bucket_key = tuple(int(p > 0) for p in projections)
                bucket_str = "".join(str(b) for b in bucket_key)
                if bucket_str not in table_buckets:
                    table_buckets[bucket_str] = []
                table_buckets[bucket_str].append(row_idx)
            buckets.append(table_buckets)

        return cls(hyperplanes, buckets)

    def query(self, vector: np.ndarray) -> set[int]:
        """Find candidate neighbors via LSH with multiprobe fallback.

        Args:
            vector: shape (D,) float array, unit-normalized

        Returns: set of row indices to consider for ranking"""
        candidates = set()

        for table_idx in range(self.L):
            projections = np.dot(self.hyperplanes[table_idx], vector)
            bucket_key = tuple(int(p > 0) for p in projections)

            for radius in range(config.MULTIPROBE_MAX_RADIUS + 1):
                if radius == 0:
                    bucket_str = "".join(str(b) for b in bucket_key)
                    if bucket_str in self.buckets[table_idx]:
                        candidates.update(self.buckets[table_idx][bucket_str])
                else:
                    for flipped_positions in combinations(range(len(bucket_key)), radius):
                        neighbor_key = list(bucket_key)
                        for pos in flipped_positions:
                            neighbor_key[pos] = 1 - neighbor_key[pos]
                        neighbor_str = "".join(str(b) for b in neighbor_key)
                        if neighbor_str in self.buckets[table_idx]:
                            candidates.update(self.buckets[table_idx][neighbor_str])

                if len(candidates) >= config.MIN_CANDIDATES:
                    break

            if len(candidates) >= config.MIN_CANDIDATES:
                break

        return candidates

    def save(self, artifacts_dir: Path) -> None:
        """Persist hyperplanes and buckets."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifacts_dir / "hyperplanes.npy", self.hyperplanes)
        with open(artifacts_dir / "buckets.json", "w") as f:
            json.dump(self.buckets, f)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "LSHIndex":
        """Load persisted index."""
        hyperplanes = np.load(artifacts_dir / "hyperplanes.npy")
        with open(artifacts_dir / "buckets.json") as f:
            buckets = json.load(f)
        return cls(hyperplanes, buckets)
