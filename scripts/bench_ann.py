#!/usr/bin/env python3
"""Benchmark LSH ANN vs brute-force cosine similarity search."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import store


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def main():
    print("=" * 80)
    print("LSH ANN vs Brute-Force Cosine Similarity Benchmark")
    print("=" * 80)

    engine_store = store.load()
    embeddings = np.load("artifacts/embeddings.npy")
    n_tracks = len(engine_store.track_ids)
    print(f"\nDataset: {n_tracks} tracks, {embeddings.shape[1]}-d embeddings\n")

    # Random test queries
    np.random.seed(42)
    query_indices = np.random.choice(n_tracks, size=20, replace=False)

    lsh_times = []
    brute_times = []
    jaccard_scores = []
    candidates_examined = []

    for query_idx in query_indices:
        query_vector = embeddings[query_idx]

        # LSH search
        start = time.perf_counter()
        lsh_candidates = engine_store.lsh_index.query(query_vector)
        lsh_time = (time.perf_counter() - start) * 1000
        lsh_times.append(lsh_time)
        candidates_examined.append(len(lsh_candidates))

        # Brute-force search
        start = time.perf_counter()
        scores = np.dot(embeddings, query_vector)
        top_k_indices = np.argsort(-scores)[:10]
        brute_time = (time.perf_counter() - start) * 1000
        brute_times.append(brute_time)

        # Compute Jaccard similarity of top-10
        # For LSH candidates, rank by dot-product and take top 10
        lsh_scores = [(idx, np.dot(embeddings[idx], query_vector)) for idx in lsh_candidates]
        lsh_top_10 = sorted(lsh_scores, key=lambda x: x[1], reverse=True)[:10]
        lsh_set = set(idx for idx, _ in lsh_top_10)
        brute_set = set(top_k_indices)
        jaccard = jaccard_similarity(lsh_set, brute_set)
        jaccard_scores.append(jaccard)

    print("Results over 20 random queries:\n")
    print(f"  LSH Search Time:      {np.mean(lsh_times):.3f}ms (median={np.median(lsh_times):.3f}ms)")
    print(f"  Brute-Force Time:     {np.mean(brute_times):.3f}ms (median={np.median(brute_times):.3f}ms)")
    print(f"  Speedup:              {np.mean(brute_times) / np.mean(lsh_times):.1f}x")
    print()
    print(f"  LSH Candidates Examined: {np.mean(candidates_examined):.0f} ({np.mean(candidates_examined)/n_tracks*100:.1f}% of dataset)")
    print(f"  Brute-Force Candidates:  {n_tracks} (100% of dataset)")
    print()
    print(f"  Top-10 Jaccard Similarity (LSH vs Brute-Force): {np.mean(jaccard_scores):.3f}")
    print(f"    (Perfect recall = 1.0, LSH should be >= 0.8 for good approximation)\n")

    if np.mean(jaccard_scores) >= 0.8:
        print("✓ LSH approximation quality is good (Jaccard >= 0.8)")
    else:
        print("⚠ LSH approximation quality is suboptimal (Jaccard < 0.8)")
        print("  Consider increasing LSH_K or LSH_L in engine/config.py")

    print("\n" + "=" * 80)
    print(f"Note: at N={n_tracks}, brute-force is fast enough for interactive use.")
    print(f"LSH wins at ~1M+ tracks where brute-force becomes prohibitive (>100ms).")
    print("=" * 80)


if __name__ == "__main__":
    main()
