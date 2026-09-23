"""Offline build of embedding artifacts.

Reads all tracks from the database (in ID order), builds text descriptors and
audio vectors, computes embeddings, and persists artifacts. Run this once after
loading the database. The artifacts are:

- artifacts/embeddings.npy: (N, 393) float array
- artifacts/track_ids.json: list of track IDs in the same row order
- artifacts/norm_stats.json: min/max bounds for audio feature normalization

Later components (engine/store.py) will assert that track_ids matches the DB
loaded at query time, so embedding vectors never drift from their tracks."""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import audio_norm, embeddings, text_descriptor, config, lsh_index, inverted_index


def build():
    db_path = config.DB_PATH
    artifacts_dir = config.ARTIFACTS_DIR

    print(f"Reading tracks from {db_path}...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    rows = [dict(row) for row in rows]
    conn.close()

    print(f"  loaded {len(rows)} tracks")

    # Extract IDs and texts
    track_ids = [row["id"] for row in rows]
    texts = [text_descriptor.semantic_text(row) for row in rows]

    # Normalize audio features
    print("Computing audio normalization stats...")
    stats = audio_norm.compute_stats(rows)
    audio_matrix = np.array(
        [audio_norm.apply(row, stats) for row in rows],
        dtype=np.float32,
    )
    print(f"  audio_matrix shape: {audio_matrix.shape}")

    # Embed texts
    print(f"Embedding {len(texts)} texts...")
    text_matrix = embeddings.embed_texts(texts)
    print(f"  text_matrix shape: {text_matrix.shape}")

    # Compose hybrid vectors
    print("Composing hybrid vectors...")
    combined = embeddings.compose_vectors(text_matrix, audio_matrix)
    print(f"  combined shape: {combined.shape}")

    # Verify all rows are unit-normalized
    norms = np.linalg.norm(combined, axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-6
    ), f"Not all rows unit-normalized: norm min={norms.min():.6f}, max={norms.max():.6f}"

    # Write artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.save(artifacts_dir / "embeddings.npy", combined)
    print(f"  wrote embeddings.npy")

    with (artifacts_dir / "track_ids.json").open("w") as f:
        json.dump(track_ids, f)
    print(f"  wrote track_ids.json")

    audio_norm.save_stats(stats, artifacts_dir / "norm_stats.json")
    print(f"  wrote norm_stats.json")

    # Build LSH index
    print("\nBuilding LSH ANN index...")
    lsh_idx = lsh_index.LSHIndex.build(combined)
    lsh_idx.save(artifacts_dir)
    print(f"  wrote hyperplanes.npy and buckets.json")

    # Build inverted index
    print("Building BM25 inverted index...")
    token_lists = [text_descriptor.lexical_tokens(row) for row in rows]
    inv_idx = inverted_index.InvertedIndex.build(token_lists)
    inv_idx.save(artifacts_dir)
    print(f"  wrote inverted_index.json")

    print("\nBuild complete.")
    print(f"  embeddings: {combined.shape}")
    print(f"  track_ids: {len(track_ids)}")
    print(f"  LSH index: L={config.LSH_L}, K={config.LSH_K}")
    print(f"  inverted index: {len(inv_idx.postings)} terms")


if __name__ == "__main__":
    build()
