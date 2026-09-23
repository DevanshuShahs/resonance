"""Startup loader for all search artifacts and the database.

Verifies alignment between embeddings and database: the embedding vectors
must be in the exact same row order as the database tracks (ORDER BY id),
otherwise searches will silently pair the wrong vector to the wrong track.
This loader reads track_ids.json (built by build_embeddings.py) and asserts
it matches a fresh SELECT FROM tracks ORDER BY id."""

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from engine import config, lsh_index, inverted_index


@dataclass
class EngineStore:
    track_ids: list[str]
    tracks: dict  # id -> row dict
    embeddings: np.ndarray
    conn: sqlite3.Connection
    lsh_index: "lsh_index.LSHIndex"
    inverted_index: "inverted_index.InvertedIndex"
    track_id_to_idx: dict[str, int]  # O(1) lookup: track_id -> embedding row index


def load() -> EngineStore:
    """Load all artifacts and perform alignment checks.

    Raises RuntimeError if embeddings.npy and the database are out of sync."""

    # Load embeddings + track IDs
    embeddings = np.load(config.ARTIFACTS_DIR / "embeddings.npy")
    with open(config.ARTIFACTS_DIR / "track_ids.json") as f:
        artifact_track_ids = json.load(f)

    # Load database tracks in the same order
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    db_rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
    db_track_ids = [row["id"] for row in db_rows]

    # ALIGNMENT CHECK: critical for correctness
    if artifact_track_ids != db_track_ids:
        conn.close()
        raise RuntimeError(
            "Track ID mismatch between embeddings artifact and database. "
            "embeddings.npy and track_ids.json may be stale - rerun "
            "scripts/build_embeddings.py to rebuild all artifacts."
        )

    # Build tracks dict for metadata lookups
    tracks = {row["id"]: dict(row) for row in db_rows}

    # Build O(1) track_id -> embedding row index lookup
    track_id_to_idx = {tid: i for i, tid in enumerate(artifact_track_ids)}

    # Load search indexes
    lsh_idx = lsh_index.LSHIndex.load(config.ARTIFACTS_DIR)
    inv_idx = inverted_index.InvertedIndex.load(config.ARTIFACTS_DIR)

    return EngineStore(
        track_ids=artifact_track_ids,
        tracks=tracks,
        embeddings=embeddings,
        conn=conn,
        lsh_index=lsh_idx,
        inverted_index=inv_idx,
        track_id_to_idx=track_id_to_idx,
    )
