"""SQL WHERE clause builder with a relaxation ladder.

Turns a ParsedQuery into a prefilter result that gracefully degrades if too
restrictive: full filters -> genre-only -> unrestricted."""

import sqlite3
from dataclasses import dataclass, field
from typing import List, Tuple

from engine import config, query_parser

AudioFilter = Tuple[str, str, float]
RELAXATION_LEVELS = ["full", "genre_only", "unrestricted"]


@dataclass
class PrefilterResult:
    ids: set[str]
    level: str
    dropped_genres: List[str] = field(default_factory=list)
    dropped_audio_filters: List[AudioFilter] = field(default_factory=list)


def _validate_filter(feature: str, op: str) -> None:
    """Guard against SQL injection - only allow known features/operators."""
    if feature not in config.AUDIO_FEATURES:
        raise ValueError(f"unknown feature: {feature}")
    if op not in (">", "<"):
        raise ValueError(f"unknown operator: {op}")


def _build_where(parsed: query_parser.ParsedQuery, level: str) -> Tuple[str, List]:
    """Build WHERE clause and parameters for a given relaxation level.

    level="full": genre IN (...) AND all audio_filters (audio filters ANDed together).
    level="genre_only": genre IN (...) only.
    level="unrestricted": empty WHERE clause."""

    if level == "unrestricted":
        return "", []

    conditions = []
    params = []

    if parsed.genres:
        placeholders = ",".join("?" * len(parsed.genres))
        conditions.append(f"genre IN ({placeholders})")
        params.extend(parsed.genres)

    if level == "full":
        for feature, op, threshold in parsed.audio_filters:
            _validate_filter(feature, op)
            conditions.append(f"{feature} {op} ?")
            params.append(threshold)

    where = " AND ".join(conditions) if conditions else ""
    return where, params


def prefilter_ids(
    conn: sqlite3.Connection, parsed: query_parser.ParsedQuery
) -> PrefilterResult:
    """Try relaxation levels in order until we get >= MIN_POOL ids.

    Returns the ids at the first level that reaches MIN_POOL, or
    "unrestricted" if even that doesn't."""
    for level in RELAXATION_LEVELS:
        where, params = _build_where(parsed, level)
        query = f"SELECT id FROM tracks{' WHERE ' + where if where else ''}"
        rows = conn.execute(query, params).fetchall()
        ids = {row[0] for row in rows}

        if len(ids) >= config.MIN_POOL or level == "unrestricted":
            dropped_genres = [] if level != "genre_only" else parsed.genres
            dropped_filters = [] if level != "full" else parsed.audio_filters
            return PrefilterResult(
                ids=ids,
                level=level,
                dropped_genres=dropped_genres,
                dropped_audio_filters=dropped_filters,
            )

    return PrefilterResult(ids=set(), level="unrestricted")


def satisfied_dropped_count(
    track: dict, result: PrefilterResult
) -> int:
    """Count how many of the dropped predicates this track still satisfies.

    Used to award a RELAXED_PREDICATE_BONUS to tracks that happen to match
    filters the relaxation ladder had to drop."""
    count = 0

    for genre in result.dropped_genres:
        if track["genre"] == genre:
            count += 1

    for feature, op, threshold in result.dropped_audio_filters:
        value = float(track[feature])
        if (op == ">" and value > threshold) or (op == "<" and value < threshold):
            count += 1

    return count
