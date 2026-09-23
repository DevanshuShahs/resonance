"""Hybrid ranking of recommendations combining audio features with contextual metadata.

Reranks ReccoBeats recommendations using multiple signals:
- Audio similarity (how close the audio features are)
- Popularity alignment (prefer similar popularity levels)
- Artist diversity (balance between discovering new artists and matching style)
- Duration alignment (avoid extreme outliers)
- Acoustic profile (acoustic vs. produced tracks)
"""

import numpy as np
from typing import Any


def normalize_score(values: list[float]) -> list[float]:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not values or len(values) == 0:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def compute_audio_similarity(
    track_features: dict, favorite_features_list: list[dict]
) -> float:
    """Compute cosine similarity between a track and the average of favorites.

    Uses: danceability, energy, valence, acousticness, instrumentalness.
    """
    audio_dims = ["danceability", "energy", "valence", "acousticness", "instrumentalness"]

    # Average the favorite tracks' audio vectors
    favorite_vector = np.array(
        [
            np.mean([f.get(dim, 0.5) for f in favorite_features_list])
            for dim in audio_dims
        ]
    )

    # Get this track's audio vector
    track_vector = np.array([track_features.get(dim, 0.5) for dim in audio_dims])

    # Cosine similarity
    dot_product = np.dot(favorite_vector, track_vector)
    norm_fav = np.linalg.norm(favorite_vector)
    norm_track = np.linalg.norm(track_vector)

    if norm_fav == 0 or norm_track == 0:
        return 0.5
    return float(np.clip(dot_product / (norm_fav * norm_track), 0, 1))


def compute_popularity_alignment(
    track_popularity: int, favorite_popularities: list[int]
) -> float:
    """Penalize tracks with very different popularity than favorites.

    If your favorites are all indie artists (low popularity),
    suggesting a #1 hit is a bad recommendation.
    """
    if not favorite_popularities:
        return 0.5

    avg_popularity = np.mean(favorite_popularities)
    popularity_gap = abs(track_popularity - avg_popularity)

    if avg_popularity == 0:
        avg_popularity = 50

    # Gap of 0 = score 1.0. Gap of 100 (max possible) = score 0.0
    alignment = max(0, 1 - (popularity_gap / 100))
    return float(alignment)


def compute_acoustic_alignment(
    track_acousticness: float, favorite_acousticness_list: list[float]
) -> float:
    """Penalize if recommendation has very different acoustic profile.

    If your favorites are acoustic, don't recommend heavy synth.
    If your favorites are electronic, don't recommend purely acoustic.
    """
    if not favorite_acousticness_list:
        return 0.5

    avg_acoustic = np.mean(favorite_acousticness_list)
    acoustic_gap = abs(track_acousticness - avg_acoustic)

    # Gap of 0 = score 1.0. Gap of 1.0 = score 0.0
    alignment = max(0, 1 - acoustic_gap)
    return float(alignment)


def compute_duration_alignment(
    track_duration_ms: int, favorite_durations_ms: list[int]
) -> float:
    """Penalize tracks with very different length.

    If your favorites are 3-4 min songs, don't recommend 20 min ambient pieces.
    """
    if not favorite_durations_ms:
        return 0.5

    avg_duration = np.mean(favorite_durations_ms)
    duration_ratio = track_duration_ms / avg_duration if avg_duration > 0 else 1

    # Penalize if duration is < 50% or > 200% of average
    if 0.5 <= duration_ratio <= 2.0:
        return 1.0
    else:
        gap = max(abs(duration_ratio - 1) - 0.5, 0)  # Allow 50% swing
        return float(max(0, 1 - gap))


def compute_artist_freshness(
    track_artists: list[dict], favorite_artist_names: set[str]
) -> float:
    """Score for artist discovery vs. familiarity.

    If a recommendation is by the same artist as a favorite, lower score (less discovery).
    If it's by a completely different artist, higher score (more discovery).

    This balances: you want some familiar artists, but mostly new ones.
    """
    track_artist_names = {a.get("name", "").lower() for a in track_artists}
    track_artist_names = {n for n in track_artist_names if n}

    same_artists = len(track_artist_names.intersection(favorite_artist_names))
    unknown_artists = len(track_artist_names) - same_artists

    if same_artists > 0:
        return 0.3  # Penalize same artist (seen it before)
    elif unknown_artists > 0:
        return 1.0  # Reward new artists
    else:
        return 0.5  # Neutral


def hybrid_rerank(
    recommendations: list[dict],
    favorite_tracks: list[dict],
    favorite_features: list[dict],
) -> list[tuple[dict, float]]:
    """Rerank recommendations using hybrid scoring.

    Args:
        recommendations: List of track dicts from ReccoBeats recommendation endpoint
        favorite_tracks: Original favorite tracks (with artist info)
        favorite_features: Audio features for favorite tracks

    Returns:
        List of (track_dict, hybrid_score) tuples, sorted by score descending.
        hybrid_score is 0-1, where 1.0 is the best match.
    """
    if not recommendations or not favorite_features:
        return [(t, 0.5) for t in recommendations]

    # Extract favorite metadata for comparison
    favorite_artist_names = set()
    for track in favorite_tracks:
        artists = track.get("artists", [])
        for artist in artists:
            name = artist.get("name", "").lower()
            if name:
                favorite_artist_names.add(name)

    favorite_popularities = [t.get("popularity", 50) for t in favorite_tracks]
    favorite_acousticness = [f.get("acousticness", 0.5) for f in favorite_features]
    favorite_durations = [t.get("durationMs", 180000) for t in favorite_tracks]

    scored = []

    for rec in recommendations:
        rec_id = rec.get("id")
        rec_artists = rec.get("artists", [])

        # Find audio features for this recommendation
        rec_features = next(
            (f for f in favorite_features if f.get("id") == rec_id),
            {},
        )
        if not rec_features:
            scored.append((rec, 0.5))
            continue

        # Compute component scores (all 0-1)
        audio_sim = compute_audio_similarity(rec_features, favorite_features)
        pop_align = compute_popularity_alignment(
            rec.get("popularity", 50), favorite_popularities
        )
        acoustic_align = compute_acoustic_alignment(
            rec_features.get("acousticness", 0.5), favorite_acousticness
        )
        duration_align = compute_duration_alignment(
            rec.get("durationMs", 180000), favorite_durations
        )
        artist_fresh = compute_artist_freshness(rec_artists, favorite_artist_names)

        # Weighted hybrid score
        hybrid_score = (
            audio_sim * 0.35  # Audio features are most important
            + pop_align * 0.20  # Popularity alignment matters
            + acoustic_align * 0.15  # Acoustic profile matters
            + duration_align * 0.10  # Duration is a soft constraint
            + artist_fresh * 0.20  # Discovery and artist freshness
        )

        scored.append((rec, float(hybrid_score)))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
