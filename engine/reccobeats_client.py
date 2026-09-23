"""HTTP client for ReccoBeats API.

Free API (https://api.reccobeats.com/v1) providing live track search,
recommendations, and audio features. No authentication required.
"""

import requests
from typing import Optional

BASE_URL = "https://api.reccobeats.com/v1"
TIMEOUT_SECONDS = 8


class ReccoBeatsError(Exception):
    """Raised on any ReccoBeats API failure."""
    pass


_session = requests.Session()


def search_tracks(query: str, limit: int) -> list[dict]:
    """Search for tracks by title/artist.

    Args:
        query: Search text (title or artist)
        limit: Max results to return (1-50)

    Returns:
        List of track dicts: {id, trackTitle, artists, durationMs, popularity, href, ...}

    Raises:
        ReccoBeatsError: On API failure, timeout, or network error.
    """
    try:
        resp = _session.get(
            f"{BASE_URL}/track/search",
            params={"searchText": query, "limit": limit},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("content", [])
    except requests.exceptions.RequestException as e:
        raise ReccoBeatsError(f"Failed to search tracks: {e}")


def get_recommendations(
    seed_ids: list[str], size: int, target_features: Optional[dict[str, float]] = None
) -> list[dict]:
    """Get similar track recommendations using ReccoBeats' similarity engine.

    Args:
        seed_ids: 1-5 track IDs to base recommendations on
        size: Number of results (1-100)
        target_features: Optional dict with keys like 'target_valence', 'target_energy'
                        to bias the recommendations

    Returns:
        List of recommended track dicts in ranking order.

    Raises:
        ReccoBeatsError: On API failure, timeout, or network error.
    """
    params = {
        "seeds": ",".join(seed_ids),
        "size": size,
    }
    if target_features:
        params.update(target_features)

    try:
        resp = _session.get(
            f"{BASE_URL}/track/recommendation",
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("content", [])
    except requests.exceptions.RequestException as e:
        raise ReccoBeatsError(f"Failed to get recommendations: {e}")


def get_audio_features_batch(track_ids: list[str]) -> dict[str, dict]:
    """Fetch Spotify-compatible audio features for multiple tracks in one call.

    Args:
        track_ids: List of track IDs (up to ~25 recommended)

    Returns:
        Dict mapping track_id -> {danceability, energy, valence, tempo, loudness,
        speechiness, acousticness, instrumentalness, liveness, key, mode}

    Raises:
        ReccoBeatsError: On API failure, timeout, or network error.
    """
    try:
        resp = _session.get(
            f"{BASE_URL}/audio-features",
            params={"ids": ",".join(track_ids)},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return {item["id"]: item for item in data.get("content", [])}
    except requests.exceptions.RequestException as e:
        raise ReccoBeatsError(f"Failed to fetch audio features: {e}")


def get_tracks_batch(track_ids: list[str]) -> dict[str, dict]:
    """Fetch track metadata for multiple tracks in one call.

    Args:
        track_ids: List of track IDs

    Returns:
        Dict mapping track_id -> {id, trackTitle, artists, durationMs, popularity, href, ...}

    Raises:
        ReccoBeatsError: On API failure, timeout, or network error.
    """
    try:
        params = {"ids": ",".join(f"ids={tid}" for tid in track_ids)}
        resp = _session.get(
            f"{BASE_URL}/track",
            params={"ids": ",".join(track_ids)},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return {item["id"]: item for item in data.get("content", [])}
    except requests.exceptions.RequestException as e:
        raise ReccoBeatsError(f"Failed to fetch track metadata: {e}")
