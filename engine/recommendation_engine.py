"""Live recommendation engine using audio feature similarity.

Instead of using ReccoBeats' server-side recommendations, we:
1. Generate smart search queries from favorite tracks
2. Search ReccoBeats for candidate tracks
3. Compute our own similarity scores using audio feature vectors
4. Re-rank with contextual metadata (popularity, artist, acoustic profile)

This gives us full control over recommendations while using live data.
"""

import numpy as np
from typing import Optional
from engine import reccobeats_client, hybrid_ranking


def generate_search_queries(track_title: str, artist_name: str, mood: Optional[str] = None) -> list[str]:
    """Generate multiple search queries to find similar tracks.

    Helps discover related tracks by searching for:
    - Track title
    - Artist name
    - Title + mood (if provided)
    - Genre-adjacent keywords based on track characteristics
    """
    queries = []

    # Core queries
    if track_title:
        queries.append(track_title)
    if artist_name:
        queries.append(artist_name)

    # Mood-based queries
    if mood:
        mood_keywords = {
            "happy": "upbeat,feel-good,positive",
            "sad": "melancholic,emotional,sad",
            "energetic": "high energy,energetic,intense",
            "chill": "relaxing,mellow,chill",
            "aggressive": "aggressive,intense,hard",
            "romantic": "romantic,love,acoustic",
            "melancholic": "melancholic,emotional,dark",
            "uplifting": "uplifting,inspiring,positive",
            "dreamy": "dreamy,atmospheric,ambient",
            "tense": "tense,dark,intense",
            "nostalgic": "nostalgic,retro,classic",
            "playful": "playful,fun,upbeat",
        }
        if mood in mood_keywords:
            queries.append(f"{mood_keywords[mood]}")

    # Artist + mood combo
    if artist_name and mood:
        queries.append(f"{artist_name} {mood}")

    # Title variations
    if track_title:
        first_word = track_title.split()[0] if track_title else ""
        if len(first_word) > 3:
            queries.append(first_word)

    return queries[:5]  # Limit to 5 queries


def compute_audio_vector(features: dict) -> np.ndarray:
    """Convert audio features dict to a normalized vector."""
    dims = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
        "tempo",  # Normalize tempo: 0-200 BPM range
        "loudness",  # -60 to 0 range
    ]

    vector = []
    for dim in dims:
        value = features.get(dim, 0.5)

        # Normalize tempo (0-200 BPM -> 0-1)
        if dim == "tempo":
            value = min(value / 200.0, 1.0)
        # Normalize loudness (-60 to 0 -> 0-1)
        elif dim == "loudness":
            value = (value + 60) / 60.0
            value = min(max(value, 0), 1.0)

        vector.append(value)

    vector = np.array(vector, dtype=np.float32)

    # Unit normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def compute_similarity_scores(
    candidate_tracks: list[dict],
    favorite_features: list[dict],
) -> list[float]:
    """Compute cosine similarity between candidates and the centroid of favorites.

    Args:
        candidate_tracks: Recommended tracks from ReccoBeats search
        favorite_features: Audio features for favorite tracks

    Returns:
        List of similarity scores (0-1) for each candidate, same length as candidate_tracks
    """
    if not favorite_features or not candidate_tracks:
        return [0.5] * len(candidate_tracks)

    # Compute centroid of favorite tracks
    favorite_vectors = [compute_audio_vector(f) for f in favorite_features]
    centroid = np.mean(favorite_vectors, axis=0)

    # Normalize centroid
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    # Compute similarity for each candidate
    similarities = []
    for track in candidate_tracks:
        # Get features from the track dict (should be populated by caller)
        features = track.get("_audio_features", {})
        if not features:
            similarities.append(0.5)
            continue

        candidate_vector = compute_audio_vector(features)

        # Cosine similarity
        similarity = float(np.dot(centroid, candidate_vector))
        similarities.append(max(0, similarity))  # Clip to 0-1

    return similarities


def search_for_candidates(
    favorite_tracks: list[dict], limit: int = 50
) -> list[dict]:
    """Search ReccoBeats for candidate tracks using favorite track metadata.

    Generates queries from favorite track titles, artists, and moods,
    then performs multiple searches to get a diverse candidate pool.

    Args:
        favorite_tracks: List of favorite track dicts with id, trackTitle, artists
        limit: Total candidate tracks to gather

    Returns:
        List of candidate track dicts from ReccoBeats
    """
    all_candidates = {}  # Use dict to deduplicate by ID

    for fav_track in favorite_tracks:
        title = fav_track.get("trackTitle", "")
        artists = fav_track.get("artists", [])
        artist_name = artists[0].get("name", "") if artists else ""

        queries = generate_search_queries(title, artist_name)

        for query in queries:
            try:
                results = reccobeats_client.search_tracks(query, limit=20)
                for track in results:
                    track_id = track.get("id")
                    if track_id:
                        all_candidates[track_id] = track
            except reccobeats_client.ReccoBeatsError:
                continue

    # Return as list, limited to requested size
    candidates = list(all_candidates.values())
    return candidates[:limit]


def recommend(
    favorite_track_ids: list[str],
    favorite_tracks: list[dict],
    favorite_features: list[dict],
    mood: Optional[str],
    top_k: int = 10,
) -> list[tuple[dict, float]]:
    """Recommend tracks using our own similarity algorithm.

    Flow:
    1. Search ReccoBeats for candidates using favorite metadata
    2. Fetch audio features for all candidates
    3. Compute cosine similarity scores
    4. Apply hybrid ranking with contextual metadata
    5. Return top-K results

    Args:
        favorite_track_ids: IDs of favorite tracks (to exclude from results)
        favorite_tracks: Full track dicts for favorites
        favorite_features: Audio features for favorite tracks
        mood: Optional mood for query generation
        top_k: Number of results to return

    Returns:
        List of (track_dict, hybrid_score) tuples, sorted by score descending
    """
    # Search for candidates
    candidates = search_for_candidates(favorite_tracks, limit=top_k * 3)

    if not candidates:
        return []

    # Fetch audio features for all candidates
    candidate_ids = [t.get("id") for t in candidates if t.get("id")]
    try:
        features_map = reccobeats_client.get_audio_features_batch(candidate_ids)
    except reccobeats_client.ReccoBeatsError:
        return []

    # Attach audio features to candidates
    for track in candidates:
        track_id = track.get("id")
        track["_audio_features"] = features_map.get(track_id, {})

    # Compute similarity scores
    similarities = compute_similarity_scores(candidates, favorite_features)

    # Attach similarity scores
    for track, similarity in zip(candidates, similarities):
        track["_similarity_score"] = similarity

    # Apply hybrid ranking
    scored = hybrid_ranking.hybrid_rerank(candidates, favorite_tracks, favorite_features)

    # Filter out favorite tracks from results
    favorite_ids_set = set(favorite_track_ids)
    scored = [(t, s) for t, s in scored if t.get("id") not in favorite_ids_set]

    return scored[:top_k]
