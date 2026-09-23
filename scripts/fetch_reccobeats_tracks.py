#!/usr/bin/env python3
"""Fetch real tracks from ReccoBeats API and load into the database.

ReccoBeats is a free music database with audio features - no auth required!

Usage:
    python scripts/fetch_reccobeats_tracks.py --limit 5000
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import requests

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import vocab

RECCOBEATS_BASE = "https://api.reccobeats.com/v1"

# Genre keywords to search ReccoBeats - multiple queries per genre for diversity
SEARCH_QUERIES = [
    # Pop
    "pop", "pop music", "pop rock", "dance pop",
    # Rock
    "rock", "rock music", "hard rock", "alternative rock",
    # Jazz
    "jazz", "jazz music", "smooth jazz", "jazz fusion",
    # Hip-hop
    "hip-hop", "hip hop", "rap", "hip-hop music",
    # Country
    "country", "country music", "country rock", "americana",
    # Electronic
    "electronic", "edm", "house", "techno",
    # Indie
    "indie", "indie rock", "indie pop", "alternative",
    # Classical
    "classical", "classical music", "symphony", "piano",
    # Folk
    "folk", "folk music", "acoustic", "singer-songwriter",
    # Metal
    "metal", "heavy metal", "rock metal", "metal music",
    # Reggae
    "reggae", "reggae music", "dub", "dancehall",
    # Blues
    "blues", "blues music", "electric blues", "blues rock",
    # R&B
    "r&b", "rnb", "r&b music", "soul music",
    # Soul
    "soul", "soul music", "funk", "neo-soul",
]

# Map ReccoBeats search/genre keywords to our canonical genres
GENRE_MAPPING = {
    # Pop
    "pop": "pop",
    "pop music": "pop",
    "pop rock": "pop",
    "dance pop": "pop",
    # Rock
    "rock": "rock",
    "rock music": "rock",
    "hard rock": "rock",
    "alternative rock": "indie",
    # Jazz
    "jazz": "jazz",
    "jazz music": "jazz",
    "smooth jazz": "jazz",
    "jazz fusion": "jazz",
    # Hip-hop
    "hip-hop": "hip-hop",
    "hip hop": "hip-hop",
    "rap": "hip-hop",
    "hip-hop music": "hip-hop",
    # Country
    "country": "country",
    "country music": "country",
    "country rock": "rock",
    "americana": "country",
    # Electronic
    "electronic": "electronic",
    "edm": "electronic",
    "house": "electronic",
    "dance": "electronic",
    "techno": "electronic",
    # Indie
    "indie": "indie",
    "indie pop": "indie",
    "indie rock": "indie",
    "alternative": "indie",
    # Classical
    "classical": "classical",
    "classical music": "classical",
    "symphony": "classical",
    "piano": "classical",
    # Folk
    "folk": "folk",
    "folk music": "folk",
    "acoustic": "folk",
    "singer-songwriter": "folk",
    # Metal
    "metal": "metal",
    "heavy metal": "metal",
    "rock metal": "metal",
    "metal music": "metal",
    # Reggae
    "reggae": "reggae",
    "reggae music": "reggae",
    "dub": "reggae",
    "dancehall": "reggae",
    # Blues
    "blues": "blues",
    "blues music": "blues",
    "electric blues": "blues",
    "blues rock": "blues",
    # R&B
    "r&b": "r&b",
    "rnb": "r&b",
    "r&b music": "r&b",
    "soul music": "soul",
    # Soul
    "soul": "soul",
    "funk": "soul",
    "neo-soul": "soul",
}


def search_reccobeats(query, limit=50):
    """Search ReccoBeats for tracks.

    Args:
        query: Search query (keyword)
        limit: Max results

    Returns:
        List of track dicts
    """
    try:
        resp = requests.get(
            f"{RECCOBEATS_BASE}/track/search",
            params={"searchText": query, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("content", [])
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def get_audio_features(track_id):
    """Get audio features for a track from ReccoBeats.

    Args:
        track_id: ReccoBeats track ID

    Returns:
        Audio features dict or None
    """
    try:
        resp = requests.get(
            f"{RECCOBEATS_BASE}/track/{track_id}/audio-features",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


def normalize_audio_features(features):
    """Normalize ReccoBeats audio features to our schema.

    ReccoBeats returns Spotify-compatible audio features:
    - danceability, energy, valence, speechiness, acousticness,
      instrumentalness, liveness: [0, 1]
    - tempo: > 0
    - loudness: typically [-60, 5]
    - key: [0, 11]
    - mode: 0 or 1
    - time_signature: typically 3-7 (ReccoBeats doesn't provide this)
    - popularity: [0, 100] (in track metadata, not audio features)
    """
    if not features:
        return None

    return {
        "danceability": max(0, min(1, features.get("danceability", 0.5))),
        "energy": max(0, min(1, features.get("energy", 0.5))),
        "valence": max(0, min(1, features.get("valence", 0.5))),
        "tempo": max(1, features.get("tempo", 120)),
        "loudness": max(-60, min(5, features.get("loudness", -5))),
        "speechiness": max(0, min(1, features.get("speechiness", 0))),
        "acousticness": max(0, min(1, features.get("acousticness", 0))),
        "instrumentalness": max(0, min(1, features.get("instrumentalness", 0))),
        "liveness": max(0, min(1, features.get("liveness", 0))),
        "duration_ms": features.get("durationMs", 180000),
        "key": features.get("key", 0) % 12,
        "mode": features.get("mode", 1),
        "time_signature": max(3, min(7, features.get("timeSignature", 4))),
    }


def infer_genre(search_genre):
    """Infer genre from search query."""
    if search_genre in GENRE_MAPPING:
        return GENRE_MAPPING[search_genre]
    return "indie"  # Default


def fetch_and_load_tracks(limit=5000, db_path=None):
    """Fetch tracks from ReccoBeats and load into database.

    Args:
        limit: max number of tracks to fetch
        db_path: path to SQLite database
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "db" / "resonance.db"

    # Initialize database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Create schema
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        cursor.executescript(f.read())

    tracks_added = 0
    tracks_rejected = 0
    seen_ids = set()

    print(f"Fetching tracks from ReccoBeats (limit: {limit})...")
    print("(No authentication required - ReccoBeats is free!)\n")

    for search_genre in SEARCH_QUERIES:
        if tracks_added >= limit:
            break

        print(f"  Searching: {search_genre}...", end=" ", flush=True)
        batch_added = 0

        # Search ReccoBeats for tracks
        tracks = search_reccobeats(search_genre, limit=50)

        for track in tracks:
            if tracks_added >= limit:
                break

            track_id = track.get("id")
            if not track_id or track_id in seen_ids:
                continue

            seen_ids.add(track_id)

            # Extract basic info
            title = track.get("trackTitle", "Unknown")
            artists = track.get("artists", [])
            if artists and len(artists) > 0:
                artist_name = artists[0].get("name", "Unknown")
            else:
                artist_name = "Unknown"

            popularity = track.get("popularity", 50)

            # Fetch audio features
            audio_features = get_audio_features(track_id)
            if not audio_features:
                tracks_rejected += 1
                continue

            # Normalize features
            features = normalize_audio_features(audio_features)
            if not features:
                tracks_rejected += 1
                continue

            # Determine genre
            genre = infer_genre(search_genre)
            if genre not in vocab.GENRES:
                genre = "indie"

            # Insert into database
            try:
                cursor.execute(
                    """
                    INSERT INTO tracks (
                        id, title, artist, genre, mood_tags,
                        danceability, energy, valence, tempo, loudness,
                        speechiness, acousticness, instrumentalness, liveness,
                        duration_ms, key, mode, time_signature, popularity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        track_id,
                        title,
                        artist_name,
                        genre,
                        "",  # mood_tags empty
                        features["danceability"],
                        features["energy"],
                        features["valence"],
                        features["tempo"],
                        features["loudness"],
                        features["speechiness"],
                        features["acousticness"],
                        features["instrumentalness"],
                        features["liveness"],
                        features["duration_ms"],
                        features["key"],
                        features["mode"],
                        features["time_signature"],
                        popularity,
                    ),
                )
                tracks_added += 1
                batch_added += 1
            except sqlite3.IntegrityError:
                continue
            except Exception as e:
                print(f"\nError inserting {title}: {e}")
                tracks_rejected += 1

        print(f"{batch_added} added")

    conn.commit()
    conn.close()

    print(f"\n✓ Loaded {tracks_added} real tracks from ReccoBeats")
    print(f"✗ Rejected {tracks_rejected} tracks")
    print(f"\nNext steps:")
    print(f"  python scripts/build_embeddings.py")
    print(f"  uvicorn api.main:app --port 8000")
    print(f"\nYour recommendation engine now uses REAL MUSIC! 🎵")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ReccoBeats tracks and load into database")
    parser.add_argument("--limit", type=int, default=5000, help="Max tracks to fetch")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    fetch_and_load_tracks(limit=args.limit, db_path=args.db)
