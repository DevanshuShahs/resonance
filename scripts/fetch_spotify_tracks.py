#!/usr/bin/env python3
"""Fetch real tracks from Spotify API and load into the database.

Usage:
    export SPOTIFY_CLIENT_ID="your_id"
    export SPOTIFY_CLIENT_SECRET="your_secret"
    python scripts/fetch_spotify_tracks.py --limit 5000
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import vocab

# Map Spotify artist genres to our canonical genres
GENRE_MAPPING = {
    "pop": "pop",
    "rock": "rock",
    "jazz": "jazz",
    "hip-hop": "hip-hop",
    "hip hop": "hip-hop",
    "rap": "hip-hop",
    "country": "country",
    "electronic": "electronic",
    "edm": "electronic",
    "house": "electronic",
    "dance": "electronic",
    "indie": "indie",
    "indie pop": "indie",
    "indie rock": "indie",
    "alternative": "indie",
    "classical": "classical",
    "folk": "folk",
    "acoustic": "folk",
    "metal": "metal",
    "heavy metal": "metal",
    "reggae": "reggae",
    "blues": "blues",
    "r&b": "r&b",
    "rnb": "r&b",
    "soul": "soul",
}

# Search queries to get diverse tracks
SEARCH_QUERIES = [
    "genre:pop",
    "genre:rock",
    "genre:jazz",
    "genre:hip-hop",
    "genre:country",
    "genre:electronic",
    "genre:indie",
    "genre:classical",
    "genre:folk",
    "genre:metal",
    "genre:reggae",
    "genre:blues",
    "genre:r&b",
    "genre:soul",
]


def get_spotify_client():
    """Initialize Spotify API client."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(
            "Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables not set"
        )
        print("Get them from https://developer.spotify.com/dashboard")
        sys.exit(1)

    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
    )


def map_genre(spotify_genres):
    """Map Spotify artist genres to our canonical genres.

    Args:
        spotify_genres: list of genre strings from Spotify

    Returns:
        canonical genre string, or None if no match
    """
    if not spotify_genres:
        return None

    for spotify_genre in spotify_genres:
        genre_lower = spotify_genre.lower().strip()
        if genre_lower in GENRE_MAPPING:
            return GENRE_MAPPING[genre_lower]

    # Fuzzy match: check if any part of a Spotify genre matches our genres
    for spotify_genre in spotify_genres:
        for key, canonical in GENRE_MAPPING.items():
            if key in spotify_genre.lower():
                return canonical

    return None


def normalize_audio_features(features):
    """Normalize Spotify audio features to our constraints.

    Spotify's audio_features endpoint returns:
    - danceability, energy, valence, speechiness, acousticness,
      instrumentalness, liveness: [0, 1]
    - tempo: > 0
    - loudness: typically [-60, 5] (in dB)
    - key: [0, 11] (pitch class)
    - mode: 0 (minor) or 1 (major)
    - time_signature: typically 3-7
    - popularity: [0, 100]
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
        "duration_ms": features.get("duration_ms", 180000),
        "key": features.get("key", 0) % 12,  # Ensure 0-11
        "mode": features.get("mode", 1),  # 0 or 1
        "time_signature": max(3, min(7, features.get("time_signature", 4))),
    }


def fetch_and_load_tracks(limit=5000, db_path=None):
    """Fetch tracks from Spotify and load into database.

    Args:
        limit: max number of tracks to fetch
        db_path: path to SQLite database (default: db/resonance.db)
    """
    if db_path is None:
        db_path = Path(__file__).parent.parent / "db" / "resonance.db"

    sp = get_spotify_client()

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

    print(f"Fetching tracks from Spotify (limit: {limit})...")

    for query in SEARCH_QUERIES:
        if tracks_added >= limit:
            break

        print(f"  Searching: {query}...", end=" ", flush=True)
        offset = 0
        batch_added = 0

        while tracks_added < limit:
            try:
                results = sp.search(q=query, type="track", limit=50, offset=offset)
            except Exception as e:
                print(f"Error: {e}")
                break

            if not results["tracks"]["items"]:
                break

            for track in results["tracks"]["items"]:
                if tracks_added >= limit:
                    break

                track_id = track["id"]
                if track_id in seen_ids:
                    continue

                seen_ids.add(track_id)

                # Extract basic info
                title = track.get("name", "Unknown")
                artist = ", ".join([a["name"] for a in track.get("artists", [])])
                popularity = track.get("popularity", 50)

                # Fetch audio features
                try:
                    audio_features = sp.audio_features(track_id)[0]
                    if not audio_features:
                        continue
                except Exception as e:
                    print(f"\nSkipping {title}: {e}")
                    tracks_rejected += 1
                    continue

                # Normalize features
                features = normalize_audio_features(audio_features)
                if not features:
                    tracks_rejected += 1
                    continue

                # Determine genre from artist info
                try:
                    artist_info = sp.artist(track.get("artists", [{}])[0].get("id"))
                    artist_genres = artist_info.get("genres", [])
                except:
                    artist_genres = []

                genre = map_genre(artist_genres)
                if not genre:
                    # Default to a random genre or skip
                    genre = "indie"

                # Check genre is valid
                if genre not in vocab.GENRES:
                    genre = "indie"  # Fallback

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
                            artist,
                            genre,
                            "",  # mood_tags empty (will be inferred from audio features later)
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
                    # Duplicate ID, skip
                    continue
                except Exception as e:
                    print(f"\nError inserting {title}: {e}")
                    tracks_rejected += 1

            offset += 50

        print(f"{batch_added} added")

    conn.commit()
    conn.close()

    print(f"\n✓ Loaded {tracks_added} tracks")
    print(f"✗ Rejected {tracks_rejected} tracks")
    print(f"\nNext steps:")
    print(f"  python scripts/build_embeddings.py")
    print(f"  uvicorn api.main:app --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Spotify tracks and load into database")
    parser.add_argument("--limit", type=int, default=5000, help="Max tracks to fetch")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    fetch_and_load_tracks(limit=args.limit, db_path=args.db)
