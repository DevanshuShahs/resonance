"""Generate a synthetic ~5,000 track dataset shaped like Spotify's public
track/audio-features data (Kaggle's SpotifyFeatures.csv or the Web API's
audio-features endpoint), for use until a real credentialed source is wired in.

Usage:
    python scripts/generate_dataset.py [--rows 5000] [--out data/tracks.csv] [--seed 42]
"""

import argparse
import csv
import random
import string
import sys
from pathlib import Path

GENRES = [
    "pop", "rock", "hip-hop", "electronic", "jazz", "classical",
    "country", "r&b", "reggae", "metal", "folk", "indie", "latin", "blues",
]

MOODS = [
    "happy", "sad", "energetic", "chill", "aggressive", "romantic",
    "melancholic", "uplifting", "dreamy", "tense", "nostalgic", "playful",
]

# Per-genre center points for (danceability, energy, valence, acousticness,
# instrumentalness, tempo, loudness) so the synthetic data has realistic
# genre-conditioned structure instead of pure uniform noise.
GENRE_PROFILES = {
    "pop":         dict(dance=0.70, energy=0.65, valence=0.60, acoustic=0.15, instr=0.02, tempo=118, loud=-5.5),
    "rock":        dict(dance=0.50, energy=0.80, valence=0.55, acoustic=0.10, instr=0.05, tempo=128, loud=-6.0),
    "hip-hop":     dict(dance=0.75, energy=0.65, valence=0.50, acoustic=0.10, instr=0.01, tempo=100, loud=-6.5),
    "electronic":  dict(dance=0.72, energy=0.78, valence=0.50, acoustic=0.05, instr=0.35, tempo=126, loud=-5.0),
    "jazz":        dict(dance=0.50, energy=0.40, valence=0.55, acoustic=0.55, instr=0.30, tempo=110, loud=-11.0),
    "classical":   dict(dance=0.30, energy=0.25, valence=0.45, acoustic=0.90, instr=0.80, tempo=100, loud=-18.0),
    "country":     dict(dance=0.55, energy=0.55, valence=0.60, acoustic=0.35, instr=0.02, tempo=112, loud=-7.0),
    "r&b":         dict(dance=0.65, energy=0.50, valence=0.45, acoustic=0.20, instr=0.02, tempo=95,  loud=-7.5),
    "reggae":      dict(dance=0.70, energy=0.55, valence=0.65, acoustic=0.25, instr=0.05, tempo=90,  loud=-8.0),
    "metal":       dict(dance=0.40, energy=0.90, valence=0.35, acoustic=0.02, instr=0.10, tempo=140, loud=-4.5),
    "folk":        dict(dance=0.45, energy=0.35, valence=0.50, acoustic=0.75, instr=0.10, tempo=105, loud=-10.0),
    "indie":       dict(dance=0.55, energy=0.55, valence=0.50, acoustic=0.35, instr=0.10, tempo=115, loud=-8.5),
    "latin":       dict(dance=0.80, energy=0.70, valence=0.70, acoustic=0.20, instr=0.02, tempo=104, loud=-5.5),
    "blues":       dict(dance=0.45, energy=0.45, valence=0.40, acoustic=0.50, instr=0.15, tempo=95,  loud=-9.0),
}

ADJECTIVES = [
    "Midnight", "Golden", "Broken", "Electric", "Silent", "Neon", "Velvet",
    "Lonely", "Wild", "Fading", "Crystal", "Distant", "Burning", "Hollow",
    "Endless", "Restless", "Bitter", "Sweet", "Faded", "Radiant",
]
NOUNS = [
    "Horizon", "Heart", "Dream", "River", "Shadow", "Skyline", "Echo",
    "Memory", "Fire", "Rain", "Road", "Ocean", "Ghost", "Light", "Storm",
    "Garden", "City", "Star", "Wave", "Bloom",
]
ARTIST_FIRST = [
    "Ava", "Kai", "Nova", "Leo", "Mira", "Jax", "Sena", "Rio", "Talia",
    "Ezra", "Wren", "Cole", "Nadia", "Finn", "Iris", "Reo", "Zara", "Toma",
]
ARTIST_LAST = [
    "Vance", "Cross", "Reyes", "Blake", "Marsh", "Quinn", "Hale", "Voss",
    "Ryder", "Stone", "Lark", "Frost", "Cade", "Wells", "Shaw", "Knox",
]


def make_id(rng: random.Random) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(22))


def make_title(rng: random.Random) -> str:
    return f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}"


def make_artist(rng: random.Random) -> str:
    if rng.random() < 0.15:
        return f"The {rng.choice(NOUNS)}s"
    return f"{rng.choice(ARTIST_FIRST)} {rng.choice(ARTIST_LAST)}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_mood_tags(rng: random.Random, valence: float, energy: float) -> str:
    weighted = []
    for mood in MOODS:
        weight = 1.0
        if mood in ("happy", "uplifting", "playful") and valence > 0.6:
            weight = 3.0
        if mood in ("sad", "melancholic", "nostalgic") and valence < 0.4:
            weight = 3.0
        if mood in ("energetic", "aggressive", "tense") and energy > 0.7:
            weight = 3.0
        if mood in ("chill", "dreamy") and energy < 0.4:
            weight = 3.0
        weighted.append((mood, weight))
    n_tags = rng.choice([1, 2, 2, 3])
    pool = [m for m, _ in weighted]
    weights = [w for _, w in weighted]
    chosen = []
    for _ in range(n_tags):
        pick = rng.choices(pool, weights=weights, k=1)[0]
        if pick not in chosen:
            chosen.append(pick)
    return "|".join(chosen)


def generate_row(rng: random.Random, row_num: int) -> dict:
    genre = rng.choice(GENRES)
    profile = GENRE_PROFILES[genre]

    danceability = clamp(rng.gauss(profile["dance"], 0.12), 0.0, 1.0)
    energy = clamp(rng.gauss(profile["energy"], 0.12), 0.0, 1.0)
    valence = clamp(rng.gauss(profile["valence"], 0.15), 0.0, 1.0)
    acousticness = clamp(rng.gauss(profile["acoustic"], 0.15), 0.0, 1.0)
    instrumentalness = clamp(rng.gauss(profile["instr"], 0.12), 0.0, 1.0)
    tempo = clamp(rng.gauss(profile["tempo"], 12), 40.0, 220.0)
    loudness = clamp(rng.gauss(profile["loud"], 2.5), -60.0, 0.0)
    speechiness = clamp(rng.gauss(0.06 if genre != "hip-hop" else 0.20, 0.05), 0.0, 1.0)
    liveness = clamp(rng.gauss(0.15, 0.10), 0.0, 1.0)
    duration_ms = int(clamp(rng.gauss(215_000, 40_000), 60_000, 480_000))
    key = rng.randint(0, 11)
    mode = rng.choice([0, 1])
    time_signature = rng.choice([3, 4, 4, 4, 5])
    popularity = int(clamp(rng.gauss(45, 20), 0, 100))

    return {
        "id": make_id(rng),
        "title": make_title(rng),
        "artist": make_artist(rng),
        "genre": genre,
        "mood_tags": make_mood_tags(rng, valence, energy),
        "danceability": round(danceability, 3),
        "energy": round(energy, 3),
        "valence": round(valence, 3),
        "tempo": round(tempo, 1),
        "loudness": round(loudness, 2),
        "speechiness": round(speechiness, 3),
        "acousticness": round(acousticness, 3),
        "instrumentalness": round(instrumentalness, 3),
        "liveness": round(liveness, 3),
        "duration_ms": duration_ms,
        "key": key,
        "mode": mode,
        "time_signature": time_signature,
        "popularity": popularity,
    }


def corrupt_row(rng: random.Random, row: dict, row_num: int) -> dict:
    """Inject one realistic data-quality defect so the validator has
    something real to catch, mirroring the kind of mess real exports have."""
    defect = rng.choice([
        "missing_title", "missing_artist", "bad_valence", "bad_tempo",
        "bad_duration", "bad_mode", "empty_genre",
    ])
    row = dict(row)
    if defect == "missing_title":
        row["title"] = ""
    elif defect == "missing_artist":
        row["artist"] = ""
    elif defect == "bad_valence":
        row["valence"] = 1.8
    elif defect == "bad_tempo":
        row["tempo"] = -5.0
    elif defect == "bad_duration":
        row["duration_ms"] = -1000
    elif defect == "bad_mode":
        row["mode"] = 7
    elif defect == "empty_genre":
        row["genre"] = ""
    return row


FIELDNAMES = [
    "id", "title", "artist", "genre", "mood_tags", "danceability", "energy",
    "valence", "tempo", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "duration_ms", "key", "mode",
    "time_signature", "popularity",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("data/tracks.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--defect-rate", type=float, default=0.02,
        help="fraction of rows to intentionally corrupt, to exercise the validator",
    )
    parser.add_argument(
        "--duplicate-rate", type=float, default=0.005,
        help="fraction of rows duplicated (same id) to exercise uniqueness checks",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [generate_row(rng, i) for i in range(args.rows)]

    n_defects = int(args.rows * args.defect_rate)
    for i in rng.sample(range(args.rows), k=min(n_defects, args.rows)):
        rows[i] = corrupt_row(rng, rows[i], i)

    n_dupes = int(args.rows * args.duplicate_rate)
    for _ in range(n_dupes):
        src = rng.randrange(len(rows))
        dst = rng.randrange(len(rows))
        rows[dst] = dict(rows[dst], id=rows[src]["id"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out} "
          f"({n_defects} intentionally corrupted, {n_dupes} duplicated ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
