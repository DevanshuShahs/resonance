"""Load the track CSV into SQLite, validating every row on the way in.

Valid rows land in `tracks`. Anything that fails validation (missing fields,
out-of-range audio features, duplicate ids, bad types) is kept in
`rejected_rows` with its errors instead of being silently dropped, so a bad
source file is auditable rather than invisible.

Usage:
    python scripts/load_and_validate.py [--csv data/tracks.csv] [--db db/resonance.db]
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

RANGE_FIELDS_01 = [
    "danceability", "energy", "valence", "speechiness", "acousticness",
    "instrumentalness", "liveness",
]

REQUIRED_TEXT_FIELDS = ["id", "title", "artist", "genre"]


def to_float(value: str):
    return float(value)


def to_int(value: str):
    return int(float(value))


def validate_and_coerce(row: dict, seen_ids: Set[str]) -> Tuple[Optional[dict], List[str]]:
    errors: List[str] = []
    out: dict = {}

    for field in REQUIRED_TEXT_FIELDS:
        value = (row.get(field) or "").strip()
        if not value:
            errors.append(f"{field} is missing/empty")
        out[field] = value

    out["mood_tags"] = (row.get("mood_tags") or "").strip()

    for field in RANGE_FIELDS_01:
        raw = row.get(field)
        try:
            value = to_float(raw)
        except (TypeError, ValueError):
            errors.append(f"{field} is not numeric: {raw!r}")
            continue
        if not (0.0 <= value <= 1.0):
            errors.append(f"{field}={value} out of range [0,1]")
        out[field] = value

    try:
        tempo = to_float(row.get("tempo"))
        if not (0 < tempo <= 300):
            errors.append(f"tempo={tempo} out of range (0,300]")
        out["tempo"] = tempo
    except (TypeError, ValueError):
        errors.append(f"tempo is not numeric: {row.get('tempo')!r}")

    try:
        loudness = to_float(row.get("loudness"))
        if not (-60 <= loudness <= 5):
            errors.append(f"loudness={loudness} out of range [-60,5]")
        out["loudness"] = loudness
    except (TypeError, ValueError):
        errors.append(f"loudness is not numeric: {row.get('loudness')!r}")

    try:
        duration_ms = to_int(row.get("duration_ms"))
        if duration_ms <= 0:
            errors.append(f"duration_ms={duration_ms} must be positive")
        out["duration_ms"] = duration_ms
    except (TypeError, ValueError):
        errors.append(f"duration_ms is not numeric: {row.get('duration_ms')!r}")

    try:
        key = to_int(row.get("key"))
        if not (0 <= key <= 11):
            errors.append(f"key={key} out of range [0,11]")
        out["key"] = key
    except (TypeError, ValueError):
        errors.append(f"key is not numeric: {row.get('key')!r}")

    try:
        mode = to_int(row.get("mode"))
        if mode not in (0, 1):
            errors.append(f"mode={mode} must be 0 or 1")
        out["mode"] = mode
    except (TypeError, ValueError):
        errors.append(f"mode is not numeric: {row.get('mode')!r}")

    try:
        time_signature = to_int(row.get("time_signature"))
        if not (3 <= time_signature <= 7):
            errors.append(f"time_signature={time_signature} out of range [3,7]")
        out["time_signature"] = time_signature
    except (TypeError, ValueError):
        errors.append(f"time_signature is not numeric: {row.get('time_signature')!r}")

    try:
        popularity = to_int(row.get("popularity"))
        if not (0 <= popularity <= 100):
            errors.append(f"popularity={popularity} out of range [0,100]")
        out["popularity"] = popularity
    except (TypeError, ValueError):
        errors.append(f"popularity is not numeric: {row.get('popularity')!r}")

    track_id = out.get("id")
    if track_id:
        if track_id in seen_ids:
            errors.append(f"duplicate id: {track_id}")
        else:
            seen_ids.add(track_id)

    if errors:
        return None, errors
    return out, errors


TRACK_COLUMNS = [
    "id", "title", "artist", "genre", "mood_tags", "danceability", "energy",
    "valence", "tempo", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "duration_ms", "key", "mode",
    "time_signature", "popularity",
]


def load(csv_path: Path, db_path: Path) -> int:
    if not csv_path.exists():
        print(f"error: {csv_path} not found. Run scripts/generate_dataset.py first.",
              file=sys.stderr)
        return 1

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript((SCRIPT_DIR / "schema.sql").read_text())

    seen_ids: Set[str] = set()
    error_counter: Counter = Counter()
    total = 0
    valid_count = 0
    rejected_count = 0
    sample_rejections: List[Tuple[int, List[str]]] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            total += 1
            clean, errors = validate_and_coerce(row, seen_ids)
            if clean is None:
                rejected_count += 1
                for err in errors:
                    error_counter[err.split("=")[0].split(" is")[0].split(":")[0]] += 1
                if len(sample_rejections) < 10:
                    sample_rejections.append((row_num, errors))
                conn.execute(
                    "INSERT INTO rejected_rows (row_num, raw_json, errors) VALUES (?, ?, ?)",
                    (row_num, json.dumps(row), "; ".join(errors)),
                )
            else:
                valid_count += 1
                conn.execute(
                    f"INSERT INTO tracks ({', '.join(TRACK_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in TRACK_COLUMNS)})",
                    tuple(clean[c] for c in TRACK_COLUMNS),
                )

    conn.commit()

    print(f"Read {total} rows from {csv_path}")
    print(f"  loaded:   {valid_count}")
    print(f"  rejected: {rejected_count}")
    if error_counter:
        print("\nRejection reasons (by field, top offenders):")
        for field, count in error_counter.most_common(10):
            print(f"  {field}: {count}")
    if sample_rejections:
        print("\nSample rejected rows:")
        for row_num, errors in sample_rejections:
            print(f"  row {row_num}: {errors}")

    row_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    genre_counts = conn.execute(
        "SELECT genre, COUNT(*) FROM tracks GROUP BY genre ORDER BY 2 DESC"
    ).fetchall()
    print(f"\n{db_path} now has {row_count} rows in `tracks`.")
    print("Genre distribution:")
    for genre, count in genre_counts:
        print(f"  {genre:12s} {count}")

    conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/tracks.csv"))
    parser.add_argument("--db", type=Path, default=Path("db/resonance.db"))
    args = parser.parse_args()
    return load(args.csv, args.db)


if __name__ == "__main__":
    sys.exit(main())
