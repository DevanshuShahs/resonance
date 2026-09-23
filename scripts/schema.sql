-- Resonance track store. Targets SQLite; only trivial changes (SERIAL/TIMESTAMPTZ
-- instead of INTEGER PRIMARY KEY/TEXT defaults) are needed to run this on Postgres.

DROP TABLE IF EXISTS rejected_rows;
DROP TABLE IF EXISTS tracks;

CREATE TABLE tracks (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    artist            TEXT NOT NULL,
    genre             TEXT NOT NULL,
    mood_tags         TEXT NOT NULL DEFAULT '',
    danceability      REAL NOT NULL CHECK (danceability BETWEEN 0 AND 1),
    energy            REAL NOT NULL CHECK (energy BETWEEN 0 AND 1),
    valence           REAL NOT NULL CHECK (valence BETWEEN 0 AND 1),
    tempo             REAL NOT NULL CHECK (tempo > 0),
    loudness          REAL NOT NULL CHECK (loudness BETWEEN -60 AND 5),
    speechiness       REAL NOT NULL CHECK (speechiness BETWEEN 0 AND 1),
    acousticness      REAL NOT NULL CHECK (acousticness BETWEEN 0 AND 1),
    instrumentalness  REAL NOT NULL CHECK (instrumentalness BETWEEN 0 AND 1),
    liveness          REAL NOT NULL CHECK (liveness BETWEEN 0 AND 1),
    duration_ms       INTEGER NOT NULL CHECK (duration_ms > 0),
    key               INTEGER NOT NULL CHECK (key BETWEEN 0 AND 11),
    mode              INTEGER NOT NULL CHECK (mode IN (0, 1)),
    time_signature    INTEGER NOT NULL CHECK (time_signature BETWEEN 3 AND 7),
    popularity        INTEGER NOT NULL CHECK (popularity BETWEEN 0 AND 100)
);

CREATE INDEX idx_tracks_genre ON tracks (genre);
CREATE INDEX idx_tracks_artist ON tracks (artist);

-- Rows from the source file that failed validation, kept for audit instead
-- of silently dropped.
CREATE TABLE rejected_rows (
    row_num    INTEGER NOT NULL,
    raw_json   TEXT NOT NULL,
    errors     TEXT NOT NULL,
    loaded_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
