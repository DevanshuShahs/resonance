"""Search API routes."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas import (
    SearchRequest, SearchResponse, SearchResultSchema,
    TrackSearchResponse, TrackSummarySchema, MoodsResponse,
    RecommendRequest, RecommendResponse, RecommendedResultSchema,
)
from engine import search, vocab, reccobeats_client, query_parser, hybrid_ranking, recommendation_engine

router = APIRouter()


@router.get("/tracks/search", response_model=TrackSearchResponse)
async def search_tracks_typeahead(
    request: Request, q: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=25)
) -> TrackSearchResponse:
    """Typeahead search for tracks by title/artist via live ReccoBeats.

    Returns matching tracks ordered by ReccoBeats' ranking."""
    try:
        tracks = reccobeats_client.search_tracks(q, limit)
    except reccobeats_client.ReccoBeatsError as e:
        raise HTTPException(status_code=502, detail="Music service temporarily unavailable")

    if not tracks:
        return TrackSearchResponse(query=q, results=[])

    track_ids = [t["id"] for t in tracks]

    try:
        features_map = reccobeats_client.get_audio_features_batch(track_ids)
    except reccobeats_client.ReccoBeatsError as e:
        raise HTTPException(status_code=502, detail="Music service temporarily unavailable")

    results = []
    for track in tracks:
        track_id = track["id"]
        features = features_map.get(track_id, {})
        valence = features.get("valence", 0.5)
        energy = features.get("energy", 0.5)
        mood_tags = query_parser.derive_mood_tags(valence, energy)

        artists = track.get("artists", [])
        artist_name = ", ".join(a.get("name", "Unknown") for a in artists) if artists else "Unknown"

        results.append(
            TrackSummarySchema(
                track_id=track_id,
                title=track.get("trackTitle", "Unknown"),
                artist=artist_name,
                mood_tags=mood_tags,
                spotify_url=track.get("href"),
            )
        )

    return TrackSearchResponse(query=q, results=results)


@router.get("/moods", response_model=MoodsResponse)
async def list_moods() -> MoodsResponse:
    """List all canonical moods."""
    return MoodsResponse(moods=vocab.MOODS)


@router.post("/search", response_model=SearchResponse)
async def search_tracks(request: Request, req: SearchRequest) -> SearchResponse:
    """Execute a music search query.

    Returns top-K tracks ranked by a blended score of semantic similarity,
    full-text relevance (BM25), and popularity, with smart SQL prefiltering
    that relaxes if too restrictive."""
    engine_store = request.app.state.engine_store

    results = search.search(engine_store, req.query, top_k=req.top_k)

    return SearchResponse(
        query=req.query,
        filter_level=results[0].filter_level if results else "unrestricted",
        results=[
            SearchResultSchema(
                rank=r.rank,
                track_id=r.track_id,
                title=r.title,
                artist=r.artist,
                genre=r.genre,
                mood_tags=r.mood_tags,
                score=r.score,
                cosine=r.cosine,
                bm25=r.bm25,
                popularity_norm=r.popularity_norm,
                relaxation_bonus=r.relaxation_bonus,
                filter_level=r.filter_level,
            )
            for r in results
        ],
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_tracks(request: Request, req: RecommendRequest) -> RecommendResponse:
    """Recommend tracks using our own audio similarity algorithm.

    Flow:
    1. Fetch favorite track metadata and audio features
    2. Search ReccoBeats for candidates using favorite track metadata
    3. Compute audio feature similarity (cosine similarity)
    4. Apply hybrid ranking with contextual metadata
    5. Return top-K results with mood tags derived from audio features

    Uses audio feature vectors + metadata for robust recommendations."""
    try:
        # Fetch metadata and audio features for favorite tracks
        tracks_map = reccobeats_client.get_tracks_batch(req.favorite_track_ids)
        features_map = reccobeats_client.get_audio_features_batch(req.favorite_track_ids)

        favorite_tracks = [
            tracks_map.get(fid, {
                "id": fid,
                "trackTitle": "",
                "artists": [],
                "durationMs": 180000,
                "popularity": 50,
            })
            for fid in req.favorite_track_ids
        ]
        favorite_features = [features_map.get(fid, {}) for fid in req.favorite_track_ids]

        # Get recommendations using our algorithm
        recommendations = recommendation_engine.recommend(
            favorite_track_ids=req.favorite_track_ids,
            favorite_tracks=favorite_tracks,
            favorite_features=favorite_features,
            mood=req.mood,
            top_k=req.top_k,
        )
    except reccobeats_client.ReccoBeatsError:
        raise HTTPException(status_code=502, detail="Music service temporarily unavailable")

    if not recommendations:
        return RecommendResponse(
            favorite_track_ids=req.favorite_track_ids,
            mood=req.mood,
            results=[],
        )

    # Get audio features for all recommended tracks
    rec_ids = [t[0].get("id") for t in recommendations]
    try:
        features_map = reccobeats_client.get_audio_features_batch(rec_ids)
    except reccobeats_client.ReccoBeatsError:
        raise HTTPException(status_code=502, detail="Music service temporarily unavailable")

    results = []
    for rank, (track, score) in enumerate(recommendations, start=1):
        track_id = track.get("id")
        features = features_map.get(track_id, {})
        valence = features.get("valence", 0.5)
        energy = features.get("energy", 0.5)
        mood_tags = query_parser.derive_mood_tags(valence, energy)

        artists = track.get("artists", [])
        artist_name = ", ".join(a.get("name", "Unknown") for a in artists) if artists else "Unknown"

        results.append(
            RecommendedResultSchema(
                rank=rank,
                track_id=track_id,
                title=track.get("trackTitle", "Unknown"),
                artist=artist_name,
                mood_tags=mood_tags,
                popularity=track.get("popularity", 0),
                spotify_url=track.get("href"),
            )
        )

    return RecommendResponse(
        favorite_track_ids=req.favorite_track_ids,
        mood=req.mood,
        results=results,
    )
