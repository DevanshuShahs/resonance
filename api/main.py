"""FastAPI application for the music search engine."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import routes
from engine import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading EngineStore...")
    app.state.engine_store = store.load()
    print(f"  Loaded {len(app.state.engine_store.track_ids)} tracks")
    print(f"  LSH index ready: L={len(app.state.engine_store.lsh_index.buckets)}")
    print(f"  Inverted index ready: {len(app.state.engine_store.inverted_index.postings)} terms")
    yield
    print("Shutting down...")
    app.state.engine_store.conn.close()


app = FastAPI(
    title="Resonance",
    description="A music search engine that finds tracks by natural-language queries",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(routes.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
