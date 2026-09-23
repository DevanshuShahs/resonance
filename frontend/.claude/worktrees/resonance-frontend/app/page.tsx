'use client';

import { useEffect, useState } from 'react';
import { getMoods, getRecommendations } from './lib/api';
import { SongPicker } from './components/SongPicker';
import { MoodPicker } from './components/MoodPicker';
import { ResultsList } from './components/ResultsList';
import type { TrackSummary, RecommendedResult } from './lib/types';

export default function Home() {
  const [favorites, setFavorites] = useState<TrackSummary[]>([]);
  const [mood, setMood] = useState<string | null>(null);
  const [moods, setMoods] = useState<string[]>([]);
  const [results, setResults] = useState<RecommendedResult[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  useEffect(() => {
    getMoods()
      .then(setMoods)
      .catch(() => {
        setError('Failed to load moods');
      });
  }, []);

  const handleSubmit = async () => {
    if (favorites.length === 0 || !mood) {
      setError('Please select at least one favorite song and a mood');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await getRecommendations({
        favorite_track_ids: favorites.map((f) => f.track_id),
        mood,
        top_k: 10,
      });
      setResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get recommendations');
      setResults([]);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isReady = favorites.length > 0 && mood !== null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-purple-900 to-slate-950">
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-md border-b border-purple-500/20">
        <div className="max-w-6xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-200 via-pink-200 to-purple-200 bg-clip-text text-transparent">
            Resonance
          </h1>
          <p className="text-purple-300/70 mt-2 text-sm tracking-wide">
            Discover music that matches your vibe
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        <div className="space-y-10">
          {/* Input section */}
          <div className="bg-gradient-to-br from-purple-900/40 to-slate-900/40 backdrop-blur-xl rounded-3xl border border-purple-500/30 p-8 space-y-8">
            <div>
              <h2 className="text-2xl font-semibold text-white mb-1">Find Your Sound</h2>
              <p className="text-purple-300/60 text-sm">Choose your favorite songs and mood</p>
            </div>

            <SongPicker selected={favorites} onChange={setFavorites} max={5} />

            {moods.length > 0 && (
              <MoodPicker moods={moods} selected={mood} onSelect={setMood} />
            )}

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-2xl text-red-300 text-sm">
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={!isReady || isSubmitting}
              className={`w-full px-8 py-4 rounded-2xl font-semibold text-white transition-all duration-300 ${
                isReady && !isSubmitting
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 shadow-lg shadow-purple-500/50 hover:shadow-pink-500/50 cursor-pointer transform hover:scale-105'
                  : 'bg-slate-700/50 text-slate-400 cursor-not-allowed'
              }`}
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin">◌</span>
                  Finding recommendations...
                </span>
              ) : (
                'Discover'
              )}
            </button>
          </div>

          {/* Results section */}
          {hasSearched && (
            <div className="space-y-6">
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">Your Recommendations</h2>
                <p className="text-purple-300/60 text-sm">
                  {results.length} songs in your taste
                </p>
              </div>
              <ResultsList results={results} loading={isSubmitting} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
