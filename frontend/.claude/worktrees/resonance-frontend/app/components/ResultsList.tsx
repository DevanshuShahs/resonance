'use client';

import type { RecommendedResult } from '../lib/types';

interface ResultsListProps {
  results: RecommendedResult[];
  loading: boolean;
}

function SkeletonCard() {
  return (
    <div className="group relative rounded-2xl overflow-hidden bg-slate-800/40 backdrop-blur-sm border border-purple-400/20 p-4 sm:p-6 animate-pulse">
      <div className="flex gap-4">
        <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl bg-slate-700/50 flex-shrink-0"></div>
        <div className="flex-1 space-y-2">
          <div className="h-5 bg-slate-700/50 rounded w-3/4"></div>
          <div className="h-4 bg-slate-700/30 rounded w-1/2"></div>
          <div className="h-3 bg-slate-700/30 rounded w-2/3 mt-2"></div>
        </div>
      </div>
    </div>
  );
}

function colorForIndex(index: number): string {
  const colors = [
    'from-purple-500/30 to-pink-500/30',
    'from-blue-500/30 to-cyan-500/30',
    'from-emerald-500/30 to-teal-500/30',
    'from-orange-500/30 to-rose-500/30',
    'from-indigo-500/30 to-purple-500/30',
    'from-pink-500/30 to-rose-500/30',
    'from-cyan-500/30 to-blue-500/30',
    'from-teal-500/30 to-emerald-500/30',
    'from-amber-500/30 to-orange-500/30',
    'from-violet-500/30 to-purple-500/30',
  ];
  return colors[index % colors.length];
}

export function ResultsList({ results, loading }: ResultsListProps) {
  if (loading) {
    return (
      <div className="space-y-3 sm:space-y-4">
        {[...Array(6)].map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-16">
        <svg className="w-16 h-16 mx-auto text-purple-400/30 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
        </svg>
        <p className="text-purple-300/60 text-sm">
          No recommendations found - try a different combination!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 sm:space-y-4">
      {results.map((track, index) => (
        <a
          key={track.track_id}
          href={track.spotify_url || '#'}
          target={track.spotify_url ? '_blank' : undefined}
          rel={track.spotify_url ? 'noopener noreferrer' : undefined}
          className="group block relative rounded-2xl overflow-hidden bg-slate-800/40 backdrop-blur-sm border border-purple-400/20 hover:border-purple-400/60 transition-all duration-300 hover:bg-slate-800/60 p-4 sm:p-5 hover:shadow-2xl hover:shadow-purple-500/20"
        >
          <div className="flex gap-4 items-start">
            {/* Album art placeholder */}
            <div className={`w-16 h-16 sm:w-20 sm:h-20 rounded-xl bg-gradient-to-br ${colorForIndex(index)} flex items-center justify-center flex-shrink-0 group-hover:scale-105 transition-transform duration-300`}>
              <svg className="w-8 h-8 sm:w-10 sm:h-10 text-white/60" fill="currentColor" viewBox="0 0 20 20">
                <path d="M18 3a1 1 0 00-1.196-.15l-10 6.5a1 1 0 000 1.7l10 6.5A1 1 0 0018 17V3z" />
              </svg>
            </div>

            {/* Track info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1 min-w-0">
                  <div className="text-white font-semibold text-base sm:text-lg leading-tight truncate">
                    {track.title}
                  </div>
                  <div className="text-purple-300/70 text-xs sm:text-sm mt-0.5 truncate">
                    {track.artist}
                  </div>
                </div>
                <div className="flex-shrink-0 text-right">
                  <div className="text-2xl sm:text-3xl font-bold text-transparent bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text">
                    {track.rank}
                  </div>
                  <div className="text-xs text-purple-300/50 uppercase tracking-widest">Rank</div>
                </div>
              </div>

              {/* Tags and metadata */}
              <div className="flex flex-wrap gap-2 items-center mt-3 sm:mt-4">
                {track.mood_tags && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-purple-500/30 border border-purple-400/50 text-purple-200 font-medium whitespace-nowrap">
                    {track.mood_tags}
                  </span>
                )}
                <span className="text-xs px-2.5 py-1 rounded-full bg-slate-700/50 border border-slate-600/50 text-slate-300 font-medium whitespace-nowrap">
                  🔥 {track.popularity}
                </span>
                {track.spotify_url && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-green-500/20 border border-green-500/50 text-green-300 font-medium flex items-center gap-1 whitespace-nowrap">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M18 3a1 1 0 00-1.196-.15l-10 6.5a1 1 0 000 1.7l10 6.5A1 1 0 0018 17V3z" />
                    </svg>
                    Play
                  </span>
                )}
              </div>
            </div>
          </div>
        </a>
      ))}
    </div>
  );
}
