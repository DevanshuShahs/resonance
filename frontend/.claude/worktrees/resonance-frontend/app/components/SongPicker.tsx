'use client';

import { useEffect, useMemo, useState } from 'react';
import { searchTracks } from '../lib/api';
import { useDebounce } from '../lib/useDebounce';
import type { TrackSummary } from '../lib/types';

interface SongPickerProps {
  selected: TrackSummary[];
  onChange: (tracks: TrackSummary[]) => void;
  max?: number;
}

export function SongPicker({ selected, onChange, max = 5 }: SongPickerProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<TrackSummary[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery.length >= 2) {
      setLoading(true);
      searchTracks(debouncedQuery, 8)
        .then(setResults)
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
      setIsOpen(true);
    } else {
      setResults([]);
      setIsOpen(false);
    }
  }, [debouncedQuery]);

  const selectedIds = useMemo(() => new Set(selected.map((t) => t.track_id)), [selected]);

  const handleAdd = (track: TrackSummary) => {
    if (!selectedIds.has(track.track_id) && selected.length < max) {
      onChange([...selected, track]);
      setQuery('');
      setResults([]);
    }
  };

  const handleRemove = (trackId: string) => {
    onChange(selected.filter((t) => t.track_id !== trackId));
  };

  return (
    <div className="space-y-6">
      <div className="relative">
        <div className="relative">
          <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-purple-400/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search songs or artists..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={selected.length >= max}
            className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-purple-400/30 rounded-xl text-white placeholder-purple-300/50 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-400/50 transition backdrop-blur-sm disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        {isOpen && (
          <div className="absolute z-20 w-full mt-2 bg-slate-900/95 backdrop-blur-xl border border-purple-400/30 rounded-2xl shadow-2xl overflow-hidden">
            {loading && (
              <div className="px-4 py-8 text-center">
                <div className="inline-block animate-spin text-purple-400">◌</div>
                <p className="text-purple-300/60 text-sm mt-2">Searching...</p>
              </div>
            )}
            {!loading && results.length === 0 && debouncedQuery && (
              <div className="px-4 py-8 text-center text-purple-300/60 text-sm">
                No songs found
              </div>
            )}
            <div className="max-h-96 overflow-y-auto">
              {results.map((track) => (
                <button
                  key={track.track_id}
                  onClick={() => handleAdd(track)}
                  disabled={selectedIds.has(track.track_id)}
                  className="w-full text-left px-4 py-3 border-b border-purple-400/10 hover:bg-purple-500/10 disabled:opacity-40 disabled:cursor-not-allowed transition group"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/30 to-pink-500/30 flex items-center justify-center mt-0.5 group-hover:from-purple-500/50 group-hover:to-pink-500/50 transition flex-shrink-0">
                      <svg className="w-5 h-5 text-purple-300" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M18 3a1 1 0 00-1.196-.15l-10 6.5a1 1 0 000 1.7l10 6.5A1 1 0 0018 17V3z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-white truncate">{track.title}</div>
                      <div className="text-sm text-purple-300/60 truncate">{track.artist}</div>
                    </div>
                    {selectedIds.has(track.track_id) && (
                      <svg className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {selected.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-purple-300">
              Selected Songs
            </span>
            <span className="text-xs px-2 py-1 rounded-full bg-purple-500/20 text-purple-300">
              {selected.length}/{max}
            </span>
          </div>
          <div className="space-y-2">
            {selected.map((track) => (
              <div
                key={track.track_id}
                className="flex items-center justify-between gap-3 px-4 py-3 rounded-xl bg-slate-800/40 border border-purple-400/20 hover:border-purple-400/40 transition group"
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-white truncate text-sm">{track.title}</div>
                  <div className="text-xs text-purple-300/60 truncate">{track.artist}</div>
                </div>
                <button
                  onClick={() => handleRemove(track.track_id)}
                  className="flex-shrink-0 w-6 h-6 rounded-lg bg-red-500/20 hover:bg-red-500/40 text-red-400 hover:text-red-300 flex items-center justify-center transition"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
