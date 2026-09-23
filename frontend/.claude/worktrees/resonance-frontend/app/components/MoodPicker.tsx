'use client';

interface MoodPickerProps {
  moods: string[];
  selected: string | null;
  onSelect: (mood: string | null) => void;
}

const MOOD_EMOJIS: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  energetic: '⚡',
  chill: '😌',
  aggressive: '💪',
  romantic: '💕',
  melancholic: '🌙',
  uplifting: '✨',
  dreamy: '☁️',
  tense: '😰',
  nostalgic: '🎬',
  playful: '🎮',
};

export function MoodPicker({ moods, selected, onSelect }: MoodPickerProps) {
  const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-purple-300 mb-4">
          How are you feeling?
        </label>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          {moods.map((mood) => {
            const isSelected = selected === mood;
            return (
              <button
                key={mood}
                onClick={() => onSelect(isSelected ? null : mood)}
                aria-pressed={isSelected}
                className={`group px-4 py-3 rounded-2xl font-medium text-sm transition-all duration-300 border ${
                  isSelected
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 border-purple-300 text-white shadow-lg shadow-purple-500/50'
                    : 'bg-slate-800/40 border-purple-400/30 text-purple-200 hover:bg-slate-800/60 hover:border-purple-400/60'
                }`}
              >
                <span className="block text-lg mb-1">{MOOD_EMOJIS[mood] || '🎵'}</span>
                {capitalize(mood)}
              </button>
            );
          })}
        </div>
      </div>
      {selected && (
        <button
          onClick={() => onSelect(null)}
          className="text-xs text-purple-400 hover:text-purple-300 transition flex items-center gap-1"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          Clear mood
        </button>
      )}
    </div>
  );
}
