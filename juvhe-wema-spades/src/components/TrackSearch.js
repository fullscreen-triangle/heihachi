'use client'
import { useState, useCallback, useRef } from 'react'

const MUSICBRAINZ_API = 'https://musicbrainz.org/ws/2'
const COVER_ART_API = 'https://coverartarchive.org'

export default function TrackSearch({ onSelect }) {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState([])
    const [loading, setLoading] = useState(false)
    const [selectedTrack, setSelectedTrack] = useState(null)
    const debounceRef = useRef(null)

    const searchTracks = useCallback(async (q) => {
        if (!q || q.length < 2) { setResults([]); return }
        setLoading(true)

        try {
            const res = await fetch(
                `${MUSICBRAINZ_API}/recording?query=${encodeURIComponent(q)}&limit=8&fmt=json`,
                { headers: { 'User-Agent': 'YokozunaPlayer/1.0 (kundai.sachikonye@wzw.tum.de)' } }
            )
            const data = await res.json()

            const tracks = (data.recordings || []).map(rec => ({
                id: rec.id,
                title: rec.title,
                artist: rec['artist-credit']?.map(a => a.name).join(', ') || 'Unknown',
                album: rec.releases?.[0]?.title || '',
                releaseId: rec.releases?.[0]?.id || null,
                year: rec['first-release-date']?.slice(0, 4) || '',
                duration: rec.length ? Math.round(rec.length / 1000) : null,
                tags: rec.tags?.map(t => t.name).slice(0, 5) || [],
                score: rec.score,
            }))

            setResults(tracks)
        } catch (e) {
            console.error('MusicBrainz search failed:', e)
            setResults([])
        }
        setLoading(false)
    }, [])

    const handleInput = useCallback((e) => {
        const v = e.target.value
        setQuery(v)
        if (debounceRef.current) clearTimeout(debounceRef.current)
        debounceRef.current = setTimeout(() => searchTracks(v), 400)
    }, [searchTracks])

    const fetchTrackDetails = useCallback(async (track) => {
        setSelectedTrack(track)

        // Try to get cover art
        if (track.releaseId) {
            try {
                const res = await fetch(`${COVER_ART_API}/release/${track.releaseId}`)
                if (res.ok) {
                    const data = await res.json()
                    const thumb = data.images?.[0]?.thumbnails?.small || data.images?.[0]?.image
                    if (thumb) {
                        setSelectedTrack(prev => ({ ...prev, coverArt: thumb }))
                    }
                }
            } catch (e) { /* no cover art available */ }
        }

        // Fetch additional recording details (genres, etc.)
        try {
            const res = await fetch(
                `${MUSICBRAINZ_API}/recording/${track.id}?inc=genres+tags+artist-credits&fmt=json`,
                { headers: { 'User-Agent': 'YokozunaPlayer/1.0 (kundai.sachikonye@wzw.tum.de)' } }
            )
            const data = await res.json()
            const genres = data.genres?.map(g => g.name) || []
            setSelectedTrack(prev => ({ ...prev, genres }))
        } catch (e) { /* ignore */ }
    }, [])

    const formatDuration = (s) => {
        if (!s) return ''
        const m = Math.floor(s / 60)
        const sec = s % 60
        return `${m}:${sec.toString().padStart(2, '0')}`
    }

    return (
        <div className="bg-black/80 backdrop-blur-md border border-white/10 rounded-lg overflow-hidden">
            {/* Search input */}
            <div className="p-3 border-b border-white/10">
                <input
                    type="text"
                    value={query}
                    onChange={handleInput}
                    placeholder="Search artist, track, or album..."
                    className="w-full bg-white/5 border border-white/10 rounded-md px-3 py-2 text-sm text-white
                        placeholder:text-white/30 focus:outline-none focus:border-white/30"
                    autoFocus
                />
            </div>

            {/* Results */}
            {loading && (
                <div className="p-4 text-center text-white/40 text-xs">Searching MusicBrainz...</div>
            )}

            {!loading && results.length > 0 && !selectedTrack && (
                <div className="max-h-72 overflow-y-auto">
                    {results.map((track) => (
                        <button
                            key={track.id}
                            onClick={() => fetchTrackDetails(track)}
                            className="w-full text-left px-4 py-2.5 hover:bg-white/10 transition-all border-b border-white/5 last:border-0"
                        >
                            <div className="text-sm text-white/90 font-medium">{track.title}</div>
                            <div className="text-xs text-white/50 mt-0.5">
                                {track.artist}
                                {track.album && <span> — {track.album}</span>}
                                {track.year && <span className="ml-2 text-white/30">({track.year})</span>}
                                {track.duration && <span className="ml-2 text-white/30">{formatDuration(track.duration)}</span>}
                            </div>
                            {track.tags.length > 0 && (
                                <div className="flex gap-1 mt-1">
                                    {track.tags.map((tag, i) => (
                                        <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-white/40">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </button>
                    ))}
                </div>
            )}

            {/* Track detail view */}
            {selectedTrack && (
                <div className="p-4">
                    <button
                        onClick={() => setSelectedTrack(null)}
                        className="text-xs text-white/40 hover:text-white/70 mb-3"
                    >
                        ← Back to results
                    </button>

                    <div className="flex gap-4">
                        {selectedTrack.coverArt && (
                            <img
                                src={selectedTrack.coverArt}
                                alt="Cover"
                                className="w-20 h-20 rounded object-cover"
                            />
                        )}
                        <div className="flex-1 min-w-0">
                            <div className="text-sm text-white font-semibold truncate">{selectedTrack.title}</div>
                            <div className="text-xs text-white/60 mt-0.5">{selectedTrack.artist}</div>
                            {selectedTrack.album && (
                                <div className="text-xs text-white/40 mt-0.5">{selectedTrack.album} {selectedTrack.year && `(${selectedTrack.year})`}</div>
                            )}
                            {selectedTrack.genres?.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                    {selectedTrack.genres.map((g, i) => (
                                        <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                                            {g}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="mt-3 pt-3 border-t border-white/10 text-xs text-white/30">
                        Metadata from MusicBrainz. To play this track, upload the audio file or paste a direct URL.
                    </div>

                    <button
                        onClick={() => onSelect?.(selectedTrack)}
                        className="mt-2 w-full py-2 rounded-md bg-white/10 border border-white/20 text-white/70
                            text-xs font-medium hover:bg-white/20 transition-all"
                    >
                        Use Track Info
                    </button>
                </div>
            )}

            {!loading && results.length === 0 && query.length >= 2 && (
                <div className="p-4 text-center text-white/30 text-xs">No results found</div>
            )}
        </div>
    )
}
