'use client'
import { useState, useCallback, useEffect, useRef } from 'react'
import { PlayerAudioProvider, usePlayerAudio } from './PlayerAudioProvider'
import { LiquidDistortion, LiquidSlider } from './LiquidDistortion'
import { useSpotify } from '../hooks/useSpotify'
import { useTrackObserver } from '../hooks/useTrackObserver'

const LOCAL_PLAYLIST = [
    { id: 'local-1', name: 'Noisia - Feed the Machine (BSE)', artist: 'Noisia', src: '/audio/BSE_NOISA_Feed_the_Machine.mp3', albumArt: '' },
    { id: 'local-2', name: 'Konflict - Messiah (Magnetude Remix)', artist: 'Konflict', src: '/audio/Konflict-Messiah-Magnetude.mp3', albumArt: '' },
    { id: 'local-3', name: 'Noisia - Stigma (Neosignal Remix)', artist: 'Noisia', src: '/audio/Noisia_Stigma(NeosignalRemix).mp3', albumArt: '' },
    { id: 'local-4', name: 'Spor - Running Man', artist: 'Spor', src: '/audio/Spor_RunningMan.mp3', albumArt: '' },
    { id: 'local-5', name: 'Squarepusher - Dark Steering', artist: 'Squarepusher', src: '/audio/Squarepusher_Dark_Steering.mp3', albumArt: '' },
]

function SearchPlayerUI() {
    const {
        audioData, trackInfo,
        loadTrack, play, pause, seek, setVolume,
    } = usePlayerAudio()

    const spotify = useSpotify()
    const observer = useTrackObserver()

    const [searchQuery, setSearchQuery] = useState('')
    const [searchResults, setSearchResults] = useState([])
    const [searching, setSearching] = useState(false)
    const [playlist, setPlaylist] = useState(LOCAL_PLAYLIST)
    const [currentIndex, setCurrentIndex] = useState(-1)
    const [started, setStarted] = useState(false)
    const [volumeVal, setVolumeVal] = useState(1)
    const indexRef = useRef(-1)

    // Observe audio data every frame
    useEffect(() => {
        if (audioData.isPlaying) observer.observe(audioData)
    }, [audioData, observer])

    // Search handler
    const handleSearch = useCallback(async (e) => {
        e?.preventDefault()
        if (!searchQuery.trim()) return

        setSearching(true)
        try {
            if (spotify.authenticated) {
                const { getValidToken } = await import('../lib/spotify')
                const token = await getValidToken()
                if (token) {
                    const { search } = await import('../lib/spotify')
                    const data = await search(token.access_token, searchQuery, 'track', 20)
                    const tracks = (data.tracks?.items || []).map(t => ({
                        id: t.id,
                        name: t.name,
                        artist: t.artists?.map(a => a.name).join(', ') || '',
                        album: t.album?.name || '',
                        albumArt: t.album?.images?.[0]?.url || '',
                        duration: t.duration_ms,
                        previewUrl: t.preview_url,
                        uri: t.uri,
                        src: t.preview_url, // 30s preview
                    }))
                    setSearchResults(tracks)
                }
            }
        } catch (err) {
            console.error('Search failed:', err)
        } finally {
            setSearching(false)
        }
    }, [searchQuery, spotify.authenticated])

    // Play a track
    const playTrack = useCallback(async (track, index, list) => {
        if (list) setPlaylist(list)
        const targetList = list || playlist
        const src = track.src || track.previewUrl
        if (!src) return

        indexRef.current = index
        setCurrentIndex(index)
        setStarted(true)

        observer.startTrack(track.id, track.name)
        await loadTrack(src, `${track.artist} - ${track.name}`)
        play()
    }, [playlist, loadTrack, play, observer])

    // Play from search result — adds to playlist
    const playFromSearch = useCallback((track, index) => {
        const newList = searchResults.filter(t => t.src || t.previewUrl)
        if (newList.length === 0) return
        const actualIndex = newList.findIndex(t => t.id === track.id)
        playTrack(track, actualIndex >= 0 ? actualIndex : 0, newList)
        setSearchResults([]) // Close search results
    }, [searchResults, playTrack])

    // Skip to next
    const handleSkip = useCallback(async () => {
        observer.finishTrack()
        const next = (indexRef.current + 1) % playlist.length
        await playTrack(playlist[next], next)
    }, [playlist, playTrack, observer])

    // Auto-advance
    useEffect(() => {
        if (!started) return
        if (trackInfo.duration > 0 && trackInfo.currentTime >= trackInfo.duration - 0.3 && !audioData.isPlaying) {
            handleSkip()
        }
    }, [audioData.isPlaying, trackInfo.currentTime, trackInfo.duration, started, handleSkip])

    const handleVolumeChange = useCallback((e) => {
        const v = parseFloat(e.target.value)
        setVolumeVal(v)
        setVolume(v)
    }, [setVolume])

    const formatTime = (s) => {
        if (!s || isNaN(s)) return '0:00'
        const m = Math.floor(s / 60)
        const sec = Math.floor(s % 60)
        return `${m}:${sec.toString().padStart(2, '0')}`
    }

    const currentTrack = currentIndex >= 0 ? playlist[currentIndex] : null
    const albumArtUrl = currentTrack?.albumArt || ''
    const albumImages = playlist.map(t => t.albumArt).filter(Boolean)
    const hasMultiple = albumImages.length > 1 && started
    const hasAlbumArt = albumArtUrl && albumArtUrl.length > 0

    return (
        <div className="relative w-full h-full bg-[#0a0a0f] overflow-hidden">

            {/* Background: liquid distortion on album art */}
            <div className="absolute inset-0 z-0">
                {hasAlbumArt ? (
                    hasMultiple ? (
                        <LiquidSlider
                            images={albumImages}
                            currentIndex={Math.min(currentIndex, albumImages.length - 1)}
                            audioData={audioData}
                            intensity={audioData.isPlaying ? 1.0 : 0.2}
                        />
                    ) : (
                        <LiquidDistortion
                            imageUrl={albumArtUrl}
                            audioData={audioData}
                            intensity={audioData.isPlaying ? 1.0 : 0.2}
                        />
                    )
                ) : (
                    <div className="w-full h-full bg-gradient-to-br from-[#0a0a1a] via-[#111128] to-[#0a0a0f]" />
                )}
            </div>

            {/* Gradient overlays */}
            <div className="absolute inset-0 z-5 bg-gradient-to-t from-black/80 via-transparent to-black/50 pointer-events-none" />

            {/* Top: Search bar + Spotify auth */}
            <div className="absolute top-0 left-0 right-0 z-20 px-6 py-4">
                <div className="flex items-center gap-3 max-w-2xl mx-auto">
                    {/* Search form */}
                    <form onSubmit={handleSearch} className="flex-1 flex gap-2">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder={spotify.authenticated ? 'Search Spotify...' : 'Connect Spotify to search'}
                            disabled={!spotify.authenticated}
                            className="flex-1 px-4 py-2.5 rounded-full bg-white/10 border border-white/20
                                text-white text-sm placeholder:text-white/30 outline-none
                                focus:border-white/40 focus:bg-white/15 transition-all
                                disabled:opacity-40 disabled:cursor-not-allowed"
                        />
                        {spotify.authenticated && (
                            <button
                                type="submit"
                                disabled={searching || !searchQuery.trim()}
                                className="px-5 py-2.5 rounded-full bg-emerald-500/20 border border-emerald-500/30
                                    text-emerald-300 text-sm font-medium hover:bg-emerald-500/30 transition-all
                                    disabled:opacity-40"
                            >
                                {searching ? '...' : 'Search'}
                            </button>
                        )}
                    </form>

                    {/* Spotify auth button */}
                    {!spotify.authenticated ? (
                        <button
                            onClick={spotify.login}
                            className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-[#1DB954]/20
                                border border-[#1DB954]/30 text-[#1DB954] text-sm font-medium
                                hover:bg-[#1DB954]/30 transition-all whitespace-nowrap"
                        >
                            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                            </svg>
                            Connect
                        </button>
                    ) : (
                        <div className="flex items-center gap-2">
                            <span className="text-white/40 text-xs">{spotify.user?.display_name}</span>
                            <button
                                onClick={spotify.logout}
                                className="text-white/30 text-xs hover:text-white/60 transition-all"
                            >
                                Logout
                            </button>
                        </div>
                    )}
                </div>

                {/* Search results dropdown */}
                {searchResults.length > 0 && (
                    <div className="max-w-2xl mx-auto mt-2 bg-black/90 backdrop-blur-lg border border-white/10 rounded-xl overflow-hidden max-h-80 overflow-y-auto">
                        {searchResults.map((track, i) => (
                            <button
                                key={track.id}
                                onClick={() => playFromSearch(track, i)}
                                disabled={!track.previewUrl}
                                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 transition-all
                                    text-left border-b border-white/5 last:border-0 disabled:opacity-30"
                            >
                                {track.albumArt && (
                                    <img src={track.albumArt} alt="" className="w-10 h-10 rounded object-cover" />
                                )}
                                <div className="flex-1 min-w-0">
                                    <div className="text-white text-sm font-medium truncate">{track.name}</div>
                                    <div className="text-white/40 text-xs truncate">{track.artist}</div>
                                </div>
                                {!track.previewUrl && (
                                    <span className="text-white/20 text-[10px]">No preview</span>
                                )}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Center: Track info overlay */}
            {currentTrack && (
                <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
                    <div className="text-center">
                        <h2 className="text-white text-2xl font-bold drop-shadow-lg md:text-xl sm:text-lg">
                            {currentTrack.name}
                        </h2>
                        <p className="text-white/60 text-sm mt-1 drop-shadow-md">
                            {currentTrack.artist}
                        </p>
                    </div>
                </div>
            )}

            {/* Bottom: Transport controls */}
            <div className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/90 to-transparent pt-16 pb-5 px-6">
                {/* Track name */}
                <div className="text-center mb-3">
                    <span className="text-white/80 text-sm font-medium">
                        {trackInfo.name || (spotify.authenticated ? 'Search for a song above' : 'Connect Spotify or press play for local jukebox')}
                    </span>
                    {started && playlist.length > 1 && (
                        <span className="text-white/30 text-xs ml-2">
                            {currentIndex + 1} / {playlist.length}
                        </span>
                    )}
                </div>

                {/* Seek bar */}
                {started && (
                    <div className="flex items-center gap-3 mb-3 max-w-lg mx-auto">
                        <span className="text-white/40 text-xs w-10 text-right font-mono">
                            {formatTime(trackInfo.currentTime)}
                        </span>
                        <input
                            type="range"
                            min={0} max={trackInfo.duration || 1} step={0.1}
                            value={trackInfo.currentTime || 0}
                            onChange={(e) => seek(parseFloat(e.target.value))}
                            className="flex-1 h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                                [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
                        />
                        <span className="text-white/40 text-xs w-10 font-mono">
                            {formatTime(trackInfo.duration)}
                        </span>
                    </div>
                )}

                {/* Controls */}
                <div className="flex items-center justify-center gap-5">
                    {/* Play local jukebox (if not connected to Spotify) */}
                    {!started && !spotify.authenticated && (
                        <button
                            onClick={() => playTrack(LOCAL_PLAYLIST[0], 0, LOCAL_PLAYLIST)}
                            className="px-5 py-2.5 rounded-full bg-white/10 border border-white/20
                                text-white text-sm font-medium hover:bg-white/20 transition-all"
                        >
                            Play Local Jukebox
                        </button>
                    )}

                    {/* Play/Pause */}
                    {started && (
                        <button
                            onClick={audioData.isPlaying ? pause : play}
                            className="w-12 h-12 flex items-center justify-center rounded-full
                                bg-white/10 border border-white/20 hover:bg-white/20 transition-all"
                        >
                            {audioData.isPlaying ? (
                                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                                    <rect x="6" y="4" width="4" height="16" rx="1" />
                                    <rect x="14" y="4" width="4" height="16" rx="1" />
                                </svg>
                            ) : (
                                <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M8 5v14l11-7z" />
                                </svg>
                            )}
                        </button>
                    )}

                    {/* Skip */}
                    {started && playlist.length > 1 && (
                        <button
                            onClick={handleSkip}
                            className="w-10 h-10 flex items-center justify-center rounded-full
                                bg-white/10 border border-white/20 hover:bg-white/20 transition-all"
                        >
                            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M6 4l12 8-12 8V4z" />
                                <rect x="18" y="4" width="2" height="16" />
                            </svg>
                        </button>
                    )}

                    {/* Volume */}
                    {started && (
                        <div className="flex items-center gap-2">
                            <svg className="w-4 h-4 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M12 6l-4 4H4v4h4l4 4V6z" />
                            </svg>
                            <input
                                type="range"
                                min={0} max={1.5} step={0.01}
                                value={volumeVal}
                                onChange={handleVolumeChange}
                                className="w-20 h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                                    [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5
                                    [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
                            />
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default function SearchPlayer() {
    return (
        <PlayerAudioProvider>
            <SearchPlayerUI />
        </PlayerAudioProvider>
    )
}
