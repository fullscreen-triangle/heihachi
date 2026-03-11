'use client'
import { useState, useRef, useCallback, useEffect } from 'react'
import { PlayerAudioProvider, usePlayerAudio } from './PlayerAudioProvider'
import PlayerVisualizer from './PlayerVisualizer'
import TrackSearch from './TrackSearch'

// Available tracks in public/audio/
const LIBRARY = [
    { name: 'Audio Omega', src: '/audio/Audio_Omega.mp3' },
    { name: 'Noisia - Feed the Machine (BSE)', src: '/audio/BSE_NOISA_Feed_the_Machine.mp3' },
    { name: 'Mindscape - Jarhead', src: '/audio/audio_mindscape_jarhead.mp3' },
    { name: 'Benga - Electro West', src: '/audio/benga_electro_west.mp3' },
    { name: "DJ's Fresh - Heavyweight", src: '/audio/djs_fresh_heavyweight.mp3' },
    { name: 'Dream Continuum - Be Free', src: '/audio/dream_continuum_be_free.mp3' },
]

function PlayerUI() {
    const {
        audioData, trackInfo, ambientCompensation,
        loadTrack, loadFile, play, pause, seek, setVolume,
        toggleAmbientCompensation
    } = usePlayerAudio()

    const [vizMode, setVizMode] = useState('desk') // 'desk' | 'raymarch'
    const [showSearch, setShowSearch] = useState(false)
    const [showLibrary, setShowLibrary] = useState(false)
    const [volumeVal, setVolumeVal] = useState(1)
    const fileInputRef = useRef(null)
    const dropRef = useRef(null)

    // Drag and drop
    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
    }, [])

    const handleDrop = useCallback(async (e) => {
        e.preventDefault()
        e.stopPropagation()
        const file = e.dataTransfer.files[0]
        if (file && file.type.startsWith('audio/')) {
            await loadFile(file)
            play()
        }
    }, [loadFile, play])

    const handleFileSelect = useCallback(async (e) => {
        const file = e.target.files[0]
        if (file) {
            await loadFile(file)
            play()
        }
    }, [loadFile, play])

    const handleLibrarySelect = useCallback(async (track) => {
        await loadTrack(track.src, track.name)
        play()
        setShowLibrary(false)
    }, [loadTrack, play])

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

    return (
        <div
            ref={dropRef}
            className="relative w-full h-full bg-[#0a0a0f] overflow-hidden"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            {/* 3D Visualizer Background */}
            <div className="absolute inset-0 z-0">
                <PlayerVisualizer mode={vizMode} audioData={audioData} />
            </div>

            {/* Top bar — viz mode toggle + search + ambient */}
            <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4">
                <div className="flex items-center gap-3">
                    {/* Viz mode toggle */}
                    <div className="flex rounded-full border border-white/20 overflow-hidden">
                        <button
                            onClick={() => setVizMode('desk')}
                            className={`px-3 py-1.5 text-xs font-medium transition-all ${vizMode === 'desk'
                                ? 'bg-white/20 text-white'
                                : 'text-white/50 hover:text-white/80'
                                }`}
                        >
                            Desk
                        </button>
                        <button
                            onClick={() => setVizMode('raymarch')}
                            className={`px-3 py-1.5 text-xs font-medium transition-all ${vizMode === 'raymarch'
                                ? 'bg-white/20 text-white'
                                : 'text-white/50 hover:text-white/80'
                                }`}
                        >
                            Raymarch
                        </button>
                    </div>

                    {/* Ambient compensation toggle */}
                    <button
                        onClick={toggleAmbientCompensation}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${ambientCompensation
                            ? 'border-cyan-400/50 bg-cyan-400/10 text-cyan-300'
                            : 'border-white/20 text-white/50 hover:text-white/80'
                            }`}
                        title="Ambient noise compensation — uses microphone to adjust EQ"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                        {ambientCompensation ? 'ANC On' : 'ANC'}
                    </button>
                </div>

                <div className="flex items-center gap-3">
                    {/* Search button */}
                    <button
                        onClick={() => { setShowSearch(!showSearch); setShowLibrary(false) }}
                        className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${showSearch
                            ? 'border-purple-400/50 bg-purple-400/10 text-purple-300'
                            : 'border-white/20 text-white/50 hover:text-white/80'
                            }`}
                    >
                        Search
                    </button>

                    {/* Library button */}
                    <button
                        onClick={() => { setShowLibrary(!showLibrary); setShowSearch(false) }}
                        className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${showLibrary
                            ? 'border-blue-400/50 bg-blue-400/10 text-blue-300'
                            : 'border-white/20 text-white/50 hover:text-white/80'
                            }`}
                    >
                        Library
                    </button>

                    {/* Upload button */}
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-3 py-1.5 rounded-full text-xs font-medium border border-white/20 text-white/50 hover:text-white/80 transition-all"
                    >
                        Upload
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={handleFileSelect}
                    />
                </div>
            </div>

            {/* Search panel */}
            {showSearch && (
                <div className="absolute top-14 right-6 z-30 w-80">
                    <TrackSearch onSelect={async (track) => {
                        if (track.previewUrl) {
                            await loadTrack(track.previewUrl, `${track.artist} - ${track.title}`)
                            play()
                        }
                        setShowSearch(false)
                    }} />
                </div>
            )}

            {/* Library panel */}
            {showLibrary && (
                <div className="absolute top-14 right-6 z-30 w-72 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg overflow-hidden">
                    <div className="p-3 border-b border-white/10">
                        <span className="text-white/70 text-xs font-medium uppercase tracking-wider">Track Library</span>
                    </div>
                    <div className="max-h-64 overflow-y-auto">
                        {LIBRARY.map((track, i) => (
                            <button
                                key={i}
                                onClick={() => handleLibrarySelect(track)}
                                className="w-full text-left px-4 py-2.5 text-sm text-white/80 hover:bg-white/10 transition-all border-b border-white/5 last:border-0"
                            >
                                {track.name}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Audio level indicators */}
            {audioData.isPlaying && (
                <div className="absolute left-6 bottom-28 z-20 flex flex-col gap-1.5">
                    <LevelBar label="B" value={audioData.bass} color="rgb(239, 68, 68)" />
                    <LevelBar label="M" value={audioData.mid} color="rgb(34, 197, 94)" />
                    <LevelBar label="T" value={audioData.treble} color="rgb(59, 130, 246)" />
                </div>
            )}

            {/* Bottom transport bar */}
            <div className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/90 to-transparent pt-12 pb-4 px-6">
                {/* Track name */}
                <div className="text-center mb-3">
                    <span className="text-white/90 text-sm font-medium">
                        {trackInfo.name || 'No track loaded — drop a file or select from library'}
                    </span>
                </div>

                {/* Seek bar */}
                <div className="flex items-center gap-3 mb-3">
                    <span className="text-white/50 text-xs w-10 text-right font-mono">
                        {formatTime(trackInfo.currentTime)}
                    </span>
                    <input
                        type="range"
                        min={0}
                        max={trackInfo.duration || 1}
                        step={0.1}
                        value={trackInfo.currentTime || 0}
                        onChange={(e) => seek(parseFloat(e.target.value))}
                        className="flex-1 h-1 bg-white/10 rounded-full appearance-none cursor-pointer
                            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                            [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
                    />
                    <span className="text-white/50 text-xs w-10 font-mono">
                        {formatTime(trackInfo.duration)}
                    </span>
                </div>

                {/* Controls row */}
                <div className="flex items-center justify-center gap-6">
                    {/* Play/Pause */}
                    <button
                        onClick={audioData.isPlaying ? pause : play}
                        className="w-12 h-12 flex items-center justify-center rounded-full bg-white/10 border border-white/20
                            hover:bg-white/20 transition-all"
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

                    {/* Volume */}
                    <div className="flex items-center gap-2">
                        <svg className="w-4 h-4 text-white/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
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
                </div>
            </div>
        </div>
    )
}

function LevelBar({ label, value, color }) {
    return (
        <div className="flex items-center gap-2">
            <span className="text-[10px] text-white/40 w-3 font-mono">{label}</span>
            <div className="w-16 h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-75"
                    style={{ width: `${Math.min(value * 100, 100)}%`, backgroundColor: color }}
                />
            </div>
        </div>
    )
}

export default function PlayerApp() {
    return (
        <PlayerAudioProvider>
            <PlayerUI />
        </PlayerAudioProvider>
    )
}
