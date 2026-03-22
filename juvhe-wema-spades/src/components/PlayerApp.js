'use client'
import { useState, useCallback, useEffect, useRef } from 'react'
import { PlayerAudioProvider, usePlayerAudio } from './PlayerAudioProvider'
import PlayerVisualizer from './PlayerVisualizer'

const PLAYLIST = [
    { name: 'Noisia - Feed the Machine (BSE)', src: '/audio/BSE_NOISA_Feed_the_Machine.mp3' },
    { name: 'Konflict - Messiah (Magnetude Remix)', src: '/audio/Konflict-Messiah-Magnetude.mp3' },
    { name: 'Noisia - Stigma (Neosignal Remix)', src: '/audio/Noisia_Stigma(NeosignalRemix).mp3' },
    { name: 'Spor - Running Man', src: '/audio/Spor_RunningMan.mp3' },
    { name: 'Squarepusher - Dark Steering', src: '/audio/Squarepusher_Dark_Steering.mp3' },
]

function PlayerUI() {
    const {
        audioData, trackInfo, ambientCompensation,
        loadTrack, play, pause, seek, setVolume,
        toggleAmbientCompensation
    } = usePlayerAudio()

    const [vizMode, setVizMode] = useState('desk')
    const [volumeVal, setVolumeVal] = useState(1)
    const [currentIndex, setCurrentIndex] = useState(0)
    const [started, setStarted] = useState(false)
    const indexRef = useRef(0)

    const playTrackAtIndex = useCallback(async (index) => {
        const track = PLAYLIST[index]
        indexRef.current = index
        setCurrentIndex(index)
        await loadTrack(track.src, track.name)
        play()
    }, [loadTrack, play])

    const handleStart = useCallback(async () => {
        setStarted(true)
        await playTrackAtIndex(0)
    }, [playTrackAtIndex])

    const handleSkip = useCallback(async () => {
        const nextIndex = (indexRef.current + 1) % PLAYLIST.length
        await playTrackAtIndex(nextIndex)
    }, [playTrackAtIndex])

    // Auto-advance when track ends
    useEffect(() => {
        if (!started) return
        if (trackInfo.duration > 0 && trackInfo.currentTime >= trackInfo.duration - 0.3 && !audioData.isPlaying) {
            const nextIndex = (indexRef.current + 1) % PLAYLIST.length
            playTrackAtIndex(nextIndex)
        }
    }, [audioData.isPlaying, trackInfo.currentTime, trackInfo.duration, started, playTrackAtIndex])

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
        <div className="relative w-full h-full bg-[#0a0a0f] overflow-hidden">
            {/* 3D Visualizer Background */}
            <div className="absolute inset-0 z-0">
                <PlayerVisualizer mode={vizMode} audioData={audioData} />
            </div>

            {/* Top bar — viz mode toggle + ambient */}
            <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-6 py-4">
                <div className="flex items-center gap-3">
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
            </div>

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
                        {trackInfo.name || 'Press play to start the jukebox'}
                    </span>
                    {started && (
                        <span className="text-white/40 text-xs ml-2">
                            {currentIndex + 1} / {PLAYLIST.length}
                        </span>
                    )}
                </div>

                {/* Seek bar */}
                {started && (
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
                )}

                {/* Controls row */}
                <div className="flex items-center justify-center gap-6">
                    {/* Play/Pause */}
                    <button
                        onClick={started ? (audioData.isPlaying ? pause : play) : handleStart}
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

                    {/* Skip */}
                    {started && (
                        <button
                            onClick={handleSkip}
                            className="w-10 h-10 flex items-center justify-center rounded-full bg-white/10 border border-white/20
                                hover:bg-white/20 transition-all"
                            title="Skip to next track"
                        >
                            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M6 4l12 8-12 8V4z" />
                                <rect x="18" y="4" width="2" height="16" />
                            </svg>
                        </button>
                    )}

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
