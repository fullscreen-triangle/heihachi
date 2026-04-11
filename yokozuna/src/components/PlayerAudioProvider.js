'use client'
import { createContext, useContext, useRef, useState, useEffect, useCallback } from 'react'

const PlayerAudioCtx = createContext()

export function PlayerAudioProvider({ children }) {
    const [audioData, setAudioData] = useState({
        frequency: 0, bass: 0, mid: 0, treble: 0, volume: 0, isPlaying: false
    })
    const [trackInfo, setTrackInfo] = useState({ name: '', duration: 0, currentTime: 0 })
    const [ambientCompensation, setAmbientCompensation] = useState(false)
    const [ambientProfile, setAmbientProfile] = useState(null) // Environmental spectrum

    const analyserRef = useRef(null)
    const dataArrayRef = useRef(null)
    const audioCtxRef = useRef(null)
    const sourceRef = useRef(null)
    const audioElRef = useRef(null)
    const animFrameRef = useRef(null)
    const connectedRef = useRef(false)
    const gainRef = useRef(null)
    const filtersRef = useRef([])

    // Ambient compensation refs
    const micSourceRef = useRef(null)
    const micAnalyserRef = useRef(null)
    const micDataRef = useRef(null)
    const micStreamRef = useRef(null)

    // Filter bank center frequencies (Hz)
    const FILTER_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

    const ensureAudioContext = useCallback(() => {
        if (!audioCtxRef.current) {
            audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)()
            analyserRef.current = audioCtxRef.current.createAnalyser()
            analyserRef.current.fftSize = 512
            analyserRef.current.smoothingTimeConstant = 0.8
            dataArrayRef.current = new Uint8Array(analyserRef.current.frequencyBinCount)

            // Create gain node
            gainRef.current = audioCtxRef.current.createGain()

            // Create parametric EQ filter bank
            filtersRef.current = FILTER_BANDS.map(freq => {
                const filter = audioCtxRef.current.createBiquadFilter()
                filter.type = 'peaking'
                filter.frequency.value = freq
                filter.Q.value = 1.4
                filter.gain.value = 0
                return filter
            })
        }
        return audioCtxRef.current
    }, [])

    const connectChain = useCallback(() => {
        if (connectedRef.current || !sourceRef.current) return

        // Chain: source → filters → gain → analyser → destination
        let lastNode = sourceRef.current
        filtersRef.current.forEach(filter => {
            lastNode.connect(filter)
            lastNode = filter
        })
        lastNode.connect(gainRef.current)
        gainRef.current.connect(analyserRef.current)
        analyserRef.current.connect(audioCtxRef.current.destination)
        connectedRef.current = true
    }, [])

    const disconnectChain = useCallback(() => {
        if (!connectedRef.current || !sourceRef.current) return
        try {
            sourceRef.current.disconnect()
            filtersRef.current.forEach(f => f.disconnect())
            gainRef.current.disconnect()
            analyserRef.current.disconnect()
        } catch (e) { /* already disconnected */ }
        connectedRef.current = false
    }, [])

    const loadTrack = useCallback(async (src, name = 'Unknown Track') => {
        const ctx = ensureAudioContext()
        if (ctx.state === 'suspended') await ctx.resume()

        // Stop current playback
        if (audioElRef.current) {
            audioElRef.current.pause()
            audioElRef.current.src = ''
        }
        if (connectedRef.current) disconnectChain()

        // Create new audio element
        audioElRef.current = new Audio(src)
        audioElRef.current.crossOrigin = 'anonymous'
        audioElRef.current.loop = false

        // Create new source and connect
        sourceRef.current = ctx.createMediaElementSource(audioElRef.current)
        connectChain()

        // Track time updates
        audioElRef.current.addEventListener('loadedmetadata', () => {
            setTrackInfo(prev => ({ ...prev, name, duration: audioElRef.current.duration }))
        })
        audioElRef.current.addEventListener('timeupdate', () => {
            setTrackInfo(prev => ({ ...prev, currentTime: audioElRef.current.currentTime }))
        })
        audioElRef.current.addEventListener('ended', () => {
            setAudioData(prev => ({ ...prev, isPlaying: false }))
        })

        setTrackInfo({ name, duration: 0, currentTime: 0 })
    }, [ensureAudioContext, connectChain, disconnectChain])

    const loadFile = useCallback(async (file) => {
        const url = URL.createObjectURL(file)
        await loadTrack(url, file.name.replace(/\.[^/.]+$/, ''))
    }, [loadTrack])

    const play = useCallback(async () => {
        if (!audioElRef.current) return
        const ctx = ensureAudioContext()
        if (ctx.state === 'suspended') await ctx.resume()
        await audioElRef.current.play()
        setAudioData(prev => ({ ...prev, isPlaying: true }))
        analyzeLoop()
    }, [ensureAudioContext])

    const pause = useCallback(() => {
        if (audioElRef.current) audioElRef.current.pause()
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
        setAudioData(prev => ({ ...prev, isPlaying: false }))
    }, [])

    const seek = useCallback((time) => {
        if (audioElRef.current) audioElRef.current.currentTime = time
    }, [])

    const setVolume = useCallback((v) => {
        if (gainRef.current) gainRef.current.gain.value = v
    }, [])

    const analyzeLoop = useCallback(() => {
        if (!analyserRef.current || !dataArrayRef.current) return
        analyserRef.current.getByteFrequencyData(dataArrayRef.current)

        const d = dataArrayRef.current
        const len = d.length

        const bass = d.slice(0, len / 4).reduce((a, b) => a + b, 0) / (len / 4) / 255
        const mid = d.slice(len / 4, len / 2).reduce((a, b) => a + b, 0) / (len / 4) / 255
        const treble = d.slice(len / 2).reduce((a, b) => a + b, 0) / (len / 2) / 255
        const volume = d.reduce((a, b) => a + b, 0) / len / 255

        setAudioData({ frequency: d[0] / 255, bass, mid, treble, volume, isPlaying: true })

        // Ambient integration: shape music to blend with environment
        if (ambientCompensation && micAnalyserRef.current && micDataRef.current) {
            micAnalyserRef.current.getByteFrequencyData(micDataRef.current)
            updateCompensationFilters(micDataRef.current)

            // Expose ambient profile for observation/library
            const md = micDataRef.current
            const mLen = md.length
            const aBass = md.slice(0, mLen / 4).reduce((a, b) => a + b, 0) / (mLen / 4) / 255
            const aMid = md.slice(mLen / 4, mLen / 2).reduce((a, b) => a + b, 0) / (mLen / 4) / 255
            const aTreble = md.slice(mLen / 2).reduce((a, b) => a + b, 0) / (mLen / 2) / 255
            const aVol = md.reduce((a, b) => a + b, 0) / mLen / 255
            setAmbientProfile({ bass: aBass, mid: aMid, treble: aTreble, volume: aVol })
        }

        animFrameRef.current = requestAnimationFrame(analyzeLoop)
    }, [ambientCompensation])

    const updateCompensationFilters = useCallback((micData) => {
        const binCount = micData.length
        const sampleRate = audioCtxRef.current?.sampleRate || 44100

        // Get the music's current spectrum for comparison
        const musicData = dataArrayRef.current
        if (!musicData) return

        filtersRef.current.forEach((filter, i) => {
            const freq = FILTER_BANDS[i]
            const bin = Math.round(freq / (sampleRate / 2) * binCount)
            const start = Math.max(0, bin - 2)
            const end = Math.min(binCount - 1, bin + 2)

            // Measure ambient energy at this band
            let ambientSum = 0
            for (let j = start; j <= end; j++) ambientSum += micData[j]
            const ambientLevel = ambientSum / (end - start + 1) / 255

            // Measure music energy at this band
            const musicBin = Math.round(freq / (sampleRate / 2) * musicData.length)
            const mStart = Math.max(0, musicBin - 2)
            const mEnd = Math.min(musicData.length - 1, musicBin + 2)
            let musicSum = 0
            for (let j = mStart; j <= mEnd; j++) musicSum += musicData[j]
            const musicLevel = musicSum / (mEnd - mStart + 1) / 255

            // Environmental audio integration:
            // Where ambient is loud AND music is also loud → slight cut (avoid masking)
            // Where ambient is loud AND music is quiet → leave alone (environment fills)
            // Where ambient is quiet AND music present → slight boost (music fills the gap)
            // The result: music and environment form a coherent acoustic image

            const overlap = ambientLevel * musicLevel  // Both present = masking risk
            const gap = Math.max(0, musicLevel - ambientLevel) // Music in quiet band = can boost
            const fill = Math.max(0, ambientLevel - musicLevel) // Ambient fills, don't fight it

            // Cut where both overlap (reduce masking), boost where music fills gaps
            const adjustDb = -overlap * 4 + gap * 3 - fill * 1
            filter.gain.value = Math.max(-6, Math.min(6, adjustDb))
        })
    }, [])

    const toggleAmbientCompensation = useCallback(async () => {
        if (!ambientCompensation) {
            try {
                const ctx = ensureAudioContext()
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
                micStreamRef.current = stream
                micSourceRef.current = ctx.createMediaStreamSource(stream)
                micAnalyserRef.current = ctx.createAnalyser()
                micAnalyserRef.current.fftSize = 512
                micDataRef.current = new Uint8Array(micAnalyserRef.current.frequencyBinCount)
                micSourceRef.current.connect(micAnalyserRef.current)
                // Don't connect mic to destination (we don't want to hear it)
                setAmbientCompensation(true)
            } catch (e) {
                console.error('Microphone access denied:', e)
            }
        } else {
            // Disable — disconnect mic, reset filters
            if (micSourceRef.current) micSourceRef.current.disconnect()
            if (micStreamRef.current) micStreamRef.current.getTracks().forEach(t => t.stop())
            filtersRef.current.forEach(f => { f.gain.value = 0 })
            setAmbientCompensation(false)
        }
    }, [ambientCompensation, ensureAudioContext])

    // Frequency data for visualizer (raw array)
    const getFrequencyData = useCallback(() => {
        if (!analyserRef.current) return null
        const data = new Uint8Array(analyserRef.current.frequencyBinCount)
        analyserRef.current.getByteFrequencyData(data)
        return data
    }, [])

    useEffect(() => {
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
            if (audioElRef.current) { audioElRef.current.pause(); audioElRef.current.src = '' }
            if (micStreamRef.current) micStreamRef.current.getTracks().forEach(t => t.stop())
            if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') audioCtxRef.current.close()
        }
    }, [])

    return (
        <PlayerAudioCtx.Provider value={{
            audioData, trackInfo, ambientCompensation, ambientProfile,
            loadTrack, loadFile, play, pause, seek, setVolume,
            toggleAmbientCompensation, getFrequencyData
        }}>
            {children}
        </PlayerAudioCtx.Provider>
    )
}

export function usePlayerAudio() {
    const ctx = useContext(PlayerAudioCtx)
    if (!ctx) throw new Error('usePlayerAudio must be inside PlayerAudioProvider')
    return ctx
}
