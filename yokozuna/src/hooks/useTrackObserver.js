'use client'
import { useRef, useCallback, useState } from 'react'
import { TrackSpectrum, computeInterference } from '../lib/categoricalAudio'

/**
 * Hook: observes audio data per frame and accumulates
 * a TrackSpectrum for each track played.
 *
 * Usage:
 *   const { observe, currentSpectrum, library, getInterference } = useTrackObserver()
 *   // Call observe(audioData) every frame while playing
 *   // Call startTrack(id, name) when a new track starts
 */
export function useTrackObserver() {
    const spectrumRef = useRef(null)
    const [library, setLibrary] = useState({}) // trackId → spectrum JSON
    const [currentTrackId, setCurrentTrackId] = useState(null)

    const startTrack = useCallback((trackId, trackName) => {
        // Save previous track to library
        if (spectrumRef.current && spectrumRef.current.count > 10) {
            const json = spectrumRef.current.toJSON()
            setLibrary(prev => ({ ...prev, [json.trackId]: json }))
        }
        // Start new observation
        spectrumRef.current = new TrackSpectrum(trackId, trackName)
        setCurrentTrackId(trackId)
    }, [])

    const observe = useCallback((audioData) => {
        if (spectrumRef.current && audioData.isPlaying) {
            spectrumRef.current.addObservation(audioData)
        }
    }, [])

    const finishTrack = useCallback(() => {
        if (spectrumRef.current && spectrumRef.current.count > 10) {
            const json = spectrumRef.current.toJSON()
            setLibrary(prev => ({ ...prev, [json.trackId]: json }))
        }
    }, [])

    const getCurrentSpectrum = useCallback(() => {
        return spectrumRef.current?.toJSON() || null
    }, [])

    // Interference between current track and any library track
    const getInterference = useCallback((targetTrackId) => {
        const current = spectrumRef.current?.toJSON()
        const target = library[targetTrackId]
        if (!current || !target) return null
        return computeInterference(current, target)
    }, [library])

    return {
        startTrack,
        observe,
        finishTrack,
        getCurrentSpectrum,
        getInterference,
        currentTrackId,
        library,
    }
}
