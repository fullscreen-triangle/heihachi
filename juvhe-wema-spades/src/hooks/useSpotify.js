'use client'
import { useState, useEffect, useCallback } from 'react'
import {
    startSpotifyAuth, exchangeCode, isAuthenticated,
    getValidToken, logout, getMe, getPlaylists,
    getPlaylistTracks, getTopTracks,
} from '../lib/spotify'

/**
 * Hook: Spotify authentication and data access.
 *
 * Handles the OAuth PKCE flow, token management, and API calls.
 * Works alongside the local jukebox — Spotify is optional.
 */
export function useSpotify() {
    const [user, setUser] = useState(null)
    const [authenticated, setAuthenticated] = useState(false)
    const [playlists, setPlaylists] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    // Check auth state on mount & handle OAuth callback
    useEffect(() => {
        const params = new URLSearchParams(window.location.search)
        const code = params.get('spotify_code')
        const spotifyError = params.get('spotify_error')

        if (spotifyError) {
            setError(spotifyError)
            // Clean URL
            window.history.replaceState({}, '', window.location.pathname)
            return
        }

        if (code) {
            // Exchange code for token
            exchangeCode(code)
                .then(() => {
                    setAuthenticated(true)
                    // Clean URL
                    window.history.replaceState({}, '', window.location.pathname)
                    return loadUserData()
                })
                .catch(err => setError(err.message))
            return
        }

        // Check existing session
        if (isAuthenticated()) {
            setAuthenticated(true)
            loadUserData()
        }
    }, [])

    const loadUserData = useCallback(async () => {
        try {
            setLoading(true)
            const token = await getValidToken()
            if (!token) return

            const [me, lists] = await Promise.all([
                getMe(token.access_token),
                getPlaylists(token.access_token),
            ])
            setUser(me)
            setPlaylists(lists.items || [])
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }, [])

    const login = useCallback(() => startSpotifyAuth(), [])

    const handleLogout = useCallback(() => {
        logout()
        setAuthenticated(false)
        setUser(null)
        setPlaylists([])
    }, [])

    const fetchPlaylistTracks = useCallback(async (playlistId) => {
        const token = await getValidToken()
        if (!token) return []
        const data = await getPlaylistTracks(token.access_token, playlistId)
        return (data.items || [])
            .filter(item => item.track)
            .map(item => ({
                id: item.track.id,
                name: item.track.name,
                artist: item.track.artists?.map(a => a.name).join(', ') || '',
                album: item.track.album?.name || '',
                albumArt: item.track.album?.images?.[0]?.url || '',
                duration: item.track.duration_ms,
                previewUrl: item.track.preview_url,
                uri: item.track.uri,
            }))
    }, [])

    const fetchTopTracks = useCallback(async () => {
        const token = await getValidToken()
        if (!token) return []
        const data = await getTopTracks(token.access_token)
        return (data.items || []).map(track => ({
            id: track.id,
            name: track.name,
            artist: track.artists?.map(a => a.name).join(', ') || '',
            album: track.album?.name || '',
            albumArt: track.album?.images?.[0]?.url || '',
            duration: track.duration_ms,
            previewUrl: track.preview_url,
            uri: track.uri,
        }))
    }, [])

    return {
        authenticated,
        user,
        playlists,
        loading,
        error,
        login,
        logout: handleLogout,
        fetchPlaylistTracks,
        fetchTopTracks,
    }
}
