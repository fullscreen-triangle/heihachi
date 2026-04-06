/**
 * Spotify OAuth callback handler
 * Receives the authorization code, redirects to player with code in URL
 * The actual token exchange happens client-side (PKCE — no secret needed)
 */
export default function handler(req, res) {
    const { code, error } = req.query

    if (error) {
        return res.redirect(`/player?spotify_error=${encodeURIComponent(error)}`)
    }

    if (!code) {
        return res.redirect('/player?spotify_error=no_code')
    }

    // Redirect to player with the auth code — client-side JS will exchange it
    res.redirect(`/player?spotify_code=${code}`)
}
