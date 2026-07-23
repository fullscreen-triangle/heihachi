/**
 * /api/resolve-meta — Stage 1 (Acquire), cheap symbolic path.
 *
 * Input : { url }  (a YouTube or SoundCloud link — 99% of requests)
 * Output: { platform, title, author, thumbnail,
 *           signature: [{pos, tStart, name, artist, title, source}],
 *           residual: [pos...] }
 *
 * This realises the paper's `Name` recogniser (Alg. 1) as metadata scraping:
 * oEmbed for title/author/thumbnail, then the page description / chapters
 * parsed into an ordered signature of least-sufficient handles. Positions with
 * no name are the residual, handed to the acoustic worker downstream. No audio
 * is downloaded here; this runs fine on Vercel serverless.
 */

import { parseTracklist } from '../../lib/tracklistParse'

const YT = /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/live\/)/i
const SC = /soundcloud\.com\//i

function detectPlatform(url) {
    if (YT.test(url)) return 'youtube'
    if (SC.test(url)) return 'soundcloud'
    return null
}

async function fetchText(url, opts = {}) {
    const res = await fetch(url, {
        headers: { 'User-Agent': 'YokozunaResolver/1.0 (+kundai.sachikonye@wzw.tum.de)' },
        ...opts,
    })
    if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`)
    return res.text()
}

async function oembed(platform, url) {
    const endpoint =
        platform === 'youtube'
            ? `https://www.youtube.com/oembed?url=${encodeURIComponent(url)}&format=json`
            : `https://soundcloud.com/oembed?url=${encodeURIComponent(url)}&format=json`
    try {
        const res = await fetch(endpoint, {
            headers: { 'User-Agent': 'YokozunaResolver/1.0' },
        })
        if (!res.ok) return {}
        return await res.json()
    } catch {
        return {}
    }
}

/**
 * Best-effort extraction of the description text from a YouTube/SoundCloud
 * page. YouTube embeds a "shortDescription" in the ytInitialPlayerResponse
 * JSON; SoundCloud embeds og:description and a JSON hydration blob. We pull
 * whatever we can and let the parser decide what is a tracklist.
 */
async function fetchDescription(platform, url) {
    let html
    try {
        html = await fetchText(url)
    } catch {
        return ''
    }

    if (platform === 'youtube') {
        // shortDescription is JSON-escaped; decode \n and \uXXXX.
        const m = html.match(/"shortDescription":"((?:[^"\\]|\\.)*)"/)
        if (m) {
            try {
                return JSON.parse(`"${m[1]}"`)
            } catch {
                return m[1].replace(/\\n/g, '\n').replace(/\\"/g, '"')
            }
        }
    }

    // Generic fallback: og:description meta tag.
    const og = html.match(
        /<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']+)["']/i
    )
    if (og) return og[1].replace(/&#10;/g, '\n').replace(/&amp;/g, '&')
    return ''
}

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' })
    }
    const { url } = req.body || {}
    if (!url || typeof url !== 'string') {
        return res.status(400).json({ error: 'No url provided' })
    }

    const platform = detectPlatform(url)
    if (!platform) {
        return res
            .status(422)
            .json({ error: 'Unsupported link — paste a YouTube or SoundCloud URL' })
    }

    try {
        const [meta, description] = await Promise.all([
            oembed(platform, url),
            fetchDescription(platform, url),
        ])

        const { signature, residual } = parseTracklist(description)

        return res.status(200).json({
            url,
            platform,
            title: meta.title || null,
            author: meta.author_name || null,
            thumbnail: meta.thumbnail_url || null,
            signature,
            residual,
            // A signature with no parseable tracklist is itself a full residual:
            // the whole item needs acoustic segmentation downstream.
            needsAcoustic: signature.length === 0 || residual.length > 0,
        })
    } catch (err) {
        return res.status(500).json({ error: err.message })
    }
}
