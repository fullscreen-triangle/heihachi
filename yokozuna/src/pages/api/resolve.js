/**
 * /api/resolve — the full Acquire orchestration (paper Alg. 1, end to end).
 *
 * 1. Symbolic path (cheap): /api/resolve-meta → partial signature from metadata.
 * 2. Acoustic descent (expensive): iff there is a residual AND a Python worker
 *    is configured (PYTHON_WORKER_URL), hand the URL + already-known names to
 *    the worker, which downloads the audio, segments it, and fingerprints ONLY
 *    the residual segments (Cor. 4.8: acoustics on the residual, nowhere else).
 * 3. Merge the acoustic names back into the residual positions and return the
 *    completed signature plus whatever residual remains (honest output —
 *    unresolved "ID" tracks stay unresolved, Rem. 6.11).
 *
 * Degrades gracefully: with no worker configured, returns the metadata-only
 * signature with its residual gaps shown honestly.
 */

const WORKER = process.env.PYTHON_WORKER_URL // e.g. http://localhost:5000
const WORKER_POLL_MS = 4000
const WORKER_MAX_POLLS = 120 // ~8 min ceiling for a long mix

async function callMeta(req, url) {
    // Call the sibling route in-process via absolute URL derived from the host.
    const proto = (req.headers['x-forwarded-proto'] || 'http').split(',')[0]
    const host = req.headers.host
    const res = await fetch(`${proto}://${host}/api/resolve-meta`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.error || 'metadata resolution failed')
    return data
}

async function callWorker(url, knownNames) {
    // Submit the resolve job.
    const submit = await fetch(`${WORKER}/api/v1/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, known_names: knownNames }),
    })
    const job = await submit.json()
    if (!submit.ok) throw new Error(job.error || 'worker submit failed')

    // Synchronous result?
    if (job.signature) return job

    // Otherwise poll the job.
    const jobId = job.job_id || job.jobId
    if (!jobId) throw new Error('worker returned no job id')
    for (let i = 0; i < WORKER_MAX_POLLS; i++) {
        await new Promise((r) => setTimeout(r, WORKER_POLL_MS))
        const s = await fetch(`${WORKER}/api/v1/resolve/jobs/${jobId}`)
        const state = await s.json()
        if (state.status === 'completed') return state.result || state
        if (state.status === 'failed') throw new Error(state.error || 'worker job failed')
    }
    throw new Error('worker timed out')
}

/** Overlay acoustic results onto the metadata signature by position. */
function merge(metaSig, acousticSig) {
    const byPos = new Map()
    for (const a of acousticSig || []) byPos.set(a.pos, a)
    const merged = metaSig.map((s) => {
        if (s.name) return s // already placed by a name; nothing owed (Thm. 4.7)
        const a = byPos.get(s.pos)
        if (a && a.name) {
            return { ...s, name: a.name, artist: a.artist || s.artist, source: 'acoustic' }
        }
        return { ...s, source: 'id' } // stays unresolved — honest residual
    })
    const residual = merged.filter((s) => !s.name).map((s) => s.pos)
    return { signature: merged, residual }
}

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' })
    }
    const { url } = req.body || {}
    if (!url) return res.status(400).json({ error: 'No url provided' })

    try {
        const meta = await callMeta(req, url)

        const hasResidual = (meta.residual || []).length > 0 || meta.signature.length === 0
        let signature = meta.signature
        let residual = meta.residual
        let acousticUsed = false
        let workerError = null

        if (hasResidual && WORKER) {
            try {
                const knownNames = meta.signature
                    .filter((s) => s.name)
                    .map((s) => ({ pos: s.pos, tStart: s.tStart, name: s.name, artist: s.artist }))
                const acoustic = await callWorker(url, knownNames)
                const m = merge(
                    meta.signature.length ? meta.signature : acoustic.signature,
                    acoustic.signature
                )
                signature = m.signature
                residual = m.residual
                acousticUsed = true
            } catch (e) {
                workerError = e.message // degrade to metadata-only, report why
            }
        }

        return res.status(200).json({
            url,
            platform: meta.platform,
            title: meta.title,
            author: meta.author,
            thumbnail: meta.thumbnail,
            signature,
            residual,
            acousticUsed,
            workerConfigured: Boolean(WORKER),
            workerError,
        })
    } catch (err) {
        return res.status(500).json({ error: err.message })
    }
}
