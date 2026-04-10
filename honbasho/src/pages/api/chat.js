/**
 * /api/chat — Purpose-trained LLM interface
 *
 * Receives: user prompt + ensemble thermodynamic state
 * Returns: target S-entropy region + Spotify search terms + explanation
 *
 * The LLM translates natural language into thermodynamic operations
 * on the ensemble. It does NOT perform matching — interference does that.
 * The LLM articulates and translates.
 *
 * Uses OpenAI GPT-4 with structured system context containing:
 *   - Ensemble thermodynamic state (T, Z, F, S_taste)
 *   - Per-track categorical observations
 *   - The categorical audio vocabulary
 */

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' })
    }

    const apiKey = process.env.OPENAI_API_KEY
    if (!apiKey) {
        return res.status(500).json({ error: 'OPENAI_API_KEY not configured' })
    }

    const { prompt, ensembleState } = req.body
    if (!prompt) {
        return res.status(400).json({ error: 'No prompt provided' })
    }

    // Build the system prompt with ensemble context
    const systemPrompt = buildSystemPrompt(ensembleState)

    try {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model: 'gpt-4o',
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: prompt },
                ],
                temperature: 0.7,
                max_tokens: 800,
                response_format: { type: 'json_object' },
            }),
        })

        if (!response.ok) {
            const err = await response.text()
            return res.status(response.status).json({ error: `OpenAI error: ${err}` })
        }

        const data = await response.json()
        const content = data.choices?.[0]?.message?.content

        let parsed
        try {
            parsed = JSON.parse(content)
        } catch {
            parsed = { explanation: content, searchTerms: [prompt], targetRegion: null }
        }

        return res.status(200).json(parsed)
    } catch (err) {
        return res.status(500).json({ error: err.message })
    }
}

function buildSystemPrompt(ensembleState) {
    const hasEnsemble = ensembleState && ensembleState.size > 0

    let ensembleContext = ''
    if (hasEnsemble) {
        const e = ensembleState
        ensembleContext = `
## Current Library Ensemble (${e.size} tracks observed)

Thermodynamic state:
- Temperature T = ${e.temperature?.toFixed(4)} (taste breadth: ${e.temperature > 0.3 ? 'eclectic' : e.temperature > 0.15 ? 'moderate' : 'focused'})
- Partition function Z = ${e.partitionFunction?.toFixed(4)}
- Free energy F = ${e.freeEnergy?.toFixed(4)}
- Taste entropy S = ${e.tasteEntropy?.toFixed(4)} (predictability: ${e.tasteEntropy > 1.5 ? 'unpredictable' : e.tasteEntropy > 0.8 ? 'moderate' : 'highly predictable'})
- Centroid: Sk=${e.centroid?.Sk?.toFixed(3)}, St=${e.centroid?.St?.toFixed(3)}, Se=${e.centroid?.Se?.toFixed(3)}

Tracks in library:
${(e.tracks || []).map((t, i) =>
    `  ${i + 1}. "${t.trackName}" — Sk=${t.sEntropy?.Sk?.toFixed(3)}, St=${t.sEntropy?.St?.toFixed(3)}, Se=${t.sEntropy?.Se?.toFixed(3)}, n*=${t.dominantPartitionDepth}, phase=[${(t.phaseSpectrum || []).map(p => p.toFixed(1)).join(',')}]`
).join('\n')}
`
    } else {
        ensembleContext = `
## Current Library Ensemble
Empty — no tracks observed yet. The user is starting fresh.
`
    }

    return `You are the search interface for a categorical audio music player. You translate natural-language music requests into structured search parameters.

## Framework

Audio signals are bounded oscillatory systems. Each track is an ensemble of oscillators with:
- S-entropy coordinates (Sk, St, Se): spectral complexity, temporal granularity, energetic range
- Partition coordinates (n, l, m, s): hierarchical harmonic structure
- Phase spectrum: accumulated phase across 8 oscillator classes

The user's library is a thermodynamic gas ensemble. Each track is a molecule. The ensemble has temperature (taste breadth), partition function, Boltzmann distribution, free energy, and taste entropy.

Finding tracks = sampling from the Boltzmann distribution.
Similarity = interference visibility between phase spectra (not distance in embedding space).

## S-Entropy Interpretation
- Sk (spectral entropy): LOW = simple/pure tones, HIGH = complex harmonics, dense textures
- St (temporal entropy): LOW = sustained/steady, HIGH = transient/rhythmically complex
- Se (energetic entropy): LOW = quiet/compressed, HIGH = dynamic/loud/aggressive

## Your Task

Given the user's prompt and their library ensemble state, respond with JSON:

{
  "searchTerms": ["spotify search query 1", "query 2"],
  "targetRegion": {
    "Sk": <target spectral entropy 0-1>,
    "St": <target temporal entropy 0-1>,
    "Se": <target energetic entropy 0-1>
  },
  "temperatureShift": <number, positive=explore, negative=refine, 0=neutral>,
  "channelPriority": [<indices 0-7 of oscillator classes to prioritize>],
  "explanation": "<1-2 sentences explaining the thermodynamic interpretation of the request, using categorical audio terms, NOT genre labels>"
}

The searchTerms should be practical Spotify search queries that would find tracks matching the target S-entropy region. Use artist names, descriptive terms (dark, aggressive, minimal, complex, atmospheric), tempo hints, and instrument characteristics. Do NOT just echo the user's words — translate their intent into effective search terms based on the ensemble state.

${ensembleContext}`
}
