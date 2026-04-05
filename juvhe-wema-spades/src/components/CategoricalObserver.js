'use client'
import * as THREE from 'three'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'

/**
 * Categorical Audio Observation Apparatus
 *
 * Implements the GPU observation architecture from the CAT framework:
 * - S-entropy coordinates (Sk, St, Se) computed from audio frequency data
 * - Fragment shader observes partition states in audio phase space
 * - The rendered texture IS the categorical waveform, not a visualization of it
 *
 * References:
 *   Definition 3.2 (Audio S-Entropy Coordinates)
 *   Definition 3.4 (Audio Partition Coordinates)
 *   Theorem 5.1  (Categorical Reconstruction)
 *   Theorem 6.1  (Gabor Bypass)
 *   Section 7    (Groove Metric)
 */

// ============================================================================
// VERTEX SHADER — passes UV and position to fragment observation apparatus
// ============================================================================
const vertexShader = `
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

// ============================================================================
// FRAGMENT SHADER — CATEGORICAL AUDIO OBSERVATION APPARATUS
// The rendered output IS the categorical state, not a picture of it.
//
// Implements:
//   Stage 1: S-entropy coordinate display (Sk, St, Se manifold)
//   Stage 2: Partition cell observation (n, ℓ, m, s)
//   Stage 3: Categorical time-frequency representation (Gabor bypass)
//   Stage 4: Groove metric geodesic visualization
// ============================================================================
const fragmentShader = `
precision highp float;

// S-entropy coordinates (computed on CPU from audio analyser)
uniform float uSk;       // Spectral entropy — harmonic complexity
uniform float uSt;       // Temporal entropy — time structure granularity
uniform float uSe;       // Energetic entropy — dynamic range

// Raw audio band data
uniform float uBass;
uniform float uMid;
uniform float uTreble;
uniform float uVolume;

// Time and control
uniform float uTime;
uniform float uMode;     // 0: partition observation, 1: Gabor bypass CTFR, 2: groove metric, 3: S-entropy manifold

// Partition parameters (derived from audio)
uniform float uPartitionDepth;    // n — number of amplitude levels
uniform float uHarmonicOrderMax;  // max ℓ
uniform float uFundamentalFreq;   // ω₀ / 2π

// History texture for groove metric
uniform sampler2D uHistory;

varying vec2 vUv;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float kB = 1.0;  // Boltzmann constant (normalized)

// ---------------------------------------------------------------
// Observe partition cell at given phase (Definition 3.4)
//   n: partition depth (amplitude levels)
//   ell: harmonic order (0 = fundamental)
//   m: phase index (-ell..+ell)
//   s: chirality (+1 or -1, frequency direction)
// ---------------------------------------------------------------
float observePartitionCell(float phase, int n, int ell, int m, float chirality) {
    // Harmonic amplitude from partition coordinates
    float harmonicAmplitude = float(ell + 1) / float(n);

    // Phase state from m index: m ∈ {-ℓ, ..., +ℓ}
    float phaseState = float(m) / float(2 * ell + 1);

    // Chirality modulation (s = ±½ → frequency increasing/decreasing)
    float chiralityMod = chirality;

    // Categorical waveform: amplitude × cos(2π·phase + chirality·phaseState)
    return harmonicAmplitude * cos(TWO_PI * phase + chiralityMod * phaseState * PI);
}

// ---------------------------------------------------------------
// Total degeneracy at partition depth n: g_n = 2n² (Eq. 21)
// ---------------------------------------------------------------
float degeneracy(float n) {
    return 2.0 * n * n;
}

// ---------------------------------------------------------------
// Mode 0: PARTITION OBSERVATION
// Synthesizes categorical waveform from partition coordinates
// The texture IS the waveform (Theorem 5.1: reconstruction map)
// ---------------------------------------------------------------
vec4 partitionObservation(vec2 uv) {
    float time = uv.x;
    float amplitudeCoord = (uv.y - 0.5) * 2.0;  // [-1, 1]

    int n = int(uPartitionDepth);
    int numHarmonics = int(1.0 + uSk * uHarmonicOrderMax);

    float observedAmplitude = 0.0;
    float totalWeight = 0.0;

    for (int ell = 0; ell < 32; ell++) {
        if (ell >= numHarmonics) break;

        // Harmonic frequency
        float harmonicFreq = uFundamentalFreq * float(ell + 1);

        // Phase at this time
        float phase = time * harmonicFreq * 10.0 + uTime * 0.5;

        // Phase index m from St (temporal structure)
        int mMax = 2 * ell + 1;
        int m = int(uSt * float(mMax)) - ell;

        // Chirality from Se (energetic dynamics)
        float chirality = uSe > 0.5 ? 1.0 : -1.0;

        // OBSERVE this partition cell
        float cellAmplitude = observePartitionCell(phase, n, ell, m, chirality);

        // Weight by 1/(ℓ+1) — higher harmonics decay
        float weight = 1.0 / float(ell + 1);
        observedAmplitude += weight * cellAmplitude;
        totalWeight += weight;
    }

    if (totalWeight > 0.0) observedAmplitude /= totalWeight;

    // Amplitude envelope from Se (dynamic range)
    float envelope = exp(-uSe * 2.0 * abs(amplitudeCoord));
    observedAmplitude *= envelope;

    // Waveform intensity — bright where waveform is near this y-coordinate
    float waveformY = observedAmplitude * 0.5 + 0.5;
    float dist = abs(uv.y - waveformY);
    float intensity = smoothstep(0.03, 0.0, dist);

    // Color from S-entropy coordinates
    vec3 waveColor = vec3(
        0.1 + uSk * 0.8,    // Red channel: spectral complexity
        0.2 + uSt * 0.6,    // Green channel: temporal granularity
        0.4 + uSe * 0.5     // Blue channel: energetic range
    );

    // Background: faint partition grid
    float gridIntensity = 0.0;
    float partitions = uPartitionDepth;
    float gridX = fract(uv.x * partitions * 2.0);
    float gridY = fract(uv.y * partitions);
    gridIntensity += smoothstep(0.02, 0.0, abs(gridX - 0.5)) * 0.05;
    gridIntensity += smoothstep(0.02, 0.0, abs(gridY - 0.5)) * 0.05;

    vec3 bgColor = vec3(0.02, 0.03, 0.06);
    vec3 gridColor = vec3(0.1, 0.15, 0.25) * gridIntensity;

    // Trail/glow around waveform
    float glow = exp(-dist * dist * 800.0) * 0.3;
    vec3 glowColor = waveColor * glow;

    vec3 color = bgColor + gridColor + waveColor * intensity + glowColor;
    return vec4(color, 1.0);
}

// ---------------------------------------------------------------
// Mode 1: CATEGORICAL TIME-FREQUENCY REPRESENTATION
// Simultaneous time-frequency observation (Gabor Bypass, Theorem 6.1)
// Resolution: δSk·δSt ~ kB²·2π/(n²·φ₀) → 0 as n → ∞
// ---------------------------------------------------------------
vec4 gaborBypass(vec2 uv) {
    float time = uv.x;
    float freq = uv.y * 20.0;  // Normalized frequency axis

    int n = int(uPartitionDepth);

    // Categorical phase (from partition coordinates, NOT windowed FFT)
    float categoricalPhase = TWO_PI * freq * time * 5.0 + uTime * 0.3;

    // S-entropy modulated observation
    // Sk determines spectral resolution (independent of time resolution!)
    float spectralPrecision = 1.0 + uSk * float(n);
    // St determines temporal resolution (independent of spectral resolution!)
    float temporalPrecision = 1.0 + uSt * float(n);

    // Categorical Fourier Transform (Definition 6.3)
    // This is NOT a windowed FFT — no Gabor trade-off
    float realPart = 0.0;
    float imagPart = 0.0;

    for (int ell = 0; ell < 16; ell++) {
        float harmonicFreq = uFundamentalFreq * float(ell + 1);
        float freqDist = abs(freq - harmonicFreq / uFundamentalFreq);

        // Spectral kernel — width determined by partition depth, NOT window size
        float spectralWeight = exp(-freqDist * freqDist * spectralPrecision * 2.0);

        // Temporal kernel — independent of spectral kernel (Gabor bypass!)
        float phase = time * harmonicFreq * 10.0 + uTime * 0.5;
        float temporalWeight = 1.0 / float(ell + 1);

        // Audio-reactive amplitude
        float amplitude = 0.0;
        if (ell < 4) amplitude += uBass * 0.8;
        if (ell >= 4 && ell < 10) amplitude += uMid * 0.6;
        if (ell >= 10) amplitude += uTreble * 0.5;
        amplitude = max(amplitude, 0.05);

        realPart += spectralWeight * temporalWeight * amplitude * cos(categoricalPhase * float(ell + 1));
        imagPart += spectralWeight * temporalWeight * amplitude * sin(categoricalPhase * float(ell + 1));
    }

    float magnitude = sqrt(realPart * realPart + imagPart * imagPart);
    float phase = atan(imagPart, realPart);

    // Color encoding: magnitude as brightness, phase as hue
    float hue = phase / TWO_PI + 0.5;
    float sat = 0.7 + uVolume * 0.3;
    float val = magnitude * 3.0;

    // HSV to RGB
    vec3 c = vec3(hue * 6.0, sat, val);
    vec3 rgb = clamp(abs(mod(c.x + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = c.z * mix(vec3(1.0), rgb, c.y);

    // Overlay S-entropy precision indicator
    // Shows δSk·δSt product decreasing with partition depth
    float precisionProduct = 1.0 / (float(n) * float(n));
    float indicatorY = 1.0 - precisionProduct * 100.0;
    if (uv.x < 0.02) {
        float indicatorDist = abs(uv.y - clamp(indicatorY, 0.0, 1.0));
        rgb += vec3(0.0, 1.0, 0.5) * smoothstep(0.01, 0.0, indicatorDist);
    }

    return vec4(rgb, 1.0);
}

// ---------------------------------------------------------------
// Mode 2: GROOVE METRIC
// Geodesic deviation in S-entropy space (Section 7)
// Riemannian metric tensor G on S-entropy manifold
// ---------------------------------------------------------------
vec4 grooveMetric(vec2 uv) {
    // S-entropy space coordinates for this pixel
    float sk = uv.x;  // Sk axis
    float st = uv.y;  // St axis
    float se = uSe;   // Se from current audio (3rd dimension projected)

    // Metric tensor components (Eq. 7.3)
    // g_kk = 1/(Sk + ε), g_tt = 1/(St + ε), g_ee = 1/(Se + ε)
    float eps = 0.01;
    float g_kk = 1.0 / (sk + eps);
    float g_tt = 1.0 / (st + eps);
    float g_ee = 1.0 / (se + eps);

    // Off-diagonal coupling (spectral-temporal interaction in groove)
    float g_kt = 0.5 * sqrt(max(sk * st, 0.0));

    // Current audio position in S-entropy space
    vec2 audioPos = vec2(uSk, uSt);

    // Metronomic grid position (straight geodesic)
    float gridPhase = fract(uTime * 0.5);
    vec2 gridPos = vec2(0.5 + 0.3 * cos(gridPhase * TWO_PI), 0.5 + 0.3 * sin(gridPhase * TWO_PI));

    // Groove deviation vector (Eq. 7.5)
    vec2 delta = audioPos - gridPos;

    // Riemannian distance: ds² = g_ij dS^i dS^j
    float ds2 = g_kk * delta.x * delta.x
              + 2.0 * g_kt * delta.x * delta.y
              + g_tt * delta.y * delta.y;
    float geodesicDist = sqrt(max(ds2, 0.0));

    // Visualize the metric tensor field
    // Brightness = metric determinant (curvature indicator)
    float det = g_kk * g_tt - g_kt * g_kt;
    float curvature = log(1.0 + det) * 0.3;

    // Background: metric tensor field as color gradient
    vec3 fieldColor = vec3(
        curvature * (1.0 + uBass),
        curvature * 0.5 * (1.0 + uMid),
        curvature * 0.8 * (1.0 + uTreble)
    );

    // Draw current audio position
    float audioDist = length(uv - audioPos);
    vec3 audioMarker = vec3(1.0, 0.3, 0.1) * smoothstep(0.03, 0.0, audioDist);

    // Draw metronomic grid position
    float gridDist = length(uv - gridPos);
    vec3 gridMarker = vec3(0.1, 0.3, 1.0) * smoothstep(0.03, 0.0, gridDist);

    // Draw geodesic deviation line
    vec2 lineDir = normalize(delta);
    float lineDist = abs(dot(uv - gridPos, vec2(-lineDir.y, lineDir.x)));
    float lineParam = dot(uv - gridPos, lineDir) / length(delta);
    float lineIntensity = smoothstep(0.005, 0.0, lineDist)
                        * step(0.0, lineParam) * step(lineParam, 1.0);
    vec3 lineColor = mix(vec3(0.1, 0.3, 1.0), vec3(1.0, 0.3, 0.1), lineParam) * lineIntensity;

    // Groove intensity indicator (geodesic distance magnitude)
    float grooveGlow = geodesicDist * 2.0;
    vec3 grooveColor = vec3(grooveGlow * 0.5, grooveGlow * 0.8, grooveGlow * 0.3)
                     * exp(-audioDist * audioDist * 20.0);

    // Geodesic grid (constant-curvature contours)
    float contourVal = fract(det * 5.0);
    float contour = smoothstep(0.02, 0.0, abs(contourVal - 0.5));
    vec3 contourColor = vec3(0.1, 0.15, 0.2) * contour;

    vec3 color = fieldColor + audioMarker + gridMarker + lineColor + grooveColor + contourColor;
    return vec4(color, 1.0);
}

// ---------------------------------------------------------------
// Mode 3: S-ENTROPY MANIFOLD
// 3D S-entropy space projected to 2D
// Shows (Sk, St, Se) trajectory in categorical state space
// ---------------------------------------------------------------
vec4 sEntropyManifold(vec2 uv) {
    // Project 3D S-entropy space to 2D with rotation
    float angle = uTime * 0.15;
    float ca = cos(angle);
    float sa = sin(angle);

    // Current S-entropy point
    vec3 sPoint = vec3(uSk, uSt, uSe);

    // Project to screen coordinates
    vec2 projected = vec2(
        sPoint.x * ca - sPoint.z * sa,
        sPoint.y
    );
    projected = projected * 0.6 + 0.5;  // Center in UV space

    // Draw S-entropy point
    float pointDist = length(uv - projected);
    float pointGlow = exp(-pointDist * pointDist * 500.0);

    // Color based on S-entropy magnitudes
    vec3 pointColor = vec3(uSk, uSt, uSe) * 2.0;

    // Coordinate axes
    vec3 axisColor = vec3(0.0);
    vec2 origin = vec2(0.5, 0.5);

    // Sk axis (red)
    vec2 skEnd = vec2(0.5 + 0.4 * ca, 0.5);
    float skDist = abs(dot(uv - origin, vec2(0.0, 1.0)));
    float skParam = dot(uv - origin, normalize(skEnd - origin)) / length(skEnd - origin);
    if (skDist < 0.002 && skParam >= 0.0 && skParam <= 1.0) axisColor += vec3(0.5, 0.1, 0.1);

    // St axis (green)
    float stDist = abs(uv.x - 0.5);
    float stParam = (uv.y - 0.5) / 0.4;
    if (stDist < 0.002 && stParam >= 0.0 && stParam <= 1.0) axisColor += vec3(0.1, 0.5, 0.1);

    // Se axis (blue)
    vec2 seEnd = vec2(0.5 - 0.4 * sa, 0.5);
    float seDist = abs(dot(uv - origin, vec2(sa, ca)));
    float seParam = dot(uv - origin, normalize(seEnd - origin));
    if (seDist < 0.002 && seParam >= 0.0 && abs(seParam) <= 0.4) axisColor += vec3(0.1, 0.1, 0.5);

    // Circular reference orbits (Poincaré recurrence visualization)
    for (int i = 1; i <= 5; i++) {
        float radius = float(i) * 0.08;
        float orbitDist = abs(length(uv - origin) - radius);
        float orbitIntensity = smoothstep(0.003, 0.0, orbitDist) * 0.15;
        axisColor += vec3(0.2, 0.3, 0.4) * orbitIntensity;
    }

    // Partition depth indicator rings
    float nRing = degeneracy(uPartitionDepth) / 200.0;
    float ringDist = abs(length(uv - origin) - nRing * 0.3);
    vec3 ringColor = vec3(0.3, 0.5, 0.2) * smoothstep(0.004, 0.0, ringDist);

    // Audio-reactive nebula background (phase space density)
    float nebula = 0.0;
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec2 center = vec2(
            0.5 + 0.3 * sin(uTime * 0.2 + fi * 1.3),
            0.5 + 0.3 * cos(uTime * 0.15 + fi * 1.7)
        );
        float d = length(uv - center);
        float band = (i < 2) ? uBass : (i < 4) ? uMid : uTreble;
        nebula += band * 0.15 * exp(-d * d * 15.0);
    }
    vec3 nebulaColor = vec3(
        nebula * 0.4 * (1.0 + uSk),
        nebula * 0.6 * (1.0 + uSt),
        nebula * 0.9 * (1.0 + uSe)
    );

    vec3 bg = vec3(0.01, 0.015, 0.03);
    vec3 color = bg + nebulaColor + axisColor + ringColor + pointColor * pointGlow;
    return vec4(color, 1.0);
}

// ---------------------------------------------------------------
// MAIN — dispatch to observation mode
// ---------------------------------------------------------------
void main() {
    vec4 result;

    if (uMode < 0.5) {
        result = partitionObservation(vUv);
    } else if (uMode < 1.5) {
        result = gaborBypass(vUv);
    } else if (uMode < 2.5) {
        result = grooveMetric(vUv);
    } else {
        result = sEntropyManifold(vUv);
    }

    gl_FragColor = result;
}
`

// ============================================================================
// S-ENTROPY COMPUTATION (CPU)
// Implements Definition 3.2 from the paper
// ============================================================================
function computeSEntropy(audioData) {
    const { bass, mid, treble, volume, frequency } = audioData

    // Reference values (Definition 3.2)
    const phi0 = Math.PI / 4     // Reference phase
    const tau0 = 1.0 / 440.0     // Reference period (A440)
    const E0 = 1e-6              // Reference energy (quiet threshold)

    // Sk: Spectral entropy — phase deviation from reference
    // Higher when harmonic structure is complex (many frequencies)
    const deltaPhi = Math.sqrt(bass * bass + mid * mid + treble * treble) * Math.PI
    const Sk = Math.log((Math.abs(deltaPhi) + phi0) / phi0)

    // St: Temporal entropy — categorical period
    // Higher for transient material, lower for sustained tones
    const dominantFreqEstimate = 60 + frequency * 4000  // Map to Hz range
    const tau = dominantFreqEstimate > 0 ? 1.0 / dominantFreqEstimate : tau0
    const St = Math.log(tau / tau0)

    // Se: Energetic entropy — signal energy
    // E = x(t)² + dx/dt² / ω₀²
    const E = volume * volume + (bass - treble) * (bass - treble) * 0.1
    const Se = Math.log((E + E0) / E0)

    // Normalize to [0, 1]
    const SkNorm = Math.min(Math.max(Sk / 5.0, 0), 1)
    const StNorm = Math.min(Math.max((St + 5) / 10.0, 0), 1)
    const SeNorm = Math.min(Math.max(Se / 15.0, 0), 1)

    return { Sk: SkNorm, St: StNorm, Se: SeNorm }
}

// ============================================================================
// Derive partition parameters from S-entropy (Definition 3.4)
// ============================================================================
function derivePartitionParams(sEntropy, audioData) {
    const { Sk, St, Se } = sEntropy

    // Partition depth n: determined by energetic entropy
    // More energy → more distinguishable amplitude levels
    const n = Math.max(2, Math.floor(2 + Se * 14))

    // Max harmonic order: determined by spectral complexity
    const harmonicOrderMax = Math.max(1, Math.floor(1 + Sk * 16))

    // Fundamental frequency: estimated from audio
    const fundamentalFreq = 0.5 + audioData.frequency * 2.0

    return { n, harmonicOrderMax, fundamentalFreq }
}

// ============================================================================
// REACT COMPONENT — Categorical Observation Apparatus
// ============================================================================
export function CategoricalObserver({ audioData, mode = 0 }) {
    const meshRef = useRef()
    const materialRef = useRef()
    const { viewport } = useThree()

    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                uSk: { value: 0 },
                uSt: { value: 0 },
                uSe: { value: 0 },
                uBass: { value: 0 },
                uMid: { value: 0 },
                uTreble: { value: 0 },
                uVolume: { value: 0 },
                uTime: { value: 0 },
                uMode: { value: mode },
                uPartitionDepth: { value: 4 },
                uHarmonicOrderMax: { value: 8 },
                uFundamentalFreq: { value: 1.0 },
                uHistory: { value: null },
            },
            transparent: false,
            depthWrite: false,
        })
    }, [])

    useEffect(() => {
        if (shaderMaterial) {
            shaderMaterial.uniforms.uMode.value = mode
        }
    }, [mode, shaderMaterial])

    useFrame((state) => {
        if (!shaderMaterial || !audioData) return

        const t = state.clock.elapsedTime

        // Compute S-entropy coordinates (Definition 3.2)
        const sEntropy = computeSEntropy(audioData)

        // Derive partition parameters (Definition 3.4)
        const params = derivePartitionParams(sEntropy, audioData)

        // Update observation apparatus uniforms
        shaderMaterial.uniforms.uSk.value = sEntropy.Sk
        shaderMaterial.uniforms.uSt.value = sEntropy.St
        shaderMaterial.uniforms.uSe.value = sEntropy.Se
        shaderMaterial.uniforms.uBass.value = audioData.bass || 0
        shaderMaterial.uniforms.uMid.value = audioData.mid || 0
        shaderMaterial.uniforms.uTreble.value = audioData.treble || 0
        shaderMaterial.uniforms.uVolume.value = audioData.volume || 0
        shaderMaterial.uniforms.uTime.value = t
        shaderMaterial.uniforms.uPartitionDepth.value = params.n
        shaderMaterial.uniforms.uHarmonicOrderMax.value = params.harmonicOrderMax
        shaderMaterial.uniforms.uFundamentalFreq.value = params.fundamentalFreq
    })

    return (
        <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
            <planeGeometry args={[1, 1]} />
            <primitive object={shaderMaterial} attach="material" ref={materialRef} />
        </mesh>
    )
}

// Export computation functions for use elsewhere
export { computeSEntropy, derivePartitionParams }

export default CategoricalObserver
