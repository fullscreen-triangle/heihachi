'use client'
import * as THREE from 'three'
import { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'

/**
 * Interference Observation Shader
 *
 * Renders the superposition of two track spectra.
 * Similarity IS interference — no matching algorithm required.
 *
 * From ray-tracing.tex Theorem 14.1:
 *   V_cell = |1/N Σ_k e^{iΦ_k}|
 *
 * Constructive interference (bright) = similar tracks
 * Destructive interference (dark) = dissimilar tracks
 *
 * The rendered texture IS the similarity observation, not a
 * visualization of it.
 */

const vertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const fragmentShader = `
precision highp float;

// Track A phase spectrum (8 oscillator classes)
uniform float uPhaseA[8];
// Track B phase spectrum (8 oscillator classes)
uniform float uPhaseB[8];

// Track A S-entropy (mean)
uniform vec3 uSEntropyA;
// Track B S-entropy (mean)
uniform vec3 uSEntropyB;

// Scalar visibility result (computed CPU-side, displayed here)
uniform float uVisibility;

uniform float uTime;

varying vec2 vUv;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;

// ─── Wave superposition at a point ─────────────────────────────────
// Each oscillator class contributes a wave. We superpose A and B.
// The interference pattern IS the similarity structure.
vec4 interferenceField(vec2 uv) {
    float realSum = 0.0;
    float imagSum = 0.0;
    float intensityA = 0.0;
    float intensityB = 0.0;

    for (int k = 0; k < 8; k++) {
        float phiA = uPhaseA[k];
        float phiB = uPhaseB[k];

        // Spatial frequency for this oscillator class
        float freq = float(k + 1) * 2.0;

        // Track A wave: amplitude modulated by S-entropy
        float ampA = 0.3 + 0.1 * float(k < 3 ? 1 : 0) * uSEntropyA.x
                        + 0.1 * float(k >= 3 && k < 6 ? 1 : 0) * uSEntropyA.y
                        + 0.1 * float(k >= 6 ? 1 : 0) * uSEntropyA.z;

        // Track B wave
        float ampB = 0.3 + 0.1 * float(k < 3 ? 1 : 0) * uSEntropyB.x
                        + 0.1 * float(k >= 3 && k < 6 ? 1 : 0) * uSEntropyB.y
                        + 0.1 * float(k >= 6 ? 1 : 0) * uSEntropyB.z;

        // Wave propagation
        float waveA = ampA * sin(TWO_PI * freq * uv.x + phiA + uTime * 0.5);
        float waveB = ampB * sin(TWO_PI * freq * uv.y + phiB + uTime * 0.3);

        // Superposition
        float superposed = waveA + waveB;
        float interference = superposed * superposed; // Intensity ∝ |A+B|²

        // Phase difference → complex phasor sum
        float deltaPhi = phiA - phiB;
        realSum += cos(deltaPhi + freq * (uv.x - uv.y) * PI);
        imagSum += sin(deltaPhi + freq * (uv.x - uv.y) * PI);

        intensityA += waveA * waveA;
        intensityB += waveB * waveB;
    }

    // Local visibility at this pixel
    float localVis = sqrt(realSum * realSum + imagSum * imagSum) / 8.0;

    // Interference phase
    float intPhase = atan(imagSum, realSum);

    return vec4(localVis, intPhase, intensityA, intensityB);
}

void main() {
    vec4 field = interferenceField(vUv);

    float localVis = field.x;
    float intPhase = field.y;

    // Color mapping:
    // Constructive (high visibility) → bright, coherent color
    // Destructive (low visibility) → dark, fragmented
    // Phase → hue rotation

    float hue = intPhase / TWO_PI + 0.5;
    float sat = 0.6 + localVis * 0.4;
    float val = localVis * 0.8 + 0.1;

    // HSV to RGB
    vec3 c = vec3(hue * 6.0, sat, val);
    vec3 rgb = clamp(abs(mod(c.x + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = c.z * mix(vec3(1.0), rgb, c.y);

    // Fringe pattern — the actual interference bands
    float fringeIntensity = 0.0;
    for (int k = 0; k < 8; k++) {
        float deltaPhi = uPhaseA[k] - uPhaseB[k];
        float freq = float(k + 1) * 3.0;
        float fringe = cos(deltaPhi + freq * (vUv.x - vUv.y) * TWO_PI + uTime * 0.2);
        fringeIntensity += fringe;
    }
    fringeIntensity /= 8.0;

    // Bright fringes where constructive, dark where destructive
    float fringeBrightness = fringeIntensity * 0.5 + 0.5;
    rgb *= (0.5 + fringeBrightness * 0.5);

    // Global visibility indicator (bottom bar)
    if (vUv.y < 0.03) {
        float barFill = step(vUv.x, uVisibility);
        vec3 barColor = mix(
            vec3(0.8, 0.1, 0.1),  // Red = destructive
            vec3(0.1, 0.8, 0.3),  // Green = constructive
            uVisibility
        );
        rgb = mix(rgb, barColor * barFill + vec3(0.05) * (1.0 - barFill), 0.9);
    }

    // S-entropy position markers
    // Track A (left side)
    vec2 posA = vec2(0.1, uSEntropyA.x * 0.8 + 0.1);
    float distA = length(vUv - posA);
    rgb += vec3(0.0, 0.5, 1.0) * smoothstep(0.015, 0.0, distA);

    // Track B (right side)
    vec2 posB = vec2(0.9, uSEntropyB.x * 0.8 + 0.1);
    float distB = length(vUv - posB);
    rgb += vec3(1.0, 0.5, 0.0) * smoothstep(0.015, 0.0, distB);

    gl_FragColor = vec4(rgb, 1.0);
}
`

export function InterferenceObserver({ spectrumA, spectrumB, visibility = 0 }) {
    const meshRef = useRef()
    const { viewport } = useThree()

    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            vertexShader,
            fragmentShader,
            uniforms: {
                uPhaseA: { value: new Array(8).fill(0) },
                uPhaseB: { value: new Array(8).fill(0) },
                uSEntropyA: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
                uSEntropyB: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
                uVisibility: { value: 0 },
                uTime: { value: 0 },
            },
            transparent: false,
            depthWrite: false,
        })
    }, [])

    useEffect(() => {
        if (!shaderMaterial) return
        if (spectrumA?.phaseSpectrum) {
            shaderMaterial.uniforms.uPhaseA.value = spectrumA.phaseSpectrum
        }
        if (spectrumB?.phaseSpectrum) {
            shaderMaterial.uniforms.uPhaseB.value = spectrumB.phaseSpectrum
        }
        if (spectrumA?.sEntropy?.mean) {
            const m = spectrumA.sEntropy.mean
            shaderMaterial.uniforms.uSEntropyA.value.set(m.Sk, m.St, m.Se)
        }
        if (spectrumB?.sEntropy?.mean) {
            const m = spectrumB.sEntropy.mean
            shaderMaterial.uniforms.uSEntropyB.value.set(m.Sk, m.St, m.Se)
        }
        shaderMaterial.uniforms.uVisibility.value = visibility
    }, [spectrumA, spectrumB, visibility, shaderMaterial])

    useFrame((state) => {
        if (shaderMaterial) {
            shaderMaterial.uniforms.uTime.value = state.clock.elapsedTime
        }
    })

    return (
        <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
            <planeGeometry args={[1, 1]} />
            <primitive object={shaderMaterial} attach="material" />
        </mesh>
    )
}

export default InterferenceObserver
