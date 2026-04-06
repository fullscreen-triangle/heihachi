'use client'
import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * Liquid Distortion Effect
 *
 * Applies water-surface interference patterns to album artwork.
 * The displacement IS the spectra — water droplet ripples on a surface.
 * Audio-reactive: bass drives large ripples, treble drives fine detail.
 *
 * Based on honbasho/main.js PixiJS DisplacementFilter approach,
 * reimplemented in three.js ShaderMaterial for consistency with
 * existing rendering pipeline.
 *
 * Single image: liquid distortion on album art
 * Playlist: slides transition via displacement wave wash
 */

// ── Liquid displacement fragment shader ────────────────────────────
const liquidVertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const liquidFragmentShader = `
precision highp float;

uniform sampler2D uImage;        // Album art
uniform sampler2D uDisplacement; // Cloud/water noise map
uniform float uTime;
uniform float uBass;
uniform float uMid;
uniform float uTreble;
uniform float uVolume;
uniform float uIntensity;        // Overall distortion strength
uniform vec2 uDisplaceSpeed;     // Displacement map scroll speed
uniform vec2 uDisplaceScale;     // Displacement strength (x, y)
uniform float uTransition;       // 0..1 slide transition progress
uniform float uTransitionDir;    // 1 = forward, -1 = backward

varying vec2 vUv;

const float PI = 3.14159265359;

void main() {
    vec2 uv = vUv;

    // Sample displacement map (scrolling — creates water flow)
    vec2 dispUv = uv * 1.5 + uTime * uDisplaceSpeed;
    vec4 dispTexel = texture2D(uDisplacement, fract(dispUv));

    // Convert displacement to signed range [-1, 1]
    vec2 displacement = (dispTexel.rg - 0.5) * 2.0;

    // Audio-reactive displacement modulation
    // Bass: large slow ripples (water surface waves)
    float bassWave = sin(uv.y * 3.0 + uTime * 1.5) * uBass * 0.3;
    // Mid: medium ripples (secondary wave pattern)
    float midWave = sin(uv.x * 8.0 + uv.y * 6.0 + uTime * 3.0) * uMid * 0.15;
    // Treble: fine detail (water surface texture)
    float trebleWave = sin(uv.x * 20.0 + uTime * 8.0) * cos(uv.y * 15.0 + uTime * 6.0) * uTreble * 0.08;

    // Water droplet interference pattern
    // Concentric ripples from center, modulated by audio
    float dist = length(uv - 0.5);
    float ripple = sin(dist * 30.0 - uTime * 4.0 * (1.0 + uBass)) * 0.5 + 0.5;
    ripple *= exp(-dist * 3.0); // Decay from center
    ripple *= uVolume * 0.5;

    // Combine all displacement sources
    vec2 totalDisp = displacement * uDisplaceScale * uIntensity;
    totalDisp += vec2(bassWave + midWave + trebleWave) * uIntensity;
    totalDisp += vec2(ripple * 0.02, ripple * 0.015) * uIntensity;

    // Slide transition displacement (water wave wash)
    if (uTransition > 0.01 && uTransition < 0.99) {
        float transWave = sin(uv.x * 4.0 + uTransition * PI * 2.0) * uTransition * (1.0 - uTransition) * 4.0;
        totalDisp += vec2(transWave * 0.3 * uTransitionDir, transWave * 0.1);
    }

    // Apply displacement to UV
    vec2 distortedUv = uv + totalDisp;

    // Clamp to prevent sampling outside texture
    distortedUv = clamp(distortedUv, 0.0, 1.0);

    // Sample album art with distorted UVs
    vec4 color = texture2D(uImage, distortedUv);

    // Subtle chromatic aberration at distortion peaks
    float aberration = length(totalDisp) * 2.0;
    if (aberration > 0.002) {
        float r = texture2D(uImage, distortedUv + vec2(aberration * 0.3, 0.0)).r;
        float b = texture2D(uImage, distortedUv - vec2(aberration * 0.3, 0.0)).b;
        color.r = mix(color.r, r, 0.3);
        color.b = mix(color.b, b, 0.3);
    }

    // Water surface caustic highlight
    float caustic = ripple * uVolume * 0.15;
    color.rgb += vec3(caustic * 0.8, caustic * 0.9, caustic * 1.0);

    // Slight vignette
    float vignette = 1.0 - dist * 0.4;
    color.rgb *= vignette;

    gl_FragColor = color;
}
`

// ── Procedural displacement map (cloud/water noise) ────────────────
function generateDisplacementTexture(size = 512) {
    const data = new Uint8Array(size * size * 4)
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = (y * size + x) * 4
            // Multi-octave value noise for water-like displacement
            let val = 0
            let amp = 1
            let freq = 1
            for (let o = 0; o < 5; o++) {
                const nx = x * freq / size
                const ny = y * freq / size
                // Simple hash-based noise
                const h = Math.sin(nx * 127.1 + ny * 311.7) * 43758.5453
                val += (h - Math.floor(h)) * amp
                amp *= 0.5
                freq *= 2
            }
            val = Math.min(Math.max(val * 0.5, 0), 1)
            const byte = Math.floor(val * 255)
            data[i] = byte
            data[i + 1] = byte
            data[i + 2] = byte
            data[i + 3] = 255
        }
    }
    const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat)
    tex.wrapS = THREE.RepeatWrapping
    tex.wrapT = THREE.RepeatWrapping
    tex.needsUpdate = true
    return tex
}

// ── Internal mesh component ────────────────────────────────────────
function LiquidPlane({ imageUrl, audioData, intensity = 1, transition = 0, transitionDir = 1 }) {
    const meshRef = useRef()
    const { viewport } = useThree()

    const displacementTex = useMemo(() => generateDisplacementTexture(512), [])

    const [imageTex, setImageTex] = useState(null)

    useEffect(() => {
        if (!imageUrl) return
        const loader = new THREE.TextureLoader()
        loader.load(imageUrl, (tex) => {
            tex.minFilter = THREE.LinearFilter
            tex.magFilter = THREE.LinearFilter
            setImageTex(tex)
        })
    }, [imageUrl])

    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            vertexShader: liquidVertexShader,
            fragmentShader: liquidFragmentShader,
            uniforms: {
                uImage: { value: null },
                uDisplacement: { value: displacementTex },
                uTime: { value: 0 },
                uBass: { value: 0 },
                uMid: { value: 0 },
                uTreble: { value: 0 },
                uVolume: { value: 0 },
                uIntensity: { value: intensity },
                uDisplaceSpeed: { value: new THREE.Vector2(0.02, 0.015) },
                uDisplaceScale: { value: new THREE.Vector2(0.015, 0.01) },
                uTransition: { value: 0 },
                uTransitionDir: { value: 1 },
            },
            transparent: false,
        })
    }, [displacementTex, intensity])

    useEffect(() => {
        if (imageTex && shaderMaterial) {
            shaderMaterial.uniforms.uImage.value = imageTex
        }
    }, [imageTex, shaderMaterial])

    useFrame((state) => {
        if (!shaderMaterial) return
        const u = shaderMaterial.uniforms
        u.uTime.value = state.clock.elapsedTime
        u.uBass.value = audioData?.bass || 0
        u.uMid.value = audioData?.mid || 0
        u.uTreble.value = audioData?.treble || 0
        u.uVolume.value = audioData?.volume || 0
        u.uIntensity.value = intensity
        u.uTransition.value = transition
        u.uTransitionDir.value = transitionDir
    })

    if (!imageTex) return null

    return (
        <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
            <planeGeometry args={[1, 1]} />
            <primitive object={shaderMaterial} attach="material" />
        </mesh>
    )
}

// ── Exported component: single image ───────────────────────────────
export function LiquidDistortion({ imageUrl, audioData, intensity = 1, className = '' }) {
    return (
        <div className={`w-full h-full ${className}`}>
            <Canvas
                camera={{ position: [0, 0, 1] }}
                gl={{ antialias: false, alpha: true }}
                style={{ width: '100%', height: '100%' }}
            >
                <LiquidPlane
                    imageUrl={imageUrl}
                    audioData={audioData}
                    intensity={intensity}
                />
            </Canvas>
        </div>
    )
}

// ── Exported component: playlist slider ────────────────────────────
export function LiquidSlider({ images, currentIndex, audioData, intensity = 1, onTransitionEnd }) {
    const [activeIndex, setActiveIndex] = useState(currentIndex)
    const [transition, setTransition] = useState(0)
    const [transitionDir, setTransitionDir] = useState(1)
    const animRef = useRef(null)

    useEffect(() => {
        if (currentIndex === activeIndex) return

        setTransitionDir(currentIndex > activeIndex ? 1 : -1)

        // Animate transition 0 → 1
        let start = null
        const duration = 1200 // ms

        const animate = (timestamp) => {
            if (!start) start = timestamp
            const progress = Math.min((timestamp - start) / duration, 1)
            // Ease in-out
            const eased = progress < 0.5
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 2) / 2

            setTransition(eased)

            if (progress < 1) {
                animRef.current = requestAnimationFrame(animate)
            } else {
                setActiveIndex(currentIndex)
                setTransition(0)
                onTransitionEnd?.()
            }
        }

        animRef.current = requestAnimationFrame(animate)
        return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
    }, [currentIndex, activeIndex, onTransitionEnd])

    // Show current image, transition shows displacement wash
    const displayIndex = transition < 0.5 ? activeIndex : currentIndex
    const imageUrl = images[displayIndex]

    return (
        <div className="w-full h-full">
            <Canvas
                camera={{ position: [0, 0, 1] }}
                gl={{ antialias: false, alpha: true }}
                style={{ width: '100%', height: '100%' }}
            >
                <LiquidPlane
                    imageUrl={imageUrl}
                    audioData={audioData}
                    intensity={intensity}
                    transition={transition}
                    transitionDir={transitionDir}
                />
            </Canvas>
        </div>
    )
}

export default LiquidDistortion
