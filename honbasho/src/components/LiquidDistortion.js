'use client'
import { useRef, useMemo, useEffect, useState, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useTexture } from '@react-three/drei'
import { EffectComposer, ShockWave, ChromaticAberration, Vignette } from '@react-three/postprocessing'
import { BlendFunction, KernelSize } from 'postprocessing'
import * as THREE from 'three'

/**
 * Liquid Distortion Effect
 *
 * Post-processing pipeline applied over album artwork:
 *   1. Album art rendered as fullscreen textured plane
 *   2. Custom water displacement pass (audio-reactive ripples)
 *   3. ShockWave effect (bass-triggered ring ripples)
 *   4. ChromaticAberration (distortion-proportional color split)
 *   5. Vignette (subtle edge darkening)
 *
 * The displacement IS the spectra — water droplet interference patterns.
 * Audio drives the physics: bass = large surface waves, treble = fine caustics.
 *
 * Single image: liquid distortion on album art
 * Playlist: slides transition via displacement wave wash between images
 */

// ── Water displacement custom shader (as post-processing effect) ───
const waterVertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const waterFragmentShader = `
precision highp float;

uniform sampler2D uImage;
uniform float uTime;
uniform float uBass;
uniform float uMid;
uniform float uTreble;
uniform float uVolume;
uniform float uIntensity;
uniform vec2 uDisplaceSpeed;
uniform float uTransition;
uniform float uTransitionDir;

varying vec2 vUv;

const float PI = 3.14159265359;

// Simplex-ish noise for water surface
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float val = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        val += amp * noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return val;
}

void main() {
    vec2 uv = vUv;

    // Water surface displacement from FBM noise (scrolling)
    vec2 noiseUv = uv * 3.0 + uTime * uDisplaceSpeed;
    float waterNoise = fbm(noiseUv) - 0.5;

    // Audio-reactive modulation
    // Bass: large slow ripples (ocean swell)
    float bassRipple = sin(uv.y * 4.0 + uTime * 2.0) * uBass * 0.4;
    bassRipple += sin(uv.x * 3.0 + uTime * 1.5) * uBass * 0.3;

    // Mid: medium wave interference
    float midWave = sin(uv.x * 10.0 + uv.y * 8.0 + uTime * 4.0) * uMid * 0.15;

    // Treble: fine surface detail (capillary waves)
    float trebleDetail = fbm(uv * 15.0 + uTime * 3.0) * uTreble * 0.08;

    // Water droplet concentric ripples from center
    float dist = length(uv - 0.5);
    float ripple = sin(dist * 25.0 - uTime * 5.0 * (1.0 + uBass * 0.5)) * 0.5 + 0.5;
    ripple *= exp(-dist * 4.0); // Decay outward
    ripple *= uVolume * 0.6;

    // Secondary ripple source (offset, driven by mid)
    float dist2 = length(uv - vec2(0.3, 0.7));
    float ripple2 = sin(dist2 * 20.0 - uTime * 4.0) * 0.5 + 0.5;
    ripple2 *= exp(-dist2 * 5.0) * uMid * 0.4;

    // Combine all displacement
    vec2 totalDisp = vec2(0.0);
    totalDisp += vec2(waterNoise * 0.02, waterNoise * 0.015);
    totalDisp += vec2(bassRipple * 0.015, bassRipple * 0.01);
    totalDisp += vec2(midWave * 0.01);
    totalDisp += vec2(trebleDetail * 0.005);
    totalDisp += vec2(ripple * 0.02, ripple * 0.015);
    totalDisp += vec2(ripple2 * 0.015, ripple2 * 0.01);
    totalDisp *= uIntensity;

    // Slide transition: wave wash
    if (uTransition > 0.01 && uTransition < 0.99) {
        float wave = sin(uv.x * 5.0 + uTransition * PI * 3.0);
        wave *= uTransition * (1.0 - uTransition) * 4.0;
        totalDisp += vec2(wave * 0.25 * uTransitionDir, wave * 0.08);
    }

    vec2 distortedUv = clamp(uv + totalDisp, 0.001, 0.999);

    // Sample with slight chromatic offset at distortion peaks
    float aberration = length(totalDisp) * 1.5;
    vec4 color;
    if (aberration > 0.003) {
        float r = texture2D(uImage, distortedUv + vec2(aberration * 0.2, 0.0)).r;
        float g = texture2D(uImage, distortedUv).g;
        float b = texture2D(uImage, distortedUv - vec2(aberration * 0.2, 0.0)).b;
        float a = texture2D(uImage, distortedUv).a;
        color = vec4(r, g, b, a);
    } else {
        color = texture2D(uImage, distortedUv);
    }

    // Water surface caustic highlights
    float caustic = ripple * uVolume * 0.12 + ripple2 * 0.08;
    color.rgb += vec3(caustic * 0.6, caustic * 0.8, caustic * 1.0);

    gl_FragColor = color;
}
`

// ── Album art plane with water displacement ────────────────────────
function WaterPlane({ imageUrl, audioData, intensity, transition, transitionDir }) {
    const meshRef = useRef()
    const { viewport } = useThree()
    const [texture, setTexture] = useState(null)

    useEffect(() => {
        if (!imageUrl) return
        const loader = new THREE.TextureLoader()
        loader.load(imageUrl, (tex) => {
            tex.minFilter = THREE.LinearFilter
            tex.magFilter = THREE.LinearFilter
            setTexture(tex)
        })
    }, [imageUrl])

    const material = useMemo(() => new THREE.ShaderMaterial({
        vertexShader: waterVertexShader,
        fragmentShader: waterFragmentShader,
        uniforms: {
            uImage: { value: null },
            uTime: { value: 0 },
            uBass: { value: 0 },
            uMid: { value: 0 },
            uTreble: { value: 0 },
            uVolume: { value: 0 },
            uIntensity: { value: 1.0 },
            uDisplaceSpeed: { value: new THREE.Vector2(0.03, 0.02) },
            uTransition: { value: 0 },
            uTransitionDir: { value: 1 },
        },
    }), [])

    useEffect(() => {
        if (texture) material.uniforms.uImage.value = texture
    }, [texture, material])

    useFrame((state) => {
        const u = material.uniforms
        u.uTime.value = state.clock.elapsedTime
        u.uBass.value = audioData?.bass || 0
        u.uMid.value = audioData?.mid || 0
        u.uTreble.value = audioData?.treble || 0
        u.uVolume.value = audioData?.volume || 0
        u.uIntensity.value = intensity
        u.uTransition.value = transition
        u.uTransitionDir.value = transitionDir
    })

    if (!texture) return null

    return (
        <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
            <planeGeometry args={[1, 1]} />
            <primitive object={material} attach="material" />
        </mesh>
    )
}

// ── Post-processing scene with audio-reactive effects ──────────────
function LiquidScene({ imageUrl, audioData, intensity = 1, transition = 0, transitionDir = 1 }) {
    const shockWaveRef = useRef()
    const chromaRef = useRef()
    const lastBassHit = useRef(0)
    const bassThreshold = useRef(0)

    useFrame((state) => {
        if (!audioData) return

        const t = state.clock.elapsedTime
        const bass = audioData.bass || 0

        // Trigger shockwave on bass transients
        if (bass > 0.6 && bass > bassThreshold.current + 0.15 && t - lastBassHit.current > 0.8) {
            if (shockWaveRef.current) {
                shockWaveRef.current.position.set(0, 0, 0)
                shockWaveRef.current.speed = 1.5 + bass
                shockWaveRef.current.amplitude = 0.02 + bass * 0.03
                shockWaveRef.current.waveSize = 0.1 + bass * 0.15
            }
            lastBassHit.current = t
        }
        bassThreshold.current = bass * 0.9 + bassThreshold.current * 0.1 // Adaptive threshold

        // Chromatic aberration follows volume
        if (chromaRef.current) {
            const vol = audioData.volume || 0
            chromaRef.current.offset.set(
                Math.sin(t * 2) * vol * 0.003,
                Math.cos(t * 1.5) * vol * 0.002
            )
        }
    })

    return (
        <>
            <WaterPlane
                imageUrl={imageUrl}
                audioData={audioData}
                intensity={intensity}
                transition={transition}
                transitionDir={transitionDir}
            />
            <EffectComposer>
                <ShockWave
                    ref={shockWaveRef}
                    position={[0, 0, 0]}
                    speed={2}
                    maxRadius={1}
                    waveSize={0.15}
                    amplitude={0.02}
                />
                <ChromaticAberration
                    ref={chromaRef}
                    blendFunction={BlendFunction.NORMAL}
                    offset={new THREE.Vector2(0.001, 0.001)}
                />
                <Vignette
                    offset={0.3}
                    darkness={0.6}
                    blendFunction={BlendFunction.NORMAL}
                />
            </EffectComposer>
        </>
    )
}

// ── Exported: single image with liquid distortion ──────────────────
export function LiquidDistortion({ imageUrl, audioData, intensity = 1, className = '' }) {
    return (
        <div className={`w-full h-full ${className}`}>
            <Canvas
                camera={{ position: [0, 0, 1] }}
                gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}
                style={{ width: '100%', height: '100%' }}
            >
                <LiquidScene
                    imageUrl={imageUrl}
                    audioData={audioData}
                    intensity={intensity}
                />
            </Canvas>
        </div>
    )
}

// ── Exported: playlist slider with liquid transitions ──────────────
export function LiquidSlider({ images, currentIndex, audioData, intensity = 1, onTransitionEnd }) {
    const [activeIndex, setActiveIndex] = useState(currentIndex)
    const [transition, setTransition] = useState(0)
    const [transitionDir, setTransitionDir] = useState(1)
    const animRef = useRef(null)

    useEffect(() => {
        if (currentIndex === activeIndex) return

        setTransitionDir(currentIndex > activeIndex ? 1 : -1)

        let start = null
        const duration = 1200

        const animate = (timestamp) => {
            if (!start) start = timestamp
            const progress = Math.min((timestamp - start) / duration, 1)
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

    const displayIndex = transition < 0.5 ? activeIndex : currentIndex
    const imageUrl = images[Math.min(displayIndex, images.length - 1)]

    return (
        <div className="w-full h-full">
            <Canvas
                camera={{ position: [0, 0, 1] }}
                gl={{ antialias: false, alpha: true, powerPreference: 'high-performance' }}
                style={{ width: '100%', height: '100%' }}
            >
                <LiquidScene
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
