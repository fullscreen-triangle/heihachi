'use client'
import { useThree, useFrame } from '@react-three/fiber'
import { useRef, useMemo } from 'react'

import {
    MeshBasicNodeMaterial,
    float,
    Loop,
    If,
    Break,
    Fn,
    uv,
    vec3,
    timerLocal,
    sin,
    cos,
    min,
    max,
    abs,
    mix,
    normalize,
    dot,
    reflect,
    vec2,
    viewportResolution,
    uniform,
} from 'three/tsl'

// Audio uniforms — updated each frame from the AudioProvider
const uBass = uniform(0)
const uMid = uniform(0)
const uTreble = uniform(0)
const uVolume = uniform(0)

const timer = timerLocal(1)

// SDF: sphere
const sdSphere = Fn(([p, r]) => {
    return p.length().sub(r)
})

// SDF: torus
const sdTorus = Fn(([p, t]) => {
    const q = vec2(vec2(p.x, p.z).length().sub(t.x), p.y)
    return q.length().sub(t.y)
})

// Smooth min for organic blending
const smin = Fn(([a, b, k]) => {
    const h = max(k.sub(abs(a.sub(b))), 0).div(k)
    return min(a, b).sub(h.mul(h).mul(k).mul(0.25))
})

// Rotation matrix around Y axis
const rotateY = Fn(([p, angle]) => {
    const c = cos(angle)
    const s = sin(angle)
    return vec3(
        p.x.mul(c).add(p.z.mul(s)),
        p.y,
        p.x.mul(s).negate().add(p.z.mul(c))
    )
})

// Main SDF — audio-reactive scene
const sdf = Fn(([pos]) => {
    // Central sphere pulses with bass
    const bassRadius = float(0.4).add(uBass.mul(0.3))
    const sphere1 = sdSphere(pos, bassRadius)

    // Orbiting sphere driven by mid frequencies
    const orbitRadius = float(1.2).add(uMid.mul(0.5))
    const orbitAngle = timer.mul(1.5)
    const orbitPos = vec3(
        sin(orbitAngle).mul(orbitRadius),
        cos(timer.mul(0.7)).mul(uMid.mul(0.5)),
        cos(orbitAngle).mul(orbitRadius)
    )
    const midRadius = float(0.2).add(uMid.mul(0.15))
    const sphere2 = sdSphere(pos.sub(orbitPos), midRadius)

    // Second orbiting sphere offset by pi
    const orbitPos2 = vec3(
        sin(orbitAngle.add(3.14159)).mul(orbitRadius),
        cos(timer.mul(0.9)).mul(uTreble.mul(0.4)),
        cos(orbitAngle.add(3.14159)).mul(orbitRadius)
    )
    const trebleRadius = float(0.15).add(uTreble.mul(0.2))
    const sphere3 = sdSphere(pos.sub(orbitPos2), trebleRadius)

    // Torus ring — rotates and scales with volume
    const torusPos = rotateY(pos, timer.mul(0.3))
    const torusSize = vec2(float(0.8).add(uVolume.mul(0.4)), float(0.05).add(uVolume.mul(0.08)))
    const torus = sdTorus(torusPos, torusSize)

    // Blend everything with smooth min — smoothness driven by volume
    const k = float(0.3).add(uVolume.mul(0.4))
    const d1 = smin(sphere1, sphere2, k)
    const d2 = smin(d1, sphere3, k)
    return smin(d2, torus, k)
})

// Normal calculation via central differences
const calcNormal = Fn(([p]) => {
    const eps = float(0.0001)
    const h = vec2(eps, 0)
    return normalize(
        vec3(
            sdf(p.add(h.xyy)).sub(sdf(p.sub(h.xyy))),
            sdf(p.add(h.yxy)).sub(sdf(p.sub(h.yxy))),
            sdf(p.add(h.yyx)).sub(sdf(p.sub(h.yyx))),
        ),
    )
})

// Lighting — audio-reactive colors
const lighting = Fn(([ro, r]) => {
    const normal = calcNormal(r)
    const viewDir = normalize(ro.sub(r))

    // Ambient — shifts with audio
    const ambient = vec3(
        float(0.05).add(uBass.mul(0.1)),
        float(0.05).add(uMid.mul(0.05)),
        float(0.1).add(uTreble.mul(0.1))
    )

    // Diffuse — main directional light
    const lightDir = normalize(vec3(1, 1, 1))
    const lightColor = vec3(
        float(0.9).add(uBass.mul(0.3)),
        float(0.85).add(uMid.mul(0.15)),
        float(1.0)
    )
    const dp = max(0, dot(lightDir, normal))
    const diffuse = dp.mul(lightColor)

    // Hemisphere light — audio shifts the sky/ground colors
    const skyColor = vec3(
        float(0.0).add(uTreble.mul(0.5)),
        float(0.15).add(uMid.mul(0.3)),
        float(0.4).add(uBass.mul(0.3))
    )
    const groundColor = vec3(
        float(0.3).add(uBass.mul(0.4)),
        float(0.1).add(uMid.mul(0.2)),
        float(0.05)
    )
    const hemiMix = normal.y.mul(0.5).add(0.5)
    const hemi = mix(groundColor, skyColor, hemiMix)

    // Phong specular — sharpness driven by treble
    const ph = normalize(reflect(lightDir.negate(), normal))
    const specPower = float(16).add(uTreble.mul(48))
    const phongValue = max(0, dot(viewDir, ph)).pow(specPower)
    const specular = vec3(phongValue).toVar()

    // Fresnel
    const fresnel = float(1).sub(max(0, dot(viewDir, normal))).pow(2)
    specular.mulAssign(fresnel)

    // Combine
    const lit = ambient.mul(0.2).toVar()
    lit.addAssign(diffuse.mul(0.5))
    lit.addAssign(hemi.mul(0.3))

    // Base color shifts with audio
    const baseColor = vec3(
        float(0.08).add(uBass.mul(0.15)),
        float(0.12).add(uMid.mul(0.1)),
        float(0.18).add(uTreble.mul(0.15))
    )

    const finalColor = baseColor.mul(lit).toVar()
    finalColor.addAssign(specular.mul(float(0.5).add(uVolume.mul(0.5))))

    return finalColor
})

// Main raymarch function
const raymarch = Fn(() => {
    const _uv = uv().mul(viewportResolution.xy).mul(2).sub(viewportResolution.xy).div(viewportResolution.y)

    const rayOrigin = vec3(0, 0, float(-3).sub(uVolume.mul(0.5)))
    const rayDirection = vec3(_uv, 1).normalize()

    const t = float(0).toVar()
    const ray = rayOrigin.add(rayDirection.mul(t)).toVar()

    Loop({ start: 1, end: 80 }, () => {
        const d = sdf(ray)
        t.addAssign(d.mul(0.8))
        ray.assign(rayOrigin.add(rayDirection.mul(t)))

        If(d.lessThan(0.005), () => {
            Break()
        })

        If(t.greaterThan(50), () => {
            Break()
        })
    })

    return lighting(rayOrigin, ray)
})()

const raymarchMaterial = new MeshBasicNodeMaterial()
raymarchMaterial.colorNode = raymarch

export const AudioReactiveRaymarch = ({ audioData }) => {
    const { width, height } = useThree((state) => state.viewport)
    const meshRef = useRef()

    // Update uniforms from audio data each frame
    useFrame(() => {
        if (audioData) {
            uBass.value = audioData.bass || 0
            uMid.value = audioData.mid || 0
            uTreble.value = audioData.treble || 0
            uVolume.value = audioData.volume || 0
        }
    })

    return (
        <mesh ref={meshRef} scale={[width, height, 1]}>
            <planeGeometry args={[1, 1]} />
            <primitive object={raymarchMaterial} attach='material' />
        </mesh>
    )
}

export default AudioReactiveRaymarch
