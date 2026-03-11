'use client'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { useMemo, useContext, createContext, useRef, useEffect, useState } from 'react'
import { useGLTF, Merged, RenderTexture, PerspectiveCamera, Text, useVideoTexture } from '@react-three/drei'
import { gsap } from 'gsap/dist/gsap'
import { Vector3 } from 'three'

THREE.ColorManagement.legacyMode = false

// ============================================
// AUDIO CONTEXT & ANALYZER
// ============================================
const AudioContext = createContext()

function AudioProvider({ children, audioSrc = '/audio/track.mp3' }) {
    const [audioData, setAudioData] = useState({
        frequency: 0,
        bass: 0,
        mid: 0,
        treble: 0,
        volume: 0,
        isPlaying: false
    })

    const analyserRef = useRef(null)
    const dataArrayRef = useRef(null)
    const audioContextRef = useRef(null)
    const sourceRef = useRef(null)
    const audioElementRef = useRef(null)
    const animationFrameRef = useRef(null)
    const connectedRef = useRef(false)

    const initAudio = async () => {
        try {
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
                analyserRef.current = audioContextRef.current.createAnalyser()
                analyserRef.current.fftSize = 256
                analyserRef.current.smoothingTimeConstant = 0.8
                dataArrayRef.current = new Uint8Array(analyserRef.current.frequencyBinCount)
            }

            if (audioContextRef.current.state === 'suspended') {
                await audioContextRef.current.resume()
            }

            if (!audioElementRef.current) {
                audioElementRef.current = new Audio(audioSrc)
                audioElementRef.current.crossOrigin = 'anonymous'
                audioElementRef.current.loop = true
            }

            if (!connectedRef.current) {
                sourceRef.current = audioContextRef.current.createMediaElementSource(audioElementRef.current)
                sourceRef.current.connect(analyserRef.current)
                analyserRef.current.connect(audioContextRef.current.destination)
                connectedRef.current = true
            }

            await audioElementRef.current.play()
            setAudioData(prev => ({ ...prev, isPlaying: true }))
            analyzeAudio()
        } catch (error) {
            console.error('Audio initialization failed:', error)
        }
    }

    const analyzeAudio = () => {
        if (!analyserRef.current || !dataArrayRef.current) return

        analyserRef.current.getByteFrequencyData(dataArrayRef.current)

        const dataArray = dataArrayRef.current
        const bufferLength = dataArray.length

        const bass = dataArray.slice(0, bufferLength / 4).reduce((a, b) => a + b, 0) / (bufferLength / 4) / 255
        const mid = dataArray.slice(bufferLength / 4, bufferLength / 2).reduce((a, b) => a + b, 0) / (bufferLength / 4) / 255
        const treble = dataArray.slice(bufferLength / 2).reduce((a, b) => a + b, 0) / (bufferLength / 2) / 255
        const volume = dataArray.reduce((a, b) => a + b, 0) / bufferLength / 255

        setAudioData({
            frequency: dataArray[0] / 255,
            bass,
            mid,
            treble,
            volume,
            isPlaying: true
        })

        animationFrameRef.current = requestAnimationFrame(analyzeAudio)
    }

    const stopAudio = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current)
        }
        if (audioElementRef.current) {
            audioElementRef.current.pause()
        }
        setAudioData(prev => ({ ...prev, isPlaying: false }))
    }

    const cleanup = () => {
        stopAudio()
        if (audioElementRef.current) {
            audioElementRef.current.pause()
            audioElementRef.current.src = ''
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close()
        }
    }

    useEffect(() => {
        return () => cleanup()
    }, [])

    return (
        <AudioContext.Provider value={{ audioData, initAudio, stopAudio }}>
            {children}
        </AudioContext.Provider>
    )
}

// Hook to use audio data
function useAudioData() {
    const context = useContext(AudioContext)
    if (!context) {
        throw new Error('useAudioData must be used within AudioProvider')
    }
    return context
}

// ============================================
// MAIN COMPONENTS
// ============================================
const context = createContext()

export function Instances({ children, ...props }) {
    const { nodes } = useGLTF('/models/computers_1-transformed.glb')
    const instances = useMemo(
        () => ({
            Object: nodes.Object_4,
            Object1: nodes.Object_16,
            Object3: nodes.Object_52,
            Object13: nodes.Object_172,
            Object14: nodes.Object_174,
            Object23: nodes.Object_22,
            Object24: nodes.Object_26,
            Object32: nodes.Object_178,
            Object36: nodes.Object_28,
            Object45: nodes.Object_206,
            Object46: nodes.Object_207,
            Object47: nodes.Object_215,
            Object48: nodes.Object_216,
            Sphere: nodes.Sphere
        }),
        [nodes]
    )
    return (
        <Merged castShadow receiveShadow meshes={instances} {...props}>
            {(instances) => <context.Provider value={instances}>{children}</context.Provider>}
        </Merged>
    )
}

export function Computers(props) {
    const { nodes: n, materials: m } = useGLTF('/models/computers_1-transformed.glb')
    const instances = useContext(context)
    const groupRef = useRef()
    const { audioData } = useAudioData()

    const wireframeMaterial = new THREE.MeshBasicMaterial({
        color: 'green',
        wireframe: true
    })

    // Audio-reactive group animation
    useFrame((state) => {
        if (groupRef.current && audioData.isPlaying) {
            // Subtle rotation based on bass
            groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * audioData.bass * 0.1

            // Scale pulse with volume
            const scale = 1 + audioData.volume * 0.05
            groupRef.current.scale.set(scale, scale, scale)
        }
    })

    useEffect(() => {
        if (!groupRef.current) return

        const explosionCenter = new Vector3(0, 0, 0)
        const explosionFactor = 1
        const duration = 3.5

        const createExplosion = () => {
            groupRef.current.children.forEach((child) => {
                const vector = child.position.clone().sub(explosionCenter).normalize()
                const targetPosition = child.position.clone().add(vector.multiplyScalar(explosionFactor))
                const originalPosition = child.position.clone()

                gsap.to(child.position, {
                    x: targetPosition.x,
                    y: targetPosition.y,
                    z: targetPosition.z,
                    duration: duration,
                    ease: 'power3.out',
                    repeat: 1,
                    yoyo: true,
                    onComplete: () => {
                        child.position.set(originalPosition.x, originalPosition.y, originalPosition.z)
                    }
                })
            })
        }

        const initialDelay = setTimeout(() => {
            createExplosion()
        }, 2000)

        const intervalId = setInterval(() => {
            createExplosion()
        }, 20000)

        return () => {
            clearTimeout(initialDelay)
            clearInterval(intervalId)
        }
    }, [])

    return (
        <group {...props} dispose={null} ref={groupRef}>
            <instances.Object position={[0.16, 0.79, -1.97]} rotation={[-0.54, 0.93, -1.12]} scale={0.5} />
            <instances.Object position={[-2.79, 0.27, 1.82]} rotation={[-1.44, 1.22, 1.43]} scale={0.5} />
            <instances.Object position={[-5.6, 4.62, -0.03]} rotation={[-1.96, 0.16, 1.2]} scale={0.5} />
            <instances.Object position={[2.62, 1.98, -2.47]} rotation={[-0.42, -0.7, -1.85]} scale={0.5} />
            <instances.Object position={[4.6, 3.46, 1.19]} rotation={[-1.24, -0.72, 0.48]} scale={0.5} />
            <instances.Object1 position={[0.63, 0, -3]} rotation={[0, 0.17, 0]} scale={1.52} />
            <instances.Object1 position={[-2.36, 0.32, -2.02]} rotation={[0, 0.53, -Math.PI / 2]} scale={1.52} />
            <mesh castShadow receiveShadow geometry={n.Object_24.geometry} material={m.Texture} position={[-2.42, 0.94, -2.25]} rotation={[0, 0.14, Math.PI / 2]} scale={-1.52} />
            <instances.Object1 position={[-3.53, 0, 0.59]} rotation={[Math.PI, -1.09, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-3.53, 1.53, 0.59]} rotation={[0, 0.91, 0]} scale={1.52} />
            <instances.Object1 position={[3.42, 0, 0]} rotation={[-Math.PI, 1.13, -Math.PI]} scale={1.52} />
            <instances.Object1 position={[4.09, 2.18, 2.41]} rotation={[0, -1.55, 1.57]} scale={1.52} />
            <instances.Object3 position={[4.31, 1.57, 2.34]} rotation={[0, -1.15, -Math.PI / 2]} scale={-1.52} />
            <instances.Object3 position={[-3.79, 0, 1.66]} rotation={[Math.PI, -1.39, 0]} scale={-1.52} />
            <instances.Object3 position={[-3.79, 1.53, 1.66]} rotation={[0, 1.22, -Math.PI]} scale={-1.52} />
            <instances.Object1 position={[-3.69, 0, 2.59]} rotation={[0, -1.57, 0]} scale={1.52} />
            <instances.Object1 position={[-5.36, 2.18, 0.81]} rotation={[0, 0.77, Math.PI / 2]} scale={1.52} />
            <instances.Object3 position={[-5.56, 1.57, 0.69]} rotation={[0, 1.17, -Math.PI / 2]} scale={-1.52} />
            <instances.Object1 position={[-5.47, 2.79, 0.74]} rotation={[Math.PI, -1.16, Math.PI / 2]} scale={1.52} />
            <instances.Object3 position={[-5.29, 3.41, 0.89]} rotation={[Math.PI, -0.76, -Math.PI / 2]} scale={-1.52} />
            <instances.Object1 position={[-5.28, 0, -2.33]} rotation={[0, 0.75, 0]} scale={1.52} />
            <instances.Object1 position={[-5.49, 0, -1.38]} rotation={[Math.PI, -0.99, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-3.01, 0, -3.79]} rotation={[0, 0.6, 0]} scale={1.52} />
            <instances.Object1 position={[-2.08, 0, -4.32]} rotation={[Math.PI, -0.6, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-1.02, 0, -4.49]} rotation={[0, 0.31, 0]} scale={1.52} />
            <instances.Object1 position={[-5.31, 1.83, -1.41]} rotation={[0, 1.06, Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-4.18, 1.83, -3.06]} rotation={[-Math.PI, -0.46, -Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-1.76, 1.83, -3.6]} rotation={[0, -1.16, Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-0.25, 1.83, -5.54]} rotation={[0, 1.55, 1.57]} scale={1.52} />
            <instances.Object1 position={[-5.28, 2.14, -2.33]} rotation={[Math.PI, -0.75, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-5.49, 2.14, -1.38]} rotation={[0, 0.99, 0]} scale={1.52} />
            <instances.Object1 position={[-3.01, 2.14, -3.79]} rotation={[Math.PI, -0.6, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-2.08, 2.14, -4.32]} rotation={[0, 0.6, 0]} scale={1.52} />
            <instances.Object1 position={[-1.02, 2.14, -4.49]} rotation={[Math.PI, -0.31, Math.PI]} scale={1.52} />
            <instances.Object1 position={[-5.31, 3.98, -1.41]} rotation={[0, 1.06, Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-4.18, 3.98, -3.06]} rotation={[-Math.PI, -0.46, -Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-1.17, 3.98, -4.45]} rotation={[0, 0.17, Math.PI / 2]} scale={1.52} />
            <instances.Object1 position={[-0.94, 3.98, -4.66]} rotation={[Math.PI, 0.02, -Math.PI / 2]} scale={1.52} />

            {/* All your existing mesh components... */}
            <mesh castShadow receiveShadow geometry={n.Object_140.geometry} material={m.Texture} position={[5.53, 2.18, 0.17]} rotation={[-Math.PI, 0, 0]} scale={-1} />
            <mesh geometry={n.Object_140.geometry} material={wireframeMaterial} position={[5.53, 2.18, 0.17]} rotation={[-Math.PI, 0, 0]} scale={-1} />

            {/* Audio-reactive video screens with effects */}
            <ScreenVideoAudioReactive
                videoSrc="/gifs/7mmpbg.gif"
                frame="Object_206"
                panel="Object_207"
                position={[0.27, 1.53, -2.61]}
                audioReactType="twist"
            />
            <ScreenVideoAudioReactive
                videoSrc="/gifs/cat-skeleton.gif"
                frame="Object_209"
                panel="Object_210"
                position={[-1.43, 2.5, -1.8]}
                rotation={[0, 1, 0]}
                audioReactType="glitch"
            />
            <ScreenText invert frame="Object_212" panel="Object_213" x={-5} y={5} position={[-2.73, 0.63, -0.52]} rotation={[0, 1.09, 0]} />
            <ScreenVideoAudioReactive
                videoSrc="/gifs/gun-shot.gif"
                frame="Object_215"
                panel="Object_216"
                position={[1.84, 0.38, -1.77]}
                rotation={[0, -Math.PI / 9, 0]}
                audioReactType="scale"
            />
            <ScreenVideoAudioReactive
                videoSrc="/gifs/rugby-paint.gif"
                frame="Object_218"
                panel="Object_219"
                position={[3.11, 2.15, -0.18]}
                rotation={[0, -0.79, 0]}
                scale={0.81}
                audioReactType="colorShift"
            />
            <ScreenVideoAudioReactive
                videoSrc="/gifs/sign.gif"
                frame="Object_221"
                panel="Object_222"
                position={[-3.42, 3.06, 1.3]}
                rotation={[0, 1.22, 0]}
                scale={0.9}
                audioReactType="twist"
            />
            <ScreenText invert frame="Object_224" panel="Object_225" position={[-3.9, 4.29, -2.64]} rotation={[0, 0.54, 0]} />
            <ScreenVideoAudioReactive
                videoSrc='/gifs/spectacular.gif'
                frame="Object_227"
                panel="Object_228"
                position={[0.96, 4.28, -4.2]}
                rotation={[0, -0.65, 0]}
                audioReactType="pixelate"
            />
            <ScreenVideoAudioReactive
                videoSrc='/gifs/wireframe-cheetah.gif'
                frame="Object_230"
                panel="Object_231"
                position={[4.68, 4.29, -1.56]}
                rotation={[0, -Math.PI / 3, 0]}
                audioReactType="rgbSplit"
            />

            <Leds instances={instances} />
        </group>
    )
}

// ============================================
// SCREEN COMPONENTS WITH AUDIO REACTIVITY
// ============================================
function Screen({ frame, panel, children, ...props }) {
    const { nodes, materials } = useGLTF('/models/computers_1-transformed.glb')
    const groupRef = useRef()
    const { audioData } = useAudioData()

    useFrame((state) => {
        if (groupRef.current && audioData.isPlaying) {
            // Twist effect based on bass
            groupRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 2) * audioData.bass * 0.3
        }
    })

    return (
        <group {...props} ref={groupRef}>
            <mesh castShadow receiveShadow geometry={nodes[frame].geometry} material={materials.Texture} />
            <mesh geometry={nodes[panel].geometry}>
                <meshBasicMaterial toneMapped={false}>
                    <RenderTexture width={512} height={512} attach="map" anisotropy={16}>
                        {children}
                    </RenderTexture>
                </meshBasicMaterial>
            </mesh>
        </group>
    )
}

function ScreenText({ invert, x = 0, y = 1.2, ...props }) {
    const textRef = useRef()
    const { audioData } = useAudioData()
    const rand = Math.random() * 10000

    useFrame((state) => {
        const audioOffset = audioData.mid * 3
        textRef.current.position.x = x + Math.sin(rand + state.clock.elapsedTime / 4) * (8 + audioOffset)
    })

    return (
        <Screen {...props}>
            <PerspectiveCamera makeDefault manual aspect={1 / 1} position={[0, 0, 15]} />
            <color attach="background" args={[invert ? 'black' : '#35c19f']} />
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} />
            <Text
                font="/fonts/Inter-Medium.woff"
                position={[x, y, 0]}
                ref={textRef}
                fontSize={4}
                letterSpacing={-0.1}
                color={!invert ? 'black' : '#35c19f'}
            >
                Fullscreen.
            </Text>
        </Screen>
    )
}

// ============================================
// AUDIO-REACTIVE VIDEO SCREEN WITH EFFECTS
// ============================================
function ScreenVideoAudioReactive({
    videoSrc,
    audioReactType = 'twist', // 'twist', 'scale', 'glitch', 'colorShift', 'pixelate', 'rgbSplit'
    ...props
}) {
    const { nodes, materials } = useGLTF('/models/computers_1-transformed.glb')
    const groupRef = useRef()
    const meshRef = useRef()
    const materialRef = useRef()
    const { audioData } = useAudioData()

    const videoTexture = useVideoTexture(videoSrc, {
        loop: true,
        muted: true,
        crossOrigin: 'anonymous',
        autoplay: true,
        playsInline: true,
    })

    // Custom shader for advanced effects
    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            uniforms: {
                uTexture: { value: videoTexture },
                uTime: { value: 0 },
                uAudioBass: { value: 0 },
                uAudioMid: { value: 0 },
                uAudioTreble: { value: 0 },
                uAudioVolume: { value: 0 },
                uEffectType: { value: 0 }, // 0: twist, 1: glitch, 2: colorShift, 3: pixelate, 4: rgbSplit
            },
            vertexShader: `
                varying vec2 vUv;
                uniform float uAudioBass;
                uniform float uTime;

                void main() {
                    vUv = uv;
                    vec3 pos = position;

                    // Vertex displacement based on audio
                    pos.z += sin(pos.x * 10.0 + uTime) * uAudioBass * 0.1;

                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D uTexture;
                uniform float uTime;
                uniform float uAudioBass;
                uniform float uAudioMid;
                uniform float uAudioTreble;
                uniform float uAudioVolume;
                uniform float uEffectType;
                varying vec2 vUv;

                // Glitch effect
                vec4 glitchEffect(vec2 uv) {
                    vec2 offset = vec2(
                        sin(uTime * 10.0 + uv.y * 20.0) * uAudioBass * 0.05,
                        cos(uTime * 15.0 + uv.x * 20.0) * uAudioMid * 0.03
                    );
                    return texture2D(uTexture, uv + offset);
                }

                // Color shift effect
                vec4 colorShiftEffect(vec2 uv) {
                    float r = texture2D(uTexture, uv + vec2(uAudioBass * 0.01, 0.0)).r;
                    float g = texture2D(uTexture, uv).g;
                    float b = texture2D(uTexture, uv - vec2(uAudioTreble * 0.01, 0.0)).b;
                    return vec4(r, g, b, 1.0);
                }

                // Pixelate effect
                vec4 pixelateEffect(vec2 uv) {
                    float pixels = 100.0 - uAudioVolume * 80.0;
                    vec2 pixelated = floor(uv * pixels) / pixels;
                    return texture2D(uTexture, pixelated);
                }

                // RGB split effect
                vec4 rgbSplitEffect(vec2 uv) {
                    float amount = uAudioBass * 0.02;
                    float r = texture2D(uTexture, uv + vec2(amount, 0.0)).r;
                    float g = texture2D(uTexture, uv).g;
                    float b = texture2D(uTexture, uv - vec2(amount, 0.0)).b;
                    return vec4(r, g, b, 1.0);
                }

                void main() {
                    vec2 uv = vUv;

                    // Twist UV coordinates based on audio
                    float angle = uAudioBass * 0.5;
                    vec2 center = vec2(0.5);
                    vec2 uvCentered = uv - center;
                    float dist = length(uvCentered);
                    float twist = angle * dist;

                    mat2 rotation = mat2(
                        cos(twist), -sin(twist),
                        sin(twist), cos(twist)
                    );

                    vec2 uvTwisted = rotation * uvCentered + center;

                    vec4 color;

                    if (uEffectType < 0.5) {
                        // Twist effect
                        color = texture2D(uTexture, uvTwisted);
                    } else if (uEffectType < 1.5) {
                        // Glitch effect
                        color = glitchEffect(uv);
                    } else if (uEffectType < 2.5) {
                        // Color shift effect
                        color = colorShiftEffect(uv);
                    } else if (uEffectType < 3.5) {
                        // Pixelate effect
                        color = pixelateEffect(uv);
                    } else {
                        // RGB split effect
                        color = rgbSplitEffect(uv);
                    }

                    // Brightness modulation
                    color.rgb *= (1.0 + uAudioVolume * 0.3);

                    gl_FragColor = color;
                }
            `,
            toneMapped: false,
        })
    }, [videoTexture])

    // Set effect type
    useEffect(() => {
        const effectMap = {
            'twist': 0,
            'glitch': 1,
            'colorShift': 2,
            'pixelate': 3,
            'rgbSplit': 4,
            'scale': 0, // Use twist for scale
        }
        shaderMaterial.uniforms.uEffectType.value = effectMap[audioReactType] || 0
    }, [audioReactType, shaderMaterial])

    useFrame((state) => {
        if (!groupRef.current || !audioData.isPlaying) return

        // Update shader uniforms
        if (shaderMaterial) {
            shaderMaterial.uniforms.uTime.value = state.clock.elapsedTime
            shaderMaterial.uniforms.uAudioBass.value = audioData.bass
            shaderMaterial.uniforms.uAudioMid.value = audioData.mid
            shaderMaterial.uniforms.uAudioTreble.value = audioData.treble
            shaderMaterial.uniforms.uAudioVolume.value = audioData.volume
        }

        // Screen twist/rotation
        if (audioReactType === 'twist' || audioReactType === 'scale') {
            groupRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 2) * audioData.bass * 0.4
        }

        // Screen scale pulsing
        if (audioReactType === 'scale') {
            const scale = 1 + audioData.volume * 0.15
            groupRef.current.scale.set(scale, scale, scale)
        }

        // Position wobble
        if (audioReactType === 'glitch') {
            groupRef.current.position.x += (Math.random() - 0.5) * audioData.treble * 0.02
            groupRef.current.position.y += (Math.random() - 0.5) * audioData.treble * 0.02
        }
    })

    return (
        <group {...props} ref={groupRef}>
            <mesh
                castShadow
                receiveShadow
                geometry={nodes[props.frame].geometry}
                material={materials.Texture}
            />
            <mesh geometry={nodes[props.panel].geometry} ref={meshRef}>
                <meshBasicMaterial toneMapped={false}>
                    <RenderTexture width={512} height={512} attach="map" anisotropy={16}>
                        <PerspectiveCamera makeDefault manual aspect={1 / 1} position={[0, 0, 1.5]} />
                        <mesh>
                            <planeGeometry args={[3, 1.8]} />
                            <primitive object={shaderMaterial} attach="material" ref={materialRef} />
                        </mesh>
                    </RenderTexture>
                </meshBasicMaterial>
            </mesh>
        </group>
    )
}

// ============================================
// AUDIO-REACTIVE LEDS
// ============================================
function Leds({ instances }) {
    const ref = useRef()
    const { nodes } = useGLTF('/models/computers_1-transformed.glb')
    const { audioData } = useAudioData()

    useMemo(() => {
        nodes.Sphere.material = new THREE.MeshBasicMaterial()
        nodes.Sphere.material.toneMapped = false
    }, [nodes])

    useFrame((state) => {
        ref.current.children.forEach((instance, i) => {
            const rand = Math.abs(2 + instance.position.x)
            const t = Math.round((1 + Math.sin(rand * 10000 + state.clock.elapsedTime * rand)) / 2)

            // Audio-reactive color
            const audioIntensity = audioData.volume * 2
            const r = audioData.treble * audioIntensity
            const g = t * (1.1 + audioData.mid * audioIntensity)
            const b = t * (1 + audioData.bass * audioIntensity)

            instance.color.setRGB(r, g, b)

            // Scale pulse
            const scale = 0.005 * (1 + audioData.volume * 0.5)
            instance.scale.set(scale, scale, scale)
        })
    })

    return (
        <group ref={ref}>
            <instances.Sphere position={[-0.41, 1.1, -2.21]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[0.59, 1.32, -2.22]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[1.77, 1.91, -1.17]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[2.44, 1.1, -0.79]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[4.87, 3.8, -0.1]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[1.93, 3.8, -3.69]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[-2.35, 3.8, -3.48]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[-4.71, 4.59, -1.81]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[-3.03, 2.85, 1.19]} scale={0.005} color={[1, 2, 1]} />
            <instances.Sphere position={[-1.21, 1.73, -1.49]} scale={0.005} color={[1, 2, 1]} />
        </group>
    )
}

// ============================================
// AUDIO CONTROL UI
// ============================================
export function AudioControls({ className = '' }) {
    const { audioData, initAudio, stopAudio } = useAudioData()

    return (
        <button
            onClick={audioData.isPlaying ? stopAudio : initAudio}
            className={`flex items-center gap-2 px-4 py-2 rounded-full border border-white/30
                bg-white/10 backdrop-blur-sm text-white text-sm font-medium
                hover:bg-white/20 transition-all duration-300 cursor-pointer ${className}`}
        >
            {audioData.isPlaying ? (
                <>
                    <span className="w-3 h-3 flex items-center justify-center">
                        <span className="block w-2 h-2 border-l-2 border-r-2 border-white" />
                    </span>
                    Pause
                </>
            ) : (
                <>
                    <span className="w-3 h-3 flex items-center justify-center">
                        <span className="block w-0 h-0 border-t-[5px] border-t-transparent border-b-[5px] border-b-transparent border-l-[8px] border-l-white" />
                    </span>
                    Play
                </>
            )}
        </button>
    )
}

// ============================================
// MAIN DESK COMPONENT
// ============================================
export { AudioProvider, useAudioData }
