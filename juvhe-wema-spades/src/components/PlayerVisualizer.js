'use client'
import { Suspense, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { BakeShadows, Environment } from '@react-three/drei'
import { Instances, Computers, AudioProvider } from './Desk'

const PlayerVisualizer = ({ mode = 'desk', audioData }) => {
    if (mode === 'raymarch') {
        return <RaymarchVisualizer audioData={audioData} />
    }
    return <DeskVisualizer />
}

function DeskVisualizer() {
    return (
        <AudioProvider>
            <Canvas
                shadows
                camera={{ position: [0, 6, 16], fov: 50 }}
                style={{ width: '100%', height: '100%' }}
            >
                <hemisphereLight intensity={0.5} />
                <ambientLight intensity={0.2} />
                <directionalLight position={[5, 5, 5]} intensity={1} castShadow />
                <pointLight position={[0, 10, 0]} intensity={0.5} />
                <fog attach="fog" args={['#0a0a0f', 16, 30]} />
                <Suspense fallback={null}>
                    <Environment preset="night" />
                    <Instances>
                        <Computers position={[0, -2, 0]} scale={1} />
                    </Instances>
                </Suspense>
                <BakeShadows />
            </Canvas>
        </AudioProvider>
    )
}

function RaymarchVisualizer({ audioData }) {
    const WebGPUCanvas = useMemo(() => {
        try {
            const mod = require('./canvas')
            return mod.default
        } catch (e) {
            return null
        }
    }, [])

    const AudioReactiveRaymarch = useMemo(() => {
        try {
            const mod = require('./raymarching/AudioReactiveRaymarch')
            return mod.AudioReactiveRaymarch
        } catch (e) {
            return null
        }
    }, [])

    if (!WebGPUCanvas || !AudioReactiveRaymarch) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-[#0a0a0f]">
                <p className="text-white/40 text-sm">WebGPU not available — requires three/tsl</p>
            </div>
        )
    }

    return (
        <WebGPUCanvas
            camera={{ position: [0, 0, 5] }}
            style={{ width: '100%', height: '100%' }}
        >
            <AudioReactiveRaymarch audioData={audioData} />
        </WebGPUCanvas>
    )
}

export default PlayerVisualizer
