'use client'
import { Suspense, useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { BakeShadows, Environment } from '@react-three/drei'
import { Instances, Computers, AudioProvider } from './Desk'
import { CategoricalObserver } from './CategoricalObserver'

const PlayerVisualizer = ({ mode = 'desk', audioData, catMode = 0 }) => {
    if (mode === 'categorical') {
        return <CategoricalVisualizer audioData={audioData} catMode={catMode} />
    }
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
    const [modules, setModules] = useState({ Canvas: null, Raymarch: null })
    const [error, setError] = useState(false)

    useEffect(() => {
        let cancelled = false
        Promise.all([
            import('./canvas').then(m => m.default),
            import('./raymarching/AudioReactiveRaymarch').then(m => m.AudioReactiveRaymarch),
        ]).then(([Canvas, Raymarch]) => {
            if (!cancelled) setModules({ Canvas, Raymarch })
        }).catch(() => {
            if (!cancelled) setError(true)
        })
        return () => { cancelled = true }
    }, [])

    if (error) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-[#0a0a0f]">
                <p className="text-white/40 text-sm">WebGPU not available — try Chrome 113+ or Edge 113+</p>
            </div>
        )
    }

    if (!modules.Canvas || !modules.Raymarch) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-[#0a0a0f]">
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white/60 rounded-full animate-spin" />
                    <p className="text-white/40 text-sm">Loading WebGPU renderer...</p>
                </div>
            </div>
        )
    }

    const WebGPUCanvas = modules.Canvas
    const AudioReactiveRaymarch = modules.Raymarch

    return (
        <WebGPUCanvas
            camera={{ position: [0, 0, 5] }}
            style={{ width: '100%', height: '100%' }}
        >
            <AudioReactiveRaymarch audioData={audioData} />
        </WebGPUCanvas>
    )
}

function CategoricalVisualizer({ audioData, catMode = 0 }) {
    return (
        <Canvas
            camera={{ position: [0, 0, 1] }}
            style={{ width: '100%', height: '100%' }}
            gl={{ antialias: true }}
        >
            <CategoricalObserver audioData={audioData} mode={catMode} />
        </Canvas>
    )
}

export default PlayerVisualizer
