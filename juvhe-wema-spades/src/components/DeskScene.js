'use client'
import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { BakeShadows, Environment } from '@react-three/drei'
import { Instances, Computers, AudioControls } from './Desk'
import { AudioProvider } from './Desk'

const DeskScene = () => {
    return (
        <AudioProvider>
            <div className="absolute inset-0 z-0">
                <Canvas
                    shadows
                    camera={{ position: [0, 6, 16], fov: 50 }}
                    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                >
                    <hemisphereLight intensity={0.5} />
                    <ambientLight intensity={0.2} />
                    <directionalLight
                        position={[5, 5, 5]}
                        intensity={1}
                        castShadow
                        shadow-mapSize-width={1024}
                        shadow-mapSize-height={1024}
                    />
                    <pointLight position={[0, 10, 0]} intensity={0.5} />
                    <fog attach="fog" args={['#272730', 16, 30]} />
                    <Suspense fallback={null}>
                        <Environment preset="night" />
                        <Instances>
                            <Computers position={[0, -2, 0]} scale={1} />
                        </Instances>
                    </Suspense>
                    <BakeShadows />
                </Canvas>
            </div>

            {/* Audio play button - bottom right */}
            <div className="absolute bottom-8 right-8 z-30">
                <AudioControls />
            </div>
        </AudioProvider>
    )
}

export default DeskScene
