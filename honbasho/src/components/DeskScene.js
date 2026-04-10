'use client'
import { Suspense, Component, useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { BakeShadows, Environment } from '@react-three/drei'
import { Instances, Computers, AudioControls } from './Desk'
import { AudioProvider, useAudioData } from './Desk'
import { useLibrary } from '../lib/LibraryProvider'
import { TrackSpectrum } from '../lib/categoricalAudio'

class ErrorBoundary extends Component {
    constructor(props) {
        super(props)
        this.state = { error: null }
    }
    static getDerivedStateFromError(error) {
        return { error }
    }
    componentDidCatch(error, info) {
        console.error('[DeskScene Error]', error, info)
    }
    render() {
        if (this.state.error) {
            return (
                <div style={{
                    position: 'absolute', inset: 0, display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                    color: '#ff4444', background: '#000', fontSize: 14,
                    fontFamily: 'monospace', padding: 20, whiteSpace: 'pre-wrap'
                }}>
                    {'[DeskScene Error]\n' + this.state.error.message + '\n\n' + this.state.error.stack}
                </div>
            )
        }
        return this.props.children
    }
}

// Observes the homepage jukebox audio and feeds the global library
function HomeAudioObserver() {
    const { audioData, trackName, trackIndex } = useAudioData()
    const library = useLibrary()
    const spectrumRef = useRef(null)
    const lastTrackIndex = useRef(-1)

    useEffect(() => {
        // Track changed — save previous, start new
        if (trackIndex !== lastTrackIndex.current) {
            if (spectrumRef.current && spectrumRef.current.count > 10) {
                library.addObservation(spectrumRef.current.toJSON())
            }
            spectrumRef.current = new TrackSpectrum(
                `home-${trackIndex}`,
                trackName || `Track ${trackIndex + 1}`
            )
            lastTrackIndex.current = trackIndex
        }
    }, [trackIndex, trackName, library])

    useEffect(() => {
        if (audioData.isPlaying && spectrumRef.current) {
            spectrumRef.current.addObservation(audioData)
        }
    }, [audioData])

    return null // Invisible — just observes
}

const DeskScene = () => {
    return (
        <ErrorBoundary>
            <AudioProvider>
                <HomeAudioObserver />
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
        </ErrorBoundary>
    )
}

export default DeskScene
