import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

const PlayerApp = dynamic(() => import("@/components/PlayerApp"), { ssr: false });

export default function PlayerPage() {
    return (
        <>
            <Head>
                <title>Yokozuna Player | Audio-Reactive Visualization</title>
                <meta name="description" content="Play audio with real-time categorical visualization, ambient noise compensation, and track annotation." />
            </Head>
            <TransitionEffect />
            <div className="relative w-full" style={{ height: "calc(100vh - 80px)" }}>
                <PlayerApp />
            </div>
        </>
    );
}
