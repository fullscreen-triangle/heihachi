import Head from "next/head";
import dynamic from "next/dynamic";
import TransitionEffect from "@/components/TransitionEffect";

const SearchPlayer = dynamic(() => import("@/components/SearchPlayer"), { ssr: false });

export default function PlayerPage() {
    return (
        <>
            <Head>
                <title>Yokozuna Player | Audio Search Engine</title>
                <meta name="description" content="Search for music, play with liquid distortion visualization. Audio analysis via categorical observation — similarity is interference." />
            </Head>
            <TransitionEffect />
            <div className="relative w-full" style={{ height: "calc(100vh - 80px)" }}>
                <SearchPlayer />
            </div>
        </>
    );
}
