import Head from "next/head";
import dynamic from "next/dynamic";
import Link from "next/link";
import { motion } from "framer-motion";
import { LinkArrow } from "@/components/Icons";
import TransitionEffect from "@/components/TransitionEffect";

const DeskScene = dynamic(() => import("@/components/DeskScene"), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Yokozuna | Categorical Audio Transport Framework</title>
        <meta
          name="description"
          content="Yokozuna: An orthogonal information channel for digital audio
        beyond the Nyquist-Shannon-Gabor limits. Categorical state counting applied
        to bounded audio phase space."
        />
      </Head>

      <TransitionEffect />

      <div
        className="relative w-full"
        style={{ height: "calc(100vh - 80px)" }}
      >
        <DeskScene />

        {/* Gradient overlay */}
        <div className="absolute inset-0 z-10 bg-gradient-to-t from-black/70 via-black/30 to-transparent pointer-events-none" />

        {/* Content overlay */}
        <div className="relative z-20 flex flex-col items-center justify-end h-full pb-24 px-8 md:pb-16 sm:pb-12 pointer-events-none">
          <motion.div
            className="max-w-3xl text-center pointer-events-auto"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.5 }}
          >
            <h1 className="text-5xl font-bold text-white mb-6 leading-tight xl:text-4xl lg:text-3xl md:text-2xl sm:text-xl drop-shadow-lg">
              An Orthogonal Information Channel for Digital Audio
            </h1>
            <p className="text-lg text-white/80 mb-8 max-w-2xl mx-auto md:text-base sm:text-sm drop-shadow-md">
              Categorical state counting in bounded phase space reveals a
              transport channel beyond the Nyquist-Shannon-Gabor limits &mdash;
              carrying information orthogonal to the physical audio signal.
            </p>

            <div className="flex items-center justify-center gap-4 sm:flex-col sm:gap-3">
              <Link
                href="/framework"
                className="flex items-center gap-2 px-6 py-3 rounded-lg bg-white text-dark font-semibold
                  hover:bg-white/90 transition-all duration-300 md:px-4 md:py-2 md:text-sm"
              >
                Explore Framework <LinkArrow className="!w-5 md:!w-4" />
              </Link>

              <Link
                href="/api-docs"
                className="px-6 py-3 rounded-lg border border-white/40 text-white font-semibold
                  hover:bg-white/10 transition-all duration-300 md:px-4 md:py-2 md:text-sm"
              >
                API Documentation
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
    </>
  );
}
