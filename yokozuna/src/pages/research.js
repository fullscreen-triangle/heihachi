import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Image from "next/image";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const panelFigures = [
  { src: "/images/panels/panel_1_triple_equivalence.png", alt: "Panel 1: Triple Equivalence", caption: "Triple equivalence verification across all four tracks." },
  { src: "/images/panels/panel_2_commutation.png", alt: "Panel 2: Commutation", caption: "Commutation correlation analysis showing |corr| < 0.06." },
  { src: "/images/panels/panel_3_irreversibility.png", alt: "Panel 3: Irreversibility", caption: "Irreversibility measurements per track." },
  { src: "/images/panels/panel_4_harmonics.png", alt: "Panel 4: Harmonics", caption: "Harmonic relation counts and cluster structure." },
  { src: "/images/panels/panel_5_s_entropy.png", alt: "Panel 5: S-Entropy", caption: "Categorical entropy distribution across states." },
  { src: "/images/panels/panel_6_trajectory.png", alt: "Panel 6: Trajectory", caption: "State-space trajectory visualisation." },
  { src: "/images/panels/panel_7_capacity.png", alt: "Panel 7: Capacity", caption: "Channel capacity beyond Nyquist-Shannon-Gabor limits." },
];

const trackData = [
  {
    name: "Audio Omega",
    genre: "Neurofunk",
    origin: "UK",
    relations: 92,
    clusters: 1,
    irreversibility: "50.08%",
  },
  {
    name: "Spor \u2014 Running Man",
    genre: "Neurofunk",
    origin: "UK",
    relations: 101,
    clusters: 1,
    irreversibility: "46.02%",
  },
  {
    name: "Noisia \u2014 Stigma",
    genre: "Neurofunk",
    origin: "Netherlands",
    relations: 53,
    clusters: 1,
    irreversibility: "42.98%",
  },
  {
    name: "Angel Techno / Zardonic remix",
    genre: "Psy-Trance",
    origin: "Colombia",
    relations: 49,
    clusters: 6,
    irreversibility: "42.91%",
  },
];

const predictions = [
  "A lossless audio codec that preserves categorical structure can achieve compression ratios unreachable by entropy coding alone.",
  "Transplanckian categorical states are experimentally detectable via the triple-equivalence invariant in any sampled audio signal.",
  "The irreversibility ratio converges to a genre-dependent constant as analysis window length increases.",
  "Harmonic cluster count distinguishes tonal complexity: single-cluster signals exhibit strict harmonic lattice structure.",
  "The orthogonal categorical channel capacity is bounded below by the logarithm of the distinct state count (log2 768 ~ 9.58 bits per frame).",
];

const itemVariants = {
  hidden: { y: 50, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: { duration: 0.5, ease: "easeInOut" },
  },
};

export default function Research() {
  return (
    <>
      <Head>
        <title>Yokozuna | Research &amp; Validation</title>
        <meta
          name="description"
          content="Research and empirical validation results for the Categorical Audio Transport framework. Experimental predictions, panel figures, and companion papers."
        />
      </Head>
      <TransitionEffect />
      <main
        className={`mb-16 flex w-full flex-col items-center justify-center dark:text-light`}
      >
        <Layout className="pt-16">
          {/* Hero */}
          <AnimatedText
            text="Research & Validation"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:mb-8 sm:!text-6xl xs:!text-4xl"
          />

          {/* Publications */}
          <section className="mb-24 w-full">
            <h2 className="mb-8 text-4xl font-bold text-dark dark:text-light md:text-3xl">
              Publications
            </h2>

            <motion.div
              initial={{ y: 50, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1, transition: { duration: 0.5 } }}
              viewport={{ once: true }}
              className="rounded-3xl border border-solid border-dark bg-light p-8 shadow-2xl dark:border-light dark:bg-dark"
            >
              <h3 className="mb-2 text-2xl font-bold text-dark dark:text-light xs:text-lg">
                Categorical Audio Transport: An Orthogonal Information Channel for
                Digital Audio Beyond the Nyquist-Shannon-Gabor Limits
              </h3>
              <p className="mb-4 text-lg font-medium text-primary dark:text-primaryDark">
                Kundai Farai Sachikonye
              </p>
              <p className="mb-6 font-medium text-dark/75 dark:text-light/75">
                This paper establishes a categorical framework demonstrating that
                digital audio carries an orthogonal information channel beyond
                classical sampling limits. The framework is validated empirically
                across four professionally produced tracks spanning neurofunk and
                psy-trance genres.
              </p>

              <h4 className="mb-3 text-lg font-bold text-dark dark:text-light">
                Companion Papers
              </h4>
              <ul className="list-inside list-disc space-y-2 font-medium text-dark/75 dark:text-light/75">
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  <span className="font-semibold text-dark dark:text-light">Categorical State Counting</span>{" "}
                  &mdash; Enumeration and classification of transplanckian states in
                  sampled audio.
                </motion.li>
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  <span className="font-semibold text-dark dark:text-light">Thermodynamic Consequences</span>{" "}
                  &mdash; Entropy production and irreversibility in categorical
                  audio transport.
                </motion.li>
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  <span className="font-semibold text-dark dark:text-light">CatScript DSL</span>{" "}
                  &mdash; A domain-specific language for categorical audio
                  transformations.
                </motion.li>
              </ul>
            </motion.div>
          </section>

          {/* Empirical Validation Results */}
          <section className="mb-24 w-full">
            <h2 className="mb-8 text-4xl font-bold text-dark dark:text-light md:text-3xl">
              Empirical Validation Results
            </h2>

            <div className="grid grid-cols-2 gap-8 md:grid-cols-1">
              {trackData.map((track, index) => (
                <motion.div
                  key={track.name}
                  initial={{ y: 50, opacity: 0 }}
                  whileInView={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.5, delay: index * 0.1 },
                  }}
                  viewport={{ once: true }}
                  className="rounded-2xl border border-solid border-dark bg-light p-6 shadow-2xl dark:border-light dark:bg-dark"
                >
                  <h3 className="mb-1 text-xl font-bold text-dark dark:text-light xs:text-lg">
                    {track.name}
                  </h3>
                  <p className="mb-4 text-sm font-medium text-primary dark:text-primaryDark">
                    {track.genre} &middot; {track.origin}
                  </p>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <span className="block text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                        {track.relations}
                      </span>
                      <span className="text-sm font-medium text-dark/75 dark:text-light/75">
                        Harmonic Relations
                      </span>
                    </div>
                    <div>
                      <span className="block text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                        {track.clusters}
                      </span>
                      <span className="text-sm font-medium text-dark/75 dark:text-light/75">
                        Cluster{track.clusters > 1 ? "s" : ""}
                      </span>
                    </div>
                    <div>
                      <span className="block text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                        {track.irreversibility}
                      </span>
                      <span className="text-sm font-medium text-dark/75 dark:text-light/75">
                        Irreversibility
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Key Findings */}
            <motion.div
              initial={{ y: 50, opacity: 0 }}
              whileInView={{ y: 0, opacity: 1, transition: { duration: 0.5 } }}
              viewport={{ once: true }}
              className="mt-12 rounded-2xl border border-solid border-dark bg-light p-8 shadow-2xl dark:border-light dark:bg-dark"
            >
              <h3 className="mb-4 text-2xl font-bold text-dark dark:text-light">
                Key Findings
              </h3>
              <ul className="list-inside list-disc space-y-3 font-medium text-dark/75 dark:text-light/75">
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  <span className="font-semibold text-dark dark:text-light">Triple equivalence</span>{" "}
                  verified exactly across all four tracks.
                </motion.li>
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  <span className="font-semibold text-dark dark:text-light">Commutation</span>{" "}
                  |corr| &lt; 0.06 &mdash; categorical and spectral channels are
                  effectively orthogonal.
                </motion.li>
                <motion.li variants={itemVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                  All tracks share exactly{" "}
                  <span className="font-semibold text-dark dark:text-light">768 distinct categorical states</span>.
                </motion.li>
              </ul>
            </motion.div>
          </section>

          {/* Key Predictions */}
          <section className="mb-24 w-full">
            <h2 className="mb-8 text-4xl font-bold text-dark dark:text-light md:text-3xl">
              Key Predictions
            </h2>

            <ol className="space-y-4">
              {predictions.map((prediction, index) => (
                <motion.li
                  key={index}
                  initial={{ y: 50, opacity: 0 }}
                  whileInView={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.5, delay: index * 0.08 },
                  }}
                  viewport={{ once: true }}
                  className="flex items-start gap-4 rounded-xl border border-solid border-dark bg-light p-6 shadow-lg dark:border-light dark:bg-dark"
                >
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-dark text-lg font-bold text-light dark:bg-light dark:text-dark">
                    {index + 1}
                  </span>
                  <p className="font-medium text-dark/90 dark:text-light/90">
                    {prediction}
                  </p>
                </motion.li>
              ))}
            </ol>
          </section>

          {/* Panel Charts */}
          <section className="mb-16 w-full">
            <h2 className="mb-8 text-4xl font-bold text-dark dark:text-light md:text-3xl">
              Panel Charts
            </h2>

            <div className="grid grid-cols-2 gap-8 lg:grid-cols-1">
              {panelFigures.map((panel, index) => (
                <motion.div
                  key={panel.src}
                  initial={{ y: 50, opacity: 0 }}
                  whileInView={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.5, delay: index * 0.08 },
                  }}
                  viewport={{ once: true }}
                  className="overflow-hidden rounded-2xl border border-solid border-dark bg-light shadow-2xl dark:border-light dark:bg-dark"
                >
                  <div className="relative w-full">
                    <Image
                      src={panel.src}
                      alt={panel.alt}
                      width={800}
                      height={500}
                      className="h-auto w-full"
                      sizes="(max-width: 768px) 100vw,
                        (max-width: 1200px) 50vw,
                        50vw"
                    />
                  </div>
                  <p className="p-4 text-center text-sm font-medium text-dark/75 dark:text-light/75">
                    {panel.caption}
                  </p>
                </motion.div>
              ))}
            </div>
          </section>
        </Layout>
      </main>
    </>
  );
}
