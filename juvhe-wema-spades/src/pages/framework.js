import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 },
};

const staggerContainer = {
  initial: {},
  animate: { transition: { staggerChildren: 0.15 } },
};

const DerivationStep = ({ label, index, isLast }) => (
  <div className="flex items-center">
    <motion.div
      className="flex flex-col items-center justify-center rounded-xl border border-solid border-dark
        bg-light px-4 py-3 shadow-md dark:border-light dark:bg-dark sm:px-3 sm:py-2"
      initial={{ opacity: 0, scale: 0.8 }}
      whileInView={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: index * 0.12 }}
      viewport={{ once: true }}
    >
      <span className="text-sm font-bold text-primary dark:text-primaryDark xs:text-xs">
        {label}
      </span>
    </motion.div>
    {!isLast && (
      <span className="mx-2 text-2xl font-bold text-dark/50 dark:text-light/50 sm:mx-1 sm:text-lg">
        &rarr;
      </span>
    )}
  </div>
);

const MathCard = ({ title, notation, description, index }) => (
  <motion.div
    className="relative flex flex-col rounded-2xl border border-solid border-dark bg-light
      p-6 shadow-lg dark:border-light dark:bg-dark sm:p-4"
    initial={{ opacity: 0, y: 40 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay: index * 0.1 }}
    viewport={{ once: true }}
    whileHover={{ y: -5, transition: { duration: 0.2 } }}
  >
    <div
      className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl
        bg-dark dark:bg-light md:-right-2 md:w-[101%] xs:h-[102%] xs:rounded-[1.5rem]"
    />
    <span className="mb-2 text-sm font-medium uppercase tracking-wider text-primary dark:text-primaryDark">
      {title}
    </span>
    <div className="my-3 rounded-lg bg-dark/5 px-4 py-3 dark:bg-light/5">
      <p className="text-center font-mono text-lg font-bold text-dark dark:text-light md:text-base sm:text-sm">
        {notation}
      </p>
    </div>
    <p className="mt-2 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
      {description}
    </p>
  </motion.div>
);

const EnhancementItem = ({ number, title, description }) => (
  <motion.div
    className="flex gap-4 rounded-xl border border-solid border-dark/20 bg-light p-5
      dark:border-light/20 dark:bg-dark sm:p-4"
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.4, delay: number * 0.08 }}
    viewport={{ once: true }}
  >
    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full
      bg-primary text-lg font-bold text-light dark:bg-primaryDark dark:text-dark sm:h-8 sm:w-8 sm:text-base">
      {number}
    </span>
    <div>
      <h4 className="text-lg font-bold text-dark dark:text-light sm:text-base">
        {title}
      </h4>
      <p className="mt-1 font-medium text-dark/75 dark:text-light/75 sm:text-sm">
        {description}
      </p>
    </div>
  </motion.div>
);

export default function Framework() {
  const derivationSteps = [
    "Boundedness",
    "Poincaré Recurrence",
    "Oscillatory Dynamics",
    "Categorical States",
    "Entropy",
    "Information Channel",
  ];

  const mathStructures = [
    {
      title: "S-Entropy Coordinates",
      notation: "(Sₖ, Sₜ, Sₑ)",
      description:
        "Three orthogonal entropy measures capturing the spectral, temporal, and energetic degrees of freedom in any audio signal. Together they form a complete entropy basis for bounded oscillatory systems.",
    },
    {
      title: "Partition Coordinates",
      notation: "(n, l, m, s)",
      description:
        "Four quantum-number-like indices that classify categorical audio states: depth n, harmonic order l, phase index m, and chirality s. Every distinguishable audio configuration maps to a unique coordinate tuple.",
    },
    {
      title: "Triple Equivalence",
      notation: "S_osc = S_cat = S_part = k_B · M · ln(n)",
      description:
        "The central identity of the framework. The oscillatory entropy, categorical entropy, and partition entropy are all equal, unified by Boltzmann\u2019s constant k_B, mode count M, and partition depth n.",
    },
    {
      title: "Orthogonal Channel",
      notation: "C_total = C_phys + C_cat  where  C_cat = M · log₂(n)",
      description:
        "Audio carries two independent information channels. The physical channel encodes waveform data, while the categorical channel encodes structural metadata. These channels are mathematically orthogonal and do not interfere.",
    },
  ];

  const enhancements = [
    {
      title: "Categorical State Tagging",
      description:
        "Each audio frame is assigned its partition coordinates (n, l, m, s), creating a trajectory through categorical state space that accompanies the raw waveform.",
    },
    {
      title: "S-Entropy Stream Embedding",
      description:
        "A continuous stream of (Sₖ, Sₜ, Sₑ) values is computed per frame and stored alongside the PCM data, enabling entropy-aware playback and processing.",
    },
    {
      title: "Harmonic Network Construction",
      description:
        "The harmonic relationships between categorical states are encoded as a directed graph, revealing the structural skeleton of the audio content.",
    },
    {
      title: "Orthogonal Channel Separation",
      description:
        "Physical and categorical information are stored in separate, non-interfering streams, allowing independent manipulation of waveform and structure.",
    },
    {
      title: "Partition-Guided Enhancement",
      description:
        "The partition coordinates guide signal processing decisions, enabling mathematically principled noise reduction, dynamic range control, and spatial rendering.",
    },
  ];

  return (
    <>
      <Head>
        <title>Yokozuna Heihachi | The Categorical Audio Framework</title>
        <meta
          name="description"
          content="Explore the categorical audio framework: from the single axiom of boundedness to S-entropy coordinates, partition numbers, and the Yokozuna container format."
        />
      </Head>

      <TransitionEffect />
      <main
        className={`mb-16 flex w-full flex-col items-center justify-center dark:text-light`}
      >
        <Layout className="pt-16">
          {/* Hero */}
          <AnimatedText
            text="The Categorical Audio Framework"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:mb-8 sm:!text-6xl xs:!text-4xl"
          />

          {/* Core Axiom */}
          <motion.section
            className="mb-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl">
              The Core Axiom
            </h2>
            <div className="rounded-2xl border border-solid border-dark bg-light p-8 shadow-lg
              dark:border-light dark:bg-dark sm:p-5">
              <div className="mb-6 rounded-xl bg-dark/5 p-6 dark:bg-light/5 sm:p-4">
                <p className="text-center text-xl font-bold italic text-dark dark:text-light md:text-lg sm:text-base">
                  &ldquo;Audio signals are bounded physical systems: finite in amplitude,
                  finite in frequency, finite in duration.&rdquo;
                </p>
              </div>
              <p className="font-medium text-dark/80 dark:text-light/80 sm:text-sm">
                From this single, empirically verifiable axiom, the entire categorical audio
                framework follows by mathematical necessity. Boundedness is not an
                approximation or a modelling convenience &mdash; it is a physical fact about
                every real audio signal ever recorded or produced. Because the system is
                bounded, the phase space is compact. Because the phase space is compact,
                Poincar&eacute;&apos;s recurrence theorem applies. Because the dynamics are
                recurrent, they are oscillatory. And because the oscillatory modes are
                countable, they admit a categorical classification with well-defined entropy
                and information content.
              </p>
            </div>
          </motion.section>

          {/* Derivation Chain */}
          <motion.section
            className="mb-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl">
              The Derivation Chain
            </h2>
            <p className="mb-8 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
              Each step follows from the previous by a known theorem or physical law. There
              are no additional assumptions, free parameters, or empirical fits. The
              framework is fully determined by the axiom of boundedness.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-y-4 rounded-2xl
              border border-solid border-dark bg-light p-8 shadow-lg dark:border-light
              dark:bg-dark sm:p-4">
              {derivationSteps.map((step, i) => (
                <DerivationStep
                  key={step}
                  label={step}
                  index={i}
                  isLast={i === derivationSteps.length - 1}
                />
              ))}
            </div>
          </motion.section>

          {/* Key Mathematical Structures */}
          <motion.section
            className="mb-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl">
              Key Mathematical Structures
            </h2>
            <p className="mb-10 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
              The framework produces four interlocking mathematical objects. The S-entropy
              coordinates and partition coordinates provide two complementary ways to
              describe the same categorical state space, unified by the triple equivalence
              identity and connected to information theory through the orthogonal channel
              theorem.
            </p>
            <div className="grid grid-cols-2 gap-10 md:grid-cols-1 md:gap-8">
              {mathStructures.map((item, i) => (
                <MathCard
                  key={item.title}
                  title={item.title}
                  notation={item.notation}
                  description={item.description}
                  index={i}
                />
              ))}
            </div>
          </motion.section>

          {/* The Yokozuna Format */}
          <motion.section
            className="mb-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl">
              The Yokozuna Format
            </h2>
            <div className="rounded-2xl border border-solid border-dark bg-light p-8 shadow-lg
              dark:border-light dark:bg-dark sm:p-5">
              <p className="mb-6 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
                The <span className="font-mono font-bold">.ykz</span> container format is the
                practical embodiment of the categorical audio framework. Built on the ZIP
                archive structure for broad compatibility, each Yokozuna file packages five
                distinct data streams into a single, self-describing container:
              </p>
              <div className="grid grid-cols-5 gap-4 md:grid-cols-3 sm:grid-cols-2 xs:grid-cols-1">
                {[
                  { label: "PCM Audio", desc: "Raw waveform data" },
                  { label: "Categorical Trajectory", desc: "(n, l, m, s) per frame" },
                  { label: "S-Entropy Stream", desc: "(Sₖ, Sₜ, Sₑ) per frame" },
                  { label: "Partition Coordinates", desc: "Full state classification" },
                  { label: "Harmonic Network", desc: "Structural graph data" },
                ].map((stream, i) => (
                  <motion.div
                    key={stream.label}
                    className="flex flex-col items-center rounded-xl bg-dark/5 p-4
                      dark:bg-light/5 sm:p-3"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: i * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <span className="mb-2 text-center text-sm font-bold text-primary dark:text-primaryDark">
                      {stream.label}
                    </span>
                    <span className="text-center text-xs font-medium text-dark/60 dark:text-light/60">
                      {stream.desc}
                    </span>
                  </motion.div>
                ))}
              </div>
              <p className="mt-6 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
                Because the categorical channel is orthogonal to the physical channel, the
                additional metadata streams carry information that is, by construction,
                independent of the PCM waveform. This means the categorical data can be
                stripped without altering the audio, or processed independently to enable
                structural queries, automatic classification, and entropy-guided enhancement.
              </p>
            </div>
          </motion.section>

          {/* Enhancement Mechanisms */}
          <motion.section
            className="mb-8"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="mb-6 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl">
              Five Enhancement Mechanisms
            </h2>
            <p className="mb-8 font-medium text-dark/80 dark:text-light/80 sm:text-sm">
              The categorical framework does not merely describe audio &mdash; it enables a
              new class of signal processing operations that leverage structural information
              inaccessible to conventional waveform-only methods.
            </p>
            <div className="flex flex-col gap-4">
              {enhancements.map((item, i) => (
                <EnhancementItem
                  key={item.title}
                  number={i + 1}
                  title={item.title}
                  description={item.description}
                />
              ))}
            </div>
          </motion.section>
        </Layout>
      </main>
    </>
  );
}
