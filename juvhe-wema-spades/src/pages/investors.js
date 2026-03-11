import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const fadeIn = {
  hidden: { opacity: 0, y: 30 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.15, duration: 0.6, ease: "easeOut" },
  }),
};

const ApplicationCard = ({ title, description, index }) => {
  return (
    <motion.div
      className="relative rounded-2xl border border-solid border-dark bg-light p-8 shadow-xl
        dark:border-light dark:bg-dark sm:p-6"
      variants={fadeIn}
      custom={index}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
    >
      <div
        className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem] rounded-br-3xl
          bg-dark dark:bg-light md:-right-2 md:w-[101%] xs:h-[102%] xs:rounded-[1.5rem]"
      />
      <h3 className="mb-3 text-2xl font-bold text-dark dark:text-light lg:text-xl">
        {title}
      </h3>
      <p className="font-medium text-dark/80 dark:text-light/80 sm:text-sm">
        {description}
      </p>
    </motion.div>
  );
};

const SectionHeading = ({ children }) => (
  <motion.h2
    className="mb-8 text-4xl font-bold text-dark dark:text-light lg:text-3xl sm:text-2xl"
    variants={fadeIn}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    {children}
  </motion.h2>
);

const Bullet = ({ children, index = 0 }) => (
  <motion.li
    className="mb-4 text-lg font-medium text-dark/90 dark:text-light/90 sm:text-base"
    variants={fadeIn}
    custom={index}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    {children}
  </motion.li>
);

export default function Investors() {
  return (
    <>
      <Head>
        <title>Heihachi | Investors</title>
        <meta
          name="description"
          content="Investment opportunity in the Yokozuna categorical digital audio framework. A mathematically rigorous, orthogonal information channel for digital audio."
        />
      </Head>

      <TransitionEffect />
      <main
        className={`mb-16 flex w-full flex-col items-center justify-center dark:text-light`}
      >
        <Layout className="pt-16">
          {/* Hero */}
          <AnimatedText
            text="The Future of Digital Audio"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:mb-8 sm:!text-6xl xs:!text-4xl"
          />

          {/* The Opportunity */}
          <section className="mb-24 sm:mb-16">
            <SectionHeading>The Opportunity</SectionHeading>
            <ul className="ml-6 list-disc">
              <Bullet index={0}>
                Digital audio is a $30B+ market, yet every format&mdash;PCM, DSD,
                MP3, FLAC, Opus&mdash;operates within the same 80-year-old
                paradigm. All of them encode amplitude as a function of time. That
                is the only information channel any existing format has ever used.
              </Bullet>
              <Bullet index={1}>
                We have discovered a second, orthogonal information channel that
                no existing format accesses. It arises from the categorical
                structure of sample partitions and is mathematically independent
                of the waveform itself.
              </Bullet>
              <Bullet index={2}>
                The implication is concrete: a 44.1&nbsp;kHz file with categorical
                annotation carries information that is formally unreachable by a
                384&nbsp;kHz file without it. Resolution cannot substitute for
                structure.
              </Bullet>
            </ul>
          </section>

          {/* What We've Built */}
          <section className="mb-24 sm:mb-16">
            <SectionHeading>What We&apos;ve Built</SectionHeading>
            <ul className="ml-6 list-disc">
              <Bullet index={0}>
                <strong>The mathematical framework</strong>&mdash;rigorous,
                published, zero adjustable parameters. The theory makes
                quantitative predictions, not qualitative claims.
              </Bullet>
              <Bullet index={1}>
                <strong>The Yokozuna (.ykz) format</strong>&mdash;a working
                encoder and decoder that embeds categorical metadata alongside
                conventional PCM. The first audio format designed to carry both
                channels.
              </Bullet>
              <Bullet index={2}>
                <strong>Validation pipeline</strong>&mdash;tested across 4 tracks
                spanning multiple genres. Every prediction of the theory has been
                confirmed empirically.
              </Bullet>
              <Bullet index={3}>
                <strong>Triple equivalence verified</strong>, orthogonal channel
                confirmed empirically. Three independent mathematical
                characterisations of the categorical channel converge to the same
                result.
              </Bullet>
            </ul>
          </section>

          {/* Market Applications */}
          <section className="mb-24 sm:mb-16">
            <SectionHeading>Market Applications</SectionHeading>
            <div className="grid grid-cols-2 gap-10 xl:gap-8 lg:gap-6 md:grid-cols-1">
              <ApplicationCard
                title="Music Production"
                description="Groove metric, sub-sample timing analysis, and categorical mixing. Producers gain access to timing and structural information invisible to conventional DAWs."
                index={0}
              />
              <ApplicationCard
                title="Spatial Audio"
                description="Interaural time difference precision improves from 23 microseconds to 0.23 microseconds without increasing the sample rate. Categorical annotation resolves spatial position two orders of magnitude below the sample grid."
                index={1}
              />
              <ApplicationCard
                title="Audio Compression"
                description="Categorical compression operates on partition space, not spectral masking. An entirely new axis of compression orthogonal to every existing codec."
                index={2}
              />
              <ApplicationCard
                title="Audio Authentication"
                description="The categorical trajectory serves as a cryptographic fingerprint. The irreversibility theorem guarantees that the trajectory cannot be reconstructed from the waveform alone, providing tamper-evidence by construction."
                index={3}
              />
              <ApplicationCard
                title="Streaming"
                description="Progressive dual-layer streaming: PCM first for immediate playback, categorical layer delivered as bandwidth permits. Full backward compatibility with zero additional latency on constrained connections."
                index={4}
              />
            </div>
          </section>

          {/* The Team */}
          <section className="mb-24 sm:mb-16">
            <SectionHeading>The Team</SectionHeading>
            <motion.div
              className="relative rounded-2xl border border-solid border-dark bg-light p-10
                shadow-xl dark:border-light dark:bg-dark sm:p-6"
              variants={fadeIn}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
            >
              <div
                className="absolute top-0 -right-3 -z-10 h-[103%] w-[102%] rounded-[2rem]
                  rounded-br-3xl bg-dark dark:bg-light md:-right-2 md:w-[101%]
                  xs:h-[102%] xs:rounded-[1.5rem]"
              />
              <h3 className="mb-2 text-2xl font-bold text-dark dark:text-light lg:text-xl">
                Kundai Farai Sachikonye
              </h3>
              <p className="mb-4 text-lg font-medium text-primary dark:text-primaryDark sm:text-base">
                Technical University of Munich
              </p>
              <p className="font-medium text-dark/80 dark:text-light/80 sm:text-sm">
                Originator of the categorical digital audio framework. Responsible
                for the mathematical theory, the Yokozuna format specification,
                encoder/decoder implementation, and the full validation pipeline.
              </p>
            </motion.div>
          </section>

          {/* Contact / Next Steps */}
          <section className="mb-8">
            <SectionHeading>Next Steps</SectionHeading>
            <motion.div
              className="flex flex-col items-center text-center"
              variants={fadeIn}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
            >
              <p className="mb-8 max-w-2xl text-lg font-medium text-dark/90 dark:text-light/90 sm:text-base">
                We are seeking partners who understand that foundational
                technology shifts are rare, and that the window to define the
                standard is brief. If the orthogonal channel thesis is correct
                &mdash; and every experiment we have run says it is &mdash; then
                every audio file produced today is leaving information on the
                table.
              </p>
              <a
                href="mailto:kundai.sachikonye@tum.de"
                className="rounded-lg bg-dark px-8 py-3 text-lg font-semibold text-light
                  shadow-md transition-transform hover:scale-105 dark:bg-light
                  dark:text-dark sm:px-6 sm:text-base"
              >
                kundai.sachikonye@tum.de
              </a>
            </motion.div>
          </section>
        </Layout>
      </main>
    </>
  );
}
