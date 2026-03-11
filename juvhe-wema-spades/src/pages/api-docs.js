import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";

const CodeBlock = ({ children }) => (
  <pre className="overflow-x-auto rounded-lg bg-[#1e1e2e] p-4 text-sm leading-relaxed text-green-300 font-mono sm:text-xs">
    <code>{children}</code>
  </pre>
);

const EndpointCard = ({ method, path, description, params }) => (
  <div className="rounded-2xl border border-solid border-dark/20 bg-light p-6 shadow-lg dark:border-light/20 dark:bg-dark sm:p-4">
    <div className="mb-3 flex items-center gap-3 sm:flex-col sm:items-start sm:gap-2">
      <span
        className={`rounded-md px-3 py-1 text-sm font-bold text-light ${
          method === "POST" ? "bg-emerald-600" : "bg-sky-600"
        }`}
      >
        {method}
      </span>
      <code className="text-lg font-semibold text-dark dark:text-light sm:text-base">
        {path}
      </code>
    </div>
    <p className="mb-3 font-medium text-dark/80 dark:text-light/80">
      {description}
    </p>
    {params && (
      <div>
        <span className="text-sm font-bold uppercase text-dark/60 dark:text-light/60">
          Parameters
        </span>
        <div className="mt-1 flex flex-wrap gap-2">
          {params.map((p) => (
            <code
              key={p}
              className="rounded bg-dark/10 px-2 py-0.5 text-sm text-dark dark:bg-light/10 dark:text-light"
            >
              {p}
            </code>
          ))}
        </div>
      </div>
    )}
  </div>
);

export default function ApiDocs() {
  return (
    <>
      <Head>
        <title>Yokozuna Categorical Audio Framework | API Documentation</title>
        <meta
          name="description"
          content="API documentation for the Yokozuna categorical audio analysis framework. Submit audio for categorical analysis, retrieve S-entropy trajectories, partition coordinates, and harmonic networks."
        />
      </Head>

      <TransitionEffect />
      <main
        className={`mb-16 flex w-full flex-col items-center justify-center dark:text-light`}
      >
        <Layout className="pt-16">
          {/* Hero */}
          <AnimatedText
            text="API for Researchers"
            className="mb-16 !text-8xl !leading-tight lg:!text-7xl sm:mb-8 sm:!text-6xl xs:!text-4xl"
          />

          <div className="flex flex-col gap-16">
            {/* Overview */}
            <section>
              <h2 className="mb-4 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Overview
              </h2>
              <p className="mb-4 text-lg font-medium text-dark/80 dark:text-light/80">
                The Yokozuna API enables researchers to programmatically interact
                with the categorical digital audio analysis framework. With the
                API you can:
              </p>
              <ul className="list-disc space-y-2 pl-6 text-lg font-medium text-dark/80 dark:text-light/80">
                <li>Submit audio files for categorical analysis</li>
                <li>
                  Retrieve S-entropy trajectories, partition coordinates, and
                  harmonic coincidence networks
                </li>
                <li>Convert WAV files to .ykz format</li>
                <li>Access validation results</li>
              </ul>
            </section>

            {/* Authentication */}
            <section>
              <h2 className="mb-4 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Authentication
              </h2>
              <p className="mb-4 text-lg font-medium text-dark/80 dark:text-light/80">
                All API requests require an API key passed via the{" "}
                <code className="rounded bg-dark/10 px-2 py-0.5 dark:bg-light/10">
                  Authorization
                </code>{" "}
                header. API keys are scoped to individual research accounts and
                can be generated from your dashboard.
              </p>
              <CodeBlock>
                {`Authorization: Bearer ykz_live_xxxxxxxxxxxxxxxxxxxx`}
              </CodeBlock>
              <p className="mt-4 rounded-lg border border-amber-500/40 bg-amber-50 p-4 text-base font-medium text-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
                API key management is coming soon. During the preview period,
                requests are authenticated via institutional email verification.
              </p>
            </section>

            {/* Endpoints */}
            <section>
              <h2 className="mb-6 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Endpoints
              </h2>
              <div className="grid grid-cols-1 gap-6">
                <EndpointCard
                  method="POST"
                  path="/api/analyze"
                  description="Submit an audio file for categorical analysis. Returns a track ID for retrieving results."
                  params={["file", "duration", "n_modes", "partition_depth"]}
                />
                <EndpointCard
                  method="POST"
                  path="/api/convert"
                  description="Convert a WAV file to .ykz format, encoding categorical structure directly into the audio container."
                />
                <EndpointCard
                  method="GET"
                  path="/api/results/:trackId"
                  description="Retrieve the full analysis results for a given track as JSON, including S-entropy values, partition coordinates, and equivalence classes."
                />
                <EndpointCard
                  method="GET"
                  path="/api/trajectory/:trackId"
                  description="Get the S-entropy trajectory data for a track, representing categorical information flow over time."
                />
                <EndpointCard
                  method="GET"
                  path="/api/harmonics/:trackId"
                  description="Get the harmonic coincidence network for a track, mapping spectral relationships as a categorical graph."
                />
              </div>
            </section>

            {/* Response Format */}
            <section>
              <h2 className="mb-4 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Response Format
              </h2>
              <p className="mb-4 text-lg font-medium text-dark/80 dark:text-light/80">
                All endpoints return JSON. Below is an example response from the{" "}
                <code className="rounded bg-dark/10 px-2 py-0.5 dark:bg-light/10">
                  /api/results/:trackId
                </code>{" "}
                endpoint:
              </p>
              <CodeBlock>
{`{
  "track_id": "ykz_8f3a1b2c",
  "title": "Analysis Result",
  "duration_sec": 30,
  "sample_rate": 44100,
  "n_modes": 6,
  "partition_depth": 4,
  "s_entropy_trajectory": [0.42, 0.87, 1.13, 0.95, 1.21, ...],
  "partition_coordinates": {
    "simplicial_dim": 3,
    "vertices": [[0.1, 0.4, 0.5], [0.3, 0.3, 0.4], ...],
    "faces": [[0, 1, 2], [1, 2, 3], ...]
  },
  "triple_equivalence": {
    "class_count": 12,
    "classes": [
      {"id": 0, "members": [0, 4, 7], "morphism_type": "iso"},
      {"id": 1, "members": [1, 5], "morphism_type": "epi"}
    ]
  },
  "harmonic_clusters": [
    {"center_hz": 440.0, "radius": 12.3, "nodes": [0, 1, 3]},
    {"center_hz": 880.0, "radius": 8.7, "nodes": [2, 5, 6]}
  ],
  "validation": {
    "permutation_p_value": 0.003,
    "baseline_divergence": 0.41,
    "passed": true
  }
}`}
              </CodeBlock>
            </section>

            {/* Python SDK */}
            <section>
              <h2 className="mb-4 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Python SDK
              </h2>
              <p className="mb-4 text-lg font-medium text-dark/80 dark:text-light/80">
                The <code className="rounded bg-dark/10 px-2 py-0.5 dark:bg-light/10">yokozuna</code> Python
                package wraps the API for convenient use in research scripts and
                notebooks:
              </p>
              <CodeBlock>
{`from yokozuna import CategoricalAnalyzer

analyzer = CategoricalAnalyzer(api_key="...")
result = analyzer.analyze("track.wav", duration=30)

print(result.triple_equivalence)
print(result.harmonic_clusters)`}
              </CodeBlock>
              <p className="mt-4 text-lg font-medium text-dark/80 dark:text-light/80">
                Install via pip:
              </p>
              <CodeBlock>{`pip install yokozuna`}</CodeBlock>
            </section>

            {/* Rate Limits & Access */}
            <section>
              <h2 className="mb-4 text-3xl font-bold text-dark dark:text-light sm:text-2xl">
                Rate Limits &amp; Access
              </h2>
              <p className="text-lg font-medium text-dark/80 dark:text-light/80">
                Academic and research access to the Yokozuna API is free.
                Requests are rate-limited to 60 per minute and 1,000 per day per
                API key. If you need higher throughput for large-scale studies,
                contact us to discuss dedicated capacity. Commercial usage
                inquiries are welcome and handled on a case-by-case basis.
              </p>
            </section>
          </div>
        </Layout>
      </main>
    </>
  );
}
