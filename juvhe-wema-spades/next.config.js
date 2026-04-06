/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack(config) {
    // Mark three/tsl and three/webgpu as external for dynamic import
    // These ESM-only modules are loaded at runtime via import()
    config.externals = config.externals || []
    config.module.rules.push({
      test: /three\.webgpu\.js$/,
      type: 'javascript/auto',
    })
    return config
  },
}

module.exports = nextConfig
