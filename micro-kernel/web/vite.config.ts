import { defineConfig } from 'vite';

// The IDE is a static page. It has no server of its own: everything it
// needs comes from the daemon, which the user pairs with by token.
export default defineConfig({
  server: { port: 5273, strictPort: false },
  build: { outDir: 'dist', sourcemap: true, target: 'es2022' },
});
