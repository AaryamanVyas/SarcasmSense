import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Vite configuration.
 *
 * The backend URL is never hard-coded in application code: it comes from
 * `VITE_API_BASE_URL` at build time (see `.env.example`). In development the
 * dev-server proxy below forwards `/api/*` to the local backend so the browser
 * makes same-origin requests and CORS is not involved at all.
 */
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_PROXY_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: { host: '0.0.0.0', port: 4173 },
  build: { outDir: 'dist', sourcemap: mode !== 'production' },
}));
