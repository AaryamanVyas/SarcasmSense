import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['src/**/*.{test,spec}.{js,jsx}'],
    globals: true,
  },
  esbuild: {
    jsx: 'automatic',
    jsxDev: true,
  },
  resolve: {
    extensions: ['.js', '.jsx', '.mjs', '.ts', '.tsx', '.json'],
  },
});
