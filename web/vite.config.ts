import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
    tailwindcss(),
  ],
  optimizeDeps: {
    exclude: ['pyodide'], // prevent Vite from trying to pre-bundle pyodide
    include: ["@latex2js/vue"],
  },
  ssr: {
    noExternal: ['@latex2js/vue'],
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  worker: {
    format: 'es' // <-- important! use 'es', not 'iife' or 'umd'
  }
})