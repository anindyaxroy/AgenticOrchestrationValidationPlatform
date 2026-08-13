import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: { host: true, proxy: { '/api': 'http://backend:8000', '/ws': { target: 'ws://backend:8000', ws: true } } }
})
