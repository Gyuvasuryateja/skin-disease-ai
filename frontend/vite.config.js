import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration. The production API base URL is provided at build time
// through VITE_API_BASE_URL (see frontend/.env.example and README.md).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
