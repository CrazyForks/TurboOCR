import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// In `npm run dev`, /api is proxied to a TurboOCR server on :8000 (override
// with OCR_BACKEND_URL). In Docker, nginx does the same proxy — see nginx.conf.
const backend = process.env.OCR_BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: backend,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
