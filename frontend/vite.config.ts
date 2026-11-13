import { defineConfig } from "vite";
import fs from "fs";
import path from "path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// SSO sign-in only works from an https *.gemini.edu origin (see README), so
// local dev can serve https://schedule-local.gemini.edu:5173 with a local certificate.
const httpsCert = process.env.SCHEDULER_HTTPS_CERT;
const httpsKey = process.env.SCHEDULER_HTTPS_KEY;
const https =
  httpsCert && httpsKey
    ? { cert: fs.readFileSync(httpsCert), key: fs.readFileSync(httpsKey) }
    : undefined;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: ["localhost", ".gemini.edu"],
    https,
  },
  preview: {
    host: "0.0.0.0",
    allowedHosts: [".gemini.edu"],
  },
  css: {
    devSourcemap: true,
  },
});
