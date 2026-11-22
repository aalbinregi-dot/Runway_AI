import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";

// Port 5180 rather than Vite's default 5173: that port is commonly already
// taken by another project, and silently landing on a different one makes the
// backend's CORS allow-list wrong. strictPort makes a collision an error you
// can see instead of a mystery.
/** Turn a failed hot update into a full reload instead of a broken tab.
 *  Deleting or renaming a module - which happened here during a refactor -
 *  leaves an open page holding a module graph it can no longer resolve, and the
 *  visible result is a blank screen with nothing in the console. */
const reloadOnBrokenHmr = () => ({
  name: "reload-on-broken-hmr",
  handleHotUpdate({ modules, server }) {
    if (modules.length === 0) {
      server.ws.send({ type: "full-reload" });
    }
  },
});

export default defineConfig({
  plugins: [react(), reloadOnBrokenHmr()],
  server: {
    port: 5180,
    strictPort: true,
    host: true,          // also bind the LAN address, so a phone can load it
    // HTTPS is required for iOS Safari to allow camera access (getUserMedia is
    // blocked on plain HTTP for any non-localhost origin). Certificate files are
    // generated once with: mkcert 192.168.1.8 localhost 127.0.0.1
    // The CA must also be installed on the iOS device — see README for steps.
    https: (() => {
      try {
        return {
          key:  fs.readFileSync("./192.168.1.8+2-key.pem"),
          cert: fs.readFileSync("./192.168.1.8+2.pem"),
        };
      } catch {
        return undefined; // fall back to HTTP if certs are missing
      }
    })(),
    // MediaPipe WASM uses SharedArrayBuffer, which requires these two headers
    // to be present in every response. Without them Safari (both iOS and macOS)
    // silently refuses to run the pose model.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    // Proxying the API through the dev server makes every request same-origin,
    // which removes CORS from the dev loop entirely.
    proxy: {
      "/api": {
        // Overridable so the dev server can be pointed at a second backend -
        // needed when a crashed uvicorn leaves port 8000 bound and Windows
        // keeps the dead listener around for minutes.
        target: process.env.API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  preview: {
    port: 4173,
    strictPort: true,
    host: true,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
