# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.

## Runway AI — web client

React + Vite front end for the analysis API.

```bash
npm install
npm run dev        # http://localhost:5180
```

The dev server proxies `/api` to `http://127.0.0.1:8000`, so the API must be
running (`run.bat api`, or `uvicorn app.api:app`). Proxying keeps every request
same-origin, which removes CORS from the dev loop entirely.

For a production build, point the bundle at a real host:

```bash
VITE_API_URL=https://api.example.com npm run build
```

### Design notes

Strictly monochrome. Score quality is carried by **luminance and weight**, not
hue — bright heavy white reads as excellent, dim grey as poor. That satisfies
the black-and-white brief and, usefully, keeps the interface readable for anyone
with colour vision deficiency.

Glass panels pair `backdrop-filter` with a hairline specular top edge and sit on
a slowly drifting light field; without something bright behind them, translucent
panels read as flat rectangles rather than glass. A `@supports not` fallback
gives opaque panels on browsers without `backdrop-filter`, since 6%-opacity
surfaces would otherwise be nearly invisible.

Mobile-first: single column, fluid type, and a clip picker rather than stacking
every result — a five-clip keypoint file rendered stacked produced a page
33,000 px tall on a phone.

### Live coaching controls

The live webcam view (`/live`) gates analysis behind explicit controls:
* **Start camera**: Activates webcam and skeleton preview for alignment and framing verification without triggering premature analysis while standing still or positioning.
* **Start coaching**: Resets previous motion and begins rolling-window gait classification, checklist tracking, and pass logging.
* **Stop coaching**: Pauses analysis while retaining metrics and pass history for review.

