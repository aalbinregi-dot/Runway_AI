import { StrictMode, Suspense, lazy, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import Shell from "./Shell.jsx";
import Analyse from "./pages/Analyse.jsx";
import History from "./pages/History.jsx";
import WalkDetail from "./pages/WalkDetail.jsx";
import Compare from "./pages/Compare.jsx";
import Group from "./pages/Group.jsx";
import Drills from "./pages/Drills.jsx";
import Progress from "./pages/Progress.jsx";
import Account from "./pages/Account.jsx";
import About from "./pages/About.jsx";
import Guide from "./pages/Guide.jsx";
import Splash from "./components/Splash.jsx";
import { api } from "./api.js";

// Live pulls in the MediaPipe WASM runtime, which is tens of megabytes.
const Live = lazy(() => import("./pages/Live.jsx"));

/**
 * MASTER SWITCH: Set to false to completely disable the splash screen.
 */
export const ENABLE_SPLASH = true;



function Loading() {
  return (
    <div className="glass card loading">
      <span className="pulse" />
      <p>Loading</p>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [showSplash, setShowSplash] = useState(() => ENABLE_SPLASH);

  // Check app prerequisites (health endpoint and user auth)
  useEffect(() => {
    let active = true;

    // Safety fallback: if the backend is slow or offline, mark ready after 2.5s
    const fallback = setTimeout(() => {
      if (active) setReady(true);
    }, 2500);

    Promise.allSettled([
      api.health(),
      api.me().then((r) => {
        if (active && r?.user) setUser(r.user);
      }),
    ]).finally(() => {
      if (active) setReady(true);
    });

    return () => {
      active = false;
      clearTimeout(fallback);
    };
  }, []);

  return (
    <>
      {/* Splash screen overlays the viewport on the first load of the session */}
      {showSplash && (
        <Splash
          ready={ready}
          minMs={3200}
          onDone={() => {
            setShowSplash(false);
          }}
        />
      )}

      {/* The router and home page are mounted immediately behind the splash */}
      <BrowserRouter>
        <Routes>
          <Route element={<Shell user={user} setUser={setUser} />}>
            <Route index element={<Analyse user={user} />} />
            <Route
              path="live"
              element={
                <Suspense fallback={<Loading />}>
                  <Live />
                </Suspense>
              }
            />
            <Route path="group" element={<Group user={user} />} />
            <Route path="drills" element={<Drills user={user} />} />
            <Route path="history" element={<History user={user} />} />
            <Route path="history/:id" element={<WalkDetail user={user} />} />
            <Route path="compare" element={<Compare user={user} />} />
            <Route path="compare/:a/:b" element={<Compare user={user} />} />
            <Route path="progress" element={<Progress user={user} />} />
            <Route path="account" element={<Account user={user} onUser={setUser} />} />
            <Route path="guide" element={<Guide />} />
            <Route path="about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}
