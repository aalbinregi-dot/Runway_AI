/**
 * Application shell: header, navigation, and the routed outlet.
 *
 * Navigation had to change shape. Four tabs fitted a pill row; eight sections
 * do not, least of all on a 390 px phone. So the nav splits by viewport rather
 * than being squeezed: a bottom bar on mobile, where thumbs are, and a single
 * horizontal row in the header on desktop, where there is room. Both render
 * from one route table, so they cannot drift apart.
 *
 * Every section is a real URL. That is the point of the change - a walk you
 * want to show someone is now a link, the back button behaves, and a refresh
 * keeps you where you were.
 */
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "./api.js";

/** Primary destinations. `auth` marks routes that need an account. */
export const NAV = [
  { to: "/", label: "Analyse", end: true, icon: "upload" },
  { to: "/live", label: "Live", icon: "camera" },
  { to: "/group", label: "Group", icon: "group" },
  { to: "/drills", label: "Drills", icon: "target", auth: true },
  { to: "/history", label: "History", icon: "list", auth: true },
  { to: "/progress", label: "Progress", icon: "trend", auth: true },
  { to: "/guide", label: "Guide", icon: "help" },
  { to: "/account", label: "Account", icon: "person" },
];

const PATHS = {
  upload: "M12 15V4m0 0L8 8m4-4 4 4M4 17v3h16v-3",
  camera: "M3 8.5h4l1.5-2h7L17 8.5h4v11H3zM12 17a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z",
  list: "M4 6h16M4 12h16M4 18h10",
  trend: "M3 17l5.5-5.5 3.5 3.5L21 6M21 6h-5M21 6v5",
  person: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4.5 20a7.5 7.5 0 0 1 15 0",
  help: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM9.6 9.2a2.5 2.5 0 1 1 3.3 2.4c-.6.2-.9.7-.9 1.3v.6M12 16.6v.5",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 13.2a1.2 1.2 0 1 0 0-2.4 1.2 1.2 0 0 0 0 2.4z",
  group: "M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM16.5 11.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5M2.5 19a6.5 6.5 0 0 1 13 0M16 13.5c2.5 0 4.5 1.9 4.5 4.4",
};

function Icon({ name }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d={PATHS[name]} stroke="currentColor" strokeWidth="1.4"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Shell({ user, setUser }) {
  const [offline, setOffline] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    api.me().then((r) => setUser(r.user)).catch(() => setOffline(true));
    localStorage.removeItem("runway.user");   // pre-accounts leftover
  }, [setUser]);

  // A route change should start at the top; otherwise arriving at a long page
  // from halfway down another one looks like a rendering fault.
  //
  // The braces are load-bearing. Written concisely as
  // `useEffect(() => window.scrollTo(0, 0), [pathname])` the arrow *returns*
  // whatever scrollTo gives back, and React treats any non-undefined return as
  // the cleanup function - then throws "destroy is not a function" on the next
  // route change, taking the whole tree down. Chromium returns undefined here
  // so it looked fine under test; another browser returning anything else does
  // not.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <>
      <div className="field" aria-hidden="true" />
      <div className="shell">
        <header className="top">
          <NavLink to="/" className="wordmark-link">
            <h1 className="wordmark">Runway <span>ai</span></h1>
          </NavLink>
          <nav className="nav-desktop" aria-label="Sections">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end}
                       className={({ isActive }) => `nav-link${isActive ? " on" : ""}`}>
                {n.label}
              </NavLink>
            ))}
          </nav>
        </header>

        {offline && (
          <div className="notice" style={{ marginBottom: "var(--gap)" }}>
            {/* Both launchers are named. The banner appears precisely when the
                API is unreachable, so there is no server to ask which platform
                this is - and naming only the Windows one sent every Mac user
                looking for a batch file. */}
            <span>
              Cannot reach the analysis service. Start it with{" "}
              <code>run.bat api</code> on Windows or <code>./run.sh api</code> on
              macOS and Linux, then reload.
            </span>
          </div>
        )}

        <main><Outlet /></main>

        <footer className="foot">
          <span>Twelve criteria · 102 features · MediaPipe Pose</span>
          <span className="foot-links">
            <NavLink to="/guide" className="foot-link">Guide</NavLink>
            <NavLink to="/about" className="foot-link">How it works</NavLink>
          </span>
        </footer>
      </div>

      <nav className="nav-mobile" aria-label="Sections">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end}
                   className={({ isActive }) => `nav-item${isActive ? " on" : ""}`}>
            <Icon name={n.icon} />
            <span>{n.label}</span>
          </NavLink>
        ))}
      </nav>
    </>
  );
}

/** Shown in place of a page that needs an account. */
export function SignInRequired({ what }) {
  return (
    <div className="glass card">
      <h3>{what}</h3>
      <p style={{ color: "var(--ink-2)" }}>
        Sign in to keep your walks and see this. Analysing a walk works without
        an account — it just is not saved.
      </p>
      <NavLink to="/account" className="btn solid" style={{ marginTop: "1rem",
               display: "inline-block", textDecoration: "none" }}>
        Sign in
      </NavLink>
    </div>
  );
}
