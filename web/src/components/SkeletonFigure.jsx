/**
 * A stored walk, replayed as a skeleton on a blank canvas.
 *
 * No video is kept - the project stores keypoints, not footage - so a walk
 * pulled from history can only be shown as the geometry that was measured.
 * That is a genuine loss of context and the caller should say so rather than
 * let a stick figure imply the footage still exists.
 *
 * Coordinates are re-centred and scaled per clip, so two walks filmed at
 * different distances are drawn the same size and can be compared by shape.
 */
import { useEffect, useRef, useState } from "react";

const C_LEFT = "#f0844c";
const C_RIGHT = "#4ea8de";
const C_SPINE = "rgba(246,245,243,0.9)";

const BONES = [
  ["left_shoulder", "right_shoulder", C_SPINE],
  ["left_hip", "right_hip", C_SPINE],
  ["left_shoulder", "left_hip", C_SPINE],
  ["right_shoulder", "right_hip", C_SPINE],
  ["left_shoulder", "left_elbow", C_LEFT],
  ["left_elbow", "left_wrist", C_LEFT],
  ["left_hip", "left_knee", C_LEFT],
  ["left_knee", "left_ankle", C_LEFT],
  ["right_shoulder", "right_elbow", C_RIGHT],
  ["right_elbow", "right_wrist", C_RIGHT],
  ["right_hip", "right_knee", C_RIGHT],
  ["right_knee", "right_ankle", C_RIGHT],
];

export default function SkeletonFigure({ track, height = 220, autoplay = true,
                                        seekRef, onTime }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const frameRef = useRef(0);
  const reportRef = useRef(0);
  const onTimeRef = useRef(onTime);
  const [playing, setPlaying] = useState(autoplay);

  // Held in a ref so a new callback identity does not restart the animation and
  // snap the figure back to frame zero.
  useEffect(() => { onTimeRef.current = onTime; }, [onTime]);

  // The figure owns the clock, and lends a handle on it: clicking a moment in the
  // timeline below has to move the replay, or the evidence strip is decorative.
  useEffect(() => {
    if (!seekRef || !track?.fps) return;
    seekRef.current = (seconds) => {
      frameRef.current = Math.max(0, Math.min(track.n_frames - 1, seconds * track.fps));
      onTimeRef.current?.(frameRef.current / track.fps);
    };
    return () => { seekRef.current = null; };
  }, [seekRef, track]);

  useEffect(() => {
    if (!track?.xy?.length) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const idx = Object.fromEntries(track.landmarks.map((n, i) => [n, i]));

    // One fixed transform for the whole clip, computed from its full extent:
    // rescaling per frame would make the figure pulse as the subject walks
    // toward the camera, which reads as motion that is not there.
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const frame of track.xy) {
      for (const [x, y] of frame) {
        if (x < lo[0]) lo[0] = x;
        if (y < lo[1]) lo[1] = y;
        if (x > hi[0]) hi[0] = x;
        if (y > hi[1]) hi[1] = y;
      }
    }
    const spanX = hi[0] - lo[0] || 1;
    const spanY = hi[1] - lo[1] || 1;

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const scale = Math.min(rect.width / spanX, rect.height / spanY) * 0.82;
      const ox = rect.width / 2 - (lo[0] + spanX / 2) * scale;
      const oy = rect.height / 2 - (lo[1] + spanY / 2) * scale;

      const i = Math.min(frameRef.current | 0, track.n_frames - 1);
      const pts = track.xy[i];
      const vis = track.visibility[i];
      const P = (n) => [ox + pts[idx[n]][0] * scale, oy + pts[idx[n]][1] * scale];

      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(rect.width * 0.012, 1.8);
      for (const [a, b, colour] of BONES) {
        if (idx[a] == null || idx[b] == null) continue;
        if (vis[idx[a]] < 0.4 || vis[idx[b]] < 0.4) continue;
        ctx.strokeStyle = colour;
        ctx.beginPath();
        const [ax, ay] = P(a); const [bx, by] = P(b);
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
      for (const n of track.landmarks) {
        if (vis[idx[n]] < 0.4) continue;
        const [x, y] = P(n);
        ctx.fillStyle = n.startsWith("left") ? C_LEFT
                      : n.startsWith("right") ? C_RIGHT : C_SPINE;
        ctx.beginPath();
        ctx.arc(x, y, Math.max(rect.width * 0.014, 2), 0, Math.PI * 2);
        ctx.fill();
      }

      if (playing) {
        frameRef.current = (frameRef.current + (track.fps > 45 ? 2 : 1)) % track.n_frames;
      }
      // Reported a few times a second, not every frame: a setState per frame
      // re-renders the whole page sixty times a second to move one line.
      const now = performance.now();
      if (onTimeRef.current && now - reportRef.current > 90) {
        reportRef.current = now;
        onTimeRef.current(i / (track.fps || 30));
      }
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [track, playing]);

  if (!track?.xy?.length) {
    return (
      <div className="figure-empty" style={{ height }}>
        <span>no replay stored</span>
      </div>
    );
  }

  return (
    <div className="figure" style={{ height }} onClick={() => setPlaying((p) => !p)}
         role="button" tabIndex={0}
         onKeyDown={(e) => { if (e.key === "Enter") setPlaying((p) => !p); }}
         title={playing ? "Pause" : "Play"}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
