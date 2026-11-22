/**
 * Plays the uploaded video with the tracked skeleton drawn over it.
 *
 * This is the part that makes the score believable. A number alone cannot be
 * checked; seeing the skeleton lock onto the body — or fail to — tells the user
 * immediately whether the analysis had anything real to work with.
 *
 * Works for uploads and for links. An upload is played straight from the File
 * via an object URL; a linked video is played back from the server, which keeps
 * the download briefly for exactly this purpose (see runway_ai.media).
 *
 * Beyond playback it carries the controls an analysis needs and a player does
 * not: quarter speed, single-frame stepping, and motion trails. The trails are
 * the cheapest useful addition here — the path a wrist or an ankle traced through
 * the clip is a shape, and a shape is far easier to judge than the amplitude
 * number derived from it.
 */
import { useEffect, useRef, useState } from "react";
import TimelineExplorer from "./TimelineExplorer.jsx";

/* Left limbs warm, right limbs cool, so left/right asymmetry is visible
   directly rather than having to be inferred from the score. */
const C_LEFT = "#f0844c";
const C_RIGHT = "#4ea8de";
const C_SPINE = "rgba(246,245,243,0.92)";

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

const MIN_VIS = 0.4;

/** Which joints leave a trail, and in what colour. Wrists and ankles carry the
 *  swing amplitudes; the nose carries head steadiness. */
const TRAILS = [
  ["left_wrist", C_LEFT], ["right_wrist", C_RIGHT],
  ["left_ankle", C_LEFT], ["right_ankle", C_RIGHT],
  ["nose", C_SPINE],
];
const TRAIL_FRAMES = 44;

const SPEEDS = [0.25, 0.5, 1];

export default function SkeletonPlayer({ file, src, track, timeline, seekRef,
                                         onTime, onSelect }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const [url, setUrl] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showTrails, setShowTrails] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [t, setT] = useState(0);

  // Expose a seek handle so findings elsewhere on the page can jump the video
  // to the moment that produced them.
  useEffect(() => {
    if (!seekRef) return;
    seekRef.current = (seconds) => {
      const v = videoRef.current;
      if (!v) return;

      // Assigning currentTime before metadata has loaded is silently dropped -
      // the element simply stays at 0 - so the seek has to wait for the
      // duration to be known. Clicking a finding right after an analysis is
      // exactly when the video is still loading, which is how this was missed.
      const go = () => {
        const dur = Number.isFinite(v.duration) ? v.duration : null;
        const target = dur ? Math.max(0, Math.min(seconds, dur - 0.05)) : Math.max(0, seconds);
        v.currentTime = target;
        setT(target);
        onTime?.(target);

        // The player loops, so starting playback within a fraction of a second
        // of the end wraps straight back to zero and the user never sees the
        // frame they asked for. Near the end, hold on the frame instead.
        const room = dur ? dur - target : Infinity;
        if (room > 0.8) v.play().catch(() => {});
        else v.pause();
      };

      if (v.readyState >= 1) go();
      else v.addEventListener("loadedmetadata", go, { once: true });

      v.scrollIntoView({ behavior: "smooth", block: "center" });
    };
    return () => { seekRef.current = null; };
  }, [seekRef, onTime]);

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setUrl(objectUrl);
      // Revoking matters: without it the whole video stays pinned in memory for
      // the lifetime of the tab.
      return () => URL.revokeObjectURL(objectUrl);
    }
    setUrl(src ?? null);
    return undefined;
  }, [file, src]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playbackRate = speed;
  }, [speed, url]);

  useEffect(() => {
    if (!track || !url) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const index = Object.fromEntries(track.landmarks.map((n, i) => [n, i]));

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw);
      const ctx = canvas.getContext("2d");
      if (!ctx || !video.videoWidth) return;

      // Match the canvas to the video's displayed box, accounting for DPR so
      // the strokes are not soft on a retina screen.
      const rect = video.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!showSkeleton && !showTrails) return;

      // The track was sampled at its own rate, which may differ from the
      // video's, so map by time rather than by frame number.
      const i = Math.min(
        track.n_frames - 1,
        Math.max(0, Math.round(video.currentTime * track.fps)),
      );
      const pts = track.xy[i];
      const vis = track.visibility[i];
      if (!pts) return;

      // The video is letterboxed inside its box by object-fit: contain, so the
      // drawn area is not the element's full rect.
      const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
      const w = video.videoWidth * scale;
      const h = video.videoHeight * scale;
      const ox = (rect.width - w) / 2;
      const oy = (rect.height - h) / 2;
      const at = (frame, k) => [ox + frame[k][0] * w, oy + frame[k][1] * h];
      const P = (name) => at(pts, index[name]);
      const V = (name) => vis[index[name]];

      if (showTrails) {
        // The path each joint has just travelled, fading out backwards. Drawn
        // under the skeleton so the current pose stays readable on top of it.
        ctx.lineWidth = Math.max(rect.width * 0.004, 1.5);
        ctx.lineCap = "round";
        const from = Math.max(0, i - TRAIL_FRAMES);
        for (const [name, colour] of TRAILS) {
          const k = index[name];
          if (k == null) continue;
          for (let j = from + 1; j <= i; j += 1) {
            const a = track.xy[j - 1];
            const b = track.xy[j];
            if (!a || !b) continue;
            if (track.visibility[j][k] < MIN_VIS) continue;
            ctx.strokeStyle = colour;
            ctx.globalAlpha = 0.08 + 0.5 * ((j - from) / Math.max(1, i - from));
            ctx.beginPath();
            const [ax, ay] = at(a, k);
            const [bx, by] = at(b, k);
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
          }
        }
        ctx.globalAlpha = 1;
      }

      if (!showSkeleton) return;

      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(rect.width * 0.007, 2);
      for (const [a, b, colour] of BONES) {
        if (V(a) < MIN_VIS || V(b) < MIN_VIS) continue;
        const [ax, ay] = P(a);
        const [bx, by] = P(b);
        ctx.strokeStyle = colour;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }

      const r = Math.max(rect.width * 0.008, 2.5);
      for (const name of track.landmarks) {
        if (V(name) < MIN_VIS) continue;
        const [x, y] = P(name);
        ctx.fillStyle = name.startsWith("left")
          ? C_LEFT
          : name.startsWith("right")
            ? C_RIGHT
            : C_SPINE;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [track, url, showSkeleton, showTrails]);

  if (!url || !track) return null;

  // State follows the element's own play/pause events rather than being set
  // here, so it stays correct when playback starts or stops by any other route
  // (looping, the native controls, or the media keys).
  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  };

  const seek = (seconds) => {
    const v = videoRef.current;
    if (!v || v.readyState < 1) return;
    const dur = Number.isFinite(v.duration) ? v.duration : timeline?.duration_s ?? 0;
    const target = Math.max(0, Math.min(seconds, dur ? dur - 0.02 : seconds));
    v.currentTime = target;
    setT(target);
    onTime?.(target);
  };

  /** One frame of the *track*, which is the resolution the measurements have. */
  const step = (frames) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    seek(v.currentTime + frames / (track.fps || 30));
  };

  return (
    <section className="glass card">
      <h3>What the system tracked</h3>
      <div className="player">
        <video
          ref={videoRef}
          src={url}
          crossOrigin={file ? undefined : "anonymous"}
          playsInline
          muted
          loop
          onTimeUpdate={(e) => {
            setT(e.currentTarget.currentTime);
            onTime?.(e.currentTarget.currentTime);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onClick={toggle}
        />
        <canvas ref={canvasRef} />
        {!playing && (
          <button className="player-play" onClick={toggle} aria-label="Play">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          </button>
        )}
      </div>

      {timeline && (
        <TimelineExplorer
          timeline={timeline}
          current={t}
          onSeek={seek}
          onSelect={onSelect}
        />
      )}

      <div className="player-bar">
        <button className="btn" onClick={toggle}>{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => step(-1)} aria-label="Previous frame">−1f</button>
        <button className="btn" onClick={() => step(1)} aria-label="Next frame">+1f</button>
        <div className="speeds" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`chip clip-pick${speed === s ? " on" : ""}`}
              aria-pressed={speed === s}
              onClick={() => setSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </div>
        <button className="btn" onClick={() => setShowSkeleton((s) => !s)}>
          {showSkeleton ? "Hide skeleton" : "Show skeleton"}
        </button>
        <button className="btn" onClick={() => setShowTrails((s) => !s)}>
          {showTrails ? "Hide trails" : "Show trails"}
        </button>
      </div>
      <p className="player-note">
        {track.n_frames} frames tracked at {track.fps.toFixed(1)} fps · orange = left,
        blue = right{showTrails ? " · trails show the last 1.5 seconds of travel" : ""}
      </p>
    </section>
  );
}
