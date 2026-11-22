/**
 * One place that turns a measured number into a string.
 *
 * The pipeline's features span jerk indices in the thousands and torso ratios
 * below one, so a fixed number of decimals is unreadable at one end or the
 * other. Worse, the same quantity arrives with two spellings: the timeline
 * labels its traces `deg` and `torso`, the rubric breakdown labels its bands `°`
 * and `torso`. Three components had begun formatting these themselves and were
 * already disagreeing about how many decimals a torso length gets.
 */

/** Adaptive precision, for a number whose unit is unknown. */
export const fmt = (v) => {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  // Counts arrive through here too — "8 steps detected" should not read "8.000".
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  return v.toFixed(3);
};

/**
 * A measurement with its unit, ready to render.
 *
 * `share` becomes a percentage: "eyeline forward 0.62" reads as a raw quantity
 * of something, "62% of the walk" reads as what it is.
 */
export function withUnit(value, unit) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  switch (unit) {
    case "°":
    case "deg":
      return `${value.toFixed(1)}°`;
    case "torso":
      return `${value.toFixed(3)} torso`;
    case "share":
      return `${Math.round(value * 100)}%`;
    case "cv":
    case "index":
    case "cycles":
      return value.toFixed(3);
    case "jerk":
      return value.toFixed(0);
    case "Hz":
      return `${value.toFixed(2)} Hz`;
    case "s":
      return `${value.toFixed(2)}s`;
    case "spm":
      return `${Math.round(value)} spm`;
    case "fps":
      return `${Math.round(value)} fps`;
    case "×":
      return `${value.toFixed(2)}×`;
    case "/s":
      return `${value.toFixed(2)}/s`;
    default:
      return fmt(value);
  }
}

/** Short form for a chart axis or a dense legend: no unit, tight precision. */
export function tight(value, unit) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (unit === "°" || unit === "deg") return value.toFixed(1);
  if (unit === "share") return `${Math.round(value * 100)}%`;
  if (unit === "jerk" || unit === "spm" || unit === "fps") return value.toFixed(0);
  const a = Math.abs(value);
  return a >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/**
 * What a rubric band is asking for, in words.
 *
 * The three modes are not interchangeable and the difference matters to a reader
 * deciding what to change: "under 12°" and "7° give or take 5°" describe the
 * same band only if you already know that `lower` mode does not penalise being
 * *more* upright than the target.
 */
export function targetText(band) {
  const { mode, ideal, tol, unit } = band;
  if (mode === "lower") return `under ${withUnit(ideal + tol, unit)}`;
  if (mode === "upper") return `at least ${withUnit(ideal - tol, unit)}`;
  return `${withUnit(ideal, unit)} ± ${withUnit(tol, unit)}`;
}

/**
 * A plotting range for one band's gauge, and whether the measurement fits in it.
 *
 * Fixed at three tolerances either side of the target rather than stretched to
 * include the measurement. Stretching was the first attempt and it is quietly
 * useless: a badly degraded recording can measure twenty tolerances out, and an
 * axis that reaches far enough to include it squeezes the entire acceptable
 * region into a few pixels — so every bad measurement renders at the same place,
 * at 1/1.15 of the width, whatever its value.
 *
 * Three tolerances is where the rubric's falloff has already reached 0.1 of a
 * mark, so nothing beyond it changes a decision. Anything further out is reported
 * as off the scale, with the number itself printed beside the gauge.
 */
export function bandAxis(band) {
  const { mode, ideal, tol, measured } = band;
  const reach = 3;
  let lo;
  let hi;
  if (mode === "lower") {
    // Every `lower` band reads a magnitude — an absolute mean, a spread, a jerk
    // index — so zero is the natural floor and the axis starts there.
    lo = 0;
    hi = ideal + reach * tol;
  } else if (mode === "upper") {
    lo = Math.max(0, ideal - reach * tol);
    hi = ideal + tol;
  } else {
    lo = Math.max(0, ideal - reach * tol);
    hi = ideal + reach * tol;
  }
  if (!(hi > lo)) hi = lo + Math.max(Math.abs(tol), 1e-6);
  return [lo, hi, measured > hi, measured < lo];
}

/** Position of `value` inside `[lo, hi]`, clamped, as a 0-100 percentage. */
export const pct = (value, lo, hi) =>
  Math.max(0, Math.min(100, ((value - lo) / (hi - lo || 1)) * 100));
