/** Display names for the twelve criteria.
 *
 * Kept in one place because the API returns snake_case keys and every page that
 * shows a score has to turn them into something readable. Two pages had grown
 * their own copy of this map, which is how a third page ended up rendering raw
 * keys - the copy it was written against used the long form of the names.
 */
export const LABELS = {
  posture: "Posture",
  balance: "Balance",
  confidence: "Confidence",
  shoulder_alignment: "Shoulder Alignment",
  head_position: "Head Position",
  arm_swing: "Arm Swing",
  hip_movement: "Hip Movement",
  rhythm: "Rhythm",
  stride_length: "Stride Length",
  foot_placement: "Foot Placement",
  symmetry: "Symmetry",
  stage_presence: "Stage Presence",
};

/** Readable name for a criterion key, falling back to the key itself. */
export const label = (key) => LABELS[key] ?? String(key ?? "").replace(/_/g, " ");

/** One-word forms, for radar axes, body-map hotspots and chip rows where the
 *  full name would either wrap or be clipped. Twelve of them have to fit round a
 *  circle on a 390 px phone, which "Shoulder Alignment" does not. */
export const SHORT = {
  posture: "Posture",
  balance: "Balance",
  confidence: "Confidence",
  shoulder_alignment: "Shoulders",
  head_position: "Head",
  arm_swing: "Arms",
  hip_movement: "Hips",
  rhythm: "Rhythm",
  stride_length: "Stride",
  foot_placement: "Feet",
  symmetry: "Symmetry",
  stage_presence: "Presence",
};

export const short = (key) => SHORT[key] ?? label(key);

/** Display order for the radar and the body map: the twelve in the order the
 *  server weights them, so neighbouring axes are related rather than arbitrary.
 *  Kept here so both components trace the same polygon. */
export const ORDER = [
  "posture", "shoulder_alignment", "head_position", "stage_presence",
  "confidence", "rhythm", "stride_length", "arm_swing",
  "hip_movement", "foot_placement", "symmetry", "balance",
];
