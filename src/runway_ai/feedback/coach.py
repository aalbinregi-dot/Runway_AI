"""Coaching layer: turn an assessment into a session plan a person can act on.

Two design choices worth stating.

**Findings are ranked by cost, then capped.** A weak walk trips a dozen bands at
once. Handing back twelve simultaneous corrections is useless coaching - nobody
can attend to twelve things while walking - so the plan leads with the two or
three faults carrying the most weight and holds the rest back.

**Drills are attached to faults, not to scores.** The library below is keyed by
criterion and keeps the specific corrective exercise a coach would actually
prescribe, so the advice stays concrete instead of restating the score as prose.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from ..config import CRITERION_LABELS
from ..scoring.assess import Assessment

MAX_FOCUS_AREAS = 3


@dataclass
class Drill:
    name: str
    instruction: str
    duration: str

    def as_dict(self) -> dict:
        return asdict(self)


#: Corrective drills per criterion, in the order a coach would introduce them.
DRILLS: dict[str, list[Drill]] = {
    "posture": [
        Drill("Wall alignment", "Stand with heels, hips, shoulders and head touching a wall. "
              "Hold, step away keeping the shape, and walk five metres.", "5 min daily"),
        Drill("Book walk", "Walk with a book balanced on your head. Keep the ribcage stacked "
              "over the pelvis rather than leaning to hold it.", "10 min"),
    ],
    "balance": [
        Drill("Single-leg hold", "Stand on one leg, eyes forward, 30 seconds each side. "
              "Progress to eyes closed.", "3 sets each side"),
        Drill("Line walk", "Walk heel-to-toe along a taped line, slowly, without stepping off.",
              "10 lengths"),
    ],
    "confidence": [
        Drill("Tempo commit", "Set a metronome to your target cadence and walk the full length "
              "without changing pace, including the turn.", "8 lengths"),
        Drill("Eyeline fix", "Pick a point at the end of the runway at eye height and do not "
              "break from it for the whole pass.", "6 lengths"),
    ],
    "shoulder_alignment": [
        Drill("Shoulder reset", "Roll both shoulders up, back and down before each pass; "
              "walk keeping the collarbones wide.", "before each pass"),
        Drill("Mirror check", "Walk toward a mirror and watch the shoulder line stay level "
              "across every step.", "10 min"),
    ],
    "head_position": [
        Drill("Chin parallel", "Walk with the chin level and the crown of the head lifting. "
              "The gaze stays at the far end, not on the floor.", "10 min"),
        Drill("Wall-slide neck", "Lengthen the back of the neck against a wall, then carry that "
              "position into the walk.", "5 min daily"),
    ],
    "arm_swing": [
        Drill("Relaxed pendulum", "Walk letting the arms swing from the shoulder, elbows soft. "
              "Do not drive them; let the stride move them.", "10 lengths"),
        Drill("Even swing", "Walk counting the swing of each arm; match the two amplitudes.",
              "8 lengths"),
    ],
    "hip_movement": [
        Drill("Lateral hip strength", "Side-lying leg raises and banded lateral walks to hold "
              "the pelvis level in single-leg stance.", "3 x 15 each side"),
        Drill("Hip-level walk", "Walk slowly with hands on the hip crests, feeling for the drop "
              "on each step and resisting it.", "10 min"),
    ],
    "rhythm": [
        Drill("Metronome walk", "Set a metronome to your step rate and land each foot on the "
              "beat for the whole pass.", "10 min"),
        Drill("Count the pass", "Count steps aloud down the runway; the count should come out "
              "the same on every pass.", "6 lengths"),
    ],
    "stride_length": [
        Drill("Marked stride", "Tape floor markers at your target stride and walk landing on "
              "each one until the length feels natural.", "10 lengths"),
        Drill("Slow-motion stride", "Walk at half speed with a full stride to feel the reach "
              "without rushing.", "8 lengths"),
    ],
    "foot_placement": [
        Drill("One-line walk", "Walk placing each foot directly on a taped line, one in front "
              "of the other.", "10 lengths"),
        Drill("Heel-to-toe roll", "Practise landing heel first and rolling through to the toe "
              "on the line.", "10 min"),
    ],
    "symmetry": [
        Drill("Video both sides", "Film from the front and check the left and right swing "
              "match in size and timing.", "weekly"),
        Drill("Unilateral strength", "Work the weaker side alone - single-leg squats and step-ups "
              "- to even out the two limbs.", "3 x 12 each side"),
    ],
    "stage_presence": [
        Drill("Chest open hold", "Draw the shoulder blades gently together and walk holding the "
              "chest open the whole way.", "10 lengths"),
        Drill("Full pass rehearsal", "Rehearse the entire pass - entrance, walk, pose, turn, exit "
              "- as one continuous performance.", "5 full passes"),
    ],
}


@dataclass
class CoachingPlan:
    headline: str
    summary: str
    focus_areas: list[dict]
    strengths: list[str]
    drills: list[dict]
    next_session: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _headline(a: Assessment) -> str:
    if a.overall >= 85:
        return f"{a.grade} walk - {a.overall:.0f}/100. Polish, not repair."
    if a.overall >= 70:
        return f"{a.grade} walk - {a.overall:.0f}/100. Solid base with clear next steps."
    if a.overall >= 55:
        return f"{a.grade} walk - {a.overall:.0f}/100. The fundamentals need attention."
    return f"{a.grade} walk - {a.overall:.0f}/100. Start with posture and balance."


def build_plan(a: Assessment) -> CoachingPlan:
    """Produce a prioritised coaching plan from an assessment."""
    # One entry per criterion, keeping only its most severe finding, so the plan
    # does not spend all three slots on three symptoms of the same fault.
    by_criterion: dict[str, object] = {}
    for f in a.findings:
        if f.criterion not in by_criterion:
            by_criterion[f.criterion] = f
    ranked = sorted(by_criterion.values(), key=lambda f: -f.severity)[:MAX_FOCUS_AREAS]

    focus = [{
        "criterion": f.criterion,
        "label": CRITERION_LABELS[f.criterion],
        "score": round(a.scores[f.criterion], 1),
        "issue": f.message,
        "feature": f.feature,
        "measured": f.measured,
        "target": f.ideal,
    } for f in ranked]

    drills = []
    for f in ranked:
        for d in DRILLS.get(f.criterion, [])[:2]:
            # ``for`` is the human label and ``criterion`` the key. Both are
            # emitted because the client needs the label to read and the key to
            # post back when the user takes a drill on for tracking - deriving
            # one from the other client-side would mean a second label map that
            # could drift out of step with this one.
            drills.append({"for": CRITERION_LABELS[f.criterion],
                           "criterion": f.criterion, **d.as_dict()})

    if focus:
        worst = focus[0]["label"].lower()
        summary = (
            f"Across twelve criteria your strongest work is "
            f"{(a.strengths[0].lower() if a.strengths else 'consistent effort')}, "
            f"and the biggest single gain available is in {worst}. "
            f"Work the {len(focus)} focus area(s) below before adding anything else."
        )
    else:
        summary = ("No criterion fell outside tolerance. Keep the current routine and "
                   "work on consistency across repeated passes.")

    next_session = [f"Film another pass after working {f['label'].lower()}" for f in focus[:2]]
    next_session.append("Record from the same distance and angle so scores stay comparable.")

    return CoachingPlan(
        headline=_headline(a),
        summary=summary,
        focus_areas=focus,
        strengths=a.strengths,
        drills=drills,
        next_session=next_session,
    )
