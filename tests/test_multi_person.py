"""Group analysis: crop geometry, track association, and refusing to guess."""
from __future__ import annotations

import numpy as np
import pytest

from runway_ai.analyze import _match_template
from runway_ai.pose import multi


def test_body_box_covers_a_whole_body_below_the_face():
    # A face 100px wide at (500, 200) in a 1080x1920 frame.
    x0, y0, x1, y1 = multi._body_box((500, 200, 100, 100), 1080, 1920)
    assert x0 < 500 and x1 > 600, "crop must be wider than the face"
    assert y0 < 200, "crop must start above the head"
    assert y1 > 200 + 7 * 100, "crop must reach well below the feet"


def test_body_box_is_clipped_to_the_frame():
    # A face at the very edge would otherwise produce negative coordinates,
    # which numpy silently reads as slicing from the far end of the array.
    x0, y0, x1, y1 = multi._body_box((5, 5, 80, 80), 640, 480)
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 480


def test_continuity_prefers_the_nearer_detection():
    track = multi.PersonTrack(track_id=0)
    track.boxes.append((100, 0, 50, 50))
    near = multi._continuity(track, (110, 0, 50, 50), 1000)
    far = multi._continuity(track, (900, 0, 50, 50), 1000)
    assert near > far


def test_continuity_cannot_outweigh_a_face_difference():
    """The tiebreaker must never pull a track onto a different person.

    Two distinct faces sit far apart in cosine; the position bonus is capped
    well below that gap, so appearance always decides and position only orders
    candidates that already look alike.
    """
    track = multi.PersonTrack(track_id=0)
    track.boxes.append((0, 0, 50, 50))
    assert multi._continuity(track, (0, 0, 50, 50), 1000) == pytest.approx(
        multi.CONTINUITY_WEIGHT)
    assert multi.CONTINUITY_WEIGHT < 0.2


def test_template_is_unit_length():
    t = multi.PersonTrack(track_id=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        t.embeddings.append(rng.normal(size=128) * 7.0)
    assert np.linalg.norm(t.template) == pytest.approx(1.0)


def _enrolled(**people) -> dict:
    return {i: {"template": v, "name": k}
            for i, (k, v) in enumerate(people.items(), start=1)}


def test_match_names_a_clearly_enrolled_walker():
    rng = np.random.default_rng(1)
    alice = rng.normal(size=128)
    bob = rng.normal(size=128)
    probe = alice + 0.05 * rng.normal(size=128)      # same person, slight noise
    got = _match_template(probe, _enrolled(alice=alice, bob=bob))
    assert got is not None and got["name"] == "alice"


def test_match_refuses_when_nobody_is_close():
    """A stranger must come back as nobody, not as the nearest enrolled user.

    Attribution errors are worse here than a missing name: the wrong label
    attaches one person's scores to another person's history.
    """
    rng = np.random.default_rng(2)
    enrolled = _enrolled(alice=rng.normal(size=128), bob=rng.normal(size=128))
    assert _match_template(rng.normal(size=128), enrolled) is None


def test_match_refuses_when_two_candidates_are_equally_close():
    """No name at all beats a coin-flip between two similar-looking accounts."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=128)
    twin_a = base + 0.01 * rng.normal(size=128)
    twin_b = base + 0.01 * rng.normal(size=128)
    got = _match_template(base, _enrolled(a=twin_a, b=twin_b))
    assert got is None


def test_match_with_no_enrolled_users_is_none():
    assert _match_template(np.ones(128), {}) is None
