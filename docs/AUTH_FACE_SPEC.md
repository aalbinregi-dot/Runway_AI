# Runway AI — Accounts, Sessions, and Opt‑In Face Attribution
## Implementation Specification v1

**Target repo:** `C:\Users\muhammed faris c a\OneDrive\Desktop\albins`
**Written against:** Python 3.14.0, sqlite3 3.50.4, OpenCV 4.13.0, argon2‑cffi 25.1.0, cryptography 46.0.3, FastAPI 0.127.0, React + Vite in `web/`
**Status of every number below:** each threshold carries either a measurement taken on this machine (§1) or an explicit "NOT MEASURED" flag. Nothing is quoted from a model card as if it were validated here.

---

## 0. Scope, and the contradictions between the four designs — resolved

### 0.1 What ships in v1

1. Email + password accounts, opaque server‑side sessions in an httpOnly cookie.
2. Per‑account walk history, replacing the free‑text `?user=` string as the identity boundary.
3. Opt‑in face enrolment (webcam capture only) and face **verification** of the uploader, plus a *claim* path so a walk that matches a different enrolled account can be collected by that account — without ever naming that account to the uploader.
4. A `PRAGMA user_version` migration runner that the live 4‑user / 11‑session database survives.

### 0.2 Contradictions between the four designs, and the choice made

| # | Conflict | Decision | Why |
|---|---|---|---|
| C1 | Design 1 puts `Depends(current_user)` on `/analyze` **and** claims `?save=false` preserves anonymous analysis. Design 4 says auth is never a gate. | **Design 4 wins.** `/analyze*` uses `current_user_optional -> int \| None`. Anonymous analysis returns a full score and persists nothing, exactly as today. | `Depends(current_user)` raises 401 before the handler reads `save`; the two halves of Design 1's own bullet are mutually exclusive. Anonymous scoring is current working behaviour (`user` is optional at `app/api.py:113`) and removing it is a product regression, not a security fix. |
| C2 | Designs 1 & 4 add `password_hash` to `users`; Design 3 says "NO CHANGE to the users table" and then contradicts itself with "Version 1 = auth columns". | **Side table `user_auth`.** No credential column is ever added to `users`. | `store.create_user`'s INSERT hard‑codes five columns; `user_history` spreads `sessions` rows with `{**dict(s), ...}` and the same pattern invites a future `SELECT *` on a widened `users`. A side table is a `CREATE TABLE` (which genuinely applies to the live DB) rather than a column addition, and `cli.py:41` / `cli.py:142` keep working untouched. |
| C3 | Designs 2 & 3 both create `src/runway_ai/face/` as a package. | **Neither. `src/runway_ai/face.py` is amended in place.** No `face/` directory is created. | `src/runway_ai/face.py` already exists (305 lines: `detect_in_frame`, `_pick_walker`, `embeddings_from_video`, `build_template`, `identify`, `Identification`). A package of the same name silently shadows it — no ImportError, no test failure — and would orphan the only face calibration ever run on this system. |
| C4 | Design 2 sets `tau_accept = 0.50 / margin 0.10`; Design 3 sets `0.45 / 0.06`; `face.py` ships `0.363 / 0.10`. | **Keep `ACCEPT_COSINE = 0.363`, `MIN_MARGIN = 0.10`.** | Design 2's 0.50 is extrapolated from a Gaussian fit to a *single* impostor identity, well past the p99.9 it can observe. Design 3's 0.45 is applied to un‑aggregated per‑photo templates, where my own measurement puts 4.1 % of genuine single‑frame pairs *below even 0.363* (§1.3). Neither is measured. 0.363 plus the aggregation rule is the only combination with local evidence behind it. |
| C5 | Design 2 gates probe frames on `det_conf >= 0.90 AND width >= 60 px`. | **Rejected for probes.** Probes keep the shipped `DETECT_SCORE = 0.6` plus `_pick_walker`'s nose anchoring. A stricter gate applies to **enrolment only**. | Measured, composed end‑to‑end, on the reference clip: 16 uniform samples → **16 walker faces at conf 0.6, 15 at conf ≥ 0.90, and only 5 at conf ≥ 0.90 AND width ≥ 60 px** (§1.2). The 60 px gate starves the sampler below `MIN_PROBE_FRAMES` on the very clip the design was derived from. |
| C6 | Designs 2 & 3 propose enrolment from uploaded still photos; Design 4 offers a file‑picker fallback. | **Cut entirely. Webcam capture only, bound to a server‑issued nonce.** | Uploaded photos make enrolment a way to enrol a *non‑consenting third party*, which is the one thing the brief forbids. SFace has no liveness capability and a JPEG of X is indistinguishable from X. |
| C7 | Design 1 binds `GET /media/{token}` to the session cookie. | **Rejected as a hard binding.** `media.store(source, user_id=None)` and `media_file` enforce ownership **only when `entry.user_id is not None`**. | `SkeletonPlayer.jsx:193` renders `crossOrigin={file ? undefined : "anonymous"}` — credentials mode `same-origin`. Cookie‑binding works through the Vite proxy and silently 401s the moment `VITE_API_URL` points elsewhere. Anonymous link analyses have no owner and must stay pure capability URLs. |
| C8 | Design 1 proposes `PRAGMA journal_mode = WAL` inside the migration list. | **Cut.** No WAL. | The DB is at `…\OneDrive\Desktop\albins\data\runway.db`. WAL adds `-wal` and `-shm` sidecars that a sync client can upload out of step with the main file. Also `journal_mode` cannot execute inside the `BEGIN IMMEDIATE` transaction the migration now requires. |
| C9 | Designs 1 & 4 expose `POST /auth/claim-legacy {legacy_name}`. | **Cut. Operator CLI only: `python -m runway_ai claim --user-id N --email …`.** | Possession of the string `"Faris"` is the only credential the endpoint checks. `users.name` has no UNIQUE constraint and `cli.py` mints fresh unclaimed rows forever, so the window never closes. Taking `--user-id` also resolves the name‑ambiguity problem Design 4's own schema note raises and then violates. |
| C10 | Design 3 stores 1–5 template rows per user and matches by max; Design 4 stores one row per frame plus `thumb BLOB NOT NULL`. | **One L2‑normalised centroid per user, no images.** | `face.identify(probes, templates: dict[int, np.ndarray])` takes exactly one template per user. Max‑over‑N inflates impostor scores as fast as genuine ones. `thumb BLOB` converts a template store into a face‑photo database and contradicts `face.py`'s own documented privacy position. |
| C11 | Design 1 proposes per‑IP rate limiting + `X-Forwarded-For` + `TRUSTED_PROXY`. | **Cut. Per‑username exponential backoff only.** | `vite.config.js` does not set `xfwd`, so every browser request reaches uvicorn as `127.0.0.1`. The per‑IP bucket is one global bucket: it throttles all users together and cannot distinguish an attacker. |
| C12 | Design 1 proposes a mandatory `X-Runway-Auth: 1` CSRF header. | **Cut, with a stated trigger to reinstate.** `SameSite=Lax` is the CSRF control. | A cross‑site multipart form POST does not carry a `SameSite=Lax` cookie, so it arrives unauthenticated and 401s with no token check needed. **Reinstate a double‑submit token the day the cookie must become `SameSite=None`** (i.e. a cross‑origin `VITE_API_URL` deployment). |

### 0.3 Explicitly deferred to v2, and why

- **Cross‑account auto‑attribution that names the other user to the uploader.** There is no safe way to tell an uploader "this looks like Ana" without disclosing that Ana enrolled, to someone with no relationship to her. v1 instead files the walk under the uploader and raises a *claim* that only the matched account can see and accept (§3.7). If a group/coach relation is ever added, naming can be re‑scoped to it.
- **Coach ↔ student links, group match scope, `CandidatePicker`.** No relation table exists, no user has one; with names withheld the candidate list always has one entry.
- **`POST /face/identify` as a standalone endpoint.** Never ships. Identification runs only as a step inside an authenticated `/analyze*` on a video the caller uploaded. A standalone route is a membership‑inference and hill‑climbing oracle.
- **Per‑face delete, `GET /faces/{id}/thumb`, `POST /face/scope`, `face_match_log`, `GET /account/export`, key rotation, dormancy sweeps, `POST /auth/logout-all`, `POST /sessions/adopt` + `result_token`.** All cut. Rationale per item in §7.

---

## 1. Measurements taken on this machine, today

Every command below was run in the project root against `video_2026-07-28_14-41-27.mp4` (1080×1920, 369 frames) and its cached pose track `data/interim/_last_video.csv`.

### 1.1 Environment facts (verified, not assumed)

| Claim | Result |
|---|---|
| `passlib` 1.7.4 + `bcrypt` 5.0.0 | **BROKEN.** `(trapped) error reading bcrypt version` → `AttributeError: module 'bcrypt' has no attribute '__about__'`, then `ValueError: password cannot be longer than 72 bytes`. Python 3.14 also has no `crypt` module. **Do not use passlib.** |
| `argon2-cffi` standalone | Works. `PasswordHasher().hash()` = **0.052 s**, 97‑char PHC string, no length limit. |
| `data/runway.db` | `user_version = 0`, `journal_mode = delete`, `secure_delete = 0`, **`busy_timeout = 5000`**, 4 users (Demo, BatTest, Guest, Faris), 11 sessions. |
| "store.py has no busy_timeout" (asserted by three designs) | **FALSE.** Python's `sqlite3.connect()` default `timeout=5.0` is already in effect. No change needed; do not claim one. |
| `BEGIN IMMEDIATE` + re‑read `user_version` under 6 concurrent threads on a copy of the live DB | **0 errors**, final `user_version = 1`, 4 users and 11 sessions intact. |
| AES‑GCM blob for a 128‑float32 vector | plaintext 512 B, ciphertext+tag 528 B, **stored `nonce(12)‖ct‖tag` = 540 B**. (Design 3's "528 bytes (measured)" omitted the nonce.) |
| `src/runway_ai/face.py` | Exists, 305 lines, imported by nothing in `src/`, `app/` or `tests/`. |
| `.gitignore` | Does **not** match `data/runway.db`. Repo is not yet a git repo. |
| `requirements.txt` | `opencv-python` is **commented out**; `cryptography` and `argon2-cffi` are absent. |

### 1.2 Face detection yield — the composed sampler + gate

16 frames sampled uniformly over the middle 80 % of the clip, walker face chosen by `_pick_walker` against the pose nose track:

```
frame  width  conf        frame  width  conf
   36     50  0.935         193     53  0.927
   55     50  0.930         213     57  0.917
   75     50  0.923         233     59  0.935
   95     49  0.923         253     66  0.920
  114     52  0.850         272     68  0.927
  134     49  0.914         292     81  0.932
  154     50  0.925         312     86  0.932
  174     50  0.922         332    100  0.931
```

- Survivors at `conf ≥ 0.6` (shipped `DETECT_SCORE`): **16 / 16**
- Survivors at `conf ≥ 0.90`: **15 / 16**
- Survivors at `conf ≥ 0.90 AND width ≥ 60 px`: **5 / 16** ← would fail or barely scrape `MIN_PROBE_FRAMES = 5`

Cost: `embeddings_from_video(n_frames=12)` = **2.04 s**, `n_frames=16` = **2.63 s**, warm, full resolution.

**Conclusion:** ship `n_frames = 16` with the existing `DETECT_SCORE = 0.6`. Do not add a probe‑side width gate.

### 1.3 Genuine similarity, measured here

55 walker embeddings sampled every 6 frames over frames 20–344.

| Comparison | min | p1 | median | max |
|---|---|---|---|---|
| single frame vs single frame (800 random pairs) | **0.1628** | 0.2133 | 0.7953 | — |
| 5‑frame centroid from the **far** half vs each **near**‑half frame | **0.7764** | — | 0.8396 | 0.8903 |
| 5‑frame centroid from the **near** half vs each **far**‑half frame | **0.2026** | — | 0.8079 | 0.8880 |
| far‑half centroid vs near‑half centroid | — | — | **0.8969** | — |

**4.13 %** of genuine single‑frame pairs fall below 0.363.

**What this justifies:**
- Single‑frame identification must be refused. `MIN_PROBE_FRAMES = 5` with a **median** across probes stands.
- A 5‑frame centroid is robust across the largest condition change available in this dataset (49 px far face → 100 px near face): worst case 0.7764, more than 2× the accept bar.
- The `enrol=near / probe=far` row (min 0.2026) is the reason the decision statistic is the median over ≥5 probes and never a single frame.

**What this does NOT justify:** any false‑accept rate. See §1.4.

### 1.4 What has NOT been measured, and must never be claimed

- **The impostor distribution is not measured.** One subject was available. `scripts/calibrate_face.py` prints `IMPOSTOR pairs: NOT MEASURED - only one subject was supplied`. **No FAR, FRR, accuracy percentage, or "99.x %" figure may appear in code comments, the UI, docs, or a report.**
- **Cross‑session genuine similarity is not measured.** Every number in §1.3 comes from *one* 369‑frame clip: one camera, one lighting setup, one session, ~0.2 s between frames. The real deployment enrols from a webcam selfie and probes from a runway clip filmed on another day. That gap is unquantified.
- **`MIN_MARGIN = 0.10` is a conservative guess, not a measurement.** It cannot be calibrated without multi‑person footage.
- **The enrolment quality gate (`det ≥ 0.90`, `min(w,h) ≥ 80 px`) is untested against a real webcam.** See §6.
- **Group attribution inherits every one of the above.** `analyze._match_template` reuses `ACCEPT_COSINE` and `MIN_MARGIN` unchanged, so naming a walker in a group clip is exactly as well- or ill-founded as sign-in identification, and no separate accuracy claim may be made for it. It is deliberately built to abstain: below the threshold, or within the margin of a runner-up, it returns no name and the walker stays "Person 2". The failure it is tuned against is *misattribution* — a wrong name would graft one person's posture scores onto another person's history.
- **Within-clip track association (`multi.TRACK_COSINE = 0.30`) is a different problem and a looser threshold, on purpose.** Its job is to keep one walker's frames together for a few seconds under consistent lighting, where the cost of a near-miss is a broken track rather than a wrong identity. It is never used to put a name to anyone. What *has* been measured, on a two-person fixture built by compositing one subject twice — the hardest case, since the faces are identical: **zero identity swaps** across every frame where both tracks were present, with the same-frame constraint and the position tiebreaker doing the separating that appearance cannot.

---

## PART A — Schema and migrations

### A.1 New file: `src/runway_ai/db/migrate.py`

The entire migration mechanism. `store.SCHEMA` is **not** edited; it keeps its existing five tables.

```python
"""Versioned schema migrations, keyed on PRAGMA user_version.

store.SCHEMA re-runs CREATE TABLE IF NOT EXISTS on every connection, which is a
silent no-op for column additions on an existing table. The live data/runway.db
would therefore never receive anything added to SCHEMA, while every tmp_path
test would - a failure invisible to the whole suite. Migrations run once per
database file, inside BEGIN IMMEDIATE so a crash rolls the whole step back.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[list[str]] = [
    # index 0 -> applied when user_version < 1, bumps to 1
    [ ...auth statements, verbatim in A.2... ],
    # index 1 -> applied when user_version < 2, bumps to 2
    [ ...face statements, verbatim in A.3... ],
]

_done: set[str] = set()          # resolved absolute db paths, NOT a bool
_lock = threading.Lock()


def apply(conn: sqlite3.Connection, db_path: Path) -> None:
    key = str(Path(db_path).resolve())
    with _lock:
        if key in _done:
            return
    prev = conn.isolation_level
    try:
        conn.isolation_level = None          # own the transaction explicitly
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("BEGIN IMMEDIATE")      # take the write lock BEFORE deciding
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        for i in range(v, len(MIGRATIONS)):
            for stmt in MIGRATIONS[i]:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    msg = str(exc).lower()
                    if "duplicate column name" in msg or "already exists" in msg:
                        continue            # recover a half-applied earlier attempt
                    raise
            conn.execute(f"PRAGMA user_version = {i + 1}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.isolation_level = prev
    with _lock:
        _done.add(key)
```

**Non‑negotiable properties, each of which has a test in §5:**
- Memoised per **resolved absolute path**, never a process‑wide boolean. The suite creates a fresh `tmp_path` DB per test in one pytest process; a process‑level guard would migrate only the first and leave every later one at `user_version = 0`.
- `BEGIN IMMEDIATE` before reading `user_version` (verified concurrency‑safe under 6 threads, §1.1).
- Called from `store.connect()` **after** `conn.executescript(SCHEMA)`, never before — `PRAGMA foreign_keys = ON` is the first statement of `SCHEMA`, and running DDL ahead of it means running with FK enforcement off.
- **No `PRAGMA journal_mode` statement anywhere.** (C8.)

### A.2 Migration v1 — accounts (exact SQL)

```sql
CREATE TABLE IF NOT EXISTS user_auth (
    user_id       INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    email         TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    display_name  TEXT,
    created_at    TEXT    NOT NULL,
    last_login_at TEXT
);
```
```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_auth_email ON user_auth(lower(email));
```
```sql
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at REAL    NOT NULL,
    expires_at REAL    NOT NULL
);
```
```sql
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
```
```sql
CREATE TABLE IF NOT EXISTS login_attempts (
    email        TEXT PRIMARY KEY,
    n            INTEGER NOT NULL,
    first_at     REAL    NOT NULL,
    locked_until REAL    NOT NULL DEFAULT 0
);
```
```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_name_nocase ON users(lower(name));
```

Notes, each load‑bearing:
- `expires_at` / `locked_until` / `created_at` on session and rate‑limit rows are **REAL epoch seconds**, not ISO text. `store._now()` produces `'…+00:00'` text; comparing that lexically for expiry is a latent bug class. `users.created_at` and `user_auth.created_at` stay ISO text to match the existing convention.
- `auth_sessions.token_hash` stores **only** `hashlib.sha256(token.encode()).hexdigest()`. SHA‑256 rather than Argon2 because the token is already 256 bits of entropy and needs no stretching — a per‑request Argon2 would cost 52 ms on every authenticated call.
- `idx_users_name_nocase` — verified applies cleanly to the live DB (`demo`, `battest`, `guest`, `faris`, no collisions). It is what finally backstops `get_or_create_user`'s check‑then‑insert. **Do not** claim `idx_user_auth_email` does this; it indexes a different table.
- **No `ALTER TABLE users`.** (C2.)

### A.3 Migration v2 — face (exact SQL)

```sql
CREATE TABLE IF NOT EXISTS face_templates (
    user_id         INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    template_enc    BLOB    NOT NULL,
    key_id          TEXT    NOT NULL,
    model_tag       TEXT    NOT NULL,
    dim             INTEGER NOT NULL,
    n_frames        INTEGER NOT NULL,
    created_at      TEXT    NOT NULL,
    last_matched_at TEXT
);
```
```sql
CREATE TABLE IF NOT EXISTS face_settings (
    user_id       INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    match_enabled INTEGER NOT NULL DEFAULT 0 CHECK (match_enabled IN (0,1)),
    updated_at    TEXT    NOT NULL
);
```
```sql
CREATE TABLE IF NOT EXISTS face_consent (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    action         TEXT    NOT NULL CHECK (action IN ('granted','paused','resumed','erased')),
    policy_version TEXT    NOT NULL,
    recorded_at    TEXT    NOT NULL
);
```
```sql
CREATE INDEX IF NOT EXISTS idx_face_consent_user ON face_consent(user_id, recorded_at);
```
```sql
CREATE TRIGGER IF NOT EXISTS face_consent_no_update
BEFORE UPDATE ON face_consent
BEGIN SELECT RAISE(ABORT, 'face_consent is append-only'); END;
```
```sql
CREATE TRIGGER IF NOT EXISTS face_consent_no_delete
BEFORE DELETE ON face_consent
WHEN (SELECT 1 FROM users WHERE user_id = OLD.user_id) IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'consent events cannot be deleted while the user exists'); END;
```
```sql
CREATE TABLE IF NOT EXISTS session_claims (
    session_id        INTEGER PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    candidate_user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    uploader_user_id  INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    score             REAL    NOT NULL,
    margin            REAL    NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'pending'
                              CHECK (state IN ('pending','accepted','rejected')),
    created_at        TEXT    NOT NULL,
    resolved_at       TEXT
);
```
```sql
CREATE INDEX IF NOT EXISTS idx_session_claims_candidate
    ON session_claims(candidate_user_id, state);
```
```sql
ALTER TABLE sessions ADD COLUMN attributed_by TEXT;
```

Notes:
- The `face_consent_no_delete` trigger's `WHEN` clause is written against the parent's existence so that the FK cascade still works with `recursive_triggers` at its default of off. This construction was independently verified by a reviewer on sqlite 3.50.4: direct DELETE blocked, UPDATE blocked, `DELETE FROM users` cascades cleanly.
- Absence of a `face_settings` row means **OFF**. The four existing users therefore need no backfill.
- `ALTER TABLE sessions ADD COLUMN attributed_by TEXT` is nullable, so `store.save_session`'s explicit column list is unaffected. It appears in `user_history` output via the existing `{**dict(s), …}` spread, which is intended: the client shows provenance.
- `face_templates` holds **one row per user** — the L2‑normalised centroid `face.build_template()` produces. (C10.)

### A.4 `store.py` edits

`src/runway_ai/db/store.py`:

1. **`DB_PATH` becomes overridable** (line 22):
   ```python
   DB_PATH = Path(os.environ.get("RUNWAY_DB_PATH") or (PROJECT_ROOT / "data" / "runway.db"))
   ```
   The default is unchanged so nothing breaks today. §6.4 records the OneDrive consequence honestly.
2. **`connect()` calls the migrator** after `executescript(SCHEMA)`:
   ```python
   conn.executescript(SCHEMA)
   migrate.apply(conn, path)
   yield conn
   ```
3. **`get_or_create_user` is fixed** — one connection, case‑insensitive, ambiguity refused:
   ```python
   def get_or_create_user(name: str, db_path=None, **kw) -> int:
       """Console-local trust path. NEVER call this from an HTTP handler:
       it creates a row for any string, which is the exact pre-auth
       vulnerability this release removes."""
       with connect(db_path) as c:
           rows = c.execute(
               "SELECT user_id FROM users WHERE lower(name) = lower(?) LIMIT 2",
               (name,)).fetchall()
           if len(rows) > 1:
               raise ValueError(f"{name!r} matches more than one account; pass --user-id")
           if rows:
               return int(rows[0]["user_id"])
           cur = c.execute(
               "INSERT INTO users (name, email, height_cm, experience, created_at) "
               "VALUES (?,?,?,?,?)",
               (name, kw.get("email"), kw.get("height_cm"),
                kw.get("experience", "beginner"), _now()))
           return int(cur.lastrowid)
   ```
   Signature is byte‑identical, so `cli.py:41` and `cli.py:142` are untouched.
4. **New functions** (all take `db_path=None`, all open their own connection, matching the existing style): `create_account`, `find_account_by_email`, `touch_last_login`, `create_auth_session`, `resolve_auth_session`, `delete_auth_session`, `sweep_auth_sessions`, `login_attempt_state`, `login_attempt_fail`, `login_attempt_clear`, `put_face_template`, `get_face_template`, `candidate_templates`, `delete_face_data`, `face_status`, `set_match_enabled`, `record_consent`, `create_claim`, `pending_claims`, `resolve_claim`, `reassign_session`.

`user_history` and `progress` need **no change** — both are already `WHERE user_id = ?`. Only the source of `user_id` changes.

### A.5 `candidate_templates` — the single reader of `face_templates`

This is the only function in the codebase permitted to `SELECT` from `face_templates`. A test greps for violations (§5.6).

```sql
SELECT t.user_id, t.template_enc, t.key_id, t.model_tag, t.dim
FROM face_templates t
JOIN face_settings s ON s.user_id = t.user_id
WHERE s.match_enabled = 1
  AND t.model_tag = ?
  AND (SELECT action FROM face_consent c
       WHERE c.user_id = t.user_id
       ORDER BY c.event_id DESC LIMIT 1) IN ('granted','resumed');
```

Consent state is joined into the query so there is no way to obtain a candidate set that bypasses it. Returning `[]` means `identify()` returns "no users have enrolled a face" and analysis proceeds normally.

---

## PART B — Backend

### B.1 Dependencies

`requirements.txt` — add to the **Core** block:

```
opencv-python-headless>=4.10   # face detection/recognition (YuNet + SFace ship with it)
argon2-cffi>=23.1              # password hashing. Do NOT use passlib: broken on this env.
cryptography>=42               # AES-GCM at rest for face templates
```

`scripts/check_env.py`, in `check_api()`'s import list:

```python
("argon2", "argon2-cffi"), ("cryptography", "cryptography"), ("cv2", "opencv-python-headless"),
```

Without this, a fresh clone fails as a raw `ImportError` inside `from runway_ai.db import store` **after** `run.bat` has already printed `ok`.

### B.2 New file: `src/runway_ai/auth.py`

Pure functions, no FastAPI import. ~150 lines.

```python
import hashlib, hmac, os, secrets, time
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

# OWASP low-memory profile. NOT argon2.profiles.RFC_9106_LOW_MEMORY, which is
# byte-identical to PasswordHasher()'s defaults (m=65536,t=3,p=4) and therefore
# reduces memory by exactly zero - the mitigation Design 1 proposed is a no-op.
_PH = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)

# Verified against a real hash on import so the timing-equalising branch cannot
# silently become a no-op.
_DUMMY_HASH = _PH.hash("runway-dummy-password-for-timing-equalisation")

SESSION_TTL_S = 14 * 24 * 3600
MIN_PASSWORD_LEN = 10
MAX_PASSWORD_LEN = 1024

def hash_password(pw: str) -> str: ...
def verify_password(stored_hash: str | None, pw: str) -> bool:
    """Always runs a full argon2 verify, even when stored_hash is None."""
def needs_rehash(stored_hash: str) -> bool: ...
def new_session_token() -> tuple[str, str]:
    """Returns (token, sha256_hex). 32 bytes via secrets.token_urlsafe."""
def token_hash(token: str) -> str: ...
def validate_password(pw: str) -> None:
    """Raises ValueError. >= 10 chars, <= 1024 bytes, not in COMMON_PASSWORDS."""
def backoff_seconds(n: int) -> float:
    """min(2 ** max(0, n - 3), 300). No hard lockout: a hard lockout is an
    account-DoS any anonymous caller can trigger on a known email."""
```

Rules that must not be regressed:
- `verify_password(None, pw)` verifies against `_DUMMY_HASH` and returns `False`. Without this, an unknown email returns in microseconds while a known one takes 52 ms — a clean enumeration oracle readable over a LAN without statistics.
- Every login/signup handler is declared `def`, **not** `async def`, so FastAPI runs it in the threadpool. `app/api.py` already mixes both styles (`analyze_upload` is `async`, `analyze_link` is not), so this is easy to get wrong. 52 ms of Argon2 on the event loop stalls every other request.
- A fresh token is minted on **every** login and on password change. Never reuse a pre‑login token (session fixation).
- `sweep_auth_sessions` (`DELETE FROM auth_sessions WHERE expires_at <= ?`) and a `login_attempts` sweep run on the login path, so neither table grows without bound.

### B.3 New file: `src/runway_ai/face_crypto.py`

~70 lines. AES‑256‑GCM over the 512‑byte template.

```python
KEY_ENV      = "RUNWAY_FACE_KEY"        # base64, for server deploys
KEY_FILE_ENV = "RUNWAY_FACE_KEY_FILE"
KEY_ID       = "v1"

def key_path() -> Path:
    """%LOCALAPPDATA%\\runway_ai\\face_key_v1.bin on Windows,
    ~/.config/runway_ai/face_key_v1.bin (0600) elsewhere.
    Refuses to resolve inside PROJECT_ROOT."""

def seal(vec: np.ndarray, user_id: int, model_tag: str) -> bytes:
    """nonce(12) || ct || tag  ==  540 bytes for a 128-float32 vector (measured).
    AAD = f'rw-face|v1|user={user_id}|model={model_tag}'.encode()"""

def open_(blob: bytes, user_id: int, model_tag: str) -> np.ndarray:
    """Returns a WRITEABLE float32 copy. np.frombuffer alone yields a read-only
    array and any later in-place op raises ValueError."""
```

**What this does and does not buy — state this in the module docstring and nowhere weaker.** It does *not* stop an attacker with read access to the whole machine: the keyfile is readable by the same user. It *does* stop every leak of the database file alone — the OneDrive replica, an accidental `git add .`, a copied backup, `sqlite3 runway.db "select * from face_templates"`. Because encryption happens above SQLite, plaintext never reaches the rollback journal or freelist pages either. The AAD binding means a row copied onto another `user_id` fails to decrypt — but only against an attacker who does **not** hold the key, which on a single laptop is the same person. Do not claim otherwise.

Key creation: `os.open(path, O_CREAT | O_EXCL | O_WRONLY, 0o600)`.

### B.4 `src/runway_ai/face.py` — amendments **in place** (no `face/` package)

Keep: `Detection`, `Identification`, `available()`, `cosine()`, `detect_in_frame()`, `embeddings_from_video()`, `build_template()`, `identify()`, and all five constants.

**Change 1 — `_pick_walker` must never fall back to largest‑face.** Lines 165–168 currently do `return max(dets, key=area)` when `nose_xy is None`. Replace:

```python
def _pick_walker(dets, nose_xy, width, height):
    if not dets or nose_xy is None:
        return None
    nx, ny = float(nose_xy[0]) * width, float(nose_xy[1]) * height
    if not (np.isfinite(nx) and np.isfinite(ny)):
        return None                       # _blank_frame writes [nan, nan]
    ...unchanged distance loop...
    return best if best_d <= MAX_NOSE_DISTANCE else None
```
Justification: measured at 16 uniform samples, the largest detection is **not** the walker at frame 0 (80.5 px non‑walker at conf 0.331 vs the walker's 48.2 px at conf 0.920), and there is a 151 px non‑walker detection at frame 294. The fallback silently attributes a bystander's face to the scored body.

**Change 2 — correct frame→row mapping.** `embeddings_from_video` currently indexes `nose_track[min(int(f), len-1)]`, which is `row = original_frame` and is wrong on the path this project actually runs. Verified: with no MediaPipe in the main env, `_analyze_path` takes the `extract_delegated` branch → CSV → `long_to_sequences`, which stores the true mapping in `seq.meta['source_frames']` and is **1‑indexed** (`to_long` emits `np.arange(1, t+1)`; the cached CSV's first row is `frame,1`). Add a `source_frames` parameter and map explicitly:

```python
def embeddings_from_video(path, n_frames=16, nose_track=None, source_frames=None):
    ...
    if source_frames is not None:
        row = int(np.searchsorted(np.asarray(source_frames), f + 1))
    else:
        row = int(f // max(1, stride))     # in-process extract path
    row = min(row, len(nose_track) - 1)
```
Also **read the seek back**: after `cap.read()`, take `int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1` as the actual frame and use that for the lookup. Exact seeking was verified on this file (16 seeks, 0 mismatches) but does not generalise to VFR uploads.

**Change 3 — `extract_from_video` must record `stride`.** In `src/runway_ai/pose/extract.py:255`, add `"stride": stride` to the meta dict. Without it the in‑process branch has no way to map rows back to source frames.

**Change 4 — `n_frames` default 12 → 16.** Measured 16/16 usable at 2.63 s (§1.2).

**Change 5 — replace `embeddings_from_images` with a webcam‑capture function.** The current version silently takes the largest face and would accept a photo of a third party.

```python
ENROL_DET_SCORE = 0.90       # PROVISIONAL - not yet measured on a real webcam
ENROL_MIN_FACE_PX = 80       # PROVISIONAL - see §6.1
ENROL_MAX_SELF_COS = 0.995   # PROVISIONAL - static-frame guard, see §6.2

class EnrolReject(Exception):
    """code is one of: no_face | multiple_faces | too_small | low_confidence
    | static_frames | too_few_usable"""

def embeddings_from_captures(images: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    """Embed webcam capture frames. Returns (kept, rejected).

    Per frame, in order: detect (setInputSize per frame - see below); reject if
    len(dets) != 1; reject if score < ENROL_DET_SCORE; reject if
    min(w, h) < ENROL_MIN_FACE_PX; else embed.
    """
```

**Change 6 — `setInputSize` per frame.** Verified: a `FaceDetectorYN` created with `(320,320)` raises `cv2.error … 'Size(inputW, inputH)' is [320 x 320]` on a differently‑sized frame. `_models(w, h)` is currently rebuilt per image inside `embeddings_from_images`, which reloads the 37 MB SFace graph every time. Restructure:

```python
_DET = None; _REC = None
def _models(width, height):
    global _DET, _REC
    if _REC is None:
        _REC = cv2.FaceRecognizerSF.create(str(EMBEDDER_PATH), "")
    if _DET is None:
        _DET = cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (width, height),
                                         DETECT_SCORE, DETECT_NMS, 5000)
    _DET.setInputSize((width, height))     # MANDATORY before every detect()
    return _DET, _REC
```

**Change 7 — close the single‑template fail‑open in `identify()`.** Line 290 sets `gap = float("inf")` when only one user is enrolled, so `if gap < margin` can never fire. The first user to enrol — during exactly the period when the feature is least validated — gets the loosest possible check. Replace:

```python
    if len(scored) == 1:
        # No runner-up to compare against, so hold the single candidate to a
        # stricter bar instead of an absent one.
        if best_score < accept + margin:
            return Identification(None, best_score, 0.0,
                                  "this face does not match any enrolled user",
                                  len(probes))
        return Identification(best_uid, best_score, 0.0,
                              "matched an enrolled face", len(probes))
```

**Change 8 — fix the dangling reference.** `face.py:125` tells the user to run `python scripts/fetch_face_models.py`, which does not exist. Replace with the real path: the two `.onnx` files under `models/face/`. `scripts/calibrate_face.py:53` points at `docs/FACE.md`, which does not exist — point it at this spec or remove.

**Change 9 — `MODEL_TAG = "sface_2021dec"`** module constant; stored on every template and compared before matching. Embeddings from different recognisers occupy unrelated spaces and produce confident nonsense rather than an error.

**Unchanged and not to be touched:** `ACCEPT_COSINE = 0.363`, `MIN_MARGIN = 0.10`, `MIN_ENROL_FRAMES = 5`, `MIN_PROBE_FRAMES = 5`, `MAX_NOSE_DISTANCE = 1.0`, `DETECT_SCORE = 0.6`. (C4, C5, §1.3.)

`scripts/calibrate_face.py` stays as the single calibration tool. **`scripts/validate_face_thresholds.py` is not created** — a second calibration script that "refuses to lower a threshold below 10 users" cannot ever run against a 4‑user database.

### B.5 `src/runway_ai/media.py`

Two edits only, both keyword‑with‑default so the 5 existing media tests (`tests/test_pipeline.py:529–580`, all calling `media.store(path)` positionally with one argument) keep passing:

```python
@dataclass
class Entry:
    path: Path
    added: float
    content_type: str
    user_id: int | None = None

def store(source: Path | str, user_id: int | None = None) -> str: ...
```

`resolve()` stays auth‑unaware — the ownership comparison happens in `media_file()` in `app/api.py`. This keeps `media.py` single‑purpose and leaves all five `resolve(token)` call sites unchanged.

`_publish_media` becomes `_publish_media(results, user_id: int | None)`; its two call sites (`app/api.py:153` inside `analyze_link`, and `app/api.py:308` inside the `analyze_url_stream` worker closure) pass the resolved uploader id or `None`.

**In‑memory caveat, to be stated in the docstring:** `_entries` is a process‑local dict, so every `uvicorn --reload` restart invalidates every outstanding media token. That is existing behaviour; ownership binding does not make it durable.

### B.6 `app/api.py`

#### B.6.1 New imports and dependencies

```python
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile

COOKIE_NAME = "runway_session"
COOKIE_SECURE = os.environ.get("RUNWAY_COOKIE_SECURE", "0") == "1"

def current_user_optional(runway_session: str | None = Cookie(None)) -> int | None:
    """Resolve the session cookie to a user_id, or None. Never raises."""

def current_user(uid: int | None = Depends(current_user_optional)) -> int:
    """As above, but 401s. Used ONLY on /auth/*, /me/*, /face/* routes."""
    if uid is None:
        raise HTTPException(401, "sign in to continue")
    return uid
```

Cookie attributes on set: `httponly=True`, `samesite="lax"`, `path="/"`, `max_age=SESSION_TTL_S`, `secure=COOKIE_SECURE`. **`Secure` must be config‑driven** — `vite.config.js` sets `host: true` so the dev server is reachable at `http://192.168.x.x:5180`, which is not a secure context; a hardcoded `Secure` cookie is silently dropped there and sign‑in appears to succeed while every subsequent request comes back anonymous, with nothing in the console.

CORS (`app/api.py:40–49`): add `http://localhost:5180` and `http://127.0.0.1:5180`. **State the truth about this in a comment:** dev requests go through the Vite `/api` proxy and are same‑origin, so CORS is not consulted at all today; the addition only matters for a direct‑to‑8000 client. The phone's LAN origin can never be listed ahead of time and works only because of the proxy.

#### B.6.2 Auth endpoints

| Method + path | Signature | Behaviour |
|---|---|---|
| `POST /auth/signup` | `def signup(body: SignupIn, response: Response) -> dict` | `SignupIn{email: str, password: str, display_name: str \| None}`. Validate password (≥10 chars). Create `users` row + `user_auth` row in **one** transaction. Mint session, set cookie. Returns `{user_id, display_name, email}`. Email collision → **409 `"that email is not available"`**. |
| `POST /auth/login` | `def login(body: LoginIn, response: Response) -> dict` | Check `login_attempts` backoff **before** hashing. Always run a full argon2 verify (real hash or `_DUMMY_HASH`). Single error string for every failure: **401 `"incorrect email or password"`**. On success: clear the attempt row, mint a *fresh* token, `sweep_auth_sessions`, `touch_last_login`, rehash if `needs_rehash`. |
| `POST /auth/logout` | `def logout(response, runway_session: str \| None = Cookie(None)) -> Response` | Delete the `auth_sessions` row by `token_hash`, clear the cookie. 204. |
| `GET /auth/me` | `def me(uid: int \| None = Depends(current_user_optional)) -> dict` | **Always 200.** `{"user": null}` when signed out, so a cold load of a signed‑out app produces no console error and no modal pop. |
| `POST /auth/change-password` | `def change_password(body, uid = Depends(current_user), ...)` | Requires the **current** password even with a valid session, then deletes all other `auth_sessions` rows for that user. |

Signup's 409 is an unavoidable enumeration oracle for usability reasons. It is accepted and documented; login is not allowed to leak the same way.

`GET /me/progress` and `GET /me/sessions` replace `GET /users/{name}/progress` (line 353) and `GET /users/{name}/sessions` (line 358). **Both old routes are deleted.** They are GETs that INSERT a `users` row as a side effect for any name any anonymous caller types, and they let anyone read anyone's history.

`GET /me/claims` and `POST /me/claims/{session_id}` — see §B.7.

#### B.6.3 The four analysis endpoints

All four take `uid: int | None = Depends(current_user_optional)`. Persistence becomes `if uid is not None: …`.

- `POST /analyze` — **remove** `user: str | None = Query(None)`. Keep `fps`. Add `identify: bool = Query(False)`.
- `POST /analyze-url` — `class LinkRequest(BaseModel): url: str`. The `user` field is **removed from the model**, not merely ignored, so a client cannot supply an identity at all.
- `POST /analyze/stream` — same as `/analyze`, plus: **`uid` is resolved to a plain `int | None` in the handler body before the `run()` closure is constructed.** (Correction to Design 2 and 3, which both claimed "nothing request-scoped is reachable inside the closure": ordinary Python values close over fine — `app/api.py:279-289` already closes over `user` and `file.filename`. The real constraint is narrower: a `Request` object or a `Depends`‑injected lazy lookup must not be touched off the event loop.)
- `POST /analyze-url/stream` — same treatment for the closure at line 305.

**The temp‑file ordering must change.** `analyze_upload` unlinks at line 129–130 inside a `finally` that runs *before* the `if user:` block at 132; `analyze_stream`'s `run()` does the same at 282–283. Face probing needs the pixels. Restructure both:

```python
    try:
        results = analyze(path, models=models(), fps=fps)
        ident = None
        if identify and uid is not None and suffix in VIDEO_SUFFIXES:
            ident = _identify_walk(path, results[0], uid)   # best-effort, time-boxed
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)
```

`_identify_walk` is a module‑level helper in `app/api.py`:

```python
def _identify_walk(path, result, uploader_id) -> dict:
    """Best-effort. Any exception yields decision='unavailable'."""
    seq_meta   = result.features.get("_seq_meta") or {}
    nose_track = result.track and _nose_from_track(result.track)
    probes = face.embeddings_from_video(path, n_frames=16,
                                        nose_track=nose_track,
                                        source_frames=seq_meta.get("source_frames"))
    templates = store.candidate_templates(face.MODEL_TAG)   # decrypted here
    ident = face.identify(probes, templates)
    ...
```

**`POST /report` (line 335) is unchanged and stays unauthenticated.** It never persists. Note in a comment that it is a free full‑analysis CPU sink for anyone on the LAN.

`GET /health`, `/sources`, `/criteria`, `/stages` are unchanged and public. `/health` must **not** start reporting user counts or auth configuration.

#### B.6.4 `GET /media/{token}`

```python
@app.get("/media/{token}")
def media_file(token: str, uid: int | None = Depends(current_user_optional)) -> FileResponse:
    entry = media.resolve(token)
    if entry is None:
        raise HTTPException(404, "that video has expired; analyse the link again")
    if entry.user_id is not None and entry.user_id != uid:
        raise HTTPException(404, "that video has expired; analyse the link again")
    return FileResponse(entry.path, media_type=entry.content_type)
```

404 rather than 403 so the response does not confirm that a token exists. Anonymous entries (`user_id is None`) keep the pure‑capability behaviour that link analysis has today.

#### B.6.5 SSE error sanitisation

`app/api.py:238` currently emits `{"type": "error", "message": str(exc)}` straight to the browser. Once auth and face code run inside the worker, that channel can carry a `sqlite3` constraint string or a `cv2.error` containing an absolute build path. Change to a fixed generic message with the detail logged server‑side. The face path must raise only typed errors with fixed codes (`no_face`, `multiple_faces`, `too_small`, `low_confidence`, `static_frames`, `too_few_usable`, `not_enrolled`).

### B.7 Face endpoints — `app/face_routes.py` (new file, `APIRouter`)

Kept in one file so the consent gate and the **absence** of an identify route are reviewable together.

| Method + path | Signature | Behaviour |
|---|---|---|
| `GET /face/policy` | `def policy() -> dict` | `{policy_version, plain_text, stores, never_stores, retention}` served from `docs/face-policy-v1.md`. Unauthenticated so it is readable before signing up. |
| `POST /face/capture-nonce` | `def capture_nonce(uid = Depends(current_user)) -> dict` | `{nonce, expires_in: 120}`. In‑memory, single‑use, per‑user. |
| `POST /face/enrol` | `def enrol(nonce: str = Form(...), consent_ack: bool = Form(...), policy_version: str = Form(...), frames: list[UploadFile] = File(...), uid = Depends(current_user)) -> dict` | 400 if `consent_ack` is false, 409 if `policy_version` is stale, 400 if the nonce is unknown/expired/reused. Requires **6–10** frames. Decodes in memory (`cv2.imdecode(np.frombuffer(await f.read(), np.uint8), IMREAD_COLOR)`), **never writes to disk**. Applies `embeddings_from_captures` + the static‑frame guard. On ≥5 kept frames: `build_template`, `seal`, upsert `face_templates`, upsert `face_settings(match_enabled=1)`, append `face_consent('granted')` — **one transaction**. Returns `{ok, kept: n, rejected: [{index, code}]}`. On <5 kept: 422 `{code:"too_few_usable", kept, rejected}`. |
| `GET /face/status` | `def status(uid = Depends(current_user)) -> dict` | `{enrolled, n_frames, match_enabled, model_tag, created_at, last_matched_at, policy_version, needs_reconsent}`. Never returns an embedding, a score, or anyone else's state. |
| `POST /face/match-enabled` | `def set_matching(body: {enabled: bool}, uid = Depends(current_user)) -> dict` | Pause/resume without deleting. Appends `paused`/`resumed`. This is the "withdraw as easily as you granted" control. |
| `DELETE /face` | `def erase(uid = Depends(current_user)) -> dict` | Deletes the `face_templates` row, zeroes `face_settings.match_enabled`, appends `erased`. Runs with `PRAGMA secure_delete = ON` set on that connection, then `VACUUM` on a separate `sqlite3.connect(..., isolation_level=None)` after the transaction commits (VACUUM cannot run inside a transaction). Returns `{deleted: n}`. |
| `GET /face/consent-log` | `def consent_log(uid = Depends(current_user)) -> dict` | The caller's own ledger, newest first. |
| `GET /me/claims` | `def claims(uid = Depends(current_user)) -> dict` | Pending `session_claims` where `candidate_user_id = uid`. Returns `{session_id, created_at, overall_score, grade}` — never the uploader's identity. |
| `POST /me/claims/{session_id}` | `def resolve_claim(session_id: int, body: {accept: bool}, uid = Depends(current_user)) -> dict` | Only callable by `candidate_user_id`. On accept: `UPDATE sessions SET user_id = ?, attributed_by = 'face-claim' WHERE session_id = ?`, mark the claim accepted. On reject: mark rejected, nothing moves. |

**Deliberately absent, and a test asserts it (§5.6):** `POST /face/identify`, `GET /face/whois`, any route returning `template_enc`, a decrypted vector, a similarity matrix, a candidate list with names, or a gallery listing. There is no admin route either.

**Enrolment is refused for accounts declaring under 18** at signup, enforced server‑side. Say plainly in the policy text that age is self‑declared and not verified — do not describe this as a hard block.

### B.8 Identification semantics — the exact decision ladder

`_identify_walk` produces one of five outcomes, attached to the result object as `results[i]["attribution"]`, and each result also carries `results[i]["session_id"]` (currently `store.save_session`'s return value is discarded at `app/api.py:135, 194, 287, 312`).

| decision | condition | what the uploader is told | what happens |
|---|---|---|---|
| `self` | `ident.user_id == uploader_id` | "Matched you." | Session saves to uploader. `attributed_by='face-self'`. |
| `other` | `ident.user_id` is a different enrolled user | **"This doesn't look like your enrolled face."** No name, no id, no score. | Session saves to **uploader**. A `session_claims` row is created for the matched user, who sees it under `GET /me/claims`. |
| `no_match` | `ident.user_id is None`, reason "does not match any enrolled user" | "Couldn't confirm it's you — saved to your account." | Session saves to uploader. |
| `low_quality` | reason "only N usable face frames" | "Couldn't find a clear enough face — saved to your account." | Session saves to uploader. |
| `unavailable` | no candidates, no video (CSV/XLSX upload), models missing, or any exception | Nothing rendered. | Session saves to uploader. |

**Invariants that must hold in code, not just in copy:**
1. `store.save_session` is called with the **authenticated uploader's** `user_id` and nothing else. A face result never selects the row written to.
2. A face match never creates a session, never grants a session cookie, never mutates auth state.
3. The matched user's `user_id`, name, and the raw cosine score are **never** returned to the uploader in any of the five branches. The score is written to `session_claims` (visible only to the candidate) and to the server log.
4. The candidate set is always `store.candidate_templates(MODEL_TAG)` — users who enrolled *and* have `match_enabled = 1` *and* whose latest consent event is `granted`/`resumed`. There is no code path taking a caller‑supplied user id list.

**Rate limit:** `identify=true` is capped at 20 calls per account per hour, recorded in the same `login_attempts`‑style bucket. Repeated `no_match` from one account is what a hill‑climbing run looks like.

### B.9 `analyze()` must not decide who to match against

`analyze()` is the single entry point the CLI, the API and the dashboard all call, and the CLI has no login by design. **Face identification is therefore never invoked from inside `analyze()` or `_analyze_path`.** It is invoked from `app/api.py` only, after `analyze()` returns and before the temp file is unlinked, with an already‑scoped candidate set resolved from the authenticated caller. This makes `python -m runway_ai analyze walk.mp4` structurally incapable of running 1:N biometric matching.

`analyze()` gains one small change so the caller can do this: `_analyze_path`'s video branch must propagate `seq.meta` into `result.features["_seq_meta"]` (`source_frames`, `stride`), because that is the only correct row→frame mapping (§B.4 Change 2).

### B.10 `src/runway_ai/cli.py`

No required changes to `cmd_analyze` (line 41) or `cmd_progress` (line 142). Two additions in `main()` (line 211):

```
python -m runway_ai claim  --user-id 4 --email me@example.com [--display-name Faris]
python -m runway_ai passwd --user-id 4
```

`claim` binds credentials to an existing pre‑auth `users` row. It takes `--user-id`, never a name, so the fact that `users.name` had no UNIQUE constraint at the time those rows were written cannot produce an ambiguous claim. It refuses if a `user_auth` row already exists for that id. This replaces `POST /auth/claim-legacy` entirely (C9).

`passwd` is the password‑reset path. All four live rows have `email = NULL` and there is no mail transport, so a forgotten password is otherwise unrecoverable — and the walk history is the thing users would lose.

**A loud docstring on `get_or_create_user` plus a source‑level test (§5.6) keep it out of HTTP handlers.** It stays in the module only because the CLI needs it, and because a process with filesystem write access to `runway.db` could insert that row with the `sqlite3` shell anyway — the trust boundary is the console, not the account.

### B.11 `.gitignore` — before any face code lands

The repo is not yet a git repo, so the first `git add .` would commit biometrics into history where removal is a rewrite, not a `DELETE`. Add:

```
data/*.db
data/*.db-journal
data/*.db-wal
data/*.db-shm
*.key
face_key_*.bin
models/face/
video_*.mp4
*_analysed.mp4
Synthetic_Runway_Keypoints_Sample.xlsx
```

(Also remove the duplicated `data/media/` line.)

---

## PART C — Frontend

### C.1 Principle: auth is a panel, not a gate

The app still boots straight into Analyse. A signed‑out visitor can drop a video and get a full score; they simply get no history. Every auth surface carries an explicit "Continue without an account" escape. (C1.)

### C.2 New files

| Path | Contents |
|---|---|
| `web/src/auth.jsx` | `AuthProvider` + `useAuth()`. State `{status: 'loading'\|'anon'\|'authed', user, signIn, signUp, signOut, refresh, openSheet(view), sheet}`. Boots from `GET /auth/me` (which always 200s). Registers the 401 handler that `api.js`'s `request()` wrapper calls, so exactly one place flips status to `anon` and reopens the sheet. Context, not a store library — this is one object. |
| `web/src/components/Icons.jsx` | Exports `InfoIcon`. **This is new code.** `InfoIcon` is currently a module‑private function at `Analyse.jsx:20`, never exported, and it is a static 14×14 SVG — there is no expand/collapse disclosure pattern anywhere in `web/src`. `Analyse.jsx` imports it from here after the move. |
| `web/src/components/IdentityChip.jsx` | Occupies the flex slot the `#who` input vacates (`App.jsx:121-130`). Three states: loading skeleton; anon → ghost `.btn` "Save my progress"; authed → display name + caret menu (Account, Face, Sign out). |
| `web/src/components/AccountSheet.jsx` | Single modal instance mounted in `App`. Bottom sheet under 640 px (`max-height: 88svh`, top corners `var(--r)`), centred dialog above. `role="dialog" aria-modal`, focus trap, Esc + backdrop close, scroll lock. Views: `signin \| signup \| account \| face \| privacy`. |
| `web/src/components/AuthForms.jsx` | `<SignIn>` / `<SignUp>` using the existing `.field-input`, `label.lab`, `.btn.solid`, `.notice` classes. |
| `web/src/components/FaceEnrol.jsx` | Four‑step `useReducer`: intro/consent → capture → review → done. Owns the `MediaStream`, the captured Blobs and their object URLs. |
| `web/src/components/FaceManager.jsx` | "Face matching: on/off" toggle (distinct from delete) and a destructive "Delete my face data" with a two‑tap inline confirm (not `window.confirm`). Lists enrolment date and frame count — **no thumbnails**. |
| `web/src/components/Attribution.jsx` | `<AttributionCard>` and the single `BANDS` copy table that owns every hedging string. |
| `web/src/components/Claims.jsx` | Pending `GET /me/claims` list with Accept / Not mine. |

### C.3 Enrolment capture — the rules that are not optional

- **Camera only.** No `<input type="file">` path. `capture="user"` is advisory and ignored by every desktop browser, so a file picker is a plain file picker, and a photo of a non‑consenting person embeds exactly as well as a live face. (C6.)
- **Capture 6 frames**, ~600 ms apart, in one tap (~4 s). `face.MIN_ENROL_FRAMES = 5` and `build_template` raises below it; 6 gives one frame of headroom for a blink or a blur rejection. **Do not lower `MIN_ENROL_FRAMES`** — §1.3 measures 4.13 % of genuine single‑frame pairs below the accept bar.
- **Send the server‑issued nonce** with the multipart POST.
- **Stop every `MediaStream` track** on step change and on unmount. Under React `<StrictMode>` (which `main.jsx` uses) the cleanup runs *before* the `getUserMedia` promise resolves, so the naive effect leaks the first stream and the camera light stays on. Required shape:
  ```js
  useEffect(() => {
    let cancelled = false; let stream = null;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
      .then((s) => { stream = s; if (cancelled) { s.getTracks().forEach(t => t.stop()); return; } setStream(s); })
      .catch(() => setStep("denied"));
    return () => { cancelled = true; stream?.getTracks().forEach(t => t.stop()); };
  }, [step]);
  ```
- Preview `<video>` needs `playsInline muted autoPlay` for iOS.
- **Review shows the client's own captured blobs.** The server returns only `{kept: n, rejected: [{index, code}]}`. No thumbnail is stored server‑side and `GET /faces/{id}/thumb` does not exist. (C10.)

### C.4 Attribution UI — three states, no confidence bands, no number

`Attribution.jsx` exports one `BANDS` table and nothing renders a hedging string that is not in it.

```js
export const BANDS = {
  self:        { title: "Matched you",                        body: "This walk was saved to your account." },
  other:       { title: "This doesn't look like your face",   body: "Saved to your account anyway. If someone else was walking, they can claim it from their own account." },
  no_match:    { title: "Couldn't confirm it's you",          body: "Saved to your account." },
  low_quality: { title: "Couldn't find a clear enough face",  body: "Saved to your account. Walking toward the camera helps." },
  unavailable: null,   // render nothing
};
```

**The cosine similarity is not shown to users at all** — not in the card, not behind a disclosure, not with a "not a probability" label. Design 4 proposed hiding it behind `<InfoIcon>`; the reviewers are right that a distance measure with **no measured false‑accept rate on this system** (§1.4) is not information a user can act on, and labelling it does not stop it reading as certainty. It is available in the server log and in `session_claims` for calibration. A `VITE_DEBUG_FACE=1` build may render it; a production build may not.

`<AttributionCard>` renders above `<ScoreHero>` in `Analyse.jsx` (line ~437), driven by `results[i].attribution` — **per result, not per response**, because a keypoint file yields N results and therefore N sessions (`Analyse.jsx` already has a clip picker for exactly this).

### C.5 `web/src/api.js`

Introduce the shared wrapper the file currently lacks — **nine** independent `fetch` calls today (`health` 35, `criteria` 39, `analyze` 46, `analyzeUrl` 51, `analyzeStream` 65, `analyzeUrlStream` 69, `sources` 77, `progress` 81, `reportHtml` 87):

```js
let onUnauthorized = () => {};
export function setUnauthorizedHandler(fn) { onUnauthorized = fn; }

async function rawRequest(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, { credentials: "include", ...opts });
  if (res.status === 401) onUnauthorized();
  return res;                       // streaming callers keep the unbuffered Response
}
async function request(path, opts) { return unwrap(await rawRequest(path, opts)); }
```

- Drop the `user` argument from `analyze`, `analyzeUrl`, `analyzeStream`, `analyzeUrlStream` and the `?user=` query building at lines 45 and 64.
- `progress(user)` → `progress()` hitting `/me/progress`.
- Add `sessions()`, `me()`, `signup()`, `login()`, `logout()`, `changePassword()`, `faceStatus()`, `faceNonce()`, `faceEnrol()`, `setFaceMatching()`, `deleteFace()`, `faceConsentLog()`, `claims()`, `resolveClaim()`.
- The two streaming builders go through `rawRequest` so `Processing.jsx:121` (`const res = await request()`) keeps receiving an unbuffered `Response` and its existing `!res.ok` branch surfaces a pre‑stream 401 as readable `detail` text. No change is needed in `Processing.jsx` for auth.
- `reportHtml` is dead code (nothing in `web/src` calls it, and there is no `dangerouslySetInnerHTML` anywhere). Leave it, but do **not** cite it as evidence that the app renders server HTML.

### C.6 `web/src/App.jsx`

- Remove `const [user, setUser] = useState(() => localStorage.getItem("runway.user") ?? "")` (line 73).
- Remove the write‑back effect (lines 81–83).
- Remove the `#who` input block (lines 121–130); render `<IdentityChip />` in that slot.
- Drop `user={user.trim()}` from `<Analyse />` and `<Progress />` (lines 134–135).
- Mount `<AccountSheet />`.
- `TABS` stays at three entries. No route guard on any tab.
- Read `localStorage["runway.user"]` **once** at boot into a display‑only hint: "Your earlier walks are saved under the name *Faris*. Run `python -m runway_ai claim --user-id N --email …` to attach them to an account." Then delete the key. **The banner calls nothing** — there is no claim endpoint. (C9.)

### C.7 `web/src/components/Analyse.jsx`

- Drop the `{ user }` prop (line 314) and the two `user` args (lines 331, 344).
- Import `InfoIcon` from `Icons.jsx`.
- Render `<AttributionCard>` above `<ScoreHero>`.
- When `status === 'anon'`, render a line under the result: "Sign in before analysing to keep this in your history." **No adoption flow** — `POST /sessions/adopt` + `result_token` is cut (§7).
- After a signed‑in user's first saved session, show a one‑line dismissible‑forever nudge: "Uploading from a shared phone? Enrol your face so walks land in the right account." Face enrolment is **never** part of signup — consent given to unblock signup is not meaningful consent.

### C.8 `web/src/components/Progress.jsx`

- Drop the `{ user }` prop (line 59).
- The effect keys on the auth `user_id`, not a name string; calls `api.progress()`.
- The `if (!user)` empty state (lines 79–88) becomes a signed‑out state with a "Sign in" button calling `openSheet('signin')`.

### C.9 `web/src/components/SkeletonPlayer.jsx` — **CHANGED, not reference**

Line 193 is `crossOrigin={file ? undefined : "anonymous"}`. `anonymous` means credentials mode `same-origin`, so the moment `VITE_API_URL` points at another host the `<video>` element sends no cookie and an owner‑bound `/media` 404s with nothing useful in the console. Change to:

```jsx
crossOrigin={file ? undefined : "use-credentials"}
```

and note in `web/README.md` that a cross‑origin deployment additionally requires the exact web origin in `allow_origins` (a wildcard cannot be combined with `allow_credentials=True`).

### C.10 `web/src/styles.css`

One appended section, monochrome, reusing `--glass`, `--edge`, `--ink-2`, `--ink-3`, `var(--r)`, `var(--pad)`, `var(--gap)`: `.sheet`, `.sheet-backdrop`, `.sheet-panel`, `.cam`, `.cam-mask`, `.face-grid`, `.face-thumb`, `.band`, `.band-dot`, `.consent`, `.toast`. Do not add a second stylesheet — this file is the design system.

### C.11 `web/README.md`

Record the two footguns that produce silent, unlogged failures:
1. `RUNWAY_COOKIE_SECURE=1` breaks the LAN dev loop (`vite.config.js` `host: true` serves plain HTTP to `192.168.x.x:5180`, which is not a secure context, so the cookie is dropped). Leave it off in dev; turn it on in the deploy checklist.
2. `python -m runway_ai analyze --user Faris` keeps creating name‑keyed rows that are invisible to a web account of the same name until an operator runs `claim --user-id`.

---

## PART D — Tests

Current baseline: **61 passing** in `tests/test_pipeline.py`. Zero API tests, exactly one store test (`test_session_round_trip_and_progress` at line 331, which uses `tmp_path` and therefore only ever exercises the fresh‑DB path). Everything below is new.

### D.1 `tests/test_migrate.py`

| Test | Assertion |
|---|---|
| `test_migrates_copy_of_live_db` | Copy `data/runway.db` to `tmp_path`, open through `store.connect()`. `user_version == 2`; `users` count 4; `sessions` 11; `session_scores` 132; `session_features` 11; `findings` 140; `user_auth`, `auth_sessions`, `face_templates`, `face_settings`, `face_consent`, `session_claims` all exist. **This is the single highest‑value test in the plan** — the existing suite would not catch a silently skipped migration. |
| `test_migration_is_idempotent` | Run it twice on the same file. No exception, `user_version` still 2. |
| `test_migration_recovers_half_applied` | Manually apply the first statement of v1 and leave `user_version = 0`. Migration completes without `already exists`. |
| `test_two_tmp_dbs_in_one_process` | Open two different `tmp_path` DBs. **Both** reach `user_version = 2`. (Catches a process‑level memo instead of a per‑path one.) |
| `test_concurrent_migration` | 6 threads against one fresh copy. Zero exceptions, one final `user_version`. |
| `test_no_wal_sidecars` | After migration, no `-wal` or `-shm` file exists beside the DB. |
| `test_users_table_unchanged` | `users` columns are exactly the original six. |

### D.2 `tests/test_auth.py`

- Unknown email and wrong password return byte‑identical `401 {"detail": "incorrect email or password"}`. **Run the wrong‑password arm against a genuinely active account** — Design 1's prototype compared an unknown email against a legacy NULL‑username row, so both arms took the same branch and the property was never actually exercised.
- Timing: unknown‑email and known‑email‑wrong‑password medians within 2× over 20 trials each.
- Password below 10 chars → 400.
- `POST /auth/logout` → the next `GET /me/sessions` is 401.
- Session fixation: the cookie value after login differs from any value present before.
- Repeated failures engage backoff; a **correct** password after the backoff window succeeds (no hard lockout).
- `GET /auth/me` signed out returns `200 {"user": null}`, not 401.
- User A cannot read user B's sessions by any route.
- `POST /analyze` with no cookie returns a full score and creates **zero** rows.
- Expired `auth_sessions` rows are swept on login.

### D.3 `tests/test_face.py`

- `runway_ai.face.__file__` ends in `face.py`, **not** `face/__init__.py`. (Guards C3 forever.)
- `build_template` with 4 embeddings raises; with 5 succeeds and returns a unit‑norm `float32` of length 128.
- `identify` with one enrolled template and `best_score` between `ACCEPT` and `ACCEPT + MARGIN` returns **not identified** (guards the `inf`‑margin fail‑open, B.4 Change 7).
- `identify` with `len(probes) < 5` returns not identified with the frame‑count reason.
- `identify` with `templates == {}` returns not identified.
- `_pick_walker(dets, None, w, h)` returns `None` (no largest‑face fallback).
- `_pick_walker` with a NaN nose returns `None`.
- `face_crypto.seal` output is exactly **540** bytes; `open_` round‑trips bit‑exactly with norm 1.0; the returned array is **writeable**.
- `open_` under a different `user_id` raises (AAD binding), and the caller drops the template and logs rather than falling back to an unauthenticated compare.
- **`cv2.FaceRecognizerSF.match()` is never called** anywhere in `src/` — a source grep. It L2‑normalises **both** inputs in place (independently measured by two reviewers: input norms 53.92/60.37 and 12.47/11.95 all read 1.000000 after the call), so passing a cached template corrupts it silently.
- A regression test on the reference clip: `embeddings_from_video(n_frames=16)` yields **≥ 5** probes. Marked `@pytest.mark.skipif(not face.available() or not VIDEO.exists())`. Note: `demo_expert.mp4` is synthetic and yields **0** face embeddings — any face test must use `video_2026-07-28_14-41-27.mp4`.

### D.4 `tests/test_face_privacy.py`

- A user with no `face_settings` row is **not** in `candidate_templates()` (default OFF, no backfill needed for the four live users).
- `POST /face/enrol` without `consent_ack` → 400; with a stale `policy_version` → 409; with an unknown/reused nonce → 400.
- **No file appears anywhere under `data/` during enrolment.** Snapshot the tree before and after.
- Enrolment with 5 identical frames → `422 static_frames`.
- Enrolment with 2 faces in a frame → that frame rejected `multiple_faces`.
- `DELETE /face` empties `face_templates` for that user, and **the exact ciphertext blob no longer appears in the raw `.db` bytes** after `VACUUM`. (`PRAGMA secure_delete` is 0 by default — measured — so without this the ciphertext survives in freelist pages and an erasure request is not an erasure.)
- `DELETE FROM users WHERE user_id = ?` empties all four face tables (verified: FK enforcement really is on through `store.connect()`).
- `face_consent` UPDATE and DELETE are both blocked by trigger; `DELETE FROM users` still cascades cleanly.
- A face match never creates a `users` row, never issues a cookie, never changes `sessions.user_id`.
- `POST /analyze?identify=true` where the match is another user: the session's `user_id` is the **uploader**, a `session_claims` row exists, and the response body contains **no** `user_id`, name, or score for the match.
- `POST /me/claims/{id}` called by anyone other than `candidate_user_id` → 404.

### D.5 `tests/test_media.py` additions

- `media.store(path)` with one positional argument still works (the 5 existing tests at `tests/test_pipeline.py:529-580` must not be touched).
- `GET /media/{token}` for an entry with `user_id is None` serves to an anonymous caller.
- `GET /media/{token}` for an owned entry returns **404** (not 403) to a different user.

### D.6 Source‑level guard tests (`tests/test_boundaries.py`)

Grep‑based, deliberately crude, because each guards a property that a future PR would otherwise silently reverse:

1. `get_or_create_user` appears in **no** file under `app/`.
2. No route decorator in `app/` registers a path containing `identify` or `whois`.
3. `SELECT … FROM face_templates` appears only inside `store.candidate_templates`.
4. `cv2.FaceRecognizerSF` and `.match(` appear only in `src/runway_ai/face.py`.
5. `src/runway_ai/face/` does not exist as a directory.
6. No `passlib` import anywhere.

---

## 5. Implementation order

1. `.gitignore` + back up `data/runway.db`.
2. `requirements.txt`, `scripts/check_env.py`.
3. `src/runway_ai/db/migrate.py` + `store.connect()` hook + `tests/test_migrate.py`. **Run against a copy of the live DB before touching anything else.**
4. `store.py` account/session/rate‑limit functions + `get_or_create_user` fix.
5. `src/runway_ai/auth.py` + `tests/test_auth.py` (backend only, no client yet).
6. `app/api.py`: dependencies, `/auth/*`, `/me/*`, delete `/users/{name}/*`, strip `user` from the four persistence sites, `LinkRequest`, SSE sanitisation, CORS.
7. `src/runway_ai/cli.py`: `claim`, `passwd`.
8. Frontend auth: `auth.jsx`, `Icons.jsx`, `IdentityChip`, `AccountSheet`, `AuthForms`, `api.js` wrapper, `App.jsx`, `Progress.jsx`, `Analyse.jsx` prop removal, `SkeletonPlayer.jsx` `crossOrigin`. **Ship and use this before starting face work** — the face feature's every authorization control is stated against an auth system, and until it exists `POST /face/enrol` is "enrol a face against any account name you can type".
9. `face_crypto.py`, `face.py` amendments (Changes 1–9), `pose/extract.py` stride meta, `analyze.py` `_seq_meta`.
10. `app/face_routes.py`, `media.py` owner param, `_publish_media` signature, `_identify_walk`.
11. `FaceEnrol`, `FaceManager`, `Attribution`, `Claims`, `styles.css`, `docs/face-policy-v1.md`.
12. `tests/test_face.py`, `tests/test_face_privacy.py`, `tests/test_media.py`, `tests/test_boundaries.py`.

---

## 6. Must be VERIFIED EMPIRICALLY before shipping — not assumed

### 6.1 The enrolment quality gate — `ENROL_DET_SCORE = 0.90`, `ENROL_MIN_FACE_PX = 80`

**Status: PROVISIONAL, never measured.** These are plausible for a webcam selfie at arm's length and disastrous if wrong — a gate that rejects real captures makes enrolment fail after the user has already consented.

**How to verify:** capture 20 frames from the actual dev machine's webcam through the real `FaceEnrol` flow, at typical laptop distance and typical indoor lighting. Record `det_score` and `min(w, h)` per frame. Set `ENROL_DET_SCORE` at the measured p5 rounded down, and `ENROL_MIN_FACE_PX` at the measured p5 width rounded down to the nearest 10, then re‑run and confirm ≥ 5 of 6 frames survive. Paste the table into the module docstring.

### 6.2 The static‑frame liveness guard — `ENROL_MAX_SELF_COS = 0.995`

**Status: PROVISIONAL, never measured.** Held-up photos and screens produce near‑identical consecutive frames; live faces do not.

**How to verify:** capture two 6‑frame sets — one live, one of a phone screen displaying a photo. Compute (a) the median pairwise embedding cosine within each set and (b) the mean absolute pixel difference between consecutive aligned crops. Set the threshold at the midpoint of the two distributions, and **state plainly** that this is a heuristic, not presentation‑attack detection: YuNet and SFace have no PAD capability, and the enrolment quality gate actively *rewards* the sharp, frontal, well‑lit profile a spoof produces. This guard plus camera‑only capture plus a one‑enrolment‑per‑account cap plus the advisory‑only decision ladder are collectively the anti‑spoof story — say so, rather than implying anything is verified.

### 6.3 The impostor distribution — **the whole reason no accuracy figure may be published**

**Status: NOT MEASURED. Cannot be measured with the footage available.**

**How to verify:** obtain walk videos of **at least 5 and ideally 10 different people**, run `scripts/calibrate_face.py` with one video per person, and record `report_impostor()`'s output — max, p99, p99.9 of cross‑person cosine. Only then may `ACCEPT_COSINE` or `MIN_MARGIN` be changed, and only in the *raising* direction until N is large. Until that run exists:
- No FAR, FRR, accuracy or "confidence %" appears anywhere.
- `MIN_MARGIN = 0.10` stays labelled "a conservative default, NOT a measured one" in the source.
- The 1:N gallery is capped: if `len(candidate_templates()) > 10`, log a warning that the operating point has not been validated at that gallery size (system FAR compounds as `1-(1-p)^N`).

### 6.4 The database's location

**Status: the DB is at `…\OneDrive\Desktop\albins\data\runway.db` and is continuously replicated to Microsoft.**

Once it contains face templates, `DELETE /face` + `secure_delete` + `VACUUM` scrubs the **local** file only. OneDrive retains prior versions in version history and the cloud recycle bin; `VACUUM` rewrites the whole file and forces one more retained version. **Verify before claiming erasure:** set `RUNWAY_DB_PATH=%LOCALAPPDATA%\runway_ai\runway.db`, move the existing file, confirm the app and CLI both still work, and confirm `data/` contains no `.db`. Until that move is done, `docs/face-policy-v1.md` must say that deletion is local‑only and that the database is replicated to a third party. App‑layer encryption is what makes the interim state tolerable — only ciphertext syncs — provided the keyfile stays in `%LOCALAPPDATA%`, which is verified to be outside the OneDrive tree.

### 6.5 Cross‑session genuine similarity

**Status: NOT MEASURED.** Everything in §1.3 is within one clip. Before enabling `identify=true` by default, enrol one person from the webcam flow and probe with a walk clip filmed on a different day. If the pooled score is below ~0.6, the enrolment path is wrong for this use and should be replaced by "enrol from a walk you already analysed" — which is only possible after `media.store()` is extended to uploads, and is **out of scope for v1** (§7).

### 6.6 Load and blocking behaviour

- Confirm every `/auth/*` handler is declared `def`, not `async def` (a grep test is sufficient).
- Confirm 40 concurrent logins do not exhaust memory: at `memory_cost=19456` that is ~760 MiB worst case versus ~2.5 GiB at the library defaults.

---

## 7. Cut list, with the reason each was cut

| Cut | Reason |
|---|---|
| `POST /face/identify` (and any standalone "who is this") | A membership‑inference and template‑reconstruction oracle. Continuous scores plus arbitrary probes allow hill‑climbing toward a stored template, which — unlike a password — can never be reissued. |
| Returning candidate names/ids/scores to an uploader | Discloses that a specific person enrolled, to someone with no relationship to them. |
| Uploaded‑photo enrolment; `<input type="file">` fallback | Enrols non‑consenting third parties. The one thing the brief forbids. |
| `thumb BLOB`, `GET /faces/{id}/thumb` | Turns a template store into a face‑photo database, and contradicts `face.py`'s own documented position. |
| Per‑face `DELETE /faces/{face_id}` | With one centroid per user there is nothing to delete individually, and per‑item deletion silently degrades a template below `MIN_ENROL_FRAMES` with no signal. |
| `POST /auth/claim-legacy` | Account‑takeover primitive: the only credential is possession of a name string, and `cli.py` mints claimable rows forever. Replaced by an operator CLI. |
| `POST /sessions/adopt` + `result_token` | A second TTL store, a server‑side result cache and a fourth save path, to rescue a walk analysed in the 90 seconds before signing up. "Sign in, then analyse" is 30 seconds. |
| `POST /face/scope`, group scope, `group_id`, `<CandidatePicker>` | No relation table exists, no user has one, and with names withheld the picker has one entry. |
| `face_match_log` with `tau_accept`/`tau_margin`/`confirmed_user_id`, `POST /face/confirm` | Machinery for a calibration loop that "refuses to lower a threshold below 10 users" against a 4‑user DB. It also outlives an erasure request as a permanent per‑video linkage record. Scores go to `session_claims` (visible only to the candidate) and the log. |
| `scripts/validate_face_thresholds.py` | Duplicates `scripts/calibrate_face.py` and cannot close its own loop at this scale. |
| `key_id` rotation flow, `purge_dormant_templates`, `purge_match_log`, `GET /account/export` | A `key_id` column exists for future use; the rotation *flow*, dormancy sweeps and export are not v1 surface. |
| `X-Runway-Auth` CSRF header, double‑submit token | `SameSite=Lax` strips the cookie from a cross‑site POST including multipart, so the request arrives unauthenticated. **Reinstate the moment the cookie must become `SameSite=None`.** |
| Per‑IP rate limiting, `X-Forwarded-For`, `TRUSTED_PROXY` | The Vite proxy does not set `xfwd`, so it is one global bucket that throttles every user together. |
| `PRAGMA journal_mode = WAL` | Sidecar files in a OneDrive‑synced folder; also cannot run inside the migration transaction. `busy_timeout` is already 5000 (measured). |
| `POST /auth/logout-all`, `auth_sessions.user_agent`/`.ip` | No session‑list UI to display them; `change-password` already revokes other sessions. |
| A `RUNWAY_ALLOW_LEGACY_CLAIM` flag | With no HTTP claim endpoint there is nothing to gate. |

---

## 8. Statements that would be dishonest, and the honest form

| Do not write | Write instead |
|---|---|
| "99 % accurate face matching" / any FAR or FRR | "The false‑accept rate has not been measured on this system: only one subject's footage was available. The accept bar is SFace's published operating point, not one fitted here." |
| "Measured: worst‑case same‑person score 0.750" (from `face.py`'s current docstring) | Keep it, but qualify: that figure and mine (§1.3) both come from **one** clip sampled every 6 frames — near‑duplicate frames, one camera, one session. Cross‑session performance is unmeasured. |
| "cosine 0.50 gives a false‑accept rate of 1.3e‑7" | Never. That is a Gaussian tail fit to a single impostor identity, extrapolated far past the p99.9 = 0.3577 it can observe. |
| "Deleting your face data erases it" | "Deleting removes it from the local database and the file is rewritten. If the database is inside a synced folder, earlier copies may persist in that service's version history." |
| "Encrypted, so your biometric is safe" | "Encrypted at rest, which protects the database file if it is copied, committed, or synced. It does not protect against someone with full access to this computer, who can read the key too." |
| "Your face data is anonymous" | "A face template is biometric data. It is not a photograph and not reversible into one exactly, but published research reconstructs identity‑preserving approximations from embeddings, and the recogniser model ships in this repo. Unlike a password, it cannot be changed if it leaks." |
| "The system refuses to identify people who haven't enrolled" as an absolute | True as stated — the candidate set is a consent‑joined query — but pair it with the honest caveat that someone could enrol a face that is not their own, which is why enrolment is camera‑only, nonce‑bound, capped at one per account, and logged. |
| "store.py has no busy_timeout, so concurrent writes fail immediately" | False. Python's `sqlite3` default `timeout=5.0` is already in effect (measured: `PRAGMA busy_timeout` = 5000 on the live DB). |
| "Nothing request‑scoped is reachable inside the SSE worker closure" | False. `app/api.py:279-289` already closes over `user` and calls into `store` on the worker thread. The real constraint: a `Request` object or a `Depends` lookup must not be touched off the event loop, so resolve `user_id` to an `int | None` in the handler first. |
| "Adding localhost:5180 to CORS fixes the dev loop" | Dev requests go through the Vite `/api` proxy and are same‑origin, so CORS is never consulted. The addition matters only for a direct‑to‑8000 client. |