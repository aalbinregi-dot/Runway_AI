"""Accounts, password hashing and face enrolment storage.

Password hashing
----------------
**Argon2id**, via ``argon2-cffi``. The alternatives were both rejected on
evidence rather than preference:

* ``passlib`` is broken against the installed bcrypt 5.x - it reads
  ``bcrypt.__about__.__version__``, which bcrypt 5 removed, and then fails
  outright. Verified on this machine.
* Raw ``bcrypt`` works, but silently caps passwords at 72 bytes (5.x raises
  instead of truncating, which is better, but the limit remains) and measured
  ~206 ms per hash here against Argon2id's ~59 ms at OWASP-recommended
  parameters.

Argon2id is also the current OWASP first choice, being memory-hard and so far
harder to attack with GPUs than bcrypt.

Legacy accounts
---------------
The database predates authentication: it already holds users created by name
alone, with walks attached. Those rows are preserved with a NULL password and
are **not claimable by signing up under the same name** - that would let anyone
who guessed "Faris" inherit that person's history. Linking a legacy row to a
real account is a deliberate local operation (``runway_ai claim``), because
running the CLI already implies control of the machine and the database file.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import connect

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_templates (
    user_id      INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    embedding    BLOB    NOT NULL,     -- float32[128], never an image
    n_frames     INTEGER NOT NULL,
    consented_at TEXT    NOT NULL,     -- explicit opt-in, recorded for audit
    updated_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash  TEXT PRIMARY KEY,      -- sha256 of the bearer token
    user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at  TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    user_agent  TEXT
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);

CREATE TABLE IF NOT EXISTS login_attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,          -- the email that was tried
    at         TEXT NOT NULL,
    ok         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_attempts ON login_attempts(identifier, at);
"""

#: Columns added to the pre-existing users table.
USER_COLUMNS = {
    "password_hash": "TEXT",          # NULL = legacy row, cannot log in
    "email_lower": "TEXT",            # lowercased, uniqueness is enforced here
    "is_legacy": "INTEGER NOT NULL DEFAULT 0",
}

#: Bumped whenever this module changes the schema; migrate() is a no-op once
#: the database already reports it.
SCHEMA_VERSION = 1

SESSION_TTL_DAYS = 30
MAX_FAILED_LOGINS = 8
LOCKOUT_MINUTES = 15


class AuthError(Exception):
    """Raised for any authentication failure the caller may show a user."""


@dataclass
class User:
    id: int
    name: str
    email: str | None
    is_legacy: bool
    has_face: bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "email": self.email,
                "is_legacy": self.is_legacy, "has_face": self.has_face}


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #
def _statements(script: str) -> list[str]:
    """Split a SQL script into executable statements.

    Comments are stripped first. Splitting the raw text on ";" cuts any comment
    that happens to contain one straight down the middle, and the trailing half
    then reaches SQLite as a statement.
    """
    body = "\n".join(
        line for line in script.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def migrate(db_path: Path | str | None = None) -> list[str]:
    """Add auth columns and tables to an existing database, non-destructively.

    SQLite cannot add a column with a UNIQUE constraint, and the live database
    already contains rows, so uniqueness on email is enforced by a partial
    UNIQUE INDEX added afterwards rather than in the column definition. The
    index is partial so the many legacy rows with NULL emails do not collide.
    """
    from .store import DB_PATH, SCHEMA

    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A dedicated connection with manual transaction control. store.connect()
    # cannot be used here: it runs executescript(SCHEMA), and executescript
    # issues an implicit COMMIT, which would silently end the BEGIN IMMEDIATE
    # this relies on.
    con = sqlite3.connect(path, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)                 # base tables must exist first
        con.execute("BEGIN IMMEDIATE")            # take the write lock, then...
        if con.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            con.execute("COMMIT")                 # ...re-check under it, so two
            return []                             # starters cannot both apply

        applied = []
        existing = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        for column, decl in USER_COLUMNS.items():
            if column not in existing:
                con.execute(f"ALTER TABLE users ADD COLUMN {column} {decl}")
                applied.append(f"users.{column}")

        # Everything predating this migration is a legacy row.
        con.execute("UPDATE users SET is_legacy = 1 "
                    "WHERE password_hash IS NULL AND is_legacy = 0")

        for statement in _statements(AUTH_SCHEMA):
            con.execute(statement)
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower "
            "ON users(email_lower) WHERE email_lower IS NOT NULL")
        applied.append("auth tables")

        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        con.execute("COMMIT")
        return applied
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def _hasher():
    from argon2 import PasswordHasher

    # OWASP's second recommended profile: 19 MiB, 2 iterations, 1 lane.
    return PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)


def hash_password(password: str) -> str:
    return _hasher().hash(password)


def verify_password(stored: str | None, password: str) -> bool:
    """Constant-ish time check that tolerates a missing hash.

    A legacy row has no hash. Returning early would make "account exists but is
    legacy" measurably faster than "wrong password", so a dummy verification is
    run instead to keep the timing comparable.
    """
    from argon2.exceptions import VerificationError, VerifyMismatchError

    ph = _hasher()
    if not stored:
        try:
            ph.verify(_DUMMY_HASH, password)
        except (VerifyMismatchError, VerificationError, Exception):
            pass
        return False
    try:
        ph.verify(stored, password)
        return True
    except (VerifyMismatchError, VerificationError):
        return False


#: Pre-computed so the dummy verification costs the same as a real one.
_DUMMY_HASH = (
    "$argon2id$v=19$m=19456,t=2,p=1$"
    "c29tZXNhbHRzb21lc2FsdA$8i7GRwZDvJHNPKNRLBFmMFVLpZlxHFmA7VgQ0ATCkVA"
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 10


def validate_signup(name: str, email: str, password: str) -> None:
    if not name.strip():
        raise AuthError("Please enter your name.")
    if not EMAIL_RE.match(email.strip()):
        raise AuthError("That does not look like an email address.")
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Use at least {MIN_PASSWORD} characters.")
    # Argon2 has no length cap, but an unbounded input is still a cheap way to
    # make the server do a lot of hashing work.
    if len(password.encode()) > 1024:
        raise AuthError("That password is too long.")


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
def _now():
    from .store import _now as now
    return now()


def create_account(name: str, email: str, password: str,
                   db_path=None) -> User:
    validate_signup(name, email, password)
    email = email.strip()
    with connect(db_path) as con:
        try:
            cur = con.execute(
                "INSERT INTO users (name, email, email_lower, password_hash, "
                "experience, created_at, is_legacy) VALUES (?,?,?,?,?,?,0)",
                (name.strip(), email, email.lower(), hash_password(password),
                 "beginner", _now()))
        except sqlite3.IntegrityError as exc:
            raise AuthError("An account with that email already exists.") from exc
        return User(cur.lastrowid, name.strip(), email, is_legacy=False)


def _locked_out(con, identifier: str) -> bool:
    # _now() writes ISO-8601 with a "+00:00" offset; datetime('now') returns a
    # space-separated string with none. Comparing them as text compares 'T'
    # against ' ' at index 10 and is meaningless, so both sides are normalised
    # to a julianday number instead.
    row = con.execute(
        "SELECT COUNT(*) FROM login_attempts "
        "WHERE identifier = ? AND ok = 0 "
        "AND julianday(replace(substr(at, 1, 19), 'T', ' ')) "
        "    > julianday('now', ?)",
        (identifier, f"-{LOCKOUT_MINUTES} minutes")).fetchone()
    return bool(row and row[0] >= MAX_FAILED_LOGINS)


def authenticate(email: str, password: str, db_path=None) -> User:
    """Verify credentials. The error text never says which half was wrong."""
    identifier = email.strip().lower()
    generic = AuthError("Email or password is incorrect.")

    with connect(db_path) as con:
        locked = _locked_out(con, identifier)
        row = None if locked else con.execute(
            "SELECT user_id, name, email, password_hash, is_legacy FROM users "
            "WHERE email_lower = ?", (identifier,)).fetchone()

    if locked:
        raise AuthError(
            f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes.")

    # Always run a verification, even for an unknown email, so the response time
    # does not reveal whether the account exists.
    ok = verify_password(row["password_hash"] if row else None, password)

    # Recorded in its own committed block. store.connect() commits when the
    # block exits normally, so raising inside it - as this function must on a
    # failure - discarded the very row the lockout counts, and the lockout
    # silently never engaged.
    with connect(db_path) as con:
        con.execute("INSERT INTO login_attempts (identifier, at, ok) VALUES (?,?,?)",
                    (identifier, _now(), 1 if ok else 0))

    if not ok or row is None:
        raise generic
    return User(row["user_id"], row["name"], row["email"], bool(row["is_legacy"]))


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def _hash_token(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()


def issue_session(user_id: int, user_agent: str | None = None, db_path=None) -> str:
    """Create a bearer token and return it. Only its hash is stored.

    A stolen database therefore does not hand over live sessions, the same
    reason passwords are not stored in the clear.
    """
    token = secrets.token_urlsafe(32)
    with connect(db_path) as con:
        con.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, created_at, "
            "expires_at, user_agent) VALUES (?,?,?,datetime('now', ?),?)",
            (_hash_token(token), user_id, _now(),
             f"+{SESSION_TTL_DAYS} days", (user_agent or "")[:200]))
    return token


def resolve_session(token: str | None, db_path=None) -> User | None:
    if not token:
        return None
    with connect(db_path) as con:
        row = con.execute(
            "SELECT u.user_id, u.name, u.email, u.is_legacy, "
            "       (SELECT 1 FROM face_templates f WHERE f.user_id = u.user_id) "
            "         AS has_face "
            "FROM auth_sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > datetime('now')",
            (_hash_token(token),)).fetchone()
    if row is None:
        return None
    return User(row["user_id"], row["name"], row["email"],
                bool(row["is_legacy"]), bool(row["has_face"]))


def revoke_session(token: str, db_path=None) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM auth_sessions WHERE token_hash = ?",
                    (_hash_token(token),))


def revoke_all_sessions(user_id: int, db_path=None) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))


# --------------------------------------------------------------------------- #
# Face enrolment
# --------------------------------------------------------------------------- #
def save_face_template(user_id: int, embedding: np.ndarray, n_frames: int,
                       db_path=None) -> None:
    """Store one face template. Writing it *is* the record of consent."""
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.size != 128:
        raise ValueError(f"expected a 128-d embedding, got {vec.size}")
    now = _now()
    with connect(db_path) as con:
        con.execute(
            "INSERT INTO face_templates (user_id, embedding, n_frames, "
            "consented_at, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET embedding=excluded.embedding, "
            "n_frames=excluded.n_frames, updated_at=excluded.updated_at",
            (user_id, vec.tobytes(), int(n_frames), now, now))


def delete_face_template(user_id: int, db_path=None) -> bool:
    """Remove a user's biometric data, leaving the account intact."""
    with connect(db_path) as con:
        cur = con.execute("DELETE FROM face_templates WHERE user_id = ?", (user_id,))
        return cur.rowcount > 0


def load_face_templates(db_path=None) -> dict[int, np.ndarray]:
    """Every enrolled template. Only enrolled users can ever be matched."""
    with connect(db_path) as con:
        rows = con.execute("SELECT user_id, embedding FROM face_templates").fetchall()
    return {r["user_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
            for r in rows}


def all_face_templates(db_path=None) -> dict[int, dict]:
    """Enrolled templates with the display name each belongs to.

    Group analysis needs the name to label a walker on screen;
    :func:`load_face_templates` deliberately returns only vectors, and joining
    in the caller would mean a second query per person.
    """
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT f.user_id, f.embedding, u.name "
            "FROM face_templates f JOIN users u ON u.user_id = f.user_id"
        ).fetchall()
    return {r["user_id"]: {"template": np.frombuffer(r["embedding"], dtype=np.float32),
                           "name": r["name"]}
            for r in rows}


def has_face_template(user_id: int, db_path=None) -> bool:
    with connect(db_path) as con:
        return con.execute("SELECT 1 FROM face_templates WHERE user_id = ?",
                           (user_id,)).fetchone() is not None


# --------------------------------------------------------------------------- #
# Legacy linking
# --------------------------------------------------------------------------- #
def list_accounts(db_path=None) -> list[dict]:
    """Every account, for the operator CLI. Never exposed over HTTP."""
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT u.user_id, u.name, u.email, u.is_legacy, "
            "  (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.user_id) AS walks, "
            "  (SELECT 1 FROM face_templates f WHERE f.user_id = u.user_id) AS has_face "
            "FROM users u ORDER BY u.user_id").fetchall()
    return [dict(r) for r in rows]


def claim_legacy(legacy_user_id: int, into_user_id: int, db_path=None) -> int:
    """Move a pre-auth account's walks onto a real account.

    Intentionally not exposed over HTTP. Anyone able to run the CLI already has
    the database file; anyone on the network does not, and an endpoint would be
    checking nothing but possession of the string "Faris".

    Keyed on user_id rather than name because ``users.name`` carries no UNIQUE
    constraint and ``get_or_create_user`` keeps minting rows, so a name does not
    identify an account. List candidates with ``runway_ai accounts``.
    """
    with connect(db_path) as con:
        row = con.execute(
            "SELECT user_id FROM users WHERE user_id = ? AND is_legacy = 1",
            (legacy_user_id,)).fetchone()
        if row is None:
            raise AuthError(f"No legacy account with id {legacy_user_id}.")
        legacy_id = row["user_id"]
        if legacy_id == into_user_id:
            raise AuthError("That is the same account.")
        moved = con.execute("UPDATE sessions SET user_id = ? WHERE user_id = ?",
                            (into_user_id, legacy_id)).rowcount
        con.execute("DELETE FROM users WHERE user_id = ?", (legacy_id,))
    return moved
