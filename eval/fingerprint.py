"""Content fingerprint of a measurement.

Standard library only, deliberately: the pre-commit guard imports this and must
run under a bare interpreter with no third-party packages installed.

What it covers, and why
-----------------------
The fingerprint originally hashed only the code under test (``env/`` and
``agent/``). That was not sufficient. The runner's *seeding policy* is part of
the measurement: switching from one RNG stream spanning all episodes to a seed
derived per episode index changed the published Random baseline from
26% / 4% / -18.08 to 22% / 3% / -21.69 with no change whatsoever to ``env/`` or
``agent/``. A fingerprint that excludes the runner therefore certifies an
artifact it cannot vouch for - two different measurements of identical code
would carry the same fingerprint and be indistinguishable.

So ``eval/evaluate.py`` is hashed too. It is the only module that participates in
seeding or aggregation; the agents and environment it drives are already covered,
and everything else it imports is either the standard library or numpy (used for
means only). This module is excluded: it computes the fingerprint but takes no
part in producing the numbers.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Glob patterns, relative to the repo root, in the order they are hashed.
MEASURED_PATHS = ("env/*.py", "agent/*.py", "eval/evaluate.py")

# Git paths whose history describes the same set, for the artifact's git context.
MEASURED_GIT_PATHS = ("env", "agent", "eval/evaluate.py")


def measured_files(root: Path = ROOT) -> list:
    """Every file the fingerprint covers, in a stable order."""
    found = []
    for pattern in MEASURED_PATHS:
        found.extend(root.glob(pattern))
    return sorted(set(found), key=lambda f: f.relative_to(root).as_posix())


def measurement_fingerprint(root: Path = ROOT) -> str:
    """Hash of the code under test *and* the runner that measures it.

    Stable across rebase/amend and recomputable by anyone, so an artifact
    identifies the exact measurement it came from rather than pointing at a
    commit that may no longer exist.
    """
    h = hashlib.sha256()
    for f in measured_files(root):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()[:16]
