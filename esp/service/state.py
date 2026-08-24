"""Population and budget that survive between wake-ups.

The batch version of this project needed a dozen candidates in one sitting and
died when the provider's daily cap stopped it at three. A service does not have
that problem, because the cap is not an obstacle to a service -- it is its
rhythm. Spend what today allows, write down where you got to, stop, and carry on
tomorrow. A population accumulates over weeks that could never be bought in an
afternoon.

Everything here is therefore built around being interrupted. State is written
after every single evaluation, not at the end of a generation, because the run
is expected to be cut short by an exhausted quota rather than to finish.
"""

from __future__ import annotations

import json
import os
import time

try:
    import fcntl
except ImportError as missing:      # pragma: no cover - POSIX-only deployment
    # The service ships as a Linux container and CI is ubuntu. Refusing loudly
    # beats degrading to the check-then-write this replaced, which looked like
    # mutual exclusion and was not.
    raise RuntimeError(
        "esp.service.state needs POSIX file locking (fcntl) to make the "
        "optimiser lease mutually exclusive") from missing
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("ESP_STATE", "state"))
LEASE_SECONDS = float(os.environ.get("ESP_LEASE_SECONDS", "3600"))

# How long a model stays retired after it reports its daily budget spent.
#
# An elapsed duration rather than "until tomorrow". Keying the record to the
# UTC calendar day assumes the provider resets on the same boundary, and Google
# resets its free tier at Pacific midnight -- eight hours out. A model can
# therefore answer normally while state still records it exhausted, leaving the
# service idle on budget it has already been given back.
#
# An hour is the right shape of answer because the two mistakes are not
# symmetric. Retrying too early costs one request that returns 429. Retrying too
# late costs every evaluation that could have run in the meantime.
RETRY_EXHAUSTED_AFTER = float(os.environ.get("ESP_RETRY_EXHAUSTED_AFTER", "3600"))


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _today() -> str:
    """A stable bucket for the day's records. UTC, because the service may move
    host and a local-time boundary would double-spend or skip a day when it
    does.

    Note this is a *filing* boundary, not the provider's reset. Nothing may
    assume a model recovers when this rolls over -- see RETRY_EXHAUSTED_AFTER.

    This said UTC and called `date.today()`, which is local. On a host in IST
    -- where this was developed -- the key rolled over five and a half hours
    before the UTC day did, and every exhaustion record filed under the old key
    became unreachable: `is_exhausted` looked up today and found nothing, so the
    service treated spent models as fresh and went back to collecting 429s. Same
    family as the bug that had it idling on budget it already had back, in the
    other direction, and hidden the same way -- by a comment stating the rule
    correctly above code that broke it.
    """
    return _utcnow().date().isoformat()


@dataclass
class Evaluated:
    """One candidate the service has paid to run."""

    genome_hash: str
    origin: str
    fitness: float
    accuracy: float
    tokens: int
    agents: int
    depth: int
    generation: int
    measured_at: str
    model: str
    # The candidate itself, as Genome.canonical().
    #
    # Recording only the hash and the score was a real limitation: an evolved
    # winner could be measured and then never served again, because a hash
    # cannot be turned back into a network. serve_champion could resurrect
    # seeds and nothing else, since seeds are the one thing rebuildable from
    # source. Optional so that state written before this change still loads --
    # those records keep the old behaviour and say so.
    genome: dict | None = None


@dataclass
class ServiceState:
    """Everything the optimiser needs to pick up where it left off."""

    generation: int = 0
    evaluated: list[Evaluated] = field(default_factory=list)
    # Spend is tracked per day and per model, because that is how the provider
    # meters it. A single global counter would let one exhausted model stall a
    # service that still has budget elsewhere.
    spend: dict[str, dict[str, int]] = field(default_factory=dict)
    # day -> {model: when it reported itself spent}. The timestamp is the point:
    # a bare list could only ever mean "not before tomorrow", and tomorrow is
    # not when the provider's budget returns.
    exhausted: dict[str, dict[str, str]] = field(default_factory=dict)
    last_wake: str = ""
    last_spoke: str = ""
    wakes: int = 0

    # ----------------------------------------------------------------- budget

    def spent_today(self, model: str) -> int:
        return self.spend.get(_today(), {}).get(model, 0)

    def record_spend(self, model: str, requests: int) -> None:
        self.spend.setdefault(_today(), {}).setdefault(model, 0)
        self.spend[_today()][model] += requests

    def mark_exhausted(self, model: str) -> None:
        self.exhausted.setdefault(_today(), {})[model] = _utcnow().isoformat()

    def is_exhausted(self, model: str) -> bool:
        """Whether this model is still presumed spent.

        Expires by elapsed time rather than by calendar day. A model retired
        permanently would leave the service dead after its first bad afternoon;
        a model retired until UTC midnight leaves it idle for hours after the
        provider has already reset. Both are the same mistake in different
        sizes -- presuming we know when somebody else's budget comes back.
        """
        when = self.exhausted.get(_today(), {}).get(model)
        if when is None:
            return False
        try:
            marked = datetime.fromisoformat(when)
        except ValueError:
            return False        # unreadable record is not a reason to stay idle
        return (_utcnow() - marked).total_seconds() < RETRY_EXHAUSTED_AFTER

    def exhausted_now(self) -> list[str]:
        """The models currently presumed spent, for a wake to report."""
        return sorted(m for m in self.exhausted.get(_today(), {})
                      if self.is_exhausted(m))

    def usable(self, ladder: list[str]) -> list[str]:
        return [model for model in ladder if not self.is_exhausted(model)]

    # ------------------------------------------------------------- population

    def best(self) -> Evaluated | None:
        return max(self.evaluated, key=lambda e: e.fitness, default=None)

    def seen(self) -> set[str]:
        return {e.genome_hash for e in self.evaluated}

    def add(self, record: Evaluated) -> None:
        self.evaluated.append(record)

    # ------------------------------------------------------------------- I/O

    @classmethod
    def load(cls, directory: Path | None = None) -> ServiceState:
        path = (directory or STATE_DIR) / "state.json"
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["evaluated"] = [Evaluated(**e) for e in raw.get("evaluated", [])]
        # Older state recorded exhaustion as a bare list of model names, which
        # carried no time and so could only mean "not before tomorrow". Read it
        # as already expired: that makes the service retry once, which is the
        # correct behaviour and costs at most one 429.
        migrated: dict[str, dict[str, str]] = {}
        for day, entry in (raw.get("exhausted") or {}).items():
            if isinstance(entry, list):
                migrated[day] = {model: "1970-01-01T00:00:00+00:00"
                                 for model in entry}
            else:
                migrated[day] = entry
        raw["exhausted"] = migrated
        return cls(**raw)

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or STATE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        payload = asdict(self)
        # Written to a sibling and moved into place: a service that is killed
        # mid-write must not come back to a truncated population.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path


class Lease:
    """One optimiser at a time.

    A scheduler that fires every fifteen minutes will start a second wake while
    the first is still evaluating -- a candidate takes eight minutes and can take
    far longer behind a rate limiter. Two optimisers would spend the same daily
    budget twice and write over each other's population.
    """

    def __init__(self, directory: Path | None = None,
                 seconds: float = LEASE_SECONDS) -> None:
        self.path = (directory or STATE_DIR) / "lease.json"
        self.seconds = seconds

    def acquire(self, owner: str) -> bool:
        """Take the lease, or report that somebody else holds it.

        The read, the expiry check and the write are one critical section under
        an exclusive file lock. As three separate steps the gap between them is
        wide enough to drive the whole population through: concurrent wakes
        against one directory can each come away believing they hold the lease,
        spending the same daily budget several times over and overwriting each
        other's population -- the precise outcome this class exists to prevent.

        POSIX file locking, so this holds between processes on a host and not
        merely between threads. It does not coordinate across hosts; nothing in
        this deployment runs the optimiser on two.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        guard = self.path.with_name(self.path.name + ".guard")

        handle = os.open(guard, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            now = time.time()
            if self.path.exists():
                try:
                    held = json.loads(self.path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    held = {}                   # corrupt lease is a free lease
                if now - float(held.get("taken_at", 0)) < self.seconds:
                    return False
            self.path.write_text(
                json.dumps({"owner": owner, "taken_at": now,
                            "taken_at_iso": _utcnow().isoformat()}),
                encoding="utf-8")
            return True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def holder(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
