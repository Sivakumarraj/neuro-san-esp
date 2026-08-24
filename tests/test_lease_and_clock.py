"""Two properties the unattended service is built on, neither of which held.

The lease and the day key are what make it safe to leave this running. The
lease is the only thing stopping two wakes spending the same daily budget
twice; the day key is what the budget ledger and the exhaustion records are
filed under. Both were documented correctly and implemented wrongly, and
neither failure shows up in a single-threaded test on the machine that wrote it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from esp.service import state as state_module
from esp.service.state import Lease, ServiceState

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ the lease

def race(directory: Path, racers: int = 8) -> list[int]:
    """Every racer released on the same instant, which is the case that broke."""
    winners: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(racers)

    def go(n: int) -> None:
        lease = Lease(directory=directory)
        barrier.wait()
        if lease.acquire(f"wake-{n}"):
            with lock:
                winners.append(n)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return winners


def test_only_one_simultaneous_wake_can_hold_the_lease(tmp_path):
    """It was a check-then-write with no atomicity: eight wakes released
    together, and between two and seven of them came away holding it. That is
    the same budget spent several times and one population overwriting
    another -- the exact outcome the class exists to prevent.

    Repeated, because a race that fails one time in six passes once by luck.
    """
    for attempt in range(6):
        directory = tmp_path / f"attempt-{attempt}"
        directory.mkdir()
        winners = race(directory)
        assert len(winners) == 1, (
            f"{len(winners)} of 8 wakes acquired the same lease: {sorted(winners)}")


def test_the_lease_is_exclusive_between_processes_not_only_threads(tmp_path):
    """A scheduler firing overlapping wakes need not do it in one process, so
    thread-level exclusion would not be enough."""
    script = tmp_path / "race.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from esp.service.state import Lease\n"
        "print('WON' if Lease(directory=Path(sys.argv[1])).acquire(sys.argv[2]) "
        "else 'lost')\n", encoding="utf-8")

    directory = tmp_path / "shared"
    directory.mkdir()
    running = [
        subprocess.Popen([sys.executable, str(script), str(directory), f"p{i}"],
                         cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
        for i in range(8)
    ]
    outcomes = [p.communicate()[0].strip() for p in running]
    assert outcomes.count("WON") == 1, outcomes


def test_an_expired_lease_can_be_taken_by_exactly_one_wake(tmp_path):
    """Expiry is what stops a crashed holder wedging the service forever. The
    steal path is a race too, and has to be as exclusive as the fresh one."""
    # A holder that died an hour ago: written directly, because `seconds`
    # governs the acquirer's own expiry check, not the age of what it writes.
    (tmp_path / "lease.json").write_text(
        json.dumps({"owner": "dead-holder", "taken_at": time.time() - 7200,
                    "taken_at_iso": "1970-01-01T00:00:00+00:00"}),
        encoding="utf-8")

    winners: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def go(n: int) -> None:
        lease = Lease(directory=tmp_path, seconds=3600.0)
        barrier.wait()
        if lease.acquire(f"wake-{n}"):
            with lock:
                winners.append(n)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1, sorted(winners)


def test_a_held_lease_is_refused_and_a_released_one_is_free(tmp_path):
    first = Lease(directory=tmp_path)
    assert first.acquire("wake-1") is True
    assert Lease(directory=tmp_path).acquire("wake-2") is False
    first.release()
    assert Lease(directory=tmp_path).acquire("wake-2") is True


# --------------------------------------------------------------- the day key

def test_the_day_key_is_utc_not_the_host_timezone():
    """It called `date.today()` under a docstring promising UTC. On the IST
    host this was developed on, the key rolled over five and a half hours
    early, every exhaustion record filed under the previous key became
    unreachable, and spent models read as fresh."""
    assert state_module._today() == datetime.now(UTC).date().isoformat()


def test_an_exhausted_model_stays_exhausted_across_a_local_midnight(monkeypatch):
    """The failure this actually caused: `is_exhausted` looks the model up
    under today's key, so a key that moves early loses the record and the
    service goes back to collecting 429s on a model it knows is spent."""
    service = ServiceState()
    service.mark_exhausted("gemini-3.1-flash-lite")
    assert service.is_exhausted("gemini-3.1-flash-lite") is True

    # Anywhere east of UTC, local midnight arrives first. The record must not
    # move out from under the lookup when it does.
    keys = set(service.exhausted)
    assert keys == {datetime.now(UTC).date().isoformat()}, keys
