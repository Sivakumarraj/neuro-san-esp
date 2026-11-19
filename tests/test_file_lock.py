"""The lease's file lock, on this platform and on the Windows code path.

`esp.service.state` used to refuse to import without `fcntl`, which made every
command that touches the service state fail on Windows. The lock now uses
`msvcrt` there. CI runs the suite on Windows as well; the stand-in test below
covers the Windows branch on any machine.
"""

from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from esp.service.file_lock import FileLock

ROOT = Path(__file__).resolve().parent.parent

HOLDER = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from esp.service.file_lock import FileLock
with FileLock(Path(sys.argv[2])):
    print("held", flush=True)
    time.sleep(float(sys.argv[3]))
"""


def test_a_second_process_waits_for_the_first(tmp_path):
    """The property the lease relies on: mutual exclusion between processes."""
    guard = tmp_path / "lease.json.guard"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(ROOT), str(guard), "1.0"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        started = time.monotonic()
        with FileLock(guard):
            waited = time.monotonic() - started
        assert waited > 0.5, f"took the lock after {waited:.2f}s while another process held it"
    finally:
        holder.wait(timeout=10)


def test_release_is_idempotent_and_the_lock_is_reusable(tmp_path):
    lock = FileLock(tmp_path / "guard")
    lock.release()                      # never acquired: a no-op
    with lock:
        pass
    lock.release()                      # already released: a no-op
    with lock:
        pass


class FakeMsvcrt(types.ModuleType):
    """msvcrt's locking() semantics: a held byte raises OSError for LK_NBLCK."""

    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self, busy_attempts: int) -> None:
        super().__init__("msvcrt")
        self.busy_attempts = busy_attempts
        self.calls: list[tuple[int, int]] = []

    def locking(self, handle: int, mode: int, nbytes: int) -> None:
        self.calls.append((mode, nbytes))
        if mode == self.LK_NBLCK and self.busy_attempts > 0:
            self.busy_attempts -= 1
            raise OSError(36, "Resource deadlock avoided")


def test_windows_branch_retries_until_free_then_unlocks(tmp_path, monkeypatch):
    fake = FakeMsvcrt(busy_attempts=3)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr("esp.service.file_lock.WINDOWS_POLL_SECONDS", 0.0)

    with FileLock(tmp_path / "guard", windows=True):
        pass

    locks = [call for call in fake.calls if call[0] == FakeMsvcrt.LK_NBLCK]
    unlocks = [call for call in fake.calls if call[0] == FakeMsvcrt.LK_UNLCK]
    assert len(locks) == 4, "three busy attempts, then the one that took it"
    assert unlocks == [(FakeMsvcrt.LK_UNLCK, 1)]
    assert all(nbytes == 1 for _, nbytes in fake.calls)


def test_state_module_imports_without_fcntl():
    """The old module raised RuntimeError at import when fcntl was missing.

    Run in a fresh interpreter, so blocking fcntl cannot leak into the modules
    other tests have already imported and patch.
    """
    script = ("import sys; sys.modules['fcntl'] = None; sys.path.insert(0, sys.argv[1]); "
              "import esp.service.state; print('ok')")
    done = subprocess.run([sys.executable, "-c", script, str(ROOT)],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX branch")
def test_posix_branch_is_the_default_here(tmp_path):
    assert FileLock(tmp_path / "guard").windows is False
