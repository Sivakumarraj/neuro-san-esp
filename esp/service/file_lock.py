"""An exclusive lock on a file, held between processes, on POSIX and Windows.

The optimiser lease (`esp.service.state.Lease`) needs its read, expiry check and
write to be one critical section across processes. POSIX has `fcntl.flock`;
Windows has `msvcrt.locking`. Both are advisory locks that every holder here
takes the same way, so either gives the mutual exclusion the lease relies on.
It does not coordinate across hosts; nothing in this deployment runs the
optimiser on two.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import ModuleType

# How long to wait between attempts on Windows, where msvcrt has no blocking
# lock that waits forever: LK_LOCK gives up after ten seconds and raises.
WINDOWS_POLL_SECONDS = 0.05


class FileLock:
    """Exclusive lock on one file, released when the `with` block ends."""

    def __init__(self, path: Path, windows: bool | None = None) -> None:
        self.path = path
        # Decided at construction so a test can exercise the Windows branch on
        # any machine by passing windows=True with a stand-in msvcrt module.
        self.windows = os.name == "nt" if windows is None else windows
        self._handle: int | None = None

    def acquire(self) -> None:
        """Block until this process holds the lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if self.windows:
                self._lock_windows(handle)
            else:
                self._lock_posix(handle)
        except BaseException:
            os.close(handle)
            raise
        self._handle = handle

    def release(self) -> None:
        """Release the lock and close the file. Safe to call when not held."""
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            if self.windows:
                msvcrt = self._msvcrt()
                os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # pylint: disable=import-outside-toplevel  # POSIX only
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    @staticmethod
    def _lock_posix(handle: int) -> None:
        """Take an exclusive flock, waiting as long as another holder keeps it."""
        import fcntl  # pylint: disable=import-outside-toplevel  # POSIX only
        fcntl.flock(handle, fcntl.LOCK_EX)

    def _lock_windows(self, handle: int) -> None:
        """Lock the first byte of the file, retrying until it is free.

        msvcrt locks a byte range from the current position, so every holder
        seeks to the start and locks the same single byte.
        """
        msvcrt = self._msvcrt()
        while True:
            os.lseek(handle, 0, os.SEEK_SET)
            try:
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(WINDOWS_POLL_SECONDS)

    @staticmethod
    def _msvcrt() -> ModuleType:
        """The Windows C runtime module, imported only where it exists."""
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error  # Windows only
        return msvcrt
