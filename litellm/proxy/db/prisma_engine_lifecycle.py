"""Lifetime management for the Prisma (prisma-client-py 0.11) query engine process.

Why this module exists (bt2 memory incident, 2026-11-09):

``prisma-client-py`` runs the query engine as a *separate* process
(``node_modules/prisma/query-engine-*``).  In 0.11.0 the teardown path is

    QueryEngine.close(): self.process.send_signal(SIGINT)
                         self.process.wait(timeout=None)   # blocks forever when the engine hangs

``close()`` runs from ``Prisma.disconnect()`` inside async code, so one
unresponsive engine blocks the whole worker event loop.  ``process`` is only set
to ``None`` *after* the wait, so when the wait raises the engine process is
simply dropped on the floor: it keeps its RSS (multi-GB in the incident) and its
Postgres connections while no client can reach it any more.

``PrismaWrapper.recreate_prisma_client()`` used to swallow that exception and
replace ``self._original_prisma`` with a brand new client, so every failed
reconnect leaked one engine for good.  A worker killed by the uvicorn supervisor
(``Child process [x] died``) leaks its engine the same way.

Two guards live here:

* :func:`engine_pid_of` / :func:`kill_process` - used by
  ``PrismaWrapper.recreate_prisma_client()`` to kill the old engine *before*
  ``disconnect()``.  The subsequent ``process.wait()`` then returns immediately
  and nothing is left behind.
* :class:`OrphanEngineReaper` - a low-frequency sweep that kills query-engine
  processes whose parent is gone (``ppid == 1``).  An engine is always spawned by
  the worker that owns it, so inside the container (PID 1 == litellm main
  process) an engine reparented to PID 1 can only mean "its worker died".  This
  covers the SIGKILL case that no in-process cleanup can catch.
"""

import asyncio
import os
import signal
import time
from typing import Callable, Iterable, List, Optional

from litellm._logging import verbose_proxy_logger

# Substring of the query-engine command line.  The real one looks like
# ``/root/.cache/prisma-python/binaries/<v>/node_modules/prisma/query-engine-debian-openssl-3.0.x -p 35495``.
QUERY_ENGINE_CMDLINE_MARKER = "query-engine"

PROC_ROOT = "/proc"
DEFAULT_REAP_INTERVAL_SECONDS = 60.0


def engine_pid_of(client: object) -> int:
    """Return the query-engine PID owned by a ``prisma.Prisma`` client, or 0.

    ``Prisma._engine`` raises ``ClientNotConnectedError`` when no engine is
    attached, so every attribute access is defensive.
    """
    if client is None:
        return 0
    try:
        engine = getattr(client, "_engine", None)
        process = getattr(engine, "process", None)
        pid = getattr(process, "pid", None)
        return int(pid) if pid else 0
    except Exception:  # noqa: BLE001 - ClientNotConnectedError and friends
        return 0


def pid_is_alive(pid: int) -> bool:
    """True when a process with ``pid`` still exists (zombies count as alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def kill_process(pid: int, sig: int = signal.SIGKILL) -> bool:
    """Send ``sig`` to ``pid``; True when the signal was delivered."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except OSError as e:
        verbose_proxy_logger.debug("Failed to signal PID %s: %s", pid, e)
        return False
    return True


def reap_child(pid: int, timeout: float = 1.0) -> bool:
    """Best-effort ``waitpid`` for a child we just killed.

    ``Prisma.disconnect()`` normally reaps the engine through ``Popen.wait()``; when
    it raises before that point the killed engine is left as a zombie.  Reaping it
    here keeps PID usage flat across reconnects.  Returns True when it was reaped.
    """
    if pid <= 0:
        return False
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return False  # not our child (already reaped, or reparented)
        except OSError:
            return False
        if waited == pid:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _read_cmdline(pid: int, proc_root: str = PROC_ROOT) -> str:
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def read_ppid(pid: int, proc_root: str = PROC_ROOT) -> int:
    """Parent PID from ``/proc/<pid>/stat``; -1 when it cannot be read.

    ``comm`` may contain spaces and parentheses, so fields are parsed from the
    last ``)`` onwards: ``pid (comm) state ppid ...``.
    """
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), "rb") as fh:
            raw = fh.read()
    except OSError:
        return -1
    close = raw.rfind(b")")
    if close < 0:
        return -1
    fields = raw[close + 1 :].split()
    if len(fields) < 2:
        return -1
    try:
        return int(fields[1])
    except ValueError:
        return -1


def list_query_engine_pids(proc_root: str = PROC_ROOT) -> List[int]:
    """PIDs of every process whose command line looks like a Prisma query engine."""
    found: List[int] = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if QUERY_ENGINE_CMDLINE_MARKER in _read_cmdline(pid, proc_root):
            found.append(pid)
    return sorted(found)


def find_orphan_query_engine_pids(
    *,
    proc_root: str = PROC_ROOT,
    exclude_pids: Iterable[int] = (),
) -> List[int]:
    """Query engines whose parent is gone (``ppid == 1``), minus ``exclude_pids``."""
    exclude = {int(pid) for pid in exclude_pids if pid}
    return [
        pid
        for pid in list_query_engine_pids(proc_root)
        if pid not in exclude and read_ppid(pid, proc_root) == 1
    ]


def reap_orphan_query_engines(
    *,
    proc_root: str = PROC_ROOT,
    exclude_pids: Iterable[int] = (),
    kill: Optional[Callable[[int, int], object]] = None,
) -> List[int]:
    """SIGKILL orphaned query engines; returns the PIDs that were signalled."""
    killed: List[int] = []
    for pid in find_orphan_query_engine_pids(
        proc_root=proc_root, exclude_pids=exclude_pids
    ):
        delivered = False
        if kill is None:
            delivered = kill_process(pid)
        else:
            try:
                kill(pid, signal.SIGKILL)
                delivered = True
            except ProcessLookupError:
                delivered = False
            except OSError as e:
                verbose_proxy_logger.debug(
                    "Failed to reap orphaned query engine %s: %s", pid, e
                )
        if delivered:
            killed.append(pid)
    if killed:
        verbose_proxy_logger.warning(
            "Reaped %d orphaned prisma query-engine process(es): %s",
            len(killed),
            killed,
        )
    return killed


class OrphanEngineReaper:
    """Periodically kills query engines whose owning worker is gone."""

    def __init__(
        self,
        *,
        interval_seconds: float = DEFAULT_REAP_INTERVAL_SECONDS,
        proc_root: str = PROC_ROOT,
        exclude_pids: Optional[Callable[[], Iterable[int]]] = None,
        kill: Optional[Callable[[int, int], object]] = None,
    ) -> None:
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.proc_root = proc_root
        self._exclude_pids = exclude_pids
        self._kill = kill

    def run_once(self) -> List[int]:
        exclude: Iterable[int] = ()
        if self._exclude_pids is not None:
            try:
                exclude = self._exclude_pids() or ()
            except Exception:  # noqa: BLE001 - never let the sweep die on a callback
                exclude = ()
        return reap_orphan_query_engines(
            proc_root=self.proc_root, exclude_pids=exclude, kill=self._kill
        )

    async def run_once_async(self) -> List[int]:
        """Run the sweep in a thread so a slow ``/proc`` never blocks the loop."""
        return await asyncio.to_thread(self.run_once)

    async def loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval_seconds)
                await self.run_once_async()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep the guard alive
                verbose_proxy_logger.debug("Orphan query-engine sweep failed: %s", e)
