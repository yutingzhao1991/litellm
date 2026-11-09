import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.db.prisma_engine_lifecycle import (
    OrphanEngineReaper,
    engine_pid_of,
    find_orphan_query_engine_pids,
    kill_process,
    list_query_engine_pids,
    is_query_engine_pid,
    pid_is_alive,
    read_ppid,
    reaper_supported,
    reap_child,
    reap_orphan_query_engines,
)

ENGINE_CMD = (
    "/root/.cache/prisma-python/binaries/5.4.2/ac9d7041/node_modules/prisma/"
    "query-engine-debian-openssl-3.0.x -p 35495 --enable-metrics --enable-raw-queries"
)


def _write_process(proc_root, pid, cmdline, ppid, comm="query-engine-debian"):
    proc_dir = os.path.join(str(proc_root), str(pid))
    os.makedirs(proc_dir)
    with open(os.path.join(proc_dir, "cmdline"), "w") as fh:
        fh.write(cmdline.replace(" ", "\x00"))
    with open(os.path.join(proc_dir, "stat"), "w") as fh:
        fh.write(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194560 100 0 0 0 1 0 0 0\n")


def test_read_ppid_handles_parentheses_and_spaces_in_comm(tmp_path):
    _write_process(tmp_path, 42, ENGINE_CMD, ppid=1, comm="query engine (x)")

    assert read_ppid(42, str(tmp_path)) == 1


def test_read_ppid_missing_process_returns_minus_one(tmp_path):
    assert read_ppid(999, str(tmp_path)) == -1


def test_list_query_engine_pids_only_matches_engines(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    _write_process(tmp_path, 200, ENGINE_CMD, ppid=300)
    _write_process(tmp_path, 300, "/usr/bin/python3.13 /usr/sbin/litellm --port 4000", ppid=1)

    assert list_query_engine_pids(str(tmp_path)) == [100, 200]


def test_find_orphan_query_engine_pids_only_returns_ppid_one(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    _write_process(tmp_path, 200, ENGINE_CMD, ppid=300)
    _write_process(tmp_path, 300, "/usr/bin/python3.13 /usr/sbin/litellm --port 4000", ppid=1)

    assert find_orphan_query_engine_pids(proc_root=str(tmp_path)) == [100]


def test_find_orphan_query_engine_pids_honours_exclude(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)

    assert find_orphan_query_engine_pids(proc_root=str(tmp_path), exclude_pids=[100]) == []


def test_reap_orphan_query_engines_signals_only_orphans(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    _write_process(tmp_path, 200, ENGINE_CMD, ppid=300)
    signalled = []

    killed = reap_orphan_query_engines(
        proc_root=str(tmp_path), kill=lambda pid, sig: signalled.append((pid, sig))
    )

    assert killed == [100]
    assert [pid for pid, _ in signalled] == [100]


def test_reap_orphan_query_engines_skips_already_dead_process(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)

    def _gone(_pid, _sig):
        raise ProcessLookupError

    assert reap_orphan_query_engines(proc_root=str(tmp_path), kill=_gone) == []


def test_engine_pid_of_reads_prisma_internal_engine(tmp_path):
    class _Process:
        pid = 4321

    class _Engine:
        process = _Process()

    class _Client:
        _engine = _Engine()

    assert engine_pid_of(_Client()) == 4321


def test_engine_pid_of_without_engine_returns_zero():
    class _Client:
        @property
        def _engine(self):
            raise RuntimeError("Client is not connected to the query engine")

    assert engine_pid_of(_Client()) == 0
    assert engine_pid_of(None) == 0


def test_pid_is_alive_and_kill_process_edge_cases():
    assert pid_is_alive(os.getpid()) is True
    assert pid_is_alive(0) is False
    assert kill_process(0) is False


@pytest.mark.asyncio
async def test_reaper_run_once_uses_exclude_callback(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    signalled = []
    reaper = OrphanEngineReaper(
        interval_seconds=1,
        proc_root=str(tmp_path),
        exclude_pids=lambda: [100],
        kill=lambda pid, sig: signalled.append(pid),
    )

    assert await reaper.run_once_async() == []
    assert signalled == []


@pytest.mark.asyncio
async def test_reaper_survives_broken_exclude_callback(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    signalled = []

    def _boom():
        raise RuntimeError("db not ready")

    reaper = OrphanEngineReaper(
        interval_seconds=1,
        proc_root=str(tmp_path),
        exclude_pids=_boom,
        kill=lambda pid, sig: signalled.append(pid),
    )

    assert await reaper.run_once_async() == [100]
    assert signalled == [100]


def test_reap_child_reaps_exited_child():
    import subprocess

    proc = subprocess.Popen(["true"])
    assert reap_child(proc.pid, timeout=2.0) is True


def test_reap_child_returns_false_for_non_child():
    assert reap_child(os.getpid(), timeout=0.1) is False
    assert reap_child(0) is False


def test_is_query_engine_pid(tmp_path):
    _write_process(tmp_path, 100, ENGINE_CMD, ppid=1)
    _write_process(tmp_path, 200, "/usr/bin/python3.13 /usr/sbin/litellm --port 4000", ppid=1)

    assert is_query_engine_pid(100, str(tmp_path)) is True
    assert is_query_engine_pid(200, str(tmp_path)) is False
    assert is_query_engine_pid(999, str(tmp_path)) is False
    assert is_query_engine_pid(0, str(tmp_path)) is False


def test_reaper_supported_refuses_pid_one():
    assert reaper_supported(lambda: 1) is False
    assert reaper_supported(lambda: 4242) is True
