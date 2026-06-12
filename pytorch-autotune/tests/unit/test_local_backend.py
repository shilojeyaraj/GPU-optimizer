import subprocess
from unittest.mock import MagicMock

import pytest

from autotune.backends import local
from autotune.backends.local import TrialResult, run_trial


def _fake_completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    fake = MagicMock()
    fake.returncode = returncode
    fake.stdout = stdout
    fake.stderr = stderr
    return fake


def test_oom_is_detected_from_stderr(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _fake_completed(
            returncode=1, stderr="RuntimeError: CUDA out of memory: tried to allocate 2GiB"
        ),
    )
    result = run_trial("anything", "anything", {"batch_size": 999})
    assert result.oom is True
    assert result.throughput is None
    assert result.compile_failure is False


def test_compile_failure_is_detected_separately(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _fake_completed(
            returncode=1,
            stderr="torch._dynamo.exc.BackendCompilerFailed: ...",
        ),
    )
    result = run_trial("m", "f", {})
    assert result.compile_failure is True
    assert result.oom is False


def test_timeout_returns_timeout_flag(monkeypatch):
    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    result = run_trial("m", "f", {}, timeout_seconds=1)
    assert result.timeout is True
    assert result.error == "timeout"


def test_successful_trial_parses_payload(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _fake_completed(
            returncode=0,
            stdout='{"throughput": 1234.5, "duration_seconds": 4.2, "memory_mb": 800, "extra": {}}',
        ),
    )
    result = run_trial("m", "f", {"batch_size": 64})
    assert result.throughput == 1234.5
    assert result.duration_seconds == 4.2
    assert result.memory_mb == 800


def test_to_dict_round_trip():
    r = TrialResult(config={"a": 1}, throughput=10.0, memory_mb=100.0)
    d = r.to_dict()
    assert d["config"] == {"a": 1}
    assert d["throughput_samples_per_sec"] == 10.0
    assert d["oom"] is False
