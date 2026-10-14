from __future__ import annotations

import io
from pathlib import Path
import signal

import pytest

from experiments.gradient_reuse_exact_reference import scheduler as scheduler_module
from experiments.gradient_reuse_exact_reference.scheduler import (
    GPUStatus,
    SchedulerInterrupted,
    SchedulerPolicy,
    UnitExecutionError,
    UnitSpec,
    launch_is_admissible,
    parse_worker_request,
    resolve_worker_limit,
    run_units,
    worker_capacity,
)


def _policy(**updates: object) -> SchedulerPolicy:
    values = {
        "maximum_workers": 20,
        "minimum_total_gpu_memory_mib": 32000,
        "gpu_reserve_mib": 1024,
        "per_job_reservation_mib": 1536,
        "launch_stagger_seconds": 2.0,
        "poll_seconds": 1.0,
        "gpu_device": 0,
    }
    values.update(updates)
    return SchedulerPolicy(**values)


def test_auto_workers_use_vram_policy_and_locked_ceiling() -> None:
    status = GPUStatus(total_mib=32623, free_mib=32000)
    policy = _policy()
    assert worker_capacity(status, policy) == 20
    assert resolve_worker_limit("auto", status, policy) == 20
    assert worker_capacity(GPUStatus(32000, 32000), policy) == 20
    assert resolve_worker_limit("8", status, policy) == 8
    assert resolve_worker_limit(19, status, policy) == 19
    with pytest.raises(RuntimeError, match="locked ceiling"):
        resolve_worker_limit(21, status, policy)
    with pytest.raises(RuntimeError, match="below locked minimum"):
        resolve_worker_limit("auto", GPUStatus(24000, 24000), policy)


def test_cifar_auto_reservation_capacity_reaches_nine_worker_ceiling() -> None:
    status = GPUStatus(total_mib=32623, free_mib=32623)
    # The production CIFAR reservation is 3400 MiB; this test uses that locked
    # value and reaches all nine configured slots on the 32 GiB card.
    policy = _policy(maximum_workers=9, per_job_reservation_mib=3400)
    assert worker_capacity(status, policy) == 9
    assert resolve_worker_limit("auto", status, policy) == 9
    assert worker_capacity(GPUStatus(32000, 32000), policy) == 9


def test_parse_worker_request_rejects_nonpositive_and_ambiguous_values() -> None:
    assert parse_worker_request("AUTO") == "auto"
    assert parse_worker_request("04") == 4
    for value in (0, -1, True, "0", "four", ""):
        with pytest.raises(ValueError):
            parse_worker_request(value)


def test_launch_gate_preserves_reserve_plus_one_job() -> None:
    policy = _policy()
    threshold = 1024 + 1536
    assert launch_is_admissible(GPUStatus(32623, threshold), policy)
    assert not launch_is_admissible(GPUStatus(32623, threshold - 1), policy)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _Process:
    next_pid = 1000

    def __init__(self, *, code: int = 0, polls: int = 1) -> None:
        self.code = code
        self.polls = polls
        self.seen = 0
        self.pid = _Process.next_pid
        _Process.next_pid += 1

    def poll(self) -> int | None:
        self.seen += 1
        return None if self.seen <= self.polls else self.code


class _TerminableProcess:
    next_pid = 8000

    def __init__(self) -> None:
        self.pid = _TerminableProcess.next_pid
        _TerminableProcess.next_pid += 1
        self.running = True
        self.wait_calls = 0

    def poll(self) -> int | None:
        return None if self.running else -signal.SIGTERM

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        return -signal.SIGTERM if not self.running else 0


def _specs(tmp_path: Path, count: int) -> list[UnitSpec]:
    return [
        UnitSpec(
            unit_id=f"unit_{index}",
            command=("python", "-m", "unit", str(index)),
            log_path=(tmp_path / "logs" / f"unit_{index}.log").resolve(),
        )
        for index in range(count)
    ]


def test_scheduler_restarts_from_valid_seals_and_revalidates_full_coverage(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    sealed = {"unit_0"}
    launched: list[tuple[str, float]] = []
    states: list[str] = []

    def launcher(spec: UnitSpec, cwd: Path, env: object) -> tuple[_Process, io.StringIO]:
        del cwd, env
        launched.append((spec.unit_id, clock.now()))
        sealed.add(spec.unit_id)
        return _Process(polls=1), io.StringIO()

    result = run_units(
        _specs(tmp_path, 4),
        policy=_policy(maximum_workers=3),
        workers="auto",
        cwd=tmp_path,
        validate_if_complete=lambda unit_id: unit_id in sealed,
        validate_after_success=lambda unit_id: None,
        state_callback=lambda stage, details: states.append(stage),
        gpu_probe=lambda device: GPUStatus(32623, 10000),
        launcher=launcher,
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    assert result.full_coverage
    assert result.previously_completed_units == 1
    assert result.newly_completed_units == 3
    assert [unit_id for unit_id, _ in launched] == ["unit_1", "unit_2", "unit_3"]
    assert all(
        second - first >= 2.0
        for first, second in zip(
            [stamp for _, stamp in launched],
            [stamp for _, stamp in launched][1:],
        )
    )
    assert states[-1] == "FULL_COVERAGE"


def test_scheduler_probes_before_each_launch_and_waits_for_memory(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    sealed: set[str] = set()
    observations = iter(
        [
            GPUStatus(32623, 12000),  # capacity resolution
            GPUStatus(32623, 5000),   # below reserve + job
            GPUStatus(32623, 7000),   # admit unit 0
            GPUStatus(32623, 7000),   # admit unit 1
        ]
    )
    launches: list[str] = []

    def launcher(spec: UnitSpec, cwd: Path, env: object) -> tuple[_Process, io.StringIO]:
        del cwd, env
        launches.append(spec.unit_id)
        sealed.add(spec.unit_id)
        return _Process(polls=0), io.StringIO()

    run_units(
        _specs(tmp_path, 2),
        policy=_policy(
            maximum_workers=2,
            gpu_reserve_mib=4000,
            per_job_reservation_mib=2000,
            launch_stagger_seconds=0,
        ),
        workers="auto",
        cwd=tmp_path,
        validate_if_complete=lambda unit_id: unit_id in sealed,
        validate_after_success=lambda unit_id: None,
        gpu_probe=lambda device: next(observations),
        launcher=launcher,
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    assert launches == ["unit_0", "unit_1"]


def test_startup_gate_prevents_cpu_preflight_stampede_then_reaches_width(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    sealed: set[str] = set()
    processes: list[_Process] = []
    launches: list[tuple[str, int]] = []
    states: list[tuple[str, int, int]] = []
    probe_calls = 0

    def launcher(spec: UnitSpec, cwd: Path, env: object) -> tuple[_Process, io.StringIO]:
        del cwd, env
        process = _Process(polls=20)
        processes.append(process)
        launches.append((spec.unit_id, probe_calls))
        sealed.add(spec.unit_id)
        return process, io.StringIO()

    def process_probe(device: int) -> set[int]:
        nonlocal probe_calls
        assert device == 0
        probe_calls += 1
        # Keep the first child in CPU-only preflight for one complete
        # observation window, then expose each launched child on CUDA.
        if probe_calls == 1:
            return set()
        return {process.pid for process in processes}

    def capture(stage: str, details: object) -> None:
        assert isinstance(details, dict)
        states.append(
            (
                stage,
                len(details["active_units"]),
                int(details["unconfirmed_startups"]),
            )
        )

    result = run_units(
        _specs(tmp_path, 3),
        policy=_policy(
            maximum_workers=3,
            maximum_concurrent_startups=1,
            launch_stagger_seconds=0,
        ),
        workers=3,
        cwd=tmp_path,
        validate_if_complete=lambda unit_id: unit_id in sealed,
        validate_after_success=lambda unit_id: None,
        state_callback=capture,
        gpu_probe=lambda device: GPUStatus(32623, 12000),
        gpu_process_probe=process_probe,
        launcher=launcher,
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    assert result.full_coverage
    assert [unit_id for unit_id, _ in launches] == ["unit_0", "unit_1", "unit_2"]
    assert launches[1][1] >= 2
    assert any(stage == "WAITING_FOR_CUDA_STARTUP" for stage, _, _ in states)
    assert max(unconfirmed for _, _, unconfirmed in states) == 1
    assert max(active for _, active, _ in states) == 3


def test_gpu_using_grandchild_confirms_owned_unit_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    sealed: set[str] = set()
    processes: list[_Process] = []
    maximum_active = 0

    def fake_launch(
        spec: UnitSpec,
        *,
        cwd: Path,
        env: object,
    ) -> tuple[_Process, io.StringIO]:
        del cwd, env
        sealed.add(spec.unit_id)
        process = _Process(polls=5)
        processes.append(process)
        return process, io.StringIO()

    def fake_getpgid(pid: int) -> int:
        return pid - 100_000

    def capture(stage: str, details: object) -> None:
        nonlocal maximum_active
        del stage
        assert isinstance(details, dict)
        maximum_active = max(maximum_active, len(details["active_units"]))

    def absent_group(pid: int, signum: int) -> None:
        del pid, signum
        raise ProcessLookupError

    monkeypatch.setattr(scheduler_module, "_default_launch", fake_launch)
    monkeypatch.setattr(scheduler_module.os, "getpgid", fake_getpgid)
    monkeypatch.setattr(scheduler_module.os, "killpg", absent_group)
    run_units(
        _specs(tmp_path, 2),
        policy=_policy(
            maximum_workers=2,
            maximum_concurrent_startups=1,
            launch_stagger_seconds=0,
        ),
        workers=2,
        cwd=tmp_path,
        validate_if_complete=lambda unit_id: unit_id in sealed,
        validate_after_success=lambda unit_id: None,
        state_callback=capture,
        gpu_probe=lambda device: GPUStatus(32623, 12000),
        gpu_process_probe=lambda device: {
            process.pid + 100_000 for process in processes
        },
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    assert maximum_active == 2


def test_failure_stops_new_launches_drains_active_and_returns_no_success(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    sealed: set[str] = set()
    launched: list[str] = []

    def launcher(spec: UnitSpec, cwd: Path, env: object) -> tuple[_Process, io.StringIO]:
        del cwd, env
        launched.append(spec.unit_id)
        if spec.unit_id == "unit_0":
            return _Process(code=9, polls=1), io.StringIO()
        sealed.add(spec.unit_id)
        return _Process(code=0, polls=4), io.StringIO()

    with pytest.raises(UnitExecutionError) as captured:
        run_units(
            _specs(tmp_path, 5),
            policy=_policy(
                maximum_workers=3,
                launch_stagger_seconds=0,
                poll_seconds=1,
            ),
            workers=3,
            cwd=tmp_path,
            validate_if_complete=lambda unit_id: unit_id in sealed,
            validate_after_success=lambda unit_id: None,
            gpu_probe=lambda device: GPUStatus(32623, 10000),
            launcher=launcher,
            monotonic=clock.now,
            sleep=clock.sleep,
        )
    assert captured.value.failures[0]["unit_id"] == "unit_0"
    # unit_1 was already active and drained; the remaining queue was untouched.
    assert launched == ["unit_0", "unit_1"]
    assert "unit_1" in sealed
    assert not {"unit_2", "unit_3", "unit_4"}.intersection(sealed)


def test_zero_exit_without_valid_seal_is_a_unit_failure(tmp_path: Path) -> None:
    clock = _Clock()

    def invalid(_: str) -> None:
        raise RuntimeError("missing PASS seal")

    with pytest.raises(UnitExecutionError, match="validation_error"):
        run_units(
            _specs(tmp_path, 1),
            policy=_policy(maximum_workers=1, launch_stagger_seconds=0),
            workers=1,
            cwd=tmp_path,
            validate_if_complete=lambda unit_id: False,
            validate_after_success=invalid,
            gpu_probe=lambda device: GPUStatus(32623, 10000),
            launcher=lambda spec, cwd, env: (_Process(polls=0), io.StringIO()),
            monotonic=clock.now,
            sleep=clock.sleep,
        )


def test_state_write_failure_closes_admission_and_drains_active_unit(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    sealed: set[str] = set()
    launched: list[str] = []

    def launcher(spec: UnitSpec, cwd: Path, env: object) -> tuple[_Process, io.StringIO]:
        del cwd, env
        launched.append(spec.unit_id)
        sealed.add(spec.unit_id)
        return _Process(polls=2), io.StringIO()

    def broken_state(stage: str, details: object) -> None:
        del stage, details
        raise OSError("state disk unavailable")

    with pytest.raises(OSError, match="state disk unavailable"):
        run_units(
            _specs(tmp_path, 3),
            policy=_policy(maximum_workers=3, launch_stagger_seconds=0),
            workers="auto",
            cwd=tmp_path,
            validate_if_complete=lambda unit_id: unit_id in sealed,
            validate_after_success=lambda unit_id: None,
            state_callback=broken_state,
            gpu_probe=lambda device: GPUStatus(32623, 10000),
            launcher=launcher,
            monotonic=clock.now,
            sleep=clock.sleep,
        )
    assert launched == ["unit_0"]
    assert sealed == {"unit_0"}


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_stop_signal_terminates_and_waits_for_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_signal: signal.Signals,
) -> None:
    clock = _Clock()
    installed: dict[int, object] = {}
    restored = object()
    launches: list[str] = []
    handles: list[io.StringIO] = []
    processes: list[_TerminableProcess] = []
    group_signals: list[tuple[int, int]] = []
    states: list[str] = []

    def fake_getsignal(signum: int) -> object:
        del signum
        return restored

    def fake_signal(signum: int, handler: object) -> object:
        previous = installed.get(int(signum), restored)
        installed[int(signum)] = handler
        return previous

    def fake_launch(
        spec: UnitSpec,
        *,
        cwd: Path,
        env: object,
    ) -> tuple[_TerminableProcess, io.StringIO]:
        del cwd, env
        launches.append(spec.unit_id)
        process = _TerminableProcess()
        handle = io.StringIO()
        processes.append(process)
        handles.append(handle)
        return process, handle

    def fake_killpg(pid: int, signum: int) -> None:
        group_signals.append((pid, signum))
        matching = [process for process in processes if process.pid == pid]
        assert len(matching) == 1
        matching[0].running = False

    sent = False

    def interrupting_sleep(seconds: float) -> None:
        nonlocal sent
        clock.sleep(seconds)
        if not sent:
            sent = True
            handler = installed[int(stop_signal)]
            assert callable(handler)
            handler(int(stop_signal), None)

    monkeypatch.setattr(scheduler_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(scheduler_module.signal, "signal", fake_signal)
    monkeypatch.setattr(scheduler_module, "_default_launch", fake_launch)
    monkeypatch.setattr(scheduler_module.os, "killpg", fake_killpg)

    with pytest.raises(SchedulerInterrupted) as captured:
        run_units(
            _specs(tmp_path, 3),
            policy=_policy(
                maximum_workers=3,
                launch_stagger_seconds=0,
                shutdown_grace_seconds=0,
            ),
            workers=3,
            cwd=tmp_path,
            validate_if_complete=lambda unit_id: False,
            validate_after_success=lambda unit_id: None,
            state_callback=lambda stage, details: states.append(stage),
            gpu_probe=lambda device: GPUStatus(32623, 12000),
            gpu_process_probe=lambda device: set(),
            monotonic=clock.now,
            sleep=interrupting_sleep,
        )
    assert captured.value.signum == int(stop_signal)
    assert launches == ["unit_0"]
    assert states[-1] == "INTERRUPTING"
    assert group_signals == [
        (processes[0].pid, signal.SIGTERM),
        (processes[0].pid, signal.SIGKILL),
    ]
    assert processes[0].wait_calls == 2
    assert handles[0].closed
    assert installed[int(signal.SIGINT)] is restored
    assert installed[int(signal.SIGTERM)] is restored


def test_policy_requires_all_locked_runtime_fields() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        SchedulerPolicy.from_runtime({"maximum_concurrent_gpu_processes": 8})
    policy = SchedulerPolicy.from_runtime(
        {
            "maximum_concurrent_gpu_processes": 8,
            "minimum_total_gpu_memory_mib": 32000,
            "reserve_free_gpu_memory_mib": 4096,
            "per_process_launch_reservation_mib": 3584,
            "launch_stagger_seconds": 4,
            "scheduler_poll_seconds": 2,
            "gpu_device": 0,
        }
    )
    assert policy.maximum_workers == 8
    assert policy.per_job_reservation_mib == 3584
    assert policy.maximum_concurrent_startups == 1
    assert policy.shutdown_grace_seconds == 10.0


def test_policy_accepts_bounded_startup_and_shutdown_overrides() -> None:
    policy = SchedulerPolicy.from_runtime(
        {
            "maximum_concurrent_gpu_processes": 8,
            "minimum_total_gpu_memory_mib": 32000,
            "reserve_free_gpu_memory_mib": 4096,
            "per_process_launch_reservation_mib": 3584,
            "launch_stagger_seconds": 4,
            "scheduler_poll_seconds": 2,
            "gpu_device": 0,
            "maximum_concurrent_startups": 2,
            "scheduler_shutdown_grace_seconds": 3,
        }
    )
    assert policy.maximum_concurrent_startups == 2
    assert policy.shutdown_grace_seconds == 3
    with pytest.raises(ValueError, match="cannot exceed maximum_workers"):
        _policy(maximum_workers=2, maximum_concurrent_startups=3)
