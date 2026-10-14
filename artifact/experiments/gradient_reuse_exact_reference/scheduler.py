"""Restart-safe concurrent scheduling for the exact-reference campaigns.

The scheduler is deliberately outcome blind.  Its only inputs are a frozen
unit inventory, resource observations, process exit codes, and authenticated
unit-completion seals.  It never reads a scientific metric and it never builds
an aggregate.  The caller may aggregate only after this module returns a
full-coverage :class:`SchedulerResult`.

One process is launched per fresh GPU-memory observation.  A launch is
admitted only when ``free_mib >= reserve_mib + per_job_mib``; launches are
staggered and the number of children which have not yet appeared in the GPU
compute-process table is bounded independently.  This prevents CPU-heavy
preflights from stampeding the host before CUDA allocation becomes visible to
``nvidia-smi`` while still allowing the configured steady-state width once
each child reaches CUDA.  ``maximum_workers`` is a ceiling, while the memory
and startup gates are the live admission controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence


WorkerRequest = int | str


@dataclass(frozen=True)
class GPUStatus:
    """One authenticated ``nvidia-smi`` memory observation."""

    total_mib: int
    free_mib: int

    def __post_init__(self) -> None:
        if self.total_mib <= 0:
            raise ValueError("GPU total memory must be positive")
        if self.free_mib < 0 or self.free_mib > self.total_mib:
            raise ValueError("GPU free memory is outside the valid range")


@dataclass(frozen=True)
class SchedulerPolicy:
    """Prospectively locked operational limits for one campaign."""

    maximum_workers: int
    minimum_total_gpu_memory_mib: int
    gpu_reserve_mib: int
    per_job_reservation_mib: int
    launch_stagger_seconds: float
    poll_seconds: float = 5.0
    gpu_device: int = 0
    maximum_concurrent_startups: int = 1
    shutdown_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.maximum_workers < 1:
            raise ValueError("maximum_workers must be positive")
        if self.minimum_total_gpu_memory_mib < 1:
            raise ValueError("minimum_total_gpu_memory_mib must be positive")
        if self.gpu_reserve_mib < 0:
            raise ValueError("gpu_reserve_mib cannot be negative")
        if self.per_job_reservation_mib < 1:
            raise ValueError("per_job_reservation_mib must be positive")
        if self.launch_stagger_seconds < 0:
            raise ValueError("launch_stagger_seconds cannot be negative")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.gpu_device < 0:
            raise ValueError("gpu_device cannot be negative")
        if self.maximum_concurrent_startups < 1:
            raise ValueError("maximum_concurrent_startups must be positive")
        if self.maximum_concurrent_startups > self.maximum_workers:
            raise ValueError(
                "maximum_concurrent_startups cannot exceed maximum_workers"
            )
        if self.shutdown_grace_seconds < 0:
            raise ValueError("shutdown_grace_seconds cannot be negative")

    @classmethod
    def from_runtime(cls, runtime: Mapping[str, Any]) -> "SchedulerPolicy":
        """Build a policy from the frozen campaign lock.

        The original prospective locks predate the operational startup and
        shutdown guards.  Their fail-safe defaults therefore remain one
        unconfirmed startup and a ten-second TERM grace.  A future lock may
        override either guard without changing the scientific unit inventory.
        """

        required = {
            "maximum_concurrent_gpu_processes",
            "minimum_total_gpu_memory_mib",
            "reserve_free_gpu_memory_mib",
            "per_process_launch_reservation_mib",
            "launch_stagger_seconds",
            "scheduler_poll_seconds",
            "gpu_device",
        }
        missing = sorted(required.difference(runtime))
        if missing:
            raise RuntimeError(f"scheduler runtime fields are missing: {missing}")
        return cls(
            maximum_workers=int(runtime["maximum_concurrent_gpu_processes"]),
            minimum_total_gpu_memory_mib=int(runtime["minimum_total_gpu_memory_mib"]),
            gpu_reserve_mib=int(runtime["reserve_free_gpu_memory_mib"]),
            per_job_reservation_mib=int(
                runtime["per_process_launch_reservation_mib"]
            ),
            launch_stagger_seconds=float(runtime["launch_stagger_seconds"]),
            poll_seconds=float(runtime["scheduler_poll_seconds"]),
            gpu_device=int(runtime["gpu_device"]),
            maximum_concurrent_startups=int(
                runtime.get("maximum_concurrent_startups", 1)
            ),
            shutdown_grace_seconds=float(
                runtime.get("scheduler_shutdown_grace_seconds", 10.0)
            ),
        )


@dataclass(frozen=True)
class UnitSpec:
    """Operational description of a frozen scientific unit."""

    unit_id: str
    command: tuple[str, ...]
    log_path: Path

    def __post_init__(self) -> None:
        if not self.unit_id or self.unit_id in {".", ".."}:
            raise ValueError("unit_id must be nonempty")
        if not self.command or any(
            not isinstance(value, str) or not value for value in self.command
        ):
            raise ValueError("unit command must contain nonempty strings")
        if not self.log_path.is_absolute():
            raise ValueError("unit log path must be absolute")


@dataclass(frozen=True)
class SchedulerResult:
    """Full-coverage result returned only after every seal validates."""

    total_units: int
    previously_completed_units: int
    newly_completed_units: int
    worker_limit: int
    worker_request: str
    gpu_total_mib: int
    gpu_reserve_mib: int
    per_job_reservation_mib: int

    @property
    def full_coverage(self) -> bool:
        return (
            self.previously_completed_units + self.newly_completed_units
            == self.total_units
        )


class UnitExecutionError(RuntimeError):
    """Raised after new launches stop and already-running units drain."""

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        self.failures = tuple(dict(value) for value in failures)
        super().__init__(f"exact-reference unit failure(s): {list(self.failures)!r}")


class SchedulerInterrupted(RuntimeError):
    """Raised after an external stop signal has cleanly reaped every unit."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        try:
            name = signal.Signals(self.signum).name
        except ValueError:
            name = str(self.signum)
        super().__init__(f"exact-reference scheduler interrupted by {name}")


@dataclass
class _Active:
    spec: UnitSpec
    process: Any
    log_handle: Any
    started_monotonic: float
    owns_process_group: bool
    cuda_confirmed: bool = False


def parse_worker_request(value: WorkerRequest) -> WorkerRequest:
    """Parse ``--workers`` as ``auto`` or a positive integer."""

    if isinstance(value, bool):
        raise ValueError("workers must be 'auto' or a positive integer")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("workers must be positive")
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if not text.isdigit() or int(text) < 1:
        raise ValueError("workers must be 'auto' or a positive integer")
    return int(text)


def worker_capacity(status: GPUStatus, policy: SchedulerPolicy) -> int:
    """Report the conservative reservation-based capacity.

    This is the ``auto`` ceiling.  A static request may deliberately select a
    larger value up to ``maximum_workers``; fresh free-VRAM observations still
    control every actual launch.
    """

    if status.total_mib < policy.minimum_total_gpu_memory_mib:
        raise RuntimeError(
            f"GPU total memory {status.total_mib} MiB is below locked minimum "
            f"{policy.minimum_total_gpu_memory_mib} MiB"
        )
    usable = status.total_mib - policy.gpu_reserve_mib
    return max(0, min(policy.maximum_workers, usable // policy.per_job_reservation_mib))


def resolve_worker_limit(
    request: WorkerRequest,
    status: GPUStatus,
    policy: SchedulerPolicy,
) -> int:
    """Resolve ``auto`` or validate a static width against the locked ceiling."""

    parsed = parse_worker_request(request)
    conservative = worker_capacity(status, policy)
    if conservative < 1:
        raise RuntimeError(
            "GPU cannot fit one reserved job while retaining the locked memory reserve"
        )
    if parsed == "auto":
        # Auto fills the GPU according to the prospectively locked reservation
        # without exceeding the campaign ceiling.  The live gate remains the
        # final decision before every launch.
        return conservative
    requested = int(parsed)
    if requested > policy.maximum_workers:
        raise RuntimeError(
            f"requested {requested} workers exceeds locked ceiling "
            f"{policy.maximum_workers}"
        )
    return requested


def query_gpu_status(device: int = 0) -> GPUStatus:
    """Read total and free VRAM for exactly one physical GPU."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(device)}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"unexpected nvidia-smi memory response: {rows!r}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 2 or any(not value.isdigit() for value in values):
        raise RuntimeError(f"unexpected nvidia-smi memory response: {rows!r}")
    return GPUStatus(total_mib=int(values[0]), free_mib=int(values[1]))


def query_gpu_process_ids(device: int = 0) -> frozenset[int]:
    """Return PIDs with a live compute context on exactly one physical GPU."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(device)}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) == 1 and rows[0].lower().startswith("no running processes"):
        return frozenset()
    if any(not row.isdigit() or int(row) < 1 for row in rows):
        raise RuntimeError(f"unexpected nvidia-smi compute-process response: {rows!r}")
    return frozenset(int(row) for row in rows)


def launch_is_admissible(status: GPUStatus, policy: SchedulerPolicy) -> bool:
    """Return whether one additional process may be launched now."""

    if status.total_mib < policy.minimum_total_gpu_memory_mib:
        raise RuntimeError(
            f"GPU total memory {status.total_mib} MiB is below locked minimum "
            f"{policy.minimum_total_gpu_memory_mib} MiB"
        )
    return status.free_mib >= (
        policy.gpu_reserve_mib + policy.per_job_reservation_mib
    )


def _safe_log_handle(path: Path) -> Any:
    """Open an append-only operational log without following a final symlink."""

    parent = path.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"unsafe unit-log directory: {parent}")
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"unit log is not a singly linked regular file: {path}")
        return os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _default_launch(
    spec: UnitSpec,
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
) -> tuple[Any, Any]:
    handle = _safe_log_handle(spec.log_path)
    try:
        process = subprocess.Popen(
            list(spec.command),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        handle.close()
        raise
    return process, handle


class _SignalLatch:
    """Temporarily convert INT/TERM into a scheduler-owned stop request."""

    def __init__(self) -> None:
        self.signum: int | None = None
        self._previous: dict[int, Any] = {}
        self._handler = self._record

    def _record(self, signum: int, _frame: Any) -> None:
        if self.signum is None:
            self.signum = int(signum)

    def __enter__(self) -> "_SignalLatch":
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "exact-reference scheduler must run in the main thread so "
                "SIGINT/SIGTERM cleanup is enforceable"
            )
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
        except BaseException:
            for installed, previous in self._previous.items():
                signal.signal(installed, previous)
            self._previous.clear()
            raise
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)


def _gpu_pid_belongs_to(row: _Active, gpu_pid: int) -> bool:
    """Authenticate a GPU PID as the unit leader or one of its descendants."""

    leader = int(row.process.pid)
    if gpu_pid == leader:
        return True
    if not row.owns_process_group:
        return False
    try:
        return os.getpgid(gpu_pid) == leader
    except (OSError, ProcessLookupError):
        # A short-lived GPU process may disappear between nvidia-smi and this
        # lookup.  It is safer to leave the startup unconfirmed and retry.
        return False


def _validated_gpu_pids(values: Any) -> frozenset[int]:
    try:
        rows = tuple(values)
    except TypeError as error:
        raise RuntimeError("GPU process probe did not return an iterable") from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in rows
    ):
        raise RuntimeError("GPU process probe returned an invalid PID")
    return frozenset(rows)


def _signal_active(row: _Active, signum: int) -> str | None:
    """Signal one owned group (or one injected test process) best-effort."""

    try:
        if row.owns_process_group:
            os.killpg(int(row.process.pid), signum)
        elif signum == signal.SIGTERM and hasattr(row.process, "terminate"):
            row.process.terminate()
        elif signum == signal.SIGKILL and hasattr(row.process, "kill"):
            row.process.kill()
        elif hasattr(row.process, "send_signal"):
            row.process.send_signal(signum)
        else:
            return f"unit {row.spec.unit_id} process cannot be signalled"
    except ProcessLookupError:
        return None
    except BaseException as error:
        return f"{row.spec.unit_id} signal {signum}: {type(error).__name__}: {error}"
    return None


def _close_log(row: _Active) -> str | None:
    try:
        row.log_handle.close()
    except BaseException as error:
        return f"{row.spec.unit_id} log close: {type(error).__name__}: {error}"
    return None


def _kill_finished_process_group(row: _Active) -> str | None:
    """Remove descendants left behind after a unit leader has exited."""

    if not row.owns_process_group:
        return None
    try:
        # The leader has already been reaped.  Anything still in its dedicated
        # session is an orphan rather than part of a valid completed unit.
        os.killpg(int(row.process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return None
    except BaseException as error:
        return (
            f"{row.spec.unit_id} orphan cleanup: "
            f"{type(error).__name__}: {error}"
        )
    return None


def _terminate_and_reap(
    active: Mapping[str, _Active],
    *,
    grace_seconds: float,
) -> list[str]:
    """TERM all units together, KILL survivors, wait, and close every log."""

    rows = list(active.values())
    errors = [
        error
        for row in rows
        if (error := _signal_active(row, signal.SIGTERM)) is not None
    ]
    deadline = time.monotonic() + grace_seconds
    for row in rows:
        try:
            row.process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            errors.append(
                f"{row.spec.unit_id} TERM wait: {type(error).__name__}: {error}"
            )

    # Signal every owned group, even if its leader returned during the TERM
    # grace: a grandchild may otherwise survive after the leader is reaped.
    for row in rows:
        leader_running = False
        try:
            leader_running = row.process.poll() is None
        except BaseException as error:
            errors.append(
                f"{row.spec.unit_id} poll after TERM: {type(error).__name__}: {error}"
            )
            leader_running = True
        if row.owns_process_group or leader_running:
            error = _signal_active(row, signal.SIGKILL)
            if error is not None:
                errors.append(error)

    kill_deadline = time.monotonic() + 5.0
    for row in rows:
        try:
            row.process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
        except BaseException as error:
            errors.append(
                f"{row.spec.unit_id} KILL wait: {type(error).__name__}: {error}"
            )
        close_error = _close_log(row)
        if close_error is not None:
            errors.append(close_error)
    return errors


def run_units(
    specs: Sequence[UnitSpec],
    *,
    policy: SchedulerPolicy,
    workers: WorkerRequest,
    cwd: Path,
    validate_if_complete: Callable[[str], bool],
    validate_after_success: Callable[[str], None],
    env: Mapping[str, str] | None = None,
    state_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    gpu_probe: Callable[[int], GPUStatus] = query_gpu_status,
    gpu_process_probe: Callable[[int], Any] | None = None,
    launcher: Callable[
        [UnitSpec, Path, Mapping[str, str] | None], tuple[Any, Any]
    ]
    | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SchedulerResult:
    """Run all unsealed units concurrently and validate full coverage.

    ``validate_if_complete`` must return ``False`` only when no completion seal
    exists; if a seal exists but is invalid it must raise.  This makes restart
    behavior fail closed.  After the first process/validation/launch failure,
    no additional unit is launched.  Already-running trajectories are allowed
    to finish and seal, then :class:`UnitExecutionError` is raised.

    Production children are considered to have left CPU-only startup after
    either their leader or a descendant in their dedicated process group is
    reported by ``nvidia-smi``.  The injected ``launcher`` is a test seam and
    is treated as already CUDA-confirmed unless ``gpu_process_probe`` is also
    injected explicitly.
    """

    cwd = cwd.resolve(strict=True)
    identifiers = [spec.unit_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("unit inventory contains duplicate identifiers")

    initial_gpu = gpu_probe(policy.gpu_device)
    worker_limit = resolve_worker_limit(workers, initial_gpu, policy)
    worker_label = str(parse_worker_request(workers))

    completed: set[str] = set()
    for spec in specs:
        if validate_if_complete(spec.unit_id):
            completed.add(spec.unit_id)
    previously_completed = len(completed)
    pending = [spec for spec in specs if spec.unit_id not in completed]
    active: dict[str, _Active] = {}
    failures: list[dict[str, Any]] = []
    deferred_error: Exception | None = None
    last_launch = float("-inf")
    last_gpu = initial_gpu
    enforce_startup_gate = launcher is None or gpu_process_probe is not None
    startup_probe = query_gpu_process_ids if gpu_process_probe is None else gpu_process_probe

    def publish(stage: str, **details: Any) -> None:
        nonlocal deferred_error
        if state_callback is None:
            return
        try:
            state_callback(
                stage,
                {
                    "total_units": len(specs),
                    "completed_units": len(completed),
                    "pending_units": len(pending),
                    "active_units": [
                        {
                            "unit_id": row.spec.unit_id,
                            "pid": int(row.process.pid),
                            "log": str(row.spec.log_path),
                            "started_monotonic": row.started_monotonic,
                            "cuda_confirmed": row.cuda_confirmed,
                        }
                        for row in active.values()
                    ],
                    "worker_request": worker_label,
                    "worker_limit": worker_limit,
                    "gpu_total_mib": last_gpu.total_mib,
                    "gpu_free_mib": last_gpu.free_mib,
                    "gpu_reserve_mib": policy.gpu_reserve_mib,
                    "per_job_reservation_mib": policy.per_job_reservation_mib,
                    "maximum_concurrent_startups": (
                        policy.maximum_concurrent_startups
                    ),
                    "unconfirmed_startups": sum(
                        not row.cuda_confirmed for row in active.values()
                    ),
                    **details,
                },
            )
        except Exception as error:
            # Queue-state persistence is part of the operational contract.  A
            # write failure closes admission, but already-running scientific
            # units are still allowed to finish and seal before it is raised.
            if deferred_error is None:
                deferred_error = error

    def start(spec: UnitSpec) -> _Active:
        # Take the timestamp before Popen so a failing clock cannot strand a
        # successfully-created child outside ``active``.
        started = monotonic()
        if launcher is None:
            process, handle = _default_launch(spec, cwd=cwd, env=env)
        else:
            process, handle = launcher(spec, cwd, env)
        pid = getattr(process, "pid", None)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            row = _Active(
                spec=spec,
                process=process,
                log_handle=handle,
                started_monotonic=started,
                owns_process_group=launcher is None,
            )
            _signal_active(row, signal.SIGKILL)
            _close_log(row)
            raise RuntimeError("launcher returned a process without a valid PID")
        return _Active(
            spec=spec,
            process=process,
            log_handle=handle,
            started_monotonic=started,
            owns_process_group=launcher is None,
            cuda_confirmed=not enforce_startup_gate,
        )

    with _SignalLatch() as stop:
        try:
            while pending or active:
                if stop.signum is not None:
                    publish("INTERRUPTING", signal_number=stop.signum)
                    raise SchedulerInterrupted(stop.signum)

                # Reap first.  A nonzero exit or invalid/missing PASS seal
                # closes admission immediately; running trajectories drain.
                for unit_id, row in list(active.items()):
                    try:
                        returncode = row.process.poll()
                    except Exception as error:
                        raise RuntimeError(
                            f"cannot poll unit process {unit_id}"
                        ) from error
                    if returncode is None:
                        continue
                    del active[unit_id]
                    cleanup_error = _kill_finished_process_group(row)
                    close_error = _close_log(row)
                    if cleanup_error is not None or close_error is not None:
                        messages = [
                            value
                            for value in (cleanup_error, close_error)
                            if value is not None
                        ]
                        if deferred_error is None:
                            deferred_error = RuntimeError("; ".join(messages))
                    if returncode != 0:
                        failures.append(
                            {
                                "unit_id": unit_id,
                                "returncode": int(returncode),
                                "log": str(row.spec.log_path),
                            }
                        )
                        continue
                    try:
                        validate_after_success(unit_id)
                    except Exception as error:
                        failures.append(
                            {
                                "unit_id": unit_id,
                                "returncode": 0,
                                "log": str(row.spec.log_path),
                                "validation_error": (
                                    f"{type(error).__name__}: {error}"
                                ),
                            }
                        )
                    else:
                        completed.add(unit_id)

                if stop.signum is not None:
                    publish("INTERRUPTING", signal_number=stop.signum)
                    raise SchedulerInterrupted(stop.signum)

                if failures or deferred_error is not None:
                    if active:
                        publish(
                            "DRAINING_AFTER_FAILURE",
                            failures=failures,
                            infrastructure_error=(
                                None
                                if deferred_error is None
                                else (
                                    f"{type(deferred_error).__name__}: "
                                    f"{deferred_error}"
                                )
                            ),
                        )
                        sleep(policy.poll_seconds)
                        continue
                    if failures:
                        raise UnitExecutionError(failures)
                    assert deferred_error is not None
                    raise deferred_error

                if not pending and not active:
                    break

                # Refresh the startup gate before considering admission.  A
                # grandchild running Mammoth inherits the unit leader's process
                # group, so it confirms the corresponding unit as well.
                starting = [row for row in active.values() if not row.cuda_confirmed]
                if starting:
                    try:
                        gpu_pids = _validated_gpu_pids(
                            startup_probe(policy.gpu_device)
                        )
                    except Exception as error:
                        deferred_error = error
                    else:
                        for row in starting:
                            if any(
                                _gpu_pid_belongs_to(row, gpu_pid)
                                for gpu_pid in gpu_pids
                            ):
                                row.cuda_confirmed = True

                unconfirmed = sum(
                    not row.cuda_confirmed for row in active.values()
                )
                startup_ready = (
                    unconfirmed < policy.maximum_concurrent_startups
                )
                launched: str | None = None
                now = monotonic()
                stagger_ready = (
                    now - last_launch >= policy.launch_stagger_seconds
                )
                if (
                    deferred_error is None
                    and pending
                    and len(active) < worker_limit
                    and startup_ready
                    and stagger_ready
                ):
                    try:
                        # One fresh observation admits at most one launch.
                        last_gpu = gpu_probe(policy.gpu_device)
                        if launch_is_admissible(last_gpu, policy):
                            if stop.signum is not None:
                                publish(
                                    "INTERRUPTING", signal_number=stop.signum
                                )
                                raise SchedulerInterrupted(stop.signum)
                            spec = pending[0]
                            try:
                                row = start(spec)
                            except Exception as error:
                                failures.append(
                                    {
                                        "unit_id": spec.unit_id,
                                        "launch_error": (
                                            f"{type(error).__name__}: {error}"
                                        ),
                                        "log": str(spec.log_path),
                                    }
                                )
                            else:
                                try:
                                    active[spec.unit_id] = row
                                except BaseException:
                                    cleanup_errors = _terminate_and_reap(
                                        {spec.unit_id: row},
                                        grace_seconds=(
                                            policy.shutdown_grace_seconds
                                        ),
                                    )
                                    if cleanup_errors:
                                        raise RuntimeError(
                                            "could not track launched unit and "
                                            "cleanup was incomplete: "
                                            + "; ".join(cleanup_errors)
                                        )
                                    raise
                                pending.pop(0)
                                launched = spec.unit_id
                                last_launch = monotonic()
                    except Exception as error:
                        deferred_error = error

                if stop.signum is not None:
                    publish("INTERRUPTING", signal_number=stop.signum)
                    raise SchedulerInterrupted(stop.signum)

                unconfirmed = sum(
                    not row.cuda_confirmed for row in active.values()
                )
                if failures or deferred_error is not None:
                    stage = "DRAINING_AFTER_FAILURE" if active else "FAILED"
                elif launched is not None:
                    stage = "LAUNCHED_UNIT"
                elif len(active) >= worker_limit:
                    stage = "AT_WORKER_CEILING"
                elif unconfirmed >= policy.maximum_concurrent_startups:
                    stage = "WAITING_FOR_CUDA_STARTUP"
                elif not launch_is_admissible(last_gpu, policy):
                    stage = "WAITING_FOR_GPU_MEMORY"
                else:
                    stage = "WAITING_FOR_LAUNCH_STAGGER"
                publish(stage, launched_unit=launched, failures=failures)

                if pending or active:
                    if (
                        pending
                        and len(active) < worker_limit
                        and unconfirmed < policy.maximum_concurrent_startups
                    ):
                        remaining_stagger = max(
                            0.0,
                            policy.launch_stagger_seconds
                            - (monotonic() - last_launch),
                        )
                        delay = (
                            min(policy.poll_seconds, remaining_stagger)
                            if remaining_stagger
                            else policy.poll_seconds
                        )
                    else:
                        delay = policy.poll_seconds
                    sleep(max(0.001, delay))
        except BaseException as error:
            if active:
                cleanup_errors = _terminate_and_reap(
                    active,
                    grace_seconds=policy.shutdown_grace_seconds,
                )
                active.clear()
                if cleanup_errors and hasattr(error, "add_note"):
                    error.add_note(
                        "scheduler cleanup issue(s): " + "; ".join(cleanup_errors)
                    )
            raise

    # Revalidate every unit at the barrier.  The queue may aggregate only after
    # this succeeds; partial-scientific aggregation is therefore impossible.
    for spec in specs:
        if not validate_if_complete(spec.unit_id):
            raise RuntimeError(f"unit lacks an authenticated completion seal: {spec.unit_id}")
    if len(completed) != len(specs):
        # ``completed`` includes initially valid units plus this invocation's
        # successes.  This is a defensive invariant independent of the final
        # seal revalidation above.
        raise RuntimeError("scheduler ended without full unit coverage")
    result = SchedulerResult(
        total_units=len(specs),
        previously_completed_units=previously_completed,
        newly_completed_units=len(completed) - previously_completed,
        worker_limit=worker_limit,
        worker_request=worker_label,
        gpu_total_mib=initial_gpu.total_mib,
        gpu_reserve_mib=policy.gpu_reserve_mib,
        per_job_reservation_mib=policy.per_job_reservation_mib,
    )
    if not result.full_coverage:
        raise RuntimeError("scheduler result is not full coverage")
    publish("FULL_COVERAGE")
    if deferred_error is not None:
        raise deferred_error
    return result
