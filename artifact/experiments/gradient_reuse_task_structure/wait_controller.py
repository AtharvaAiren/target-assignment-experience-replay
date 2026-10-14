"""Wait for earlier GPU campaigns, then exec the task-structure queue."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

from .integrity import atomic_json, real_directory
from .protocol import load_lock, strict_json, validate_prerequisite_payload


SCHEMA = "gradient_reuse.task_structure_wait_controller.v1"


def prerequisite_status(
    path: Path, campaign: str
) -> tuple[str, dict[str, Any] | None]:
    if path.is_symlink():
        raise RuntimeError(f"prerequisite sentinel is symbolic: {path}")
    if not path.exists():
        return "ABSENT", None
    if not path.is_file():
        raise RuntimeError(f"prerequisite sentinel is not a file: {path}")
    value = validate_prerequisite_payload(strict_json(path), campaign)
    return "PASS", value


def run(arguments: argparse.Namespace) -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError("wait controller requires Python -B")
    project = real_directory(arguments.project, "task-structure project")
    lock = load_lock(arguments.lock)
    configured = lock["queue_prerequisites"]
    expected = {
        "assignment_strength": Path(configured["assignment_strength_done"]),
        "mean_matched": Path(configured["mean_matched_done"]),
    }
    provided = {
        "assignment_strength": arguments.assignment_strength_done,
        "mean_matched": arguments.mean_matched_done,
    }
    if {name: str(path) for name, path in provided.items()} != {
        name: str(path) for name, path in expected.items()
    }:
        raise RuntimeError("wait-controller prerequisite path differs from the lock")
    out = arguments.out.expanduser()
    if not out.is_absolute() or out.is_symlink():
        raise RuntimeError("task-structure output must be an absolute nonsymbolic path")
    out.mkdir(parents=True, exist_ok=True)
    state = out / "QUEUE_CONTROL_STATE.json"
    started = time.time()

    def update(status: str, stage: str, details: dict[str, Any]) -> None:
        atomic_json(
            state,
            {
                "schema_version": SCHEMA,
                "status": status,
                "stage": stage,
                "controller_pid": os.getpid(),
                "started_unix": started,
                "updated_unix": time.time(),
                "details": details,
            },
        )

    try:
        while True:
            statuses = {
                name: prerequisite_status(path, name)[0]
                for name, path in provided.items()
            }
            missing = sorted(name for name, status in statuses.items() if status != "PASS")
            if not missing:
                break
            update(
                "WAITING",
                "WAITING_FOR_GPU_CAMPAIGNS",
                {
                    "prerequisite_status": statuses,
                    "no_task_structure_gpu_process_started": True,
                    "poll_seconds": arguments.poll_seconds,
                },
            )
            time.sleep(arguments.poll_seconds)

        command = [
            str(arguments.python), "-B", "-m", "experiments.gradient_reuse_task_structure.queue",
            "--project", str(project), "--lock", str(arguments.lock),
            "--source-manifest", str(arguments.source_manifest),
            "--mammoth-root", str(arguments.mammoth_root),
            "--data-root", str(arguments.data_root),
            "--data-manifest", str(arguments.data_manifest),
            "--environment-manifest", str(arguments.environment_manifest),
            "--assignment-strength-done", str(arguments.assignment_strength_done),
            "--mean-matched-done", str(arguments.mean_matched_done),
            "--python", str(arguments.python), "--out", str(arguments.out),
            "--device", "cuda:0", "--max-processes", str(arguments.max_processes),
        ]
        update(
            "RUNNING",
            "RUNNING_TASK_STRUCTURE_QUEUE",
            {"prerequisite_status": {name: "PASS" for name in provided}},
        )
        completed = subprocess.run(
            command,
            cwd=project,
            env={**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"task-structure queue exited {completed.returncode}")
        update("PASS", "COMPLETE", {"queue_done": str(arguments.out / "DONE.json")})
        return 0
    except BaseException as error:
        update(
            "FAILED",
            "FAILED",
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--lock", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--mammoth-root", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--data-manifest", type=Path, required=True)
    result.add_argument("--environment-manifest", type=Path, required=True)
    result.add_argument("--assignment-strength-done", type=Path, required=True)
    result.add_argument("--mean-matched-done", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--max-processes", type=int, default=2)
    result.add_argument("--poll-seconds", type=int, default=60)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
