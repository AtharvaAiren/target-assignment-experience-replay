"""Exact current-loss coefficient for a current-only ER reference arm.

The reference arm deliberately does not instantiate or sample a replay buffer.
It instead shadows only the two pieces of ER state needed to recover the
coefficient on the current-example mean loss: the number of examples observed
before the update and the resulting buffer occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, TextIO

import torch


HEADER_SCHEMA = "gradient_reuse.exact_current_coefficient.header.v1"
UPDATE_SCHEMA = "gradient_reuse.exact_current_coefficient.update.v1"
SEMANTICS = (
    "current-only mean cross-entropy multiplied at every update by "
    "b_current/(b_current+b_replay), with b_replay equal to the "
    "canonical ER draw cardinality before adding the current batch"
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if integer <= 0 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def reference_replay_rows(
    seen_examples: int,
    reference_buffer_size: int,
    reference_minibatch_size: int,
) -> int:
    """Return the number of replay rows canonical ER would draw now."""

    if isinstance(seen_examples, bool) or not isinstance(seen_examples, int):
        raise ValueError("seen_examples must be a nonnegative integer")
    if seen_examples < 0:
        raise ValueError("seen_examples must be a nonnegative integer")
    buffer_size = _positive_int(reference_buffer_size, "reference_buffer_size")
    minibatch_size = _positive_int(
        reference_minibatch_size, "reference_minibatch_size"
    )
    if seen_examples == 0:
        return 0
    occupancy = min(seen_examples, buffer_size)
    return min(minibatch_size, occupancy)


@dataclass(frozen=True)
class CoefficientPlan:
    """Immutable coefficient plan for exactly one optimizer update."""

    update: int
    b_current: int
    seen_examples_before: int
    buffer_occupancy_before: int
    b_replay: int
    coefficient_numerator: int
    coefficient_denominator: int

    @property
    def coefficient(self) -> float:
        return self.coefficient_numerator / self.coefficient_denominator


def coefficient_plan(
    *,
    update: int,
    b_current: int,
    seen_examples: int,
    reference_buffer_size: int,
    reference_minibatch_size: int,
) -> CoefficientPlan:
    """Construct the exact current-loss coefficient for one ER update."""

    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError("update must be a nonnegative integer")
    current = _positive_int(b_current, "b_current")
    buffer_size = _positive_int(reference_buffer_size, "reference_buffer_size")
    replay = reference_replay_rows(
        seen_examples, buffer_size, reference_minibatch_size
    )
    denominator = current + replay
    return CoefficientPlan(
        update=update,
        b_current=current,
        seen_examples_before=seen_examples,
        buffer_occupancy_before=min(seen_examples, buffer_size),
        b_replay=replay,
        coefficient_numerator=current,
        coefficient_denominator=denominator,
    )


def assert_plain_sgd(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Reject optimizer semantics outside the locked plain-SGD contract."""

    if type(optimizer) is not torch.optim.SGD:
        raise RuntimeError("exact coefficient control requires torch.optim.SGD")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("exact coefficient control requires one SGD parameter group")
    group = optimizer.param_groups[0]
    forbidden = {
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "fused": None,
        "differentiable": False,
    }
    for name, expected in forbidden.items():
        observed = group.get(name, optimizer.defaults.get(name))
        if observed != expected:
            raise RuntimeError(
                f"exact coefficient control requires plain SGD: "
                f"{name}={observed!r}, expected {expected!r}"
            )
    learning_rate = group.get("lr")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise RuntimeError("exact coefficient control requires a scalar SGD learning rate")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise RuntimeError("exact coefficient control requires a finite positive SGD learning rate")
    return {
        "optimizer": "torch.optim.SGD",
        "parameter_groups": 1,
        "learning_rate": float(learning_rate),
        **forbidden,
    }


def assert_mean_cross_entropy_objective(objective: Any) -> dict[str, Any]:
    """Lock the supplied Mammoth objective to unweighted mean CE.

    The pinned PMNIST and Split CIFAR-100 datasets supply
    ``torch.nn.functional.cross_entropy``.  Supporting the equivalent module
    makes the contract directly testable without changing its semantics.
    """

    if objective is torch.nn.functional.cross_entropy:
        pass
    elif type(objective) is torch.nn.CrossEntropyLoss:
        if objective.reduction != "mean":
            raise RuntimeError("exact coefficient control requires mean cross-entropy")
        if objective.weight is not None:
            raise RuntimeError("exact coefficient control forbids class-weighted cross-entropy")
        if int(objective.ignore_index) != -100:
            raise RuntimeError("exact coefficient control requires the default CE ignore_index")
        if float(getattr(objective, "label_smoothing", 0.0)) != 0.0:
            raise RuntimeError("exact coefficient control forbids CE label smoothing")
    else:
        raise RuntimeError(
            "exact coefficient control requires the canonical mean "
            "cross-entropy objective"
        )
    return {
        "objective": "unweighted_cross_entropy",
        "reduction": "mean",
        "ignore_index": -100,
        "label_smoothing": 0.0,
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _open_new_audit(path: Path) -> TextIO:
    if not path.is_absolute():
        raise ValueError("exact coefficient audit path must be absolute")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("exact coefficient audit parent must be a real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError("exact coefficient audit is not a singly linked regular file")
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


class ExactCurrentCoefficientAudit:
    """Shadow ER occupancy and append a hash-chained record per update."""

    def __init__(
        self,
        args: Any,
        model_name: str,
        *,
        optimizer_contract: Mapping[str, Any],
        objective_contract: Mapping[str, Any],
    ) -> None:
        self.reference_buffer_size = _positive_int(
            getattr(args, "reference_buffer_size"), "reference_buffer_size"
        )
        self.reference_minibatch_size = _positive_int(
            getattr(args, "reference_minibatch_size"),
            "reference_minibatch_size",
        )
        raw_path = getattr(args, "exact_coefficient_audit_path", None)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("exact_coefficient_audit_path is required")
        self.path = Path(raw_path).expanduser()
        self._stream = _open_new_audit(self.path)
        self._seen_examples = 0
        self._update = 0
        self._previous_record_sha256: str | None = None
        header = {
            "schema_version": HEADER_SCHEMA,
            "model": model_name,
            "dataset": getattr(args, "dataset", None),
            "seed": int(getattr(args, "seed")),
            "reference_buffer_size": self.reference_buffer_size,
            "reference_minibatch_size": self.reference_minibatch_size,
            "optimizer_contract": dict(optimizer_contract),
            "objective_contract": dict(objective_contract),
            "semantics": SEMANTICS,
        }
        self._write(header)

    @property
    def seen_examples(self) -> int:
        return self._seen_examples

    @property
    def update(self) -> int:
        return self._update

    def plan(self, b_current: int) -> CoefficientPlan:
        return coefficient_plan(
            update=self._update,
            b_current=b_current,
            seen_examples=self._seen_examples,
            reference_buffer_size=self.reference_buffer_size,
            reference_minibatch_size=self.reference_minibatch_size,
        )

    def complete(
        self,
        plan: CoefficientPlan,
        *,
        current_mean_loss: float,
        scaled_loss: float,
    ) -> None:
        if plan.update != self._update or plan.seen_examples_before != self._seen_examples:
            raise RuntimeError("stale or out-of-order exact coefficient plan")
        for name, value in {
            "current_mean_loss": current_mean_loss,
            "scaled_loss": scaled_loss,
        }.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"{name} must be finite")
        expected_scaled = float(current_mean_loss) * plan.coefficient
        tolerance = 1.0e-6 * max(1.0, abs(expected_scaled))
        if abs(float(scaled_loss) - expected_scaled) > tolerance:
            raise RuntimeError("scaled loss does not match the locked coefficient")

        seen_after = self._seen_examples + plan.b_current
        record = {
            "schema_version": UPDATE_SCHEMA,
            "update": plan.update,
            "b_current": plan.b_current,
            "seen_examples_before": plan.seen_examples_before,
            "buffer_occupancy_before": plan.buffer_occupancy_before,
            "b_replay": plan.b_replay,
            "reference_total_rows": plan.coefficient_denominator,
            "coefficient_numerator": plan.coefficient_numerator,
            "coefficient_denominator": plan.coefficient_denominator,
            "current_loss_coefficient": plan.coefficient,
            "current_mean_loss": float(current_mean_loss),
            "scaled_loss": float(scaled_loss),
            "seen_examples_after": seen_after,
            "buffer_occupancy_after": min(
                seen_after, self.reference_buffer_size
            ),
        }
        self._write(record)
        self._seen_examples = seen_after
        self._update += 1

    def _write(self, payload: Mapping[str, Any]) -> None:
        chained = {
            **dict(payload),
            "previous_record_sha256": self._previous_record_sha256,
        }
        record_sha256 = _digest(chained)
        record = {**chained, "record_sha256": record_sha256}
        self._stream.write(_canonical_json(record) + "\n")
        self._stream.flush()
        self._previous_record_sha256 = record_sha256

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __del__(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is not None and not stream.closed:
            stream.close()


def add_exact_current_coefficient_args(parser: Any) -> Any:
    parser.add_argument("--reference_buffer_size", type=int, required=True)
    parser.add_argument("--reference_minibatch_size", type=int, required=True)
    parser.add_argument("--exact_coefficient_audit_path", type=str, required=True)
    return parser


def _strict_json_object(line: str, line_number: int) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(
                    f"duplicate JSON key on exact coefficient audit line {line_number}: {key}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            line,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"invalid exact coefficient audit JSON on line {line_number}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(
            f"exact coefficient audit line {line_number} is not an object"
        )
    if line != _canonical_json(value):
        raise RuntimeError(
            f"exact coefficient audit line {line_number} is not canonical JSON"
        )
    return value


def _exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{name} must be finite")
    return numeric


def summarize_exact_coefficient_audit(
    path: Path,
    *,
    expected_header: Mapping[str, Any],
    expected_updates: int | None = None,
    expected_seen_examples: int | None = None,
) -> dict[str, Any]:
    """Authenticate and summarize a completed exact-coefficient JSONL audit.

    This validator is intentionally independent of the training process.  A
    unit runner must call it after Mammoth exits and before writing a PASS
    seal.  Optional final counts allow a campaign to bind the audit to a known
    loader length in addition to its internally checked recurrence.
    """

    audit_path = Path(path)
    if audit_path.is_symlink() or not audit_path.is_file():
        raise RuntimeError("exact coefficient audit is absent, symbolic, or nonregular")
    status = audit_path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise RuntimeError("exact coefficient audit is not a singly linked regular file")
    audit_bytes = audit_path.read_bytes()
    if not audit_bytes:
        raise RuntimeError("exact coefficient audit is empty")
    if not audit_bytes.endswith(b"\n"):
        raise RuntimeError("exact coefficient audit lacks its final line terminator")
    try:
        audit_text = audit_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("exact coefficient audit is not UTF-8") from error
    raw_lines = audit_text[:-1].split("\n")
    rows = [
        _strict_json_object(line, index)
        for index, line in enumerate(raw_lines, start=1)
    ]

    required_expected_fields = {
        "model",
        "dataset",
        "seed",
        "reference_buffer_size",
        "reference_minibatch_size",
        "learning_rate",
    }
    if set(expected_header) != required_expected_fields:
        raise RuntimeError(
            "expected exact coefficient header must contain exactly: "
            + ", ".join(sorted(required_expected_fields))
        )
    expected_seed = _exact_int(expected_header["seed"], "expected seed")
    expected_model = expected_header["model"]
    expected_dataset = expected_header["dataset"]
    if not isinstance(expected_model, str) or not expected_model:
        raise RuntimeError("expected exact coefficient model must be a nonempty string")
    if not isinstance(expected_dataset, str) or not expected_dataset:
        raise RuntimeError("expected exact coefficient dataset must be a nonempty string")
    expected_learning_rate = _finite_number(
        expected_header["learning_rate"], "expected learning_rate"
    )
    if expected_learning_rate <= 0.0:
        raise RuntimeError("expected learning_rate must be positive")
    expected_optimizer = {
        "optimizer": "torch.optim.SGD",
        "parameter_groups": 1,
        "learning_rate": expected_learning_rate,
        "momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "fused": None,
        "differentiable": False,
    }
    expected_objective = {
        "objective": "unweighted_cross_entropy",
        "reduction": "mean",
        "ignore_index": -100,
        "label_smoothing": 0.0,
    }
    locked_header = {
        "schema_version": HEADER_SCHEMA,
        "model": expected_model,
        "dataset": expected_dataset,
        "seed": expected_seed,
        "reference_buffer_size": _positive_int(
            expected_header["reference_buffer_size"],
            "expected_reference_buffer_size",
        ),
        "reference_minibatch_size": _positive_int(
            expected_header["reference_minibatch_size"],
            "expected_reference_minibatch_size",
        ),
        "optimizer_contract": expected_optimizer,
        "objective_contract": expected_objective,
        "semantics": SEMANTICS,
        "previous_record_sha256": None,
    }

    previous_digest: str | None = None
    for line_number, row in enumerate(rows, start=1):
        supplied_digest = row.get("record_sha256")
        if not isinstance(supplied_digest, str) or len(supplied_digest) != 64:
            raise RuntimeError(
                f"bad exact coefficient record digest on line {line_number}"
            )
        payload = {key: value for key, value in row.items() if key != "record_sha256"}
        if payload.get("previous_record_sha256") != previous_digest:
            raise RuntimeError(
                f"broken exact coefficient hash chain on line {line_number}"
            )
        if _digest(payload) != supplied_digest:
            raise RuntimeError(
                f"bad exact coefficient record digest on line {line_number}"
            )
        previous_digest = supplied_digest

    header = rows[0]
    header_payload = {
        key: value for key, value in header.items() if key != "record_sha256"
    }
    if not isinstance(header_payload.get("model"), str):
        raise RuntimeError("exact coefficient audit model is not a string")
    if not isinstance(header_payload.get("dataset"), str):
        raise RuntimeError("exact coefficient audit dataset is not a string")
    _exact_int(header_payload.get("seed"), "audit seed")
    _exact_int(
        header_payload.get("reference_buffer_size"),
        "audit reference_buffer_size",
    )
    _exact_int(
        header_payload.get("reference_minibatch_size"),
        "audit reference_minibatch_size",
    )
    observed_optimizer = header_payload.get("optimizer_contract")
    if not isinstance(observed_optimizer, dict):
        raise RuntimeError("exact coefficient optimizer contract is not an object")
    if set(observed_optimizer) != set(expected_optimizer):
        raise RuntimeError("exact coefficient optimizer contract fields changed")
    if observed_optimizer.get("optimizer") != "torch.optim.SGD":
        raise RuntimeError("exact coefficient optimizer identity changed")
    if _exact_int(
        observed_optimizer.get("parameter_groups"), "optimizer parameter_groups"
    ) != 1:
        raise RuntimeError("exact coefficient optimizer parameter_groups changed")
    observed_learning_rate = _finite_number(
        observed_optimizer.get("learning_rate"), "optimizer learning_rate"
    )
    if observed_learning_rate <= 0.0:
        raise RuntimeError("exact coefficient optimizer learning_rate is not positive")
    for name in ("momentum", "dampening", "weight_decay"):
        if _finite_number(observed_optimizer.get(name), f"optimizer {name}") != 0.0:
            raise RuntimeError(f"exact coefficient optimizer {name} changed")
    for name in ("nesterov", "maximize", "differentiable"):
        if observed_optimizer.get(name) is not False:
            raise RuntimeError(f"exact coefficient optimizer {name} changed")
    for name in ("foreach", "fused"):
        if observed_optimizer.get(name) is not None:
            raise RuntimeError(f"exact coefficient optimizer {name} changed")
    observed_objective = header_payload.get("objective_contract")
    if not isinstance(observed_objective, dict):
        raise RuntimeError("exact coefficient objective contract is not an object")
    if set(observed_objective) != set(expected_objective):
        raise RuntimeError("exact coefficient objective contract fields changed")
    if observed_objective.get("objective") != "unweighted_cross_entropy":
        raise RuntimeError("exact coefficient objective identity changed")
    if observed_objective.get("reduction") != "mean":
        raise RuntimeError("exact coefficient objective reduction changed")
    if _exact_int(observed_objective.get("ignore_index"), "objective ignore_index") != -100:
        raise RuntimeError("exact coefficient objective ignore_index changed")
    if _finite_number(
        observed_objective.get("label_smoothing"), "objective label_smoothing"
    ) != 0.0:
        raise RuntimeError("exact coefficient objective label_smoothing changed")
    if header_payload != locked_header:
        raise RuntimeError("exact coefficient audit header does not match its unit contract")

    updates = rows[1:]
    if not updates:
        raise RuntimeError("exact coefficient audit contains no optimizer updates")

    seen = 0
    total_replay_rows = 0
    minimum_coefficient = 1.0
    maximum_coefficient = 0.0
    maximum_scaled_loss_residual = 0.0
    expected_update_keys = {
        "schema_version",
        "update",
        "b_current",
        "seen_examples_before",
        "buffer_occupancy_before",
        "b_replay",
        "reference_total_rows",
        "coefficient_numerator",
        "coefficient_denominator",
        "current_loss_coefficient",
        "current_mean_loss",
        "scaled_loss",
        "seen_examples_after",
        "buffer_occupancy_after",
        "previous_record_sha256",
        "record_sha256",
    }
    for index, row in enumerate(updates):
        if set(row) != expected_update_keys:
            raise RuntimeError(
                f"unexpected exact coefficient update fields at update {index}"
            )
        if row.get("schema_version") != UPDATE_SCHEMA:
            raise RuntimeError(f"bad exact coefficient update schema at update {index}")
        observed_update = _exact_int(row.get("update"), "update")
        b_current = _exact_int(row.get("b_current"), "b_current")
        if observed_update != index:
            raise RuntimeError("exact coefficient updates are not contiguous")
        plan = coefficient_plan(
            update=index,
            b_current=b_current,
            seen_examples=seen,
            reference_buffer_size=locked_header["reference_buffer_size"],
            reference_minibatch_size=locked_header["reference_minibatch_size"],
        )
        expected_values = {
            "seen_examples_before": plan.seen_examples_before,
            "buffer_occupancy_before": plan.buffer_occupancy_before,
            "b_replay": plan.b_replay,
            "reference_total_rows": plan.coefficient_denominator,
            "coefficient_numerator": plan.coefficient_numerator,
            "coefficient_denominator": plan.coefficient_denominator,
            "seen_examples_after": seen + plan.b_current,
            "buffer_occupancy_after": min(
                seen + plan.b_current,
                locked_header["reference_buffer_size"],
            ),
        }
        for name, expected in expected_values.items():
            if _exact_int(row.get(name), name) != expected:
                raise RuntimeError(
                    f"bad exact coefficient {name} arithmetic at update {index}"
                )
        coefficient = _finite_number(
            row.get("current_loss_coefficient"), "current_loss_coefficient"
        )
        if coefficient != plan.coefficient:
            raise RuntimeError(
                f"bad exact coefficient ratio arithmetic at update {index}"
            )
        current_loss = _finite_number(row.get("current_mean_loss"), "current_mean_loss")
        scaled_loss = _finite_number(row.get("scaled_loss"), "scaled_loss")
        expected_scaled = coefficient * current_loss
        residual = abs(scaled_loss - expected_scaled)
        tolerance = 1.0e-6 * max(1.0, abs(expected_scaled))
        if residual > tolerance:
            raise RuntimeError(
                f"bad exact coefficient scaled-loss relation at update {index}"
            )
        seen += plan.b_current
        total_replay_rows += plan.b_replay
        minimum_coefficient = min(minimum_coefficient, coefficient)
        maximum_coefficient = max(maximum_coefficient, coefficient)
        maximum_scaled_loss_residual = max(maximum_scaled_loss_residual, residual)

    if expected_updates is not None:
        if _exact_int(expected_updates, "expected_updates") != len(updates):
            raise RuntimeError("exact coefficient final update count mismatch")
    if expected_seen_examples is not None:
        if _exact_int(expected_seen_examples, "expected_seen_examples") != seen:
            raise RuntimeError("exact coefficient final example count mismatch")

    return {
        "schema_version": "gradient_reuse.exact_current_coefficient.summary.v1",
        "status": "PASS",
        "updates": len(updates),
        "seen_examples": seen,
        "final_buffer_occupancy": min(
            seen, locked_header["reference_buffer_size"]
        ),
        "total_reference_replay_rows": total_replay_rows,
        "minimum_current_loss_coefficient": minimum_coefficient,
        "maximum_current_loss_coefficient": maximum_coefficient,
        "maximum_scaled_loss_residual": maximum_scaled_loss_residual,
        "final_record_sha256": previous_digest,
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
    }
