"""Equal-severity replay swaps with paired, counter-keyed quadruple selection.

The intervention is deliberately external to the pinned Mammoth checkout.  It
owns its RNG, never consumes a global NumPy/Torch stream, and uses neither
model outputs nor losses when selecting rows.  Every selected quadruple has
two unequal-class rows from each of two historical tasks.  Both experimental
views change all four label values and preserve the exact target multiset.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ARMS = ("within_task", "across_task", "paired_probe")
SCHEMA_ROOT = "gradient_reuse.task_structure"
COUNTER_DOMAIN = "gradient_reuse.task_structure.quadruple_selection.v1"


def add_task_structure_args(parser):
    parser.add_argument("--task_structure_arm", choices=ARMS, required=True)
    parser.add_argument("--task_structure_run_seed", type=int, required=True)
    parser.add_argument("--task_structure_counter_seed", type=int, required=True)
    parser.add_argument("--task_structure_selection_seed_salt", type=int, required=True)
    parser.add_argument("--task_structure_audit_path", type=str, required=True)
    parser.add_argument("--task_structure_gradient_interval", type=int, default=100)
    return parser


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _row_multiset_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().reshape(len(value), -1).numpy())
    rows = sorted(hashlib.sha256(row.tobytes()).digest() for row in array)
    return hashlib.sha256(b"".join(rows)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("task-structure audit forbids non-finite JSON values")
    return value


@dataclass(frozen=True)
class Quadruple:
    task_a: int
    task_b: int
    positions: tuple[int, int, int, int]
    across_reversed: bool


@dataclass(frozen=True)
class TaskStructurePlan:
    within_permutation: torch.Tensor
    across_permutation: torch.Tensor
    selected_positions: torch.Tensor
    quadruples: tuple[Quadruple, ...]
    historical_rows: int
    represented_historical_tasks: tuple[int, ...]
    pair_capacity_by_task: tuple[tuple[int, int], ...]
    ineligible_reason: str | None

    @property
    def selected_quadruples(self) -> int:
        return len(self.quadruples)


def task_ids_from_split_cifar100_labels(labels: torch.Tensor) -> torch.Tensor:
    """Pinned 20-task mapping: task t owns unchanged global labels 5t..5t+4."""

    values = labels.detach().reshape(-1).to(dtype=torch.long)
    if values.numel() == 0 or bool(((values < 0) | (values >= 100)).any().item()):
        raise RuntimeError("Split CIFAR-100 labels must lie in [0,100)")
    return torch.div(values, 5, rounding_mode="floor")


def _maximum_unequal_label_pairs(
    positions: Sequence[int], labels: Sequence[int], priority: Mapping[int, float]
) -> list[tuple[int, int]]:
    """Greedily realize the maximum number of disjoint unequal-label pairs."""

    pools: dict[int, list[int]] = defaultdict(list)
    for position in positions:
        pools[int(labels[position])].append(int(position))
    for pool in pools.values():
        pool.sort(key=lambda row: (priority[row], row), reverse=True)
    pairs: list[tuple[int, int]] = []
    while True:
        available = [label for label, pool in pools.items() if pool]
        if len(available) < 2:
            break
        available.sort(
            key=lambda label: (len(pools[label]), priority[pools[label][-1]], -label),
            reverse=True,
        )
        first, second = available[:2]
        left = pools[first].pop()
        right = pools[second].pop()
        if priority[right] < priority[left]:
            left, right = right, left
        pairs.append((left, right))
    return pairs


def _maximum_cross_task_matching(
    pairs_by_task: Mapping[int, Sequence[tuple[int, int]]],
    task_priority: Mapping[int, float],
    rng: np.random.Generator,
) -> list[Quadruple]:
    pools = {task: list(pairs) for task, pairs in pairs_by_task.items() if pairs}
    result: list[Quadruple] = []
    while True:
        available = [task for task, pool in pools.items() if pool]
        if len(available) < 2:
            break
        available.sort(
            key=lambda task: (len(pools[task]), task_priority[task], -task),
            reverse=True,
        )
        task_a, task_b = available[:2]
        pair_a = pools[task_a].pop()
        pair_b = pools[task_b].pop()
        result.append(
            Quadruple(
                task_a=min(task_a, task_b),
                task_b=max(task_a, task_b),
                positions=(
                    *(pair_a if task_a < task_b else pair_b),
                    *(pair_b if task_a < task_b else pair_a),
                ),
                across_reversed=bool(rng.integers(0, 2)),
            )
        )
    return result


def build_task_structure_plan(
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    current_task: int,
    counter_seed: int,
    run_seed: int,
    selection_seed_salt: int,
    observation: int,
    channel: str = "replay_labels",
) -> TaskStructurePlan:
    target = labels.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    tasks = task_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    if len(target) != len(tasks) or len(target) < 1:
        raise RuntimeError("replay labels/task IDs must be aligned and nonempty")
    if current_task < 0 or observation < 0:
        raise RuntimeError("current task and observation must be nonnegative")
    inferred = task_ids_from_split_cifar100_labels(target)
    if not torch.equal(inferred, tasks):
        raise RuntimeError("buffer task IDs disagree with pinned label//5 mapping")
    if bool((tasks > current_task).any().item()):
        raise RuntimeError("replay buffer contains a future task")

    key = "|".join(
        map(
            str,
            (
                COUNTER_DOMAIN,
                counter_seed,
                run_seed,
                selection_seed_salt,
                observation,
                channel,
            ),
        )
    )
    seed = np.random.SeedSequence(
        [
            int(counter_seed) & 0xFFFFFFFF,
            int(run_seed) & 0xFFFFFFFF,
            int(selection_seed_salt) & 0xFFFFFFFF,
            int(observation) & 0xFFFFFFFF,
            _stable_u64(key) & 0xFFFFFFFF,
            (_stable_u64(key) >> 32) & 0xFFFFFFFF,
        ]
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    row_priority = {row: float(rng.random()) for row in range(len(target))}

    historical = [row for row, task in enumerate(tasks.tolist()) if task < current_task]
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in historical:
        grouped[int(tasks[row].item())].append(row)
    pairs_by_task = {
        task: _maximum_unequal_label_pairs(rows, target.tolist(), row_priority)
        for task, rows in sorted(grouped.items())
    }
    task_priority = {task: float(rng.random()) for task in pairs_by_task}
    quadruples = tuple(_maximum_cross_task_matching(pairs_by_task, task_priority, rng))

    identity = torch.arange(len(target), dtype=torch.long)
    within = identity.clone()
    across = identity.clone()
    selected: list[int] = []
    for quadruple in quadruples:
        a0, a1, b0, b1 = quadruple.positions
        within[[a0, a1, b0, b1]] = torch.tensor([a1, a0, b1, b0])
        cross_sources = [b1, b0, a1, a0] if quadruple.across_reversed else [b0, b1, a0, a1]
        across[[a0, a1, b0, b1]] = torch.tensor(cross_sources)
        selected.extend(quadruple.positions)

    selected_tensor = torch.tensor(sorted(selected), dtype=torch.long)
    ineligible = None
    if not quadruples:
        if not historical:
            ineligible = "no_historical_replay_rows"
        elif len(grouped) < 2:
            ineligible = "fewer_than_two_historical_tasks"
        elif sum(bool(value) for value in pairs_by_task.values()) < 2:
            ineligible = "fewer_than_two_tasks_with_distinct_class_pairs"
        else:
            ineligible = "no_cross_task_pair_matching"

    plan = TaskStructurePlan(
        within_permutation=within,
        across_permutation=across,
        selected_positions=selected_tensor,
        quadruples=quadruples,
        historical_rows=len(historical),
        represented_historical_tasks=tuple(sorted(grouped)),
        pair_capacity_by_task=tuple(
            (task, len(pairs)) for task, pairs in sorted(pairs_by_task.items())
        ),
        ineligible_reason=ineligible,
    )
    validate_plan(target, tasks, current_task=current_task, plan=plan)
    return plan


def validate_plan(
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    current_task: int,
    plan: TaskStructurePlan,
) -> None:
    target = labels.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    tasks = task_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    identity = torch.arange(len(target), dtype=torch.long)
    if sorted(plan.within_permutation.tolist()) != identity.tolist():
        raise RuntimeError("within-task mapping is not a permutation")
    if sorted(plan.across_permutation.tolist()) != identity.tolist():
        raise RuntimeError("across-task mapping is not a permutation")
    selected = set(plan.selected_positions.tolist())
    if len(selected) != 4 * len(plan.quadruples):
        raise RuntimeError("selected quadruples overlap")
    for row in range(len(target)):
        if row not in selected and not (
            plan.within_permutation[row] == row and plan.across_permutation[row] == row
        ):
            raise RuntimeError("an unselected row was reassigned")
    for quadruple in plan.quadruples:
        a0, a1, b0, b1 = quadruple.positions
        if not (
            tasks[a0] == tasks[a1] == quadruple.task_a
            and tasks[b0] == tasks[b1] == quadruple.task_b
            and quadruple.task_a < quadruple.task_b < current_task
            and target[a0] != target[a1]
            and target[b0] != target[b1]
        ):
            raise RuntimeError("selected quadruple violates task/class eligibility")
    within_targets = target[plan.within_permutation]
    across_targets = target[plan.across_permutation]
    if Counter(within_targets.tolist()) != Counter(target.tolist()):
        raise RuntimeError("within-task swap changed the target multiset")
    if Counter(across_targets.tolist()) != Counter(target.tolist()):
        raise RuntimeError("across-task swap changed the target multiset")
    if selected:
        index = torch.tensor(sorted(selected), dtype=torch.long)
        if bool((within_targets[index] == target[index]).any().item()):
            raise RuntimeError("within-task swap failed to change all selected values")
        if bool((across_targets[index] == target[index]).any().item()):
            raise RuntimeError("across-task swap failed to change all selected values")
        if not torch.equal(tasks[plan.within_permutation[index]], tasks[index]):
            raise RuntimeError("within-task target source crossed a task boundary")
        if bool((tasks[plan.across_permutation[index]] == tasks[index]).any().item()):
            raise RuntimeError("across-task target source stayed within task")


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None], parameters: Sequence[torch.Tensor]
) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for gradient, parameter in zip(gradients, parameters)
        ]
    )


def _functional_gradient(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cpu_state: torch.Tensor,
    cuda_state: list[torch.Tensor] | None,
) -> torch.Tensor:
    parameters = OrderedDict(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    torch.set_rng_state(cpu_state.clone())
    if cuda_state is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in cuda_state])
    output = torch.func.functional_call(
        model, (dict(parameters), dict(buffers)), (inputs,)
    )
    loss = objective(output, targets)
    gradients = torch.autograd.grad(
        loss, tuple(parameters.values()), allow_unused=True, retain_graph=False
    )
    return _flatten_gradients(gradients, tuple(parameters.values()))


def cross_entropy_objective(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(output, target)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second))
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(first, second).item() / denominator)


class TaskStructureIntervention:
    def __init__(self, args: Any, method: str) -> None:
        self.arm = str(getattr(args, "task_structure_arm", ""))
        self.method = str(method)
        self.run_seed = int(getattr(args, "task_structure_run_seed", -1))
        self.counter_seed = int(getattr(args, "task_structure_counter_seed", -1))
        self.selection_seed_salt = int(
            getattr(args, "task_structure_selection_seed_salt", -1)
        )
        self.gradient_interval = int(
            getattr(args, "task_structure_gradient_interval", 0)
        )
        self.observation = -1
        self.event = 0
        raw_path = getattr(args, "task_structure_audit_path", None)
        if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
            raise RuntimeError("task-structure audit path must be an absolute path")
        path = Path(raw_path)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("task-structure audit path is symbolic or nonregular")
        self.path = path.resolve()
        if self.arm not in ARMS or self.method != "er_task_structure_swap":
            raise RuntimeError("invalid task-structure arm/model")
        if self.run_seed != int(getattr(args, "seed", -2)):
            raise RuntimeError("task-structure run seed differs from Mammoth seed")
        if min(self.counter_seed, self.selection_seed_salt) < 0:
            raise RuntimeError("task-structure counter values must be nonnegative")
        if self.gradient_interval < 1:
            raise RuntimeError("task-structure gradient interval must be positive")
        if str(getattr(args, "dataset", "")) != "seq-cifar100":
            raise RuntimeError("task-structure study is locked to seq-cifar100")
        if bool(getattr(args, "permute_classes", False)):
            raise RuntimeError("task mapping forbids class permutation")
        if getattr(args, "custom_class_order", None) is not None or getattr(
            args, "custom_task_order", None
        ) is not None:
            raise RuntimeError("task mapping forbids custom class/task order")
        if float(getattr(args, "noise_rate", 0.0)) != 0.0:
            raise RuntimeError("task mapping forbids label noise")
        self.record(
            {
                "schema_version": f"{SCHEMA_ROOT}.effective_args.v1",
                "effective_args": {
                    name: getattr(args, name)
                    for name in vars(args)
                    if name.startswith("task_structure_")
                    or name
                    in {
                        "model",
                        "dataset",
                        "dataset_config",
                        "backbone",
                        "lr",
                        "n_epochs",
                        "seed",
                        "device",
                        "num_workers",
                        "buffer_size",
                        "minibatch_size",
                        "batch_size",
                        "permute_classes",
                        "custom_class_order",
                        "custom_task_order",
                        "noise_rate",
                    }
                },
                "mapping": "task_id=label//5",
            }
        )

    def begin_observation(self) -> None:
        self.observation += 1
        self.event = 0

    @property
    def should_diagnose(self) -> bool:
        return self.arm == "paired_probe" and self.observation % self.gradient_interval == 0

    def record(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise RuntimeError("task-structure audit path became unsafe")
        common = {
            "method": self.method,
            "arm": self.arm,
            "run_seed": self.run_seed,
            "counter_seed": self.counter_seed,
            "selection_seed_salt": self.selection_seed_salt,
        }
        payload = json.dumps(
            _json_safe({**common, **dict(value)}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def no_replay(self, *, current_task: int) -> None:
        self.record(
            {
                "schema_version": f"{SCHEMA_ROOT}.exposure.v1",
                "observation": self.observation,
                "event": self.event,
                "current_task": int(current_task),
                "replay_rows": 0,
                "historical_rows": 0,
                "eligible": False,
                "ineligible_reason": "empty_replay_buffer",
                "selected_quadruples": 0,
                "selected_rows": 0,
                "changed_rows": 0,
            }
        )
        self.event += 1

    def plan(
        self,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        current_task: int,
    ) -> TaskStructurePlan:
        return build_task_structure_plan(
            labels,
            task_ids,
            current_task=current_task,
            counter_seed=self.counter_seed,
            run_seed=self.run_seed,
            selection_seed_salt=self.selection_seed_salt,
            observation=self.observation,
        )

    def apply(
        self, labels: torch.Tensor, plan: TaskStructurePlan
    ) -> torch.Tensor:
        if self.arm == "within_task":
            permutation = plan.within_permutation
        elif self.arm == "across_task":
            permutation = plan.across_permutation
        else:
            return labels
        return labels[permutation.to(device=labels.device)]

    def exposure(
        self,
        labels: torch.Tensor,
        task_ids: torch.Tensor,
        plan: TaskStructurePlan,
        *,
        current_task: int,
        sampled_indexes: torch.Tensor,
        sampled_inputs: torch.Tensor,
    ) -> None:
        within = labels[plan.within_permutation.to(labels.device)]
        across = labels[plan.across_permutation.to(labels.device)]
        used = self.apply(labels, plan)
        selected = plan.selected_positions.to(labels.device)
        changed = int((used != labels).sum().item())
        expected = 0 if self.arm == "paired_probe" else int(selected.numel())
        if changed != expected:
            raise RuntimeError("live target change count violates the locked arm")
        self.record(
            {
                "schema_version": f"{SCHEMA_ROOT}.exposure.v1",
                "observation": self.observation,
                "event": self.event,
                "current_task": int(current_task),
                "replay_rows": int(len(labels)),
                "historical_rows": plan.historical_rows,
                "represented_historical_tasks": plan.represented_historical_tasks,
                "pair_capacity_by_task": plan.pair_capacity_by_task,
                "eligible": bool(plan.quadruples),
                "ineligible_reason": plan.ineligible_reason,
                "selected_quadruples": plan.selected_quadruples,
                "selected_rows": int(selected.numel()),
                "changed_rows": changed,
                "selected_fraction": float(selected.numel() / len(labels)),
                "same_selected_rows_for_both_swap_arms": True,
                "all_selected_values_change_within": bool(
                    selected.numel() == 0
                    or (within[selected] != labels[selected]).all().item()
                ),
                "all_selected_values_change_across": bool(
                    selected.numel() == 0
                    or (across[selected] != labels[selected]).all().item()
                ),
                "target_multiset_preserved_within": _row_multiset_sha256(within)
                == _row_multiset_sha256(labels),
                "target_multiset_preserved_across": _row_multiset_sha256(across)
                == _row_multiset_sha256(labels),
                "sampled_indexes_sha256": _tensor_sha256(sampled_indexes),
                "input_sha256": _tensor_sha256(sampled_inputs),
                "labels_sha256": _tensor_sha256(labels),
                "task_ids_sha256": _tensor_sha256(task_ids),
                "selected_positions_sha256": _tensor_sha256(plan.selected_positions),
                "within_permutation_sha256": _tensor_sha256(plan.within_permutation),
                "across_permutation_sha256": _tensor_sha256(plan.across_permutation),
                "within_targets_sha256": _tensor_sha256(within),
                "across_targets_sha256": _tensor_sha256(across),
                "used_targets_sha256": _tensor_sha256(used),
            }
        )
        self.event += 1

    def gradient_diagnostic(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        plan: TaskStructurePlan,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        sampled_indexes: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not self.should_diagnose or not plan.quadruples:
            return None
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        training = model.training
        parameter_grads = {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        }
        within = labels[plan.within_permutation.to(labels.device)]
        across = labels[plan.across_permutation.to(labels.device)]
        try:
            true_gradient = _functional_gradient(
                model, inputs, labels, objective, cpu_before, cuda_before
            )
            within_gradient = _functional_gradient(
                model, inputs, within, objective, cpu_before, cuda_before
            )
            across_gradient = _functional_gradient(
                model, inputs, across, objective, cpu_before, cuda_before
            )
        finally:
            torch.set_rng_state(cpu_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state_all(cuda_before)
            model.train(training)
            for name, value in model.named_parameters():
                saved = parameter_grads[name]
                value.grad = None if saved is None else saved
        record = {
            "schema_version": f"{SCHEMA_ROOT}.common_state_gradient.v1",
            "observation": self.observation,
            "event": self.event,
            "rows": int(len(labels)),
            "selected_quadruples": plan.selected_quadruples,
            "selected_rows": int(plan.selected_positions.numel()),
            "gradient_definition": "raw_full_replay_mean_cross_entropy_parameter_gradient_without_direction_normalization",
            "common_parameter_state": True,
            "training_update_uses_true_targets": True,
            "inference_role": "descriptive_only",
            "true_norm": float(torch.linalg.vector_norm(true_gradient).item()),
            "within_norm": float(torch.linalg.vector_norm(within_gradient).item()),
            "across_norm": float(torch.linalg.vector_norm(across_gradient).item()),
            "true_within_distance": float(
                torch.linalg.vector_norm(true_gradient - within_gradient).item()
            ),
            "true_across_distance": float(
                torch.linalg.vector_norm(true_gradient - across_gradient).item()
            ),
            "across_within_distance": float(
                torch.linalg.vector_norm(across_gradient - within_gradient).item()
            ),
            "true_within_dot": float(torch.dot(true_gradient, within_gradient).item()),
            "true_across_dot": float(torch.dot(true_gradient, across_gradient).item()),
            "across_within_dot": float(torch.dot(across_gradient, within_gradient).item()),
            "true_within_cosine": _cosine(true_gradient, within_gradient),
            "true_across_cosine": _cosine(true_gradient, across_gradient),
            "across_within_cosine": _cosine(across_gradient, within_gradient),
            "sampled_indexes_sha256": _tensor_sha256(sampled_indexes),
            "labels_sha256": _tensor_sha256(labels),
            "selected_positions_sha256": _tensor_sha256(plan.selected_positions),
        }
        self.record(record)
        self.event += 1
        return record


__all__ = [
    "ARMS",
    "Quadruple",
    "TaskStructureIntervention",
    "TaskStructurePlan",
    "add_task_structure_args",
    "build_task_structure_plan",
    "cross_entropy_objective",
    "task_ids_from_split_cifar100_labels",
    "validate_plan",
]
