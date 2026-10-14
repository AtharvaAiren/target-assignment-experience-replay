"""Matched four-row within/across-domain target swaps for Permuted-MNIST."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ARMS = ("no_replay", "paired", "within_domain", "across_domain")
SCHEMA_ROOT = "gradient_reuse.shared_output_structure"
COUNTER_DOMAIN = "gradient_reuse.shared_output_structure.four_row_atoms.v1"


def add_shared_output_structure_args(parser):
    parser.add_argument("--shared_structure_arm", choices=ARMS, required=True)
    parser.add_argument("--shared_structure_run_seed", type=int, required=True)
    parser.add_argument("--shared_structure_counter_seed", type=int, required=True)
    parser.add_argument("--shared_structure_selection_seed_salt", type=int, required=True)
    parser.add_argument("--shared_structure_audit_path", type=str, required=True)
    parser.add_argument("--shared_structure_gradient_interval", type=int, default=500)
    return parser


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
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
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("shared-output audit forbids non-finite values")
    return value


@dataclass(frozen=True)
class FourRowAtom:
    domain_a: int
    domain_b: int
    positions: tuple[int, int, int, int]
    across_reversed: bool


@dataclass(frozen=True)
class SharedOutputPlan:
    within_permutation: torch.Tensor
    across_permutation: torch.Tensor
    selected_positions: torch.Tensor
    atoms: tuple[FourRowAtom, ...]
    historical_rows: int
    represented_domains: tuple[int, ...]
    eligible_candidate_atoms: int
    ineligible_reason: str | None

    @property
    def selected_atoms(self) -> int:
        return len(self.atoms)


def build_shared_output_plan(
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    current_domain: int,
    counter_seed: int,
    run_seed: int,
    selection_seed_salt: int,
    observation: int,
) -> SharedOutputPlan:
    """Select the same disjoint eligible atoms for both structured swap arms."""

    target = labels.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    domains = domain_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    if len(target) != len(domains) or len(target) < 1:
        raise RuntimeError("replay labels/domain IDs must be nonempty and aligned")
    if bool(((target < 0) | (target >= 10)).any().item()):
        raise RuntimeError("Permuted-MNIST labels must lie in [0,10)")
    if current_domain < 0 or observation < 0 or bool((domains > current_domain).any().item()):
        raise RuntimeError("invalid current or buffered domain ID")

    historical = [index for index, domain in enumerate(domains.tolist()) if domain < current_domain]
    by_domain: dict[int, list[int]] = {}
    for index in historical:
        by_domain.setdefault(int(domains[index].item()), []).append(index)

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for domain_a, domain_b in itertools.combinations(sorted(by_domain), 2):
        for a0, a1 in itertools.combinations(by_domain[domain_a], 2):
            if target[a0] == target[a1]:
                continue
            for b0, b1 in itertools.combinations(by_domain[domain_b], 2):
                if len({int(target[a0]), int(target[a1]), int(target[b0]), int(target[b1])}) == 4:
                    candidates.append((domain_a, domain_b, a0, a1, b0, b1))

    key = "|".join(map(str, (COUNTER_DOMAIN, counter_seed, run_seed,
                              selection_seed_salt, observation)))
    stable = _stable_u64(key)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
        counter_seed & 0xFFFFFFFF, run_seed & 0xFFFFFFFF,
        selection_seed_salt & 0xFFFFFFFF, observation & 0xFFFFFFFF,
        stable & 0xFFFFFFFF, (stable >> 32) & 0xFFFFFFFF,
    ])))
    priorities = rng.random(len(candidates))
    order = sorted(range(len(candidates)), key=lambda i: (float(priorities[i]), candidates[i]))
    used: set[int] = set()
    atoms: list[FourRowAtom] = []
    for candidate_index in order:
        domain_a, domain_b, a0, a1, b0, b1 = candidates[candidate_index]
        positions = (a0, a1, b0, b1)
        if used.intersection(positions):
            continue
        used.update(positions)
        atoms.append(FourRowAtom(domain_a, domain_b, positions, bool(rng.integers(0, 2))))

    identity = torch.arange(len(target), dtype=torch.long)
    within = identity.clone()
    across = identity.clone()
    for atom in atoms:
        a0, a1, b0, b1 = atom.positions
        within[[a0, a1, b0, b1]] = torch.tensor([a1, a0, b1, b0])
        sources = [b1, b0, a1, a0] if atom.across_reversed else [b0, b1, a0, a1]
        across[[a0, a1, b0, b1]] = torch.tensor(sources)

    selected = torch.tensor(sorted(used), dtype=torch.long)
    reason = None
    if not atoms:
        if not historical:
            reason = "no_historical_replay_rows"
        elif len(by_domain) < 2:
            reason = "fewer_than_two_historical_domains"
        else:
            reason = "no_four_distinct_label_cross_domain_atom"
    plan = SharedOutputPlan(
        within, across, selected, tuple(atoms), len(historical),
        tuple(sorted(by_domain)), len(candidates), reason,
    )
    validate_plan(target, domains, current_domain=current_domain, plan=plan)
    return plan


def validate_plan(
    labels: torch.Tensor, domain_ids: torch.Tensor, *, current_domain: int,
    plan: SharedOutputPlan,
) -> None:
    target = labels.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    domains = domain_ids.detach().reshape(-1).to(dtype=torch.long, device="cpu")
    identity = list(range(len(target)))
    if sorted(plan.within_permutation.tolist()) != identity or sorted(plan.across_permutation.tolist()) != identity:
        raise RuntimeError("structured target map is not a permutation")
    selected = set(plan.selected_positions.tolist())
    if len(selected) != 4 * len(plan.atoms):
        raise RuntimeError("selected four-row atoms overlap")
    for atom in plan.atoms:
        a0, a1, b0, b1 = atom.positions
        values = {int(target[index]) for index in atom.positions}
        if not (
            domains[a0] == domains[a1] == atom.domain_a
            and domains[b0] == domains[b1] == atom.domain_b
            and atom.domain_a < atom.domain_b < current_domain
            and len(values) == 4
        ):
            raise RuntimeError("selected atom violates domain/label eligibility")
    within = target[plan.within_permutation]
    across = target[plan.across_permutation]
    if Counter(within.tolist()) != Counter(target.tolist()) or Counter(across.tolist()) != Counter(target.tolist()):
        raise RuntimeError("structured swap changed the target multiset")
    if selected:
        index = torch.tensor(sorted(selected), dtype=torch.long)
        if bool((within[index] == target[index]).any()) or bool((across[index] == target[index]).any()):
            raise RuntimeError("structured swap did not change every selected label")
        if not torch.equal(domains[plan.within_permutation[index]], domains[index]):
            raise RuntimeError("within-domain swap crossed a domain boundary")
        if bool((domains[plan.across_permutation[index]] == domains[index]).any()):
            raise RuntimeError("across-domain swap retained a domain source")
    for index in identity:
        if index not in selected and (plan.within_permutation[index] != index or plan.across_permutation[index] != index):
            raise RuntimeError("an unselected row was reassigned")


def _flatten(gradients: Sequence[torch.Tensor | None], parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def _functional_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cpu_state: torch.Tensor, cuda_state: list[torch.Tensor] | None,
) -> torch.Tensor:
    parameters = OrderedDict((name, value) for name, value in model.named_parameters() if value.requires_grad)
    buffers = OrderedDict((name, value.detach().clone()) for name, value in model.named_buffers())
    torch.set_rng_state(cpu_state.clone())
    if cuda_state is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in cuda_state])
    output = torch.func.functional_call(model, (dict(parameters), dict(buffers)), (inputs,))
    gradients = torch.autograd.grad(objective(output, targets), tuple(parameters.values()), allow_unused=True)
    return _flatten(gradients, tuple(parameters.values()))


def cross_entropy_objective(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(output, target.long().reshape(-1))


class SharedOutputStructureIntervention:
    def __init__(self, args: Any, method: str) -> None:
        self.arm = str(args.shared_structure_arm)
        self.method = str(method)
        self.run_seed = int(args.shared_structure_run_seed)
        self.counter_seed = int(args.shared_structure_counter_seed)
        self.selection_seed_salt = int(args.shared_structure_selection_seed_salt)
        self.interval = int(args.shared_structure_gradient_interval)
        self.path = Path(args.shared_structure_audit_path).expanduser().resolve()
        self.observation = -1
        if not (
            self.arm in ARMS and self.method == "er_shared_output_structure"
            and self.run_seed == int(args.seed) and self.interval > 0
            and str(args.dataset) == "perm-mnist"
        ):
            raise RuntimeError("shared-output intervention identity changed")
        self.record({
            "schema_version": f"{SCHEMA_ROOT}.effective_args.v1",
            "training_task_mask_applied": False,
            "training_output_classes": 10,
            "historical_domain_supplied_at_inference": False,
            "effective_args": {
                name: getattr(args, name) for name in vars(args)
                if name.startswith("shared_structure_") or name in {
                    "model", "dataset", "dataset_config", "backbone", "lr", "n_epochs",
                    "seed", "device", "num_workers", "buffer_size", "minibatch_size", "batch_size"
                }
            },
        })

    def begin_observation(self) -> None:
        self.observation += 1

    def no_replay_update(self, *, rows: int, current_domain: int) -> None:
        if self.arm != "no_replay" or rows < 1:
            raise RuntimeError("invalid no-replay update evidence")
        self.record({
            "schema_version": f"{SCHEMA_ROOT}.no_replay_update.v1",
            "observation": self.observation,
            "current_domain": int(current_domain),
            "current_rows": int(rows),
            "replay_rows": 0,
            "replay_storage_enabled": False,
            "replay_loss_enabled": False,
        })

    def plan(self, labels: torch.Tensor, domain_ids: torch.Tensor, *, current_domain: int) -> SharedOutputPlan:
        return build_shared_output_plan(
            labels, domain_ids, current_domain=current_domain,
            counter_seed=self.counter_seed, run_seed=self.run_seed,
            selection_seed_salt=self.selection_seed_salt, observation=self.observation,
        )

    def apply(self, labels: torch.Tensor, plan: SharedOutputPlan) -> torch.Tensor:
        if self.arm == "paired":
            return labels
        permutation = plan.within_permutation if self.arm == "within_domain" else plan.across_permutation
        return labels[permutation.to(labels.device)]

    def exposure(
        self, labels: torch.Tensor, domain_ids: torch.Tensor, plan: SharedOutputPlan,
        *, current_domain: int, sampled_indexes: torch.Tensor, sampled_inputs: torch.Tensor,
    ) -> None:
        within = labels[plan.within_permutation.to(labels.device)]
        across = labels[plan.across_permutation.to(labels.device)]
        used = self.apply(labels, plan)
        expected_changed = 0 if self.arm == "paired" else int(plan.selected_positions.numel())
        changed = int((used != labels).sum().item())
        if changed != expected_changed:
            raise RuntimeError("live changed-row count violates the locked arm")
        selected = plan.selected_positions.to(labels.device)
        self.record({
            "schema_version": f"{SCHEMA_ROOT}.exposure.v1",
            "observation": self.observation,
            "current_domain": int(current_domain),
            "replay_rows": int(len(labels)),
            "historical_rows": plan.historical_rows,
            "represented_domains": plan.represented_domains,
            "eligible": bool(plan.atoms),
            "ineligible_reason": plan.ineligible_reason,
            "eligible_candidate_atoms": plan.eligible_candidate_atoms,
            "selected_atoms": plan.selected_atoms,
            "selected_rows": int(selected.numel()),
            "changed_rows": changed,
            "same_selected_rows_for_both_swap_arms": True,
            "all_selected_values_change_within": bool(selected.numel() == 0 or (within[selected] != labels[selected]).all()),
            "all_selected_values_change_across": bool(selected.numel() == 0 or (across[selected] != labels[selected]).all()),
            "target_multiset_preserved_within": _row_multiset_sha256(within) == _row_multiset_sha256(labels),
            "target_multiset_preserved_across": _row_multiset_sha256(across) == _row_multiset_sha256(labels),
            "sampled_indexes_sha256": _tensor_sha256(sampled_indexes),
            "input_sha256": _tensor_sha256(sampled_inputs),
            "labels_sha256": _tensor_sha256(labels),
            "domain_ids_sha256": _tensor_sha256(domain_ids),
            "selected_positions_sha256": _tensor_sha256(plan.selected_positions),
            "within_permutation_sha256": _tensor_sha256(plan.within_permutation),
            "across_permutation_sha256": _tensor_sha256(plan.across_permutation),
            "within_targets_sha256": _tensor_sha256(within),
            "across_targets_sha256": _tensor_sha256(across),
        })

    def gradient_diagnostic(
        self, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
        plan: SharedOutputPlan, objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        if self.arm != "paired" or self.observation % self.interval or not plan.atoms:
            return
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        training = model.training
        saved_grads = {name: None if value.grad is None else value.grad.detach().clone()
                       for name, value in model.named_parameters()}
        try:
            true = _functional_gradient(model, inputs, labels, objective, cpu, cuda).double()
            within_targets = labels[plan.within_permutation.to(labels.device)]
            across_targets = labels[plan.across_permutation.to(labels.device)]
            within = _functional_gradient(model, inputs, within_targets, objective, cpu, cuda).double()
            across = _functional_gradient(model, inputs, across_targets, objective, cpu, cuda).double()
        finally:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            model.train(training)
            for name, value in model.named_parameters():
                value.grad = saved_grads[name]
        def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
            denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
            return 0.0 if denominator == 0.0 else float(torch.dot(left, right) / denominator)
        self.record({
            "schema_version": f"{SCHEMA_ROOT}.common_state_gradient.v1",
            "observation": self.observation,
            "common_parameter_state": True,
            "training_update_uses_true_targets": True,
            "gradient_definition": "raw_full_replay_mean_cross_entropy_parameter_gradient",
            "true_norm": float(torch.linalg.vector_norm(true)),
            "within_norm": float(torch.linalg.vector_norm(within)),
            "across_norm": float(torch.linalg.vector_norm(across)),
            "true_within_distance": float(torch.linalg.vector_norm(true - within)),
            "true_across_distance": float(torch.linalg.vector_norm(true - across)),
            "across_within_distance": float(torch.linalg.vector_norm(across - within)),
            "true_within_cosine": cosine(true, within),
            "true_across_cosine": cosine(true, across),
        })

    def final_evaluation(
        self, model: torch.nn.Module, dataset: Any, *, completed_domains: int,
    ) -> None:
        """Record final shared-output NLL without supplying domain identity."""

        if completed_domains != 20:
            return
        if len(dataset.test_loaders) != 20:
            raise RuntimeError("final Permuted-MNIST test-loader inventory changed")
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        training = model.training
        before = {
            name: _tensor_sha256(value)
            for name, value in model.state_dict().items()
        }
        per_domain = []
        total_loss = total_correct = total_examples = 0
        try:
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                for domain, loader in enumerate(dataset.test_loaders):
                    loss_sum = correct = examples = 0
                    for batch in loader:
                        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                            raise RuntimeError("Permuted-MNIST test loader omitted inputs or labels")
                        inputs = batch[0].to(device)
                        labels = batch[1].reshape(-1).long().to(device)
                        logits = model(inputs)
                        if logits.ndim != 2 or logits.shape[-1] != 10:
                            raise RuntimeError("final shared-output evaluation did not expose ten classes")
                        loss_sum += float(torch.nn.functional.cross_entropy(
                            logits, labels, reduction="sum"
                        ).item())
                        correct += int((logits.argmax(dim=1) == labels).sum().item())
                        examples += int(labels.numel())
                    if examples < 1:
                        raise RuntimeError("Permuted-MNIST test domain is empty")
                    per_domain.append({
                        "domain": domain,
                        "examples": examples,
                        "negative_log_likelihood": loss_sum / examples,
                        "accuracy_percentage_points": 100.0 * correct / examples,
                    })
                    total_loss += loss_sum
                    total_correct += correct
                    total_examples += examples
        finally:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            model.train(training)
        after = {
            name: _tensor_sha256(value)
            for name, value in model.state_dict().items()
        }
        if before != after:
            raise RuntimeError("read-only final NLL evaluation changed model state")
        before_digest = hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        after_digest = hashlib.sha256(
            json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.record({
            "schema_version": f"{SCHEMA_ROOT}.final_domain_il.v1",
            "completed_domains": completed_domains,
            "historical_domain_supplied_at_inference": False,
            "task_mask_applied": False,
            "examples": total_examples,
            "negative_log_likelihood": total_loss / total_examples,
            "accuracy_percentage_points": 100.0 * total_correct / total_examples,
            "per_domain": per_domain,
            "model_state_before_sha256": before_digest,
            "model_state_after_sha256": after_digest,
        })

    def record(self, value: Mapping[str, Any]) -> None:
        payload = {
            "method": self.method, "arm": self.arm, "run_seed": self.run_seed,
            "counter_seed": self.counter_seed, "selection_seed_salt": self.selection_seed_salt,
            **dict(value),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise RuntimeError("shared-output audit path became unsafe")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()


__all__ = [
    "ARMS", "FourRowAtom", "SharedOutputPlan", "SharedOutputStructureIntervention",
    "add_shared_output_structure_args", "build_shared_output_plan",
    "cross_entropy_objective", "validate_plan",
]
