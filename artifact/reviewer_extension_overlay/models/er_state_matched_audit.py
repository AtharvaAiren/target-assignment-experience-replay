"""Experience replay with read-only state-matched task-boundary probes."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import torch

from models.er_pairing import ErPairing
from models.utils.state_matched_audit import (
    deterministic_permutation,
    evaluate_old_task_branches,
    final_confusion_diagnostic,
    model_state_sha256,
    preserve_live_state,
    replay_gradient,
    tensor_multiset_sha256,
    tensor_sha256,
)
from utils.args import ArgumentParser


def add_state_matched_args(parser):
    parser.add_argument("--state_audit_path", type=str, required=True)
    parser.add_argument("--state_audit_enable", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--state_audit_final_confusion", type=int, choices=(0, 1), required=True
    )
    parser.add_argument("--state_audit_expected_tasks", type=int, default=20)
    parser.add_argument("--state_audit_update_window", type=int, default=16)
    parser.add_argument("--state_audit_replicas", type=int, default=4)
    return parser


class ErStateMatchedAudit(ErPairing):
    """Canonical ER plus functional branches that never touch the live path."""

    NAME = "er_state_matched_audit"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        ErPairing.get_parser(parser)
        return add_state_matched_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.state_audit_path = Path(args.state_audit_path).expanduser().resolve()
        self.state_audit_enable = bool(int(args.state_audit_enable))
        self.state_audit_final_confusion = bool(int(args.state_audit_final_confusion))
        self.state_audit_expected_tasks = int(args.state_audit_expected_tasks)
        self.state_audit_update_window = int(args.state_audit_update_window)
        self.state_audit_replicas = int(args.state_audit_replicas)
        if (
            self.state_audit_expected_tasks != 20
            or self.state_audit_update_window != 16
            or self.state_audit_replicas != 4
        ):
            raise RuntimeError("state-matched probe arguments differ from the lock")
        if self.state_audit_enable and self.pairing.mode != "paired":
            raise RuntimeError(
                "state-matched probes are restricted to paired trajectories"
            )
        self.update_norms: list[float] = []
        self._record(
            {
                "schema_version": "gradient_reuse.state_matched_effective_args.v1",
                "state_audit_enable": self.state_audit_enable,
                "state_audit_final_confusion": self.state_audit_final_confusion,
                "state_audit_expected_tasks": self.state_audit_expected_tasks,
                "state_audit_update_window": self.state_audit_update_window,
                "state_audit_replicas": self.state_audit_replicas,
                "gradient_definition": "replay_only_cross_entropy",
                "normalization": "g/max(l2_norm,1e-12)",
            }
        )

    def _record(self, value):
        self.state_audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.NAME,
            "benchmark": str(self.args.pairing_benchmark),
            "dataset": str(self.args.dataset),
            "intervention": self.pairing.mode,
            "run_seed": int(self.args.pairing_run_seed),
            "permutation_replica": int(self.args.pairing_replica),
            **dict(value),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self.state_audit_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()

    def _validate_plain_sgd(self):
        for group in self.opt.param_groups:
            if (
                float(group.get("momentum", 0.0)) != 0.0
                or float(group.get("weight_decay", 0.0)) != 0.0
                or bool(group.get("nesterov", False))
                or float(group.get("dampening", 0.0)) != 0.0
            ):
                raise RuntimeError(
                    "realized update-norm shortcut requires plain, momentum-free SGD"
                )

    def _realized_update_norm(self) -> float:
        self._validate_plain_sgd()
        squared = 0.0
        for group in self.opt.param_groups:
            learning_rate = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is not None:
                    squared += float(
                        torch.sum((learning_rate * parameter.grad.detach()) ** 2).item()
                    )
        value = math.sqrt(squared)
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("realized ER update norm is not finite and positive")
        return value

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        loss = super().observe(inputs, labels, not_aug_inputs, epoch=epoch)
        if self.state_audit_enable:
            self.update_norms.append(self._realized_update_norm())
        return loss

    def _probe(self, dataset, completed_tasks: int):
        if len(self.update_norms) < self.state_audit_update_window:
            raise RuntimeError("fewer than 16 realized updates precede the probe")
        old_loaders = tuple(dataset.test_loaders[: completed_tasks - 1])
        if len(old_loaders) != completed_tasks - 1:
            raise RuntimeError("old-task loader inventory is incomplete")
        guarded_loaders = tuple(dataset.test_loaders)
        with preserve_live_state(
            self.net,
            optimizer=self.opt,
            buffer=self.buffer,
            loaders=guarded_loaders,
        ) as live_state:
            replay = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            replay_indexes, replay_inputs, replay_labels = replay
            directions = {
                "paired": replay_gradient(self.net, replay_inputs, replay_labels)
            }
            original_multiset = tensor_multiset_sha256(replay_labels)
            permutations = []
            for replica in range(self.state_audit_replicas):
                permutation = deterministic_permutation(
                    len(replay_labels),
                    run_seed=int(self.args.pairing_run_seed),
                    boundary=completed_tasks,
                    replica=replica,
                ).to(replay_labels.device)
                transformed = replay_labels[permutation]
                if tensor_multiset_sha256(transformed) != original_multiset:
                    raise RuntimeError("probe permutation changed the target multiset")
                directions[f"permuted_{replica}"] = replay_gradient(
                    self.net, replay_inputs, transformed
                )
                permutations.append(tensor_sha256(permutation))
            radius = float(
                statistics.median(self.update_norms[-self.state_audit_update_window :])
            )
            metrics = evaluate_old_task_branches(
                self.net, old_loaders, directions, radius=radius
            )
        self._record(
            {
                "schema_version": "gradient_reuse.state_matched_boundary.v1",
                "completed_tasks": completed_tasks,
                "evaluated_old_tasks": completed_tasks - 1,
                "live_state_before_and_after_sha256": live_state,
                "sampled_indexes_sha256": tensor_sha256(replay_indexes),
                "replay_inputs_sha256": tensor_sha256(replay_inputs),
                "replay_targets_sha256": tensor_sha256(replay_labels),
                "replay_target_multiset_sha256": original_multiset,
                "permutation_sha256": permutations,
                "update_norm_window": self.state_audit_update_window,
                **metrics,
            }
        )

    def _final_confusion(self, dataset, completed_tasks: int):
        if completed_tasks != self.state_audit_expected_tasks:
            return
        old_loaders = tuple(dataset.test_loaders[: completed_tasks - 1])
        newest_loader = dataset.test_loaders[completed_tasks - 1]
        guarded_loaders = tuple(dataset.test_loaders)
        with preserve_live_state(
            self.net,
            optimizer=self.opt,
            buffer=self.buffer,
            loaders=guarded_loaders,
        ) as live_state:
            diagnostic = final_confusion_diagnostic(
                self.net,
                old_loaders,
                newest_loader,
                setting=str(getattr(dataset, "SETTING", "")),
            )
        self._record(
            {
                "schema_version": "gradient_reuse.final_confusion_record.v1",
                "completed_tasks": completed_tasks,
                "live_state_before_and_after_sha256": live_state,
                "diagnostic": diagnostic,
            }
        )

    def end_task(self, dataset):
        completed_tasks = int(self.current_task) + 1
        if len(dataset.test_loaders) != completed_tasks:
            raise RuntimeError("test-loader inventory differs from completed tasks")
        if self.state_audit_enable and completed_tasks >= 2:
            self._probe(dataset, completed_tasks)
        if self.state_audit_final_confusion:
            self._final_confusion(dataset, completed_tasks)
