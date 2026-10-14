"""GEM with replay-target pairing as the only experimental intervention.

The GEM constraint/projection structure follows the pinned Mammoth
``models/gem.py`` implementation, whose header points to the checkout's
separate CC-BY-NC-4.0 ``gem_license``. See ``mammoth_overlay/NOTICE.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from models.gem import overwrite_grad, project2cone2, store_grad
from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    cross_entropy_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer, fill_buffer
from utils.conf import warn_once


@torch.no_grad()
def fill_domain_task_balanced_buffer(buffer, dataset, task_index: int) -> None:
    """Fill GEM memory evenly by domain when class labels repeat across tasks.

    Mammoth's pinned ``fill_buffer`` balances by newly introduced class IDs.
    That is correct for class-incremental streams but selects no rows after the
    first task of Permuted-MNIST, where every domain reuses labels 0--9.  This
    domain-IL branch instead keeps an almost equal, deterministic quota per
    observed task and records explicit task labels for GEM's constraints.
    """

    task_index = int(task_index)
    if task_index < 0:
        raise RuntimeError("GEM task index must be nonnegative")
    task_count = task_index + 1
    capacity = int(buffer.buffer_size)
    if capacity < task_count:
        raise RuntimeError("GEM buffer cannot retain one example per observed task")
    quotas = [capacity // task_count] * task_count
    for index in range(capacity % task_count):
        quotas[index] += 1

    retained: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    if not buffer.is_empty():
        contents = buffer.get_all_data(device="cpu")
        if len(contents) != 3:
            raise RuntimeError("GEM domain buffer must contain examples, labels, task labels")
        old_examples, old_labels, old_tasks = contents
        for old_task in range(task_index):
            mask = old_tasks.reshape(-1).to(dtype=torch.long) == old_task
            required = quotas[old_task]
            if int(mask.sum().item()) < required:
                raise RuntimeError(
                    f"GEM domain buffer lacks task {old_task}: "
                    f"need {required}, found {int(mask.sum().item())}"
                )
            retained[old_task] = (
                old_examples[mask][:required].cpu(),
                old_labels[mask][:required].reshape(-1).to(dtype=torch.long).cpu(),
            )

    examples: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in dataset.train_loader:
        if len(batch) < 3:
            raise RuntimeError("GEM domain loader must expose not-augmented inputs")
        examples.append(batch[2].cpu())
        labels.append(batch[1].reshape(-1).to(dtype=torch.long).cpu())
    if not examples:
        raise RuntimeError("GEM domain loader is empty")
    current_examples = torch.cat(examples, dim=0)
    current_labels = torch.cat(labels, dim=0)
    required = quotas[task_index]
    if len(current_examples) < required:
        raise RuntimeError(
            f"GEM domain task {task_index} has too few examples: "
            f"need {required}, found {len(current_examples)}"
        )
    run_seed = int(getattr(getattr(dataset, "args", None), "seed", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_seed * 1_000_003 + task_index)
    selected = torch.randperm(len(current_examples), generator=generator)[:required]
    retained[task_index] = (
        current_examples[selected],
        current_labels[selected],
    )

    buffer.empty()
    for retained_task in range(task_count):
        task_examples, task_labels = retained[retained_task]
        buffer.add_data(
            examples=task_examples,
            labels=task_labels,
            task_labels=torch.full(
                (len(task_examples),), retained_task, dtype=torch.long
            ),
        )
    if len(buffer) != capacity:
        raise RuntimeError(
            f"GEM domain buffer fill is incomplete: {len(buffer)} != {capacity}"
        )
    observed_tasks, observed_counts = torch.unique(
        buffer.get_all_data(device="cpu")[2].to(dtype=torch.long),
        return_counts=True,
    )
    observed = {
        int(task.item()): int(count.item())
        for task, count in zip(observed_tasks, observed_counts)
    }
    expected = {task: quota for task, quota in enumerate(quotas)}
    if observed != expected:
        raise RuntimeError(f"GEM domain task quotas changed: {observed} != {expected}")


class GemPairing(ContinualModel):
    """Mammoth GEM with paired or within-task permuted memory labels."""

    NAME = "gem_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--gamma", type=float, default=0.5)
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = [parameter.data.numel() for parameter in self.parameters()]
        dimension = int(np.sum(self.grad_dims))
        self.grads_cs: list[torch.Tensor] = []
        self.grads_da = torch.zeros(dimension, device=self.device)
        self.pairing = PairingIntervention(args, self.NAME)
        try:
            import quadprog as solver
        except ImportError as error:
            warn_once("GEM pairing audit requires the pinned quadprog solver")
            raise RuntimeError("install quadprog for the GEM mechanism row") from error
        self.solver = solver

    def end_task(self, dataset):
        self.grads_cs.append(
            torch.zeros(int(np.sum(self.grad_dims)), device=self.device)
        )
        if getattr(dataset, "SETTING", None) == "domain-il":
            fill_domain_task_balanced_buffer(
                self.buffer, dataset, self.current_task
            )
        else:
            fill_buffer(
                self.buffer,
                dataset,
                self.current_task,
                required_attributes=["examples", "labels", "task_labels"],
            )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()
        memory_gradients: list[torch.Tensor] = []
        if not self.buffer.is_empty():
            replay = self.buffer.get_data(
                self.args.buffer_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            replay_indexes, replay_inputs, replay_labels, replay_tasks = replay
            for task_value in replay_tasks.unique(sorted=True):
                mask = replay_tasks == task_value
                task_inputs = replay_inputs[mask]
                task_labels = replay_labels[mask]
                task_indexes = replay_indexes[mask]
                channel = f"constraint_task_{int(task_value.item())}"
                self.pairing.gradient_diagnostic(
                    self.net,
                    task_inputs,
                    task_labels,
                    cross_entropy_objective,
                    channel=channel,
                    sampled_indexes=task_indexes,
                )
                used_labels, permutation = self.pairing.intervene(
                    task_labels, channel=channel
                )
                self.pairing.record_intervention(
                    task_labels,
                    used_labels,
                    permutation,
                    channel=channel,
                    sampled_indexes=task_indexes,
                    sampled_inputs=task_inputs,
                )
                self.opt.zero_grad()
                penalty = self.loss(self.forward(task_inputs), used_labels)
                penalty.backward()
                gradient = self.grads_cs[int(task_value.item())]
                store_grad(self.parameters, gradient, self.grad_dims)
                memory_gradients.append(gradient)

        self.opt.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if memory_gradients:
            store_grad(self.parameters, self.grads_da, self.grad_dims)
            memories = torch.stack(memory_gradients).T
            pre_constraints = torch.mm(self.grads_da.unsqueeze(0), memories).reshape(-1)
            projected = bool((pre_constraints < 0).any().item())
            if projected:
                project2cone2(
                    self.solver,
                    self.grads_da.unsqueeze(1),
                    memories,
                    margin=self.args.gamma,
                )
                overwrite_grad(self.parameters, self.grads_da, self.grad_dims)
            if self.pairing.should_diagnose:
                post_constraints = torch.mm(
                    self.grads_da.unsqueeze(0), memories
                ).reshape(-1)
                self.pairing.record(
                    {
                        "schema_version": "gradient_reuse.gem_projection.v1",
                        "method": self.NAME,
                        "intervention": self.pairing.mode,
                        "observation": self.pairing.observation,
                        "event": self.pairing.event,
                        "channel": "constraint_projection",
                        "projection_active": projected,
                        "constraints": int(len(memory_gradients)),
                        "pre_min_dot": float(pre_constraints.min().item()),
                        "post_min_dot": float(post_constraints.min().item()),
                    }
                )

        self.opt.step()
        return loss.item()
