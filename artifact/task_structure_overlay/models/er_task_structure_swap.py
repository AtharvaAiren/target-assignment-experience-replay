"""Experience replay with equal-severity within/across historical-task swaps."""

from __future__ import annotations

import torch

from models.utils.continual_model import ContinualModel
from models.utils.task_structure_intervention import (
    TaskStructureIntervention,
    add_task_structure_args,
    cross_entropy_objective,
    task_ids_from_split_cifar100_labels,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErTaskStructureSwap(ContinualModel):
    NAME = "er_task_structure_swap"
    COMPATIBILITY = ["class-il", "task-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_task_structure_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if dataset is None or not (
            getattr(dataset, "NAME", None) == "seq-cifar100"
            and int(getattr(dataset, "N_TASKS", 0)) == 20
            and int(getattr(dataset, "N_CLASSES_PER_TASK", 0)) == 5
            and int(getattr(dataset, "N_CLASSES", 0)) == 100
        ):
            raise RuntimeError("task-structure model requires pinned 20-task CIFAR-100")
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.task_structure = TaskStructureIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del epoch
        self.task_structure.begin_observation()
        current_labels = labels.reshape(-1).to(dtype=torch.long)
        current_tasks = task_ids_from_split_cifar100_labels(current_labels)
        unique_tasks = current_tasks.unique(sorted=True)
        if unique_tasks.numel() != 1:
            raise RuntimeError("current Split CIFAR-100 minibatch crosses task boundaries")
        current_task = int(unique_tasks.item())
        if current_task != int(self.current_task):
            raise RuntimeError(
                "label-derived task ID disagrees with Mammoth's current task counter"
            )

        self.opt.zero_grad()
        if self.buffer.is_empty():
            self.task_structure.no_replay(current_task=current_task)
        else:
            replay = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            if len(replay) != 4:
                raise RuntimeError("task-structure buffer omitted explicit task labels")
            replay_indexes, replay_inputs, replay_labels, replay_tasks = replay
            replay_labels = replay_labels.reshape(-1).to(dtype=torch.long)
            replay_tasks = replay_tasks.reshape(-1).to(dtype=torch.long)
            plan = self.task_structure.plan(
                replay_labels, replay_tasks, current_task=current_task
            )
            self.task_structure.gradient_diagnostic(
                self.net,
                replay_inputs,
                replay_labels,
                plan,
                cross_entropy_objective,
                sampled_indexes=replay_indexes,
            )
            used_replay_labels = self.task_structure.apply(replay_labels, plan)
            self.task_structure.exposure(
                replay_labels,
                replay_tasks,
                plan,
                current_task=current_task,
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            inputs = torch.cat((inputs, replay_inputs))
            labels = torch.cat((current_labels, used_replay_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=current_labels,
            task_labels=current_tasks,
        )
        return loss.item()
