"""Experience replay with a controlled target-assignment strength."""

from __future__ import annotations

import torch

from models.utils.assignment_strength_intervention import (
    AssignmentStrengthIntervention,
    add_assignment_strength_args,
    cross_entropy_objective,
)
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErAssignmentStrength(ContinualModel):
    """Canonical ER, changing only the target view of sampled replay rows."""

    NAME = "er_assignment_strength"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_assignment_strength_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.assignment_strength = AssignmentStrengthIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.assignment_strength.begin_observation()
        current_labels = labels
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            replay = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            replay_indexes, replay_inputs, replay_labels = replay
            self.assignment_strength.gradient_diagnostic(
                self.net,
                replay_inputs,
                replay_labels,
                cross_entropy_objective,
                channel="replay_labels",
                sampled_indexes=replay_indexes,
            )
            used_replay_labels, assignment_plan = self.assignment_strength.intervene(
                replay_labels, channel="replay_labels"
            )
            self.assignment_strength.record_intervention(
                replay_labels,
                used_replay_labels,
                assignment_plan,
                channel="replay_labels",
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            inputs = torch.cat((inputs, replay_inputs))
            labels = torch.cat((current_labels, used_replay_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=current_labels)
        return loss.item()
