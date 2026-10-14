"""Experience replay with K averaged assignment draws at a fixed mean target."""

from __future__ import annotations

import torch

from models.utils.continual_model import ContinualModel
from models.utils.variance_ladder_intervention import (
    VarianceLadderIntervention,
    add_variance_ladder_args,
    soft_cross_entropy,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErVarianceLadder(ContinualModel):
    NAME = "er_variance_ladder"
    COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_variance_ladder_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.variance_ladder = VarianceLadderIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del epoch
        self.variance_ladder.begin_observation()
        current_labels = labels.reshape(-1).long()
        self.opt.zero_grad()
        if self.buffer.is_empty():
            outputs = self.net(inputs)
            loss = self.loss(outputs, current_labels)
        else:
            replay_indexes, replay_inputs, replay_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform,
                return_index=True, device=self.device,
            )
            all_inputs = torch.cat((inputs, replay_inputs))
            outputs = self.net(all_inputs)
            classes = int(outputs.shape[-1])
            current_targets = torch.nn.functional.one_hot(
                current_labels, num_classes=classes
            ).to(dtype=outputs.dtype)
            transformed = self.variance_ladder.transform(
                replay_labels, num_classes=classes, dtype=outputs.dtype,
            )
            self.variance_ladder.exposure(
                replay_labels, transformed, sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            self.variance_ladder.audit(
                self.net, replay_inputs, replay_labels, transformed,
                sampled_indexes=replay_indexes,
            )
            targets = torch.cat((current_targets, transformed.used_targets), dim=0)
            loss = soft_cross_entropy(outputs, targets)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=current_labels)
        return loss.item()
