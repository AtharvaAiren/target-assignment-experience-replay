"""Experience replay using the deterministic mean target of reassignment."""

from __future__ import annotations

import torch

from models.utils.continual_model import ContinualModel
from models.utils.mean_matched_intervention import (
    MeanMatchedIntervention,
    add_mean_matched_args,
    soft_cross_entropy,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErMeanMatched(ContinualModel):
    NAME = "er_mean_matched"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_mean_matched_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.mean_match = MeanMatchedIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del epoch
        self.mean_match.begin_observation()
        current_labels = labels.reshape(-1).long()
        self.opt.zero_grad()
        if self.buffer.is_empty():
            outputs = self.net(inputs)
            loss = self.loss(outputs, current_labels)
        else:
            replay_indexes, replay_inputs, replay_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            all_inputs = torch.cat((inputs, replay_inputs))
            outputs = self.net(all_inputs)
            classes = int(outputs.shape[-1])
            current_targets = torch.nn.functional.one_hot(
                current_labels, num_classes=classes
            ).to(dtype=outputs.dtype)
            replay_targets = self.mean_match.transform(
                replay_labels,
                num_classes=classes,
                dtype=outputs.dtype,
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            targets = torch.cat((current_targets, replay_targets), dim=0)
            loss = soft_cross_entropy(outputs, targets)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=current_labels)
        return loss.item()
