"""Experience replay with a controlled replay-target assignment intervention."""

from __future__ import annotations

import torch

from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    cross_entropy_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErPairing(ContinualModel):
    """Canonical ER, differing only at the replay target view."""

    NAME = "er_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.pairing = PairingIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()
        current_rows = int(inputs.shape[0])
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            replay = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            replay_indexes, replay_inputs, replay_labels = replay
            self.pairing.gradient_diagnostic(
                self.net,
                replay_inputs,
                replay_labels,
                cross_entropy_objective,
                channel="replay_labels",
                sampled_indexes=replay_indexes,
            )
            used_labels, permutation = self.pairing.intervene(
                replay_labels, channel="replay_labels"
            )
            self.pairing.record_intervention(
                replay_labels,
                used_labels,
                permutation,
                channel="replay_labels",
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            inputs = torch.cat((inputs, replay_inputs))
            labels = torch.cat((labels, used_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:current_rows],
        )
        return loss.item()
