"""Meta-Experience Replay with controlled replay-target assignment."""

from __future__ import annotations

import logging

import torch

from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    cross_entropy_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class MerPairing(ContinualModel):
    """Mammoth MER Algorithm 6 with one intervention at replay labels."""

    NAME = "mer_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.set_defaults(batch_size=1)
        parser.add_argument("--beta", type=float, required=True)
        parser.add_argument("--gamma", type=float, required=True)
        parser.add_argument("--batch_num", type=int, default=1)
        parser.add_argument(
            "--mer_replay_batch_size",
            type=int,
            required=True,
            help=(
                "Replay rows per MER meta-batch. This separate name prevents "
                "Mammoth's batch_size=1 extension step from silently replacing "
                "the prespecified replay minibatch size."
            ),
        )
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if int(args.batch_size) != 1:
            logging.warning("MER is defined for batch_size=1; overriding the request")
            args.batch_size = 1
        if int(args.mer_replay_batch_size) < 1:
            raise ValueError("MER replay batch size must be positive")
        args.minibatch_size = int(args.mer_replay_batch_size)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.pairing = PairingIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()
        outer_start = self.net.get_params().data.clone()
        current_labels = labels.reshape(-1)

        for batch_number in range(int(self.args.batch_num)):
            inner_start = self.net.get_params().data.clone()
            if not self.buffer.is_empty():
                replay = self.buffer.get_data(
                    self.args.minibatch_size,
                    transform=self.transform,
                    return_index=True,
                    device=self.device,
                )
                replay_indexes, replay_inputs, replay_labels = replay
                channel = f"meta_replay_{batch_number}"
                self.pairing.gradient_diagnostic(
                    self.net,
                    replay_inputs,
                    replay_labels,
                    cross_entropy_objective,
                    channel=channel,
                    sampled_indexes=replay_indexes,
                )
                used_labels, permutation = self.pairing.intervene(
                    replay_labels, channel=channel
                )
                self.pairing.record_intervention(
                    replay_labels,
                    used_labels,
                    permutation,
                    channel=channel,
                    sampled_indexes=replay_indexes,
                    sampled_inputs=replay_inputs,
                )
                batch_inputs = torch.cat((replay_inputs, inputs))
                batch_labels = torch.cat((used_labels, current_labels))
            else:
                batch_inputs, batch_labels = inputs, current_labels

            self.opt.zero_grad()
            loss = self.loss(self.net(batch_inputs), batch_labels)
            loss.backward()
            self.opt.step()
            inner_updated = inner_start + self.args.beta * (
                self.net.get_params() - inner_start
            )
            self.net.set_params(inner_updated)

        self.buffer.add_data(examples=not_aug_inputs, labels=current_labels)
        outer_updated = outer_start + self.args.gamma * (
            self.net.get_params() - outer_start
        )
        self.net.set_params(outer_updated)
        return loss.item()
