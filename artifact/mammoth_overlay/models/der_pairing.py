"""Dark Experience Replay with controlled input--teacher assignment."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    mse_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class DerPairing(ContinualModel):
    """Canonical DER; stored logit vectors are the replay targets."""

    NAME = "der_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument("--alpha", type=float, required=True)
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.pairing = PairingIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            replay = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                return_index=True,
                device=self.device,
            )
            replay_indexes, replay_inputs, replay_logits = replay
            self.pairing.gradient_diagnostic(
                self.net,
                replay_inputs,
                replay_logits,
                mse_objective,
                channel="dark_logits",
                sampled_indexes=replay_indexes,
            )
            used_logits, permutation = self.pairing.intervene(
                replay_logits, channel="dark_logits"
            )
            self.pairing.record_intervention(
                replay_logits,
                used_logits,
                permutation,
                channel="dark_logits",
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )
            replay_outputs = self.net(replay_inputs)
            loss = loss + self.args.alpha * F.mse_loss(
                replay_outputs, used_logits
            )

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)
        return loss.item()
