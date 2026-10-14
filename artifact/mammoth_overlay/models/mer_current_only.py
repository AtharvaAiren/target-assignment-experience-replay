"""Current-example-only control matched to Meta-Experience Replay updates."""

from __future__ import annotations

import logging

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class MerCurrentOnly(ContinualModel):
    """MER's inner/outer meta-update with no memory or replay examples."""

    NAME = "mer_current_only"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.set_defaults(batch_size=1)
        parser.add_argument("--beta", type=float, required=True)
        parser.add_argument("--gamma", type=float, required=True)
        parser.add_argument("--batch_num", type=int, default=1)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if int(args.batch_size) != 1:
            logging.warning(
                "The MER-matched control is defined for batch_size=1; "
                "overriding the request"
            )
            args.batch_size = 1
        if int(args.batch_num) < 1:
            raise ValueError("MER batch_num must be positive")
        super().__init__(backbone, loss, args, transform, dataset=dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del not_aug_inputs, epoch
        outer_start = self.net.get_params().data.clone()
        current_labels = labels.reshape(-1)

        for _ in range(int(self.args.batch_num)):
            inner_start = self.net.get_params().data.clone()
            self.opt.zero_grad()
            loss = self.loss(self.net(inputs), current_labels)
            loss.backward()
            self.opt.step()
            inner_updated = inner_start + self.args.beta * (
                self.net.get_params() - inner_start
            )
            self.net.set_params(inner_updated)

        outer_updated = outer_start + self.args.gamma * (
            self.net.get_params() - outer_start
        )
        self.net.set_params(outer_updated)
        return loss.item()
