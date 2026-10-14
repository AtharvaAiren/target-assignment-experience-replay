"""A-GEM with a controlled replay-target assignment intervention."""

from __future__ import annotations

import numpy as np
import torch

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    cross_entropy_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


def project(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    denominator = torch.dot(reference, reference)
    if float(denominator.item()) <= 1.0e-30:
        return current
    return current - (torch.dot(current, reference) / denominator) * reference


class AGemPairing(ContinualModel):
    """Mammoth A-GEM with paired or target-permuted reference gradients."""

    NAME = "agem_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = [parameter.data.numel() for parameter in self.parameters()]
        dimension = int(np.sum(self.grad_dims))
        self.grad_xy = torch.empty(dimension, device=self.device)
        self.grad_er = torch.empty(dimension, device=self.device)
        self.pairing = PairingIntervention(args, self.NAME)

    def end_task(self, dataset):
        loader = dataset.train_loader
        current_labels, current_examples = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=current_examples.to(self.device),
            labels=current_labels.to(self.device),
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()
        self.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)
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
                channel="reference_labels",
                sampled_indexes=replay_indexes,
            )
            used_labels, permutation = self.pairing.intervene(
                replay_labels, channel="reference_labels"
            )
            self.pairing.record_intervention(
                replay_labels,
                used_labels,
                permutation,
                channel="reference_labels",
                sampled_indexes=replay_indexes,
                sampled_inputs=replay_inputs,
            )

            self.net.zero_grad()
            penalty = self.loss(self.net(replay_inputs), used_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            pre_dot = float(torch.dot(self.grad_xy, self.grad_er).item())
            corrected = (
                project(self.grad_xy, self.grad_er)
                if pre_dot < 0.0
                else self.grad_xy
            )
            overwrite_grad(self.parameters, corrected, self.grad_dims)
            if self.pairing.should_diagnose:
                self.pairing.record(
                    {
                        "schema_version": "gradient_reuse.agem_projection.v1",
                        "method": self.NAME,
                        "intervention": self.pairing.mode,
                        "observation": self.pairing.observation,
                        "event": self.pairing.event,
                        "channel": "reference_projection",
                        "projection_active": pre_dot < 0.0,
                        "pre_projection_dot": pre_dot,
                        "post_projection_dot": float(
                            torch.dot(corrected, self.grad_er).item()
                        ),
                    }
                )

        self.opt.step()
        return loss.item()
