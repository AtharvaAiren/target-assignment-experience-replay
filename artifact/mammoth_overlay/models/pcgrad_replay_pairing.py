"""PCGrad replay adaptation with controlled replay-target assignment.

This is explicitly a two-objective *adaptation* of PCGrad to continual replay,
not a claim that PCGrad itself is a canonical continual-learning method.  The
two objectives are current-batch cross-entropy and replay-batch
cross-entropy.  The only experimental intervention changes the assignment of
already stored replay labels to the sampled replay inputs.
"""

from __future__ import annotations

import numpy as np

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from models.utils.pairing_intervention import (
    PairingIntervention,
    add_pairing_audit_args,
    cross_entropy_objective,
    gradient_cosine,
)
from models.utils.pcgrad_replay import pcgrad_replay_merge
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class PCGradReplayPairing(ContinualModel):
    """Equal-weight PCGrad replay adaptation with paired/permuted labels."""

    NAME = "pcgrad_replay_pairing"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_pairing_audit_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = [parameter.data.numel() for parameter in self.parameters()]
        dimension = int(np.sum(self.grad_dims))
        self.current_gradient = self._new_gradient_buffer(dimension)
        self.replay_gradient = self._new_gradient_buffer(dimension)
        self.pairing = PairingIntervention(args, self.NAME)

    def _new_gradient_buffer(self, dimension):
        parameter = next(self.parameters())
        return parameter.new_empty(int(dimension))

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.pairing.begin_observation()

        self.opt.zero_grad()
        current_outputs = self.net(inputs)
        current_loss = self.loss(current_outputs, labels)
        current_loss.backward()
        store_grad(self.parameters, self.current_gradient, self.grad_dims)
        reported_loss = float(current_loss.detach().item())

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

            self.opt.zero_grad()
            replay_outputs = self.net(replay_inputs)
            replay_loss = self.loss(replay_outputs, used_labels)
            replay_loss.backward()
            store_grad(self.parameters, self.replay_gradient, self.grad_dims)

            projected = pcgrad_replay_merge(
                self.current_gradient, self.replay_gradient
            )
            overwrite_grad(self.parameters, projected.merged, self.grad_dims)
            reported_loss = 0.5 * (
                float(current_loss.detach().item())
                + float(replay_loss.detach().item())
            )

            if self.pairing.should_diagnose:
                self.pairing.record(
                    {
                        "schema_version": "gradient_reuse.pcgrad_replay_projection.v1",
                        "method": self.NAME,
                        "method_role": "PCGrad replay adaptation",
                        "intervention": self.pairing.mode,
                        "observation": self.pairing.observation,
                        "event": self.pairing.event,
                        "channel": "replay_projection",
                        "projection_active": projected.conflict,
                        "pre_projection_dot": projected.pre_dot,
                        "post_projection_dot": projected.post_dot,
                        "pre_projection_cosine": gradient_cosine(
                            self.current_gradient, self.replay_gradient
                        ),
                        "post_projection_cosine": gradient_cosine(
                            projected.current_projected,
                            projected.replay_projected,
                        ),
                        "current_gradient_norm": float(
                            self.current_gradient.double().norm().item()
                        ),
                        "replay_gradient_norm": float(
                            self.replay_gradient.double().norm().item()
                        ),
                        "merged_gradient_norm": float(
                            projected.merged.double().norm().item()
                        ),
                    }
                )

        self.opt.step()
        # Only the untouched current labels enter memory.  Replay labels are a
        # view created after sampling and are never written back to the buffer.
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)
        return reported_loss
