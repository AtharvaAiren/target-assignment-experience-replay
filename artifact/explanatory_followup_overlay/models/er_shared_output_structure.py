"""Experience replay with matched within/across-domain four-row swaps."""

from __future__ import annotations

import torch

from models.utils.continual_model import ContinualModel
from models.utils.shared_output_structure_intervention import (
    SharedOutputStructureIntervention,
    add_shared_output_structure_args,
    cross_entropy_objective,
)
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer


class ErSharedOutputStructure(ContinualModel):
    NAME = "er_shared_output_structure"
    # Mammoth uses class-IL compatibility as the signal that inference needs no
    # task argument.  The dataset guard below still restricts this model to the
    # shared ten-class Permuted-MNIST domain-IL protocol.
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return add_shared_output_structure_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if dataset is None or not (
            getattr(dataset, "NAME", None) == "perm-mnist"
            and int(getattr(dataset, "N_TASKS", 0)) == 20
            and int(getattr(dataset, "N_CLASSES", 0)) == 10
            and getattr(dataset, "SETTING", None) == "domain-il"
        ):
            raise RuntimeError("shared-output model requires the pinned 20-domain Permuted-MNIST")
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.shared_structure = SharedOutputStructureIntervention(args, self.NAME)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del epoch
        self.shared_structure.begin_observation()
        current_labels = labels.reshape(-1).long()
        if bool(((current_labels < 0) | (current_labels >= 10)).any()):
            raise RuntimeError("current Permuted-MNIST target is outside the shared output space")
        current_domain = int(self.current_task)
        self.opt.zero_grad()
        if self.shared_structure.arm == "no_replay":
            self.shared_structure.no_replay_update(
                rows=int(current_labels.numel()), current_domain=current_domain,
            )
        elif not self.buffer.is_empty():
            replay = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform,
                return_index=True, device=self.device,
            )
            if len(replay) != 4:
                raise RuntimeError("shared-output buffer omitted stored domain IDs")
            replay_indexes, replay_inputs, replay_labels, replay_domains = replay
            replay_labels = replay_labels.reshape(-1).long()
            replay_domains = replay_domains.reshape(-1).long()
            plan = self.shared_structure.plan(
                replay_labels, replay_domains, current_domain=current_domain,
            )
            self.shared_structure.gradient_diagnostic(
                self.net, replay_inputs, replay_labels, plan, cross_entropy_objective,
            )
            used_labels = self.shared_structure.apply(replay_labels, plan)
            self.shared_structure.exposure(
                replay_labels, replay_domains, plan, current_domain=current_domain,
                sampled_indexes=replay_indexes, sampled_inputs=replay_inputs,
            )
            inputs = torch.cat((inputs, replay_inputs))
            current_labels = torch.cat((current_labels, used_labels))
        outputs = self.net(inputs)
        if outputs.ndim != 2 or outputs.shape[-1] != 10:
            raise RuntimeError("shared-output training logits do not expose exactly ten unmasked classes")
        loss = self.loss(outputs, current_labels)
        loss.backward()
        self.opt.step()
        if self.shared_structure.arm != "no_replay":
            domain_labels = torch.full_like(labels.reshape(-1).long(), current_domain)
            self.buffer.add_data(
                examples=not_aug_inputs, labels=labels.reshape(-1).long(),
                task_labels=domain_labels,
            )
        return loss.item()

    def end_task(self, dataset):
        self.shared_structure.final_evaluation(
            self.net, dataset, completed_domains=int(self.current_task) + 1,
        )
