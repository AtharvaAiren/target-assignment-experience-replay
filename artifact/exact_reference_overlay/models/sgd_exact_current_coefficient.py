"""Current-only SGD with the exact per-update current coefficient of ER."""

from __future__ import annotations

from models.utils.continual_model import ContinualModel
from models.utils.exact_current_coefficient import (
    ExactCurrentCoefficientAudit,
    add_exact_current_coefficient_args,
    assert_mean_cross_entropy_objective,
    assert_plain_sgd,
)
from utils.args import ArgumentParser


class SgdExactCurrentCoefficient(ContinualModel):
    """Remove replay while preserving ER's current-loss coefficient exactly."""

    NAME = "sgd_exact_current_coefficient"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        return add_exact_current_coefficient_args(parser)

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        optimizer_contract = assert_plain_sgd(self.opt)
        objective_contract = assert_mean_cross_entropy_objective(self.loss)
        self.exact_coefficient = ExactCurrentCoefficientAudit(
            args,
            self.NAME,
            optimizer_contract=optimizer_contract,
            objective_contract=objective_contract,
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        del not_aug_inputs, epoch
        # Mammoth may rebuild ``self.opt`` at task boundaries.  Reassert the
        # locked optimizer/objective semantics on every update so the audit
        # covers the optimizer actually used, not only the constructor state.
        assert_plain_sgd(self.opt)
        assert_mean_cross_entropy_objective(self.loss)
        if inputs.ndim == 0 or labels.ndim == 0 or inputs.shape[0] != labels.shape[0]:
            raise RuntimeError("current inputs and labels must have the same batch size")
        plan = self.exact_coefficient.plan(int(labels.shape[0]))

        self.opt.zero_grad()
        outputs = self.net(inputs)
        current_mean_loss = self.loss(outputs, labels)
        if current_mean_loss.ndim != 0:
            raise RuntimeError("exact coefficient control requires a scalar mean CE loss")
        loss = plan.coefficient * current_mean_loss
        loss.backward()
        self.opt.step()

        self.exact_coefficient.complete(
            plan,
            current_mean_loss=float(current_mean_loss.detach().item()),
            scaled_loss=float(loss.detach().item()),
        )
        return loss.item()
