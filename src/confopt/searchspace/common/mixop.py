from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.dropout import Dropout
from confopt.oneshot.weightentangler import WeightEntangler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
__all__ = ["OperationChoices", "OperationBlock"]


class OperationChoices(nn.Module):
    def __init__(self, ops: list[nn.Module], is_reduction_cell: bool = False) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        states = []
        for op, alpha in zip(self.ops, alphas):
            if hasattr(op, "is_pruned") and op.is_pruned:
                continue
            states.append(op(x) * alpha)

        return sum(states)  # type: ignore


class OperationBlock(nn.Module):
    def __init__(
        self,
        ops: list[nn.Module],
        is_reduction_cell: bool,
        dropout: Dropout | None = None,
        weight_entangler: WeightEntangler | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__()
        self.device = device
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell
        self.dropout = dropout
        self.weight_entangler = weight_entangler

    def forward_method(
        self, x: torch.Tensor, ops: list[nn.Module], alphas: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.weight_entangler is not None:
            return self.weight_entangler.forward(x, ops, alphas)

        states = []
        for op, alpha in zip(ops, alphas):
            states.append(op(x) * alpha)

        return sum(states)

    def forward(
        self,
        x: torch.Tensor,
        alphas: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.dropout:
            alphas = self.dropout.apply_mask(alphas)

        return self.forward_method(x, self.ops, alphas)
