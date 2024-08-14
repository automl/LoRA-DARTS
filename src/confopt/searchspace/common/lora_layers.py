############################################################
# Copyright (c) Microsoft Corporation [Github LoRA], 2021.
############################################################

from __future__ import annotations  # noqa: I001

from abc import abstractmethod
import math
from typing import Callable

from torch import nn
import torch


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ) -> None:
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout_p = lora_dropout
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout: Callable | nn.Dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    @abstractmethod
    def _initialize_AB(self) -> None:  # noqa: N802
        pass

    def activate_lora(
        self,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout_rate: float = 0,
        merge_weights: bool = True,
    ) -> None:
        if hasattr(
            self, "_original_r"
        ):  # if it's being activated again after deactivation
            self.r = self._original_r
            self.unmerge_lora_weights()
            self.conv.weight.requires_grad = False  # type: ignore
            self.lora_A.requires_grad = True  # type: ignore
            self.lora_B.requires_grad = True  # type: ignore
            del self._original_r
            return

        assert self.r == 0, "rank can only be changed once"
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout_rate
        self.merge_weights = merge_weights
        if lora_dropout_rate > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout_rate)
        else:
            self.lora_dropout = lambda x: x
        self._initialize_AB()

    def deactivate_lora(self) -> None:
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self._original_r = self.r
            self.merge_lora_weights()
            self.r = 0
            self.conv.weight.requires_grad = True  # type: ignore
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            if self.conv.bias is not None:  # type: ignore
                self.conv.bias.requires_grad = True  # type: ignore

    def toggle_lora(self) -> None:
        assert hasattr(self, "lora_A"), "LoRA components are not initialized"
        assert hasattr(self, "lora_B"), "LoRA components are not initialized"

        if self.r > 0:
            self.deactivate_lora()
        else:
            self.activate_lora()

    def merge_lora_weights(self) -> None:
        if self.r > 0:
            # Merge the weights and mark it
            self.conv.weight.data += (self.lora_B @ self.lora_A).view(  # type: ignore
                self.conv.weight.shape  # type: ignore
            ) * self.scaling  # type: ignore
        self.merged = True

    def unmerge_lora_weights(self) -> None:
        if self.r > 0:
            # Make sure that the weights are not merged
            self.conv.weight.data -= (self.lora_B @ self.lora_A).view(  # type: ignore
                self.conv.weight.shape  # type: ignore
            ) * self.scaling  # type: ignore
        self.merged = False


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(  # type: ignore
        self,
        conv_module: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs,
    ) -> None:
        super().__init__()  # type: ignore
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # TODO Refactor this line for a better design
        LoRALayer.__init__(
            self,
            r=0,
            lora_alpha=1,
            lora_dropout=0.0,
            merge_weights=True,
        )
        if not isinstance(kernel_size, int):
            if isinstance(kernel_size, tuple):
                assert len(kernel_size) == 2
                assert kernel_size[0] == kernel_size[1], (
                    "This module is not implemented for different height and"
                    + " width kernels"
                )
                self.kernel_size: int = kernel_size[0]
            else:
                raise TypeError("Incompatible kernel size parameter")
        else:
            self.kernel_size = kernel_size

        # Actual trainable parameters
        # TODO Refactor ConvLoRA to think of a better way to initialize lora parameters
        # if r > 0:
        #     self._initialize_AB()
        self.reset_parameters()
        self.merged = False

        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def _initialize_AB(self) -> None:  # noqa: N802
        assert (
            self.r > 0
        ), "a value of rank > 0 is required to initialize LoRA components"
        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros(
                (self.r * self.kernel_size, self.in_channels * self.kernel_size)
            )
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros(
                (
                    self.out_channels // self.conv.groups * self.kernel_size,
                    self.r * self.kernel_size,
                )
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = self.lora_alpha / self.r
        # Freezing the pre-trained weight matrix
        self.conv.weight.requires_grad = False

    def reset_parameters(self) -> None:
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):  # type: ignore
        super().train(mode)
        if mode:
            if self.merge_weights and self.merged:
                self.unmerge_lora_weights()
        else:
            if self.merge_weights and not self.merged:  # noqa: PLR5501
                self.merge_lora_weights()

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None
    ) -> torch.Tensor:
        weight = self.conv.weight if weight is None else weight
        bias = self.conv.bias if bias is None else bias

        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(  # type: ignore
                x,
                weight + (self.lora_B @ self.lora_A).view(weight.shape) * self.scaling,
                bias,
            )

        return self.conv._conv_forward(x, weight, bias)


class Conv2DLoRA(ConvLoRA):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Creates a 2D convolution layer.

        Args:
            *args : Any
            **kwargs : Any

        The args order would be:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int or tuple): The size of the convolution kernel.
            stride (int or tuple, optional): The stride of the convolution operation.
            Default is 1.
            padding (int or tuple, optional): The amount of zero padding. Default is 0.
            dilation (int or tuple, optional): The spacing between kernel elements.
            Default is 1.
            groups (int, optional): The number of blocked connections from input
            channels to output channels. Default is 1.
            bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.

        Returns:
            torch.Tensor: The output tensor after applying the 2D convolution.

        Notes:
            - The input tensor should have shape (batch_size, in_channels, height,
            width).
            - The kernel size can be specified as a single integer or a tuple
            (kernel_height, kernel_width).
            - If `bias` is True, the layer learns an additive bias term for each output
            channel.
            - The LoRA modules are not initialized by default
            - One can call activate_lora() function to initialize LoRA components
            - For more information, see the PyTorch documentation:
              https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__(nn.Conv2d, *args, **kwargs)  # type: ignore
