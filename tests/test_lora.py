from __future__ import annotations

import unittest

import torch

from confopt.searchspace.common import Conv2DLoRA

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestLoRA(unittest.TestCase):
    def test_initialization(self) -> None:
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        lora_conv2d = Conv2DLoRA(in_channels, out_channels, kernel_size)

        assert lora_conv2d.conv.weight.shape == torch.Size(
            [out_channels, in_channels, *kernel_size]
        )

        assert not hasattr(lora_conv2d, "lora_A")
        assert not hasattr(lora_conv2d, "lora_B")

    def test_activate_lora(self) -> None:
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        r = 8
        lora_conv2d = Conv2DLoRA(in_channels, out_channels, kernel_size)

        assert not hasattr(lora_conv2d, "lora_A")
        assert not hasattr(lora_conv2d, "lora_B")

        lora_conv2d.activate_lora(r=8)

        assert lora_conv2d.lora_A.shape == torch.Size(
            [r * kernel_size[0], in_channels * kernel_size[0]]
        )

        assert lora_conv2d.lora_B.shape == torch.Size(
            [out_channels * kernel_size[0], r * kernel_size[0]]
        )

    def test_deactivate_lora(self) -> None:
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        r = 8
        lora_conv2d = Conv2DLoRA(in_channels, out_channels, kernel_size)

        lora_conv2d.activate_lora(r=r)

        assert hasattr(lora_conv2d, "lora_A")
        assert hasattr(lora_conv2d, "lora_B")

        weight_before = lora_conv2d.conv.weight.clone()
        lora_conv2d.deactivate_lora()
        weight_after = lora_conv2d.conv.weight.clone()

        assert torch.all(
            torch.eq(
                weight_before
                + (lora_conv2d.lora_B @ lora_conv2d.lora_A).view(weight_before.shape),
                weight_after,
            )
        )

    def test_toggle_lora(self) -> None:
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        r = 8
        lora_conv2d = Conv2DLoRA(in_channels, out_channels, kernel_size)

        assert not hasattr(lora_conv2d, "lora_A")
        assert not hasattr(lora_conv2d, "lora_B")

        weight_before_activation = lora_conv2d.conv.weight.clone()
        lora_conv2d.activate_lora()  # Activating
        weight_after_activation = lora_conv2d.conv.weight.clone()
        assert lora_conv2d.conv.weight.requires_grad == False
        assert torch.all(torch.eq(weight_before_activation, weight_after_activation))

        weight_before_first_toggle = lora_conv2d.conv.weight.clone()
        lora_conv2d.toggle_lora()  # Deactivating
        weight_after_first_toggle = lora_conv2d.conv.weight.clone()
        assert lora_conv2d.conv.weight.requires_grad == True
        assert torch.all(
            torch.eq(
                weight_before_first_toggle
                + (lora_conv2d.lora_B @ lora_conv2d.lora_A).view(
                    weight_before_first_toggle.shape
                ),
                weight_after_first_toggle,
            )
        )

        weight_before_second_toggle = lora_conv2d.conv.weight.clone()
        lora_conv2d.toggle_lora()  # Activating again
        weight_after_second_toggle = lora_conv2d.conv.weight.clone()
        assert lora_conv2d.conv.weight.requires_grad == False
        assert torch.all(
            torch.eq(
                weight_before_second_toggle
                - (lora_conv2d.lora_B @ lora_conv2d.lora_A).view(
                    weight_before_second_toggle.shape
                ),
                weight_after_second_toggle,
            )
        )

    def test_reset_parameters(self) -> None:
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        r = 8
        lora_conv2d = Conv2DLoRA(in_channels, out_channels, kernel_size)
        lora_conv2d.activate_lora(r=r)
        lora_conv2d.lora_A.data = torch.randn_like(lora_conv2d.lora_A.data)
        lora_conv2d.lora_B.data = torch.randn_like(lora_conv2d.lora_B.data)

        a = lora_conv2d.lora_A.data.clone()
        b = lora_conv2d.lora_B.data.clone()

        lora_conv2d.reset_parameters()

        assert torch.any(lora_conv2d.lora_A.data != a)
        assert torch.any(lora_conv2d.lora_B != b)

        assert torch.any(lora_conv2d.lora_B == 0)


if __name__ == "__main__":
    unittest.main()
