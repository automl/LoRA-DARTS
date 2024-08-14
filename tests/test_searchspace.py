import copy
import unittest

import torch
from torch import nn

from confopt.searchspace import (
    DARTSSearchSpace,
    RobustDARTSSearchSpace,
)
from confopt.searchspace.common.base_search import SearchSpace
from confopt.searchspace.common.lora_layers import LoRALayer
from confopt.searchspace.darts.core.model_search import Cell as DARTSSearchCell
from confopt.searchspace.darts.core.operations import (
    FactorizedReduce,
    Identity,
    SepConv,
    Zero,
)
from confopt.searchspace.robust_darts.core.model_search import (
    Cell as RobustDARTSSearchCell,
)
from confopt.searchspace.robust_darts.core.operations import NoiseOp
from confopt.searchspace.robust_darts.core.spaces import spaces_dict
from utils import get_modules_of_type  # type: ignore

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _test_deactivate_lora(search_space: SearchSpace) -> None:
    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.activate_lora(r=4)

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert module.conv.weight.requires_grad is False

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.deactivate_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 0
            assert hasattr(module, "_original_r")
            assert module._original_r == 4
            assert module.conv.weight.requires_grad is True


def _test_toggle_lora(search_space: SearchSpace) -> None:  # noqa: C901
    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.activate_lora(r=4)

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert module.conv.weight.requires_grad is False

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            module.toggle_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 0
            assert hasattr(module, "_original_r")
            assert module._original_r == 4
            assert module.conv.weight.requires_grad is True

    for _, module in search_space.named_modules(remove_duplicate=True):
        if isinstance(module, LoRALayer):
            module.toggle_lora()

    for _, module in search_space.named_modules(remove_duplicate=False):
        if isinstance(module, LoRALayer):
            assert module.r == 4
            assert not hasattr(module, "_original_r")
            assert module.conv.weight.requires_grad is False


class TestDARTSSearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = DARTSSearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2
        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = DARTSSearchSpace()
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        layers = 6
        num_classes = 11
        search_space = DARTSSearchSpace(C=C, layers=layers, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, DARTSSearchCell)
        assert len(search_cells) == layers

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 2

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_discretize_supernet(self) -> None:
        # TODO: check to have one operation on each edge of the search space
        C = 32
        layers = 6
        num_classes = 10
        search_space = DARTSSearchSpace(
            C=C,
            layers=layers,
            num_classes=num_classes,
        )

        new_model = search_space.discretize()
        new_model.drop_path_prob = 0.1  # type: ignore

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = new_model(x)

        assert logits.shape == torch.Size([2, num_classes])

    def test_optim_forward_pass(self) -> None:
        search_space = DARTSSearchSpace()
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(
            [*search_space.arch_parameters, *search_space.beta_parameters]
        )
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def test_lora_parameters(self) -> None:
        search_space = DARTSSearchSpace()
        model_optimizer = torch.optim.Adam(search_space.model_weight_parameters())
        for _, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.activate_lora(r=4)
        opt_hyperparams = model_optimizer.defaults
        model_optimizer = type(model_optimizer)(
            search_space.model_weight_parameters(), **opt_hyperparams
        )
        model_params = search_space.model_weight_parameters()

        assert model_params == model_optimizer.param_groups[0]["params"]

    def test_deactivate_lora(self) -> None:
        _test_deactivate_lora(DARTSSearchSpace())

    def test_toggle_lora(self) -> None:
        _test_toggle_lora(DARTSSearchSpace())


class TestRobustDARTSSearchSpace(unittest.TestCase):
    def test_supernet_init(self) -> None:
        C = 32
        layers = 6
        num_classes = 11
        search_space = RobustDARTSSearchSpace(
            "s1", C=C, layers=layers, num_classes=num_classes
        )

        search_cells = get_modules_of_type(search_space.model, RobustDARTSSearchCell)
        assert len(search_cells) == layers

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 2

        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

    def _test_arch_parameters(self, space: str, n_ops: int) -> None:
        search_space = RobustDARTSSearchSpace(space=space)
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2

        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)
        assert arch_params[0].shape == (14, n_ops)
        assert arch_params[0].shape == (14, n_ops)

    def test_arch_parameters_s1(self) -> None:
        self._test_arch_parameters("s1", 2)

    def test_arch_parameters_s2(self) -> None:
        self._test_arch_parameters("s2", 2)

    def test_arch_parameters_s3(self) -> None:
        self._test_arch_parameters("s3", 3)

    def test_arch_parameters_s4(self) -> None:
        self._test_arch_parameters("s4", 2)

    def test_forward_pass(self) -> None:
        search_space = RobustDARTSSearchSpace("s1")
        x = torch.randn(2, 3, 64, 64).to(DEVICE)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_init_search_spaces(self) -> None:
        s1 = RobustDARTSSearchSpace(space="s1")
        assert s1.model is not None

        s2 = RobustDARTSSearchSpace(space="s2")
        assert s1.model is not None

        s3 = RobustDARTSSearchSpace(space="s3")
        assert s1.model is not None

        s4 = RobustDARTSSearchSpace(space="s4")
        assert s1.model is not None

        with self.assertRaises(ValueError):
            RobustDARTSSearchSpace(space="s5")

        with self.assertRaises(ValueError):
            RobustDARTSSearchSpace(space="S1")

    def _test_search_space_forward(self, space: str) -> None:
        search_space = RobustDARTSSearchSpace(space=space)
        x = torch.randn(2, 3, 64, 64).to(DEVICE)
        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_search_space_forward_s1(self) -> None:
        self._test_search_space_forward("s1")

    def test_search_space_forward_s2(self) -> None:
        self._test_search_space_forward("s2")

    def test_search_space_forward_s3(self) -> None:
        self._test_search_space_forward("s3")

    def test_search_space_forward_s4(self) -> None:
        self._test_search_space_forward("s4")

    def _test_search_space_candidate_ops(
        self, space: str, candidate_ops: list[str]
    ) -> None:
        search_space = RobustDARTSSearchSpace(space=space)
        cells = search_space.model.cells

        op_mapping = {
            "skip_connect": (Identity, FactorizedReduce),
            "sep_conv_3x3": SepConv,
            "none": Zero,
            "noise": NoiseOp,
        }

        for cell in cells[:3]:
            for operation_choices in cell._ops:
                ops = operation_choices.ops

                for idx, op in enumerate(ops):
                    correct_op = op_mapping[candidate_ops[idx]]
                    if isinstance(correct_op, tuple):
                        assert isinstance(op, correct_op[0]) or isinstance(
                            op, correct_op[1]
                        )
                    else:
                        assert isinstance(op, correct_op)

    def test_search_space_s1_ops(self) -> None:
        s1 = RobustDARTSSearchSpace(space="s1")
        cells = s1.model.cells

        for cell in cells:
            for operation_choices in cell._ops:
                ops = operation_choices.ops
                assert len(ops) == 2
                assert isinstance(ops[0], nn.Module)

    def test_search_space_s2_ops(self) -> None:
        self._test_search_space_candidate_ops("s2", ["skip_connect", "sep_conv_3x3"])

    def test_search_space_s3_ops(self) -> None:
        self._test_search_space_candidate_ops(
            "s3", ["none", "skip_connect", "sep_conv_3x3"]
        )

    def test_search_space_s4_ops(self) -> None:
        self._test_search_space_candidate_ops("s4", ["noise", "sep_conv_3x3"])

    def test_optim_forward_pass(self) -> None:
        search_space = RobustDARTSSearchSpace(space="s1")
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)
        arch_optim = torch.optim.Adam(
            [*search_space.arch_parameters, *search_space.beta_parameters]
        )
        arch_optim.zero_grad()
        out = search_space(x)
        loss = loss_fn(out[1], y)
        loss.backward()
        alphas_before = copy.deepcopy(search_space.arch_parameters)
        arch_optim.step()
        alphas_after = search_space.arch_parameters
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)


if __name__ == "__main__":
    unittest.main()
