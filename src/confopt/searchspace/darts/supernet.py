from __future__ import annotations

import torch
from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import DARTSSearchModel
from .core.genotypes import DARTSGenotype

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DARTSSearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        """DARTS Search Space for Neural Architecture Search.

        This class represents a search space for neural architecture search using
        DARTS (Differentiable Architecture Search).

        Args:
            *args: Variable length positional arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.
            **kwargs: Variable length keyword arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.

        Keyword Args:
            C (int): Number of channels.
            num_classes (int): Number of output classes.
            layers (int): Number of layers in the network.
            criterion (nn.modules.loss._Loss): Loss function.
            steps (int): Number of steps in the search space cell.
            multiplier (int): Multiplier for channels in the cells.
            stem_multiplier (int): Stem multiplier for channels.

        Methods:
            - arch_parameters: Get architectural parameters.
            - beta_parameters: Get beta parameters.
            - set_arch_parameters(arch_parameters): Set architectural parameters

        Example:
            You can create an instance of DARTSSearchSpace with optional arguments as:
            >>> search_space = DARTSSearchSpace(
                                    C=32,
                                    num_classes=20,
                                    layers=10,
                                    criterion=nn.CrossEntropyLoss(),
                                    steps=5,
                                    multiplier=3,
                                    stem_multiplier=2,
                                    dropout=0.2)
        """
        model = DARTSSearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the alpha parameters of the model
        Return:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.
        """
        return self.model.arch_parameters()  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        assert len(arch_parameters) == len(self.arch_parameters)
        assert arch_parameters[0].shape == self.arch_parameters[0].shape
        (
            self.model.alphas_normal.data,
            self.model.alphas_reduce.data,
        ) = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_normal,
            self.model.alphas_reduce,
        ]

    def discretize(self) -> nn.Module:
        return self.model.discretize()  # type: ignore

    def get_genotype(self) -> DARTSGenotype:
        return self.model.genotype()  # type: ignore

    def get_num_skip_ops(self) -> tuple[int, int]:
        alphas_normal, alphas_reduce = self.model.arch_parameters()
        count_skip = lambda alphas: sum(alphas[:, 1:].argmax(dim=1) == 2)
        return count_skip(alphas_normal), count_skip(alphas_reduce)
