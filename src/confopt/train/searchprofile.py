from __future__ import annotations

import torch

from confopt.oneshot.archsampler import BaseSampler, DARTSSampler
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.lora_toggler import LoRAToggler
from confopt.oneshot.weightentangler import WeightEntangler
from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common import (
    LoRALayer,
    OperationBlock,
    OperationChoices,
    SearchSpace,
)


class Profile:
    def __init__(
        self,
        sampler: BaseSampler,
        dropout: Dropout | None = None,
        weight_entangler: WeightEntangler | None = None,
        lora_configs: dict | None = None,
        lora_toggler: LoRAToggler | None = None,
    ) -> None:
        self.sampler = sampler
        self.dropout = dropout
        self.weight_entangler = weight_entangler
        self.lora_configs = lora_configs
        self.lora_toggler = lora_toggler

    def adapt_search_space(self, search_space: SearchSpace) -> None:
        for name, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, OperationChoices):
                new_module = self._initialize_operation_block(
                    module.ops, module.is_reduction_cell
                )
                parent_name, attribute_name = self.get_parent_and_attribute(name)
                setattr(
                    eval("search_space" + parent_name),
                    attribute_name,
                    new_module,
                )
        search_space.components.append(self.sampler)

        if self.dropout:
            search_space.components.append(self.dropout)

        if self.lora_toggler:
            search_space.components.append(self.lora_toggler)

    def update_sample_function_from_sampler(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.sampler.sample_alphas)

    def default_sample_function(self, alphas: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(alphas, dim=-1)

    def reset_sample_function(self, search_space: SearchSpace) -> None:
        search_space.set_sample_function(self.default_sample_function)

    def _initialize_operation_block(
        self, ops: torch.nn.Module, is_reduction_cell: bool = False
    ) -> OperationBlock:
        op_block = OperationBlock(
            ops,
            is_reduction_cell=is_reduction_cell,
            dropout=self.dropout,
            weight_entangler=self.weight_entangler,
        )
        return op_block

    def activate_lora(
        self,
        searchspace: SearchSpace,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        merge_weights: bool = True,
    ) -> None:
        if r > 0:
            for _, module in searchspace.named_modules(remove_duplicate=False):
                if isinstance(module, LoRALayer):
                    module.activate_lora(
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout_rate=lora_dropout,
                        merge_weights=merge_weights,
                    )

    def deactivate_lora(
        self,
        searchspace: SearchSpace,
    ) -> None:
        for _, module in searchspace.named_modules(remove_duplicate=False):
            if isinstance(module, LoRALayer):
                module.deactivate_lora()

    def toggle_lora(
        self,
        searchspace: SearchSpace,
    ) -> None:
        for _, module in searchspace.named_modules(remove_duplicate=True):
            if isinstance(module, LoRALayer):
                module.toggle_lora()

    def get_parent_and_attribute(self, module_name: str) -> tuple[str, str]:
        split_index = module_name.rfind(".")
        if split_index != -1:
            parent_name = module_name[:split_index]
            attribute_name = module_name[split_index + 1 :]
        else:
            parent_name = ""
            attribute_name = module_name
            return parent_name, attribute_name
        parent_name_list = parent_name.split(".")
        for idx, comp in enumerate(parent_name_list):
            try:
                if isinstance(eval(comp), int):
                    parent_name_list[idx] = "[" + comp + "]"
            except:  # noqa: E722, S112
                continue

        parent_name = ""
        for comp in parent_name_list:
            if "[" in comp:
                parent_name += comp
            else:
                parent_name += "." + comp
        return parent_name, attribute_name


if __name__ == "__main__":
    search_space = DARTSSearchSpace()
    sampler = DARTSSampler(search_space.arch_parameters)
    profile = Profile(sampler=sampler)
    profile.adapt_search_space(search_space=search_space)
    print("success")
