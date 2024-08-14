from __future__ import annotations

import unittest

import torch

from confopt.oneshot.archsampler import (
    BaseSampler,
    DARTSSampler,
    DRNASSampler,
)
from confopt.oneshot.dropout import Dropout
from confopt.searchspace.darts.supernet import DARTSSearchSpace

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestArchSamplers(unittest.TestCase):
    def _sampler_new_step_or_epoch(
        self, sampler: BaseSampler, sample_frequency: str
    ) -> None:
        if sample_frequency == "epoch":
            sampler.new_epoch()
        elif sample_frequency == "step":
            sampler.new_step()
        else:
            raise ValueError(f"Unknown sample_frequency: {sample_frequency}")

    def assert_rows_one_hot(self, alphas: list[torch.Tensor]) -> None:
        for row in alphas:
            assert torch.sum(row == 1.0) == 1
            assert torch.sum(row == 0.0) == row.numel() - 1

    def test_darts_sampler(self) -> None:
        searchspace = DARTSSearchSpace()
        sampler = DARTSSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def _test_darts_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = DARTSSearchSpace()
        sampler = DARTSSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def test_darts_sampler_new_step(self) -> None:
        self._test_darts_sampler_new_step_epoch(sample_frequency="step")

    def test_darts_sampler_new_epoch(self) -> None:
        self._test_darts_sampler_new_step_epoch(sample_frequency="epoch")

    def test_drnas_sampler(self) -> None:
        searchspace = DARTSSearchSpace()
        sampler = DRNASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.0]).to(DEVICE))

    def _test_drnas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = DARTSSearchSpace()
        sampler = DRNASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency,
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.0]).to(DEVICE))

    def test_drnas_sampler_new_step(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="step")

    def test_drnas_sampler_new_epoch(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="epoch")

    def test_illegal_sample_frequency(self) -> None:
        arch_parameters = [torch.randn(5, 5)]
        with self.assertRaises(AssertionError):
            DARTSSampler(arch_parameters=arch_parameters, sample_frequency="illegal")

        with self.assertRaises(AssertionError):
            DRNASSampler(arch_parameters=arch_parameters, sample_frequency="illegal")


class TestDropout(unittest.TestCase):
    def test_dropout_probability(self) -> None:
        probability = 0.1
        arch_parameters = torch.ones(1000)

        dropout = Dropout(p=probability)
        output = dropout.apply_mask(arch_parameters)
        dropped_percent = (1000 - torch.count_nonzero(output)) / 1000

        self.assertAlmostEqual(
            probability, dropped_percent.numpy(), places=1
        )  # type: ignore

    def test_negative_probability(self) -> None:
        self._test_probabilities(-1.0)

    def test_too_large_probability(self) -> None:
        self._test_probabilities(1.0)

    def _test_probabilities(self, probability: float) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=probability)

    def test_illegal_anneal_frequency(self) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_frequency="illegal")

    def test_illegal_anneal_type_and_frequency(self) -> None:
        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_frequency="epoch")

        with self.assertRaises(AssertionError):
            Dropout(p=0.5, anneal_type="linear")


if __name__ == "__main__":
    unittest.main()
