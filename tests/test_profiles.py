import unittest

from confopt.profiles import (
    DartsProfile,
    DRNASProfile,
    ProfileConfig,
)


class TestProfileConfig(unittest.TestCase):
    def test_config_change(self) -> None:
        profile = ProfileConfig(
            "TEST",
            epochs=1,
            dropout=0.5,
        )

        trainer_config = {"use_data_parallel": True}

        profile.configure_trainer(**trainer_config)

        assert (
            profile.trainer_config["use_data_parallel"]
            == trainer_config["use_data_parallel"]
        )

    def test_invalid_configuration(self) -> None:
        profile = ProfileConfig(
            "TEST",
            epochs=1,
            dropout=0.5,
        )

        trainer_config = {"invalid_config": False}

        dropout_config = {"invalid_config": "test"}

        with self.assertRaises(AssertionError):
            profile.configure_trainer(**trainer_config)

        with self.assertRaises(AssertionError):
            profile.configure_dropout(**dropout_config)


class TestDartsProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        profile = DartsProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )

        assert profile.sampler_config is not None

    def test_sampler_change(self) -> None:
        profile = DartsProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"]
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")


class TestDRNASProfile(unittest.TestCase):
    def test_initialization(self) -> None:
        profile = DRNASProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )

        assert profile.sampler_config is not None

    def test_sampler_change(self) -> None:
        profile = DRNASProfile(
            epochs=100,
            sampler_sample_frequency="step",
        )
        sampler_config = {"sample_frequency": "epoch"}
        profile.configure_sampler(**sampler_config)
        assert (
            profile.sampler_config["sample_frequency"]
            == sampler_config["sample_frequency"]
        )

        with self.assertRaises(AssertionError):
            profile.configure_sampler(invalid_config="step")


if __name__ == "__main__":
    unittest.main()
