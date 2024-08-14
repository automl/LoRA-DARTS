from __future__ import annotations

from abc import abstractmethod

import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSERIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)
INIT_CHANNEL_NUM = 16


class ProfileConfig:
    def __init__(
        self,
        config_type: str,
        epochs: int = 100,
        dropout: float | None = None,
        entangle_op_weights: bool = False,
        lora_rank: int = 0,
        lora_warm_epochs: int = 0,
        lora_toggle_epochs: list[int] | None = None,
        lora_toggle_probability: float | None = None,
        seed: int = 100,
        searchspace_str: str = "darts",
    ) -> None:
        self.config_type = config_type
        self.epochs = epochs
        self.lora_warm_epochs = lora_warm_epochs
        self.seed = seed
        self.searchspace_str = searchspace_str
        self._initialize_trainer_config()
        self._initialize_sampler_config()
        self._set_lora_configs(
            lora_rank,
            lora_warm_epochs,
            toggle_epochs=lora_toggle_epochs,
            lora_toggle_probability=lora_toggle_probability,
        )
        self._set_dropout(dropout)
        self.entangle_op_weights = entangle_op_weights
        PROFILE_TYPE = "BASE"
        self.sampler_type = str.lower(PROFILE_TYPE)

    def _set_lora_configs(
        self,
        lora_rank: int = 0,
        lora_warm_epochs: int = 0,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        lora_toggle_probability: float | None = None,
        merge_weights: bool = True,
        toggle_epochs: list[int] | None = None,
    ) -> None:
        self.lora_config = {
            "r": lora_rank,
            "lora_dropout": lora_dropout,
            "lora_alpha": lora_alpha,
            "merge_weights": merge_weights,
        }
        self.lora_toggle_epochs = toggle_epochs
        self.lora_warm_epochs = lora_warm_epochs
        self.lora_toggle_probability = lora_toggle_probability

    def _set_dropout(self, dropout: float | None = None) -> None:
        self.dropout = dropout
        self._initialize_dropout_config()

    def get_config(self) -> dict:
        assert (
            self.sampler_config is not None
        ), "atleast a sampler is needed to initialize the search space"
        weight_type = (
            "weight_entanglement" if self.entangle_op_weights else "weight_sharing"
        )
        config = {
            "sampler": self.sampler_config,
            "dropout": self.dropout_config,
            "trainer": self.trainer_config,
            "lora": self.lora_config,
            "lora_extra": {
                "toggle_epochs": self.lora_toggle_epochs,
                "warm_epochs": self.lora_warm_epochs,
                "toggle_probability": self.lora_toggle_probability,
            },
            "sampler_type": self.sampler_type,
            "searchspace_str": self.searchspace_str,
            "weight_type": weight_type,
        }

        if hasattr(self, "searchspace_config") and self.searchspace_config is not None:
            config.update({"search_space": self.searchspace_config})

        if hasattr(self, "extra_config") and self.extra_config is not None:
            config.update(self.extra_config)
        return config

    @abstractmethod
    def _initialize_sampler_config(self) -> None:
        self.sampler_config = None

    @abstractmethod
    def _initialize_trainer_config(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "arch_lr": 3e-4,
            "epochs": self.epochs,
            "lora_warm_epochs": self.lora_warm_epochs,
            "optim": "sgd",
            "arch_optim": "adam",
            "optim_config": {
                "momentum": 0.9,
                "nesterov": 0,
                "weight_decay": 3e-4,
            },
            "arch_optim_config": {
                "betas": (0.5, 0.999),
                "weight_decay": 1e-3,
            },
            "scheduler": "cosine_annealing_lr",
            "criterion": "cross_entropy",
            "batch_size": 64,
            "learning_rate_min": 0.001,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.5,
            "use_data_parallel": False,
            "checkpointing_freq": 1,
            "seed": self.seed,
        }

        self.trainer_config = trainer_config

    @abstractmethod
    def _initialize_dropout_config(self) -> None:
        dropout_config = {
            "p": self.dropout if self.dropout is not None else 0.0,
            "p_min": 0.0,
            "anneal_frequency": "epoch",
            "anneal_type": "linear",
            "max_iter": self.epochs,
        }
        self.dropout_config = dropout_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        assert self.sampler_config is not None
        for config_key in kwargs:
            assert (
                config_key in self.sampler_config  # type: ignore
            ), f"{config_key} not a valid configuration for the sampler of type \
                {self.config_type}"
            self.sampler_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_trainer(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.trainer_config
            ), f"{config_key} not a valid configuration for the trainer"
            self.trainer_config[config_key] = kwargs[config_key]

    def configure_dropout(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.dropout_config
            ), f"{config_key} not a valid configuration for the dropout module"
            self.dropout_config[config_key] = kwargs[config_key]

    def configure_lora_config(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.lora_config
            ), f"{config_key} not a valid configuration for the lora layers"
            self.lora_config[config_key] = kwargs[config_key]

    @abstractmethod
    def set_searchspace_config(self, config: dict) -> None:
        if not hasattr(self, "searchspace_config"):
            self.searchspace_config = config
        else:
            self.searchspace_config.update(config)

    @abstractmethod
    def configure_extra_config(self, config: dict) -> None:
        self.extra_config = config

    def get_name_wandb_run(self) -> str:
        name_wandb_run = []
        name_wandb_run.append(f"ss_{self.searchspace_str}")
        if self.entangle_op_weights:
            name_wandb_run.append("type_we")
        else:
            name_wandb_run.append("type_ws")
        name_wandb_run.append(f"opt_{self.sampler_type}")
        if self.lora_warm_epochs > 0:
            name_wandb_run.append(f"lorarank_{self.lora_config.get('r')}")
            name_wandb_run.append(f"lorawarmup_{self.lora_warm_epochs}")
        name_wandb_run.append(f"epochs_{self.trainer_config.get('epochs')}")
        name_wandb_run.append(f"seed_{self.seed}")
        name_wandb_run_str = "-".join(name_wandb_run)
        return name_wandb_run_str
