from __future__ import annotations

import argparse
from collections import namedtuple
from enum import Enum
import json
import random
from typing import Callable
import warnings

import numpy as np
import torch
from torch.backends import cudnn
import wandb

from confopt.dataset import (
    CIFAR10Data,
    CIFAR100Data,
    ImageNet16Data,
    ImageNet16120Data,
)
from confopt.oneshot.archsampler import (
    DARTSSampler,
    DRNASSampler,
)
from confopt.oneshot.dropout import Dropout
from confopt.oneshot.lora_toggler import LoRAToggler
from confopt.oneshot.weightentangler import WeightEntangler
from confopt.profiles import (
    DiscreteProfile,
    ProfileConfig,
)
from confopt.profiles.profiles import DartsProfile
from confopt.searchspace import (
    DARTSGenotype,  # noqa: F401
    DARTSImageNetModel,
    DARTSModel,
    DARTSSearchSpace,
    RobustDARTSSearchSpace,
    SearchSpace,
)
from confopt.train import ConfigurableTrainer, DiscreteTrainer, Profile
from confopt.utils import Logger
from confopt.utils.time import check_date_format

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO Change this to real data
ADVERSERIAL_DATA = (
    torch.randn(2, 3, 32, 32).to(DEVICE),
    torch.randint(0, 9, (2,)).to(DEVICE),
)


class SearchSpaceType(Enum):
    DARTS = "darts"
    RobustDARTS = "robust_darts"


class ModelType(Enum):
    DARTS = "darts"


class SamplerType(Enum):
    DARTS = "darts"
    DRNAS = "drnas"


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMGNET16 = "imgnet16"
    IMGNET16_120 = "imgnet16_120"


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET16_120: 120,
}


class CriterionType(Enum):
    CROSS_ENTROPY = "cross_entropy"


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ASGD = "asgd"


class SchedulerType(Enum):
    CosineAnnealingLR = "cosine_annealing_lr"
    CosineAnnealingWarmRestart = "cosine_annealing_warm_restart"


class Experiment:
    def __init__(
        self,
        search_space: SearchSpaceType,
        dataset: DatasetType,
        seed: int,
        is_wandb_log: bool = False,
        debug_mode: bool = False,
        exp_name: str = "test",
        runtime: str = "",
    ) -> None:
        self.search_space_str = search_space
        self.dataset_str = dataset
        self.seed = seed
        self.is_wandb_log = is_wandb_log
        self.debug_mode = debug_mode
        self.exp_name = exp_name
        self.runtime = runtime

    def set_seed(self, rand_seed: int) -> None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(rand_seed)

    def run_with_profile(
        self,
        profile: ProfileConfig,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_benchmark: bool = False,
    ) -> ConfigurableTrainer:
        config = profile.get_config()
        run_name = profile.get_name_wandb_run()

        assert hasattr(profile, "sampler_type")
        self.sampler_str = SamplerType(profile.sampler_type)
        self.dropout = profile.dropout
        self.entangle_op_weights = profile.entangle_op_weights

        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1
        return self.runner(
            config,
            start_epoch,
            load_saved_model,
            load_best_model,
            use_benchmark,
            run_name,
        )

    def runner(
        self,
        config: dict | None = None,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_benchmark: bool = False,
        run_name: str = "supernet_run",
    ) -> ConfigurableTrainer:
        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1

        self.set_seed(self.seed)

        if load_saved_model or load_best_model or start_epoch > 0:
            last_run = False
            if not self.runtime:
                last_run = True

            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                dataset=str(self.dataset_str.value),
                seed=self.seed,
                runtime=self.runtime,
                use_supernet_checkpoint=True,
                last_run=last_run,
            )
        else:
            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                dataset=str(self.dataset_str.value),
                search_space=self.search_space_str.value,
                seed=self.seed,
                runtime=self.runtime,
                use_supernet_checkpoint=True,
            )

        self._enum_to_objects(
            self.search_space_str,
            self.sampler_str,
            config=config,
            use_benchmark=use_benchmark,
        )
        if self.is_wandb_log:
            wandb.init(  # type: ignore
                name=run_name,
                project=(
                    config.get("project_name", "LoRA-DARTS")
                    if config is not None
                    else "LoRA-DARTS"
                ),
                config=config,
            )

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(config["trainer"].keys())  # type: ignore
        )
        trainer_arguments = Arguments(**config["trainer"])  # type: ignore

        criterion = self._get_criterion(
            criterion_str=trainer_arguments.criterion  # type: ignore
        )

        data = self._get_dataset(self.dataset_str)(
            root="datasets",
            cutout=trainer_arguments.cutout,  # type: ignore
            train_portion=trainer_arguments.train_portion,  # type: ignore
        )

        model = self.search_space
        # model =

        w_optimizer = self._get_optimizer(trainer_arguments.optim)(  # type: ignore
            model.model_weight_parameters(),
            trainer_arguments.lr,  # type: ignore
            **config["trainer"].get("optim_config", {}),  # type: ignore
        )

        w_scheduler = self._get_scheduler(
            scheduler_str=trainer_arguments.scheduler,  # type: ignore
            optimizer=w_optimizer,
            num_epochs=trainer_arguments.epochs,  # type: ignore
            eta_min=trainer_arguments.learning_rate_min,  # type: ignore
            config=config["trainer"].get("scheduler_config", {}),  # type: ignore
        )

        arch_optimizer = self._get_optimizer(
            trainer_arguments.arch_optim  # type: ignore
        )(
            model.arch_parameters,
            lr=config["trainer"].get("arch_lr", 0.001),  # type: ignore
            **config["trainer"].get("arch_optim_config", {}),  # type: ignore
        )

        trainer = ConfigurableTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            arch_optimizer=arch_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=trainer_arguments.batch_size,  # type: ignore
            use_data_parallel=trainer_arguments.use_data_parallel,  # type: ignore
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            start_epoch=start_epoch,
            checkpointing_freq=trainer_arguments.checkpointing_freq,  # type: ignore
            epochs=trainer_arguments.epochs,  # type: ignore
            debug_mode=self.debug_mode,
            query_dataset=self.dataset_str.value,
            benchmark_api=self.benchmark_api,
        )

        config_str = json.dumps(config, indent=2, default=str)
        self.logger.log(
            f"Training the supernet with the following configuration: \n{config_str}"
        )

        trainer.train(
            profile=self.profile,  # type: ignore
            is_wandb_log=self.is_wandb_log,
            lora_warm_epochs=config["trainer"].get(  # type: ignore
                "lora_warm_epochs", 0
            ),
        )

        return trainer

    def _enum_to_objects(
        self,
        search_space_enum: SearchSpaceType,
        sampler_enum: SamplerType,
        config: dict | None = None,
        use_benchmark: bool = False,
    ) -> None:
        if config is None:
            config = {}  # type : ignore
        self.set_search_space(search_space_enum, config.get("search_space", {}))
        self.set_sampler(sampler_enum, config.get("sampler", {}))
        self.set_dropout(config.get("dropout", {}))

        if use_benchmark:
            if (
                search_space_enum == SearchSpaceType.RobustDARTS
                and config.get("search_space", {}).get("space") == "s4"
            ):
                warnings.warn(
                    "Argument use_benchmark was set to True with s4 space of"
                    + " RobustDARTSSearchSpace. Consider setting it to False",
                    stacklevel=1,
                )
                self.benchmark_api = None
            else:
                self.set_benchmark_api(search_space_enum, config.get("benchmark", {}))
        else:
            self.benchmark_api = None

        self.set_lora_toggler(config.get("lora", {}), config.get("lora_extra", {}))
        self.set_weight_entangler()
        self.set_profile(config)

    def set_search_space(
        self,
        search_space: SearchSpaceType,
        config: dict,
    ) -> None:
        if search_space == SearchSpaceType.DARTS:
            self.search_space = DARTSSearchSpace(**config)
        elif search_space == SearchSpaceType.RobustDARTS:
            self.search_space = RobustDARTSSearchSpace(**config)

    def set_benchmark_api(
        self,
        search_space: SearchSpaceType,
        config: dict,
    ) -> None:
        if search_space in (SearchSpaceType.DARTS, SearchSpaceType.RobustDARTS):
            from confopt.benchmarks import NB301Benchmark

            self.benchmark_api = NB301Benchmark(**config)
        else:
            print(f"Benchmark does not exist for the {search_space.value} searchspace")
            self.benchmark_api = None

    def set_sampler(
        self,
        sampler: SamplerType,
        config: dict,
    ) -> None:
        arch_params = self.search_space.arch_parameters
        if sampler == SamplerType.DARTS:
            self.sampler = DARTSSampler(**config, arch_parameters=arch_params)
        elif sampler == SamplerType.DRNAS:
            self.sampler = DRNASSampler(**config, arch_parameters=arch_params)

    def set_dropout(self, config: dict) -> None:
        if self.dropout is not None:
            self.dropout = Dropout(**config)
        else:
            self.dropout = None

    def set_weight_entangler(self) -> None:
        self.weight_entangler = WeightEntangler() if self.entangle_op_weights else None

    def set_lora_toggler(self, lora_config: dict, lora_extra: dict) -> None:
        if lora_config.get("r", 0) == 0:
            self.lora_toggler = None
            return

        toggle_epochs = lora_extra.get("toggle_epochs")
        toggle_probability = lora_extra.get("toggle_probability")
        if toggle_epochs is not None:
            assert min(toggle_epochs) > lora_extra.get(
                "warm_epochs"
            ), "The first toggle epoch should be after the warmup epochs"
            self.lora_toggler = LoRAToggler(
                searchspace=self.search_space,
                toggle_epochs=toggle_epochs,
                toggle_probability=toggle_probability,
            )
        else:
            self.lora_toggler = None

    def set_profile(self, config: dict) -> None:
        assert self.sampler is not None

        self.profile = Profile(
            sampler=self.sampler,
            dropout=self.dropout,
            weight_entangler=self.weight_entangler,
            lora_configs=config.get("lora"),
            lora_toggler=self.lora_toggler,
        )

    def _get_dataset(self, dataset: DatasetType) -> Callable | None:
        if dataset == DatasetType.CIFAR10:
            return CIFAR10Data
        elif dataset == DatasetType.CIFAR100:  # noqa: RET505
            return CIFAR100Data
        elif dataset == DatasetType.IMGNET16:
            return ImageNet16Data
        elif dataset == DatasetType.IMGNET16_120:
            return ImageNet16120Data
        return None

    def _get_criterion(self, criterion_str: str) -> torch.nn.Module:
        criterion = CriterionType(criterion_str)
        if criterion == CriterionType.CROSS_ENTROPY:
            return torch.nn.CrossEntropyLoss()

        raise NotImplementedError

    def _get_optimizer(self, optim_str: str) -> Callable | None:
        optim = OptimizerType(optim_str)
        if optim == OptimizerType.ADAM:
            return torch.optim.Adam
        elif optim == OptimizerType.SGD:  # noqa: RET505
            return torch.optim.SGD
        if optim == OptimizerType.ASGD:
            return torch.optim.ASGD
        return None

    def _get_scheduler(
        self,
        scheduler_str: str,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        eta_min: float = 0.0,
        config: dict | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if config is None:
            config = {}
        scheduler = SchedulerType(scheduler_str)
        if scheduler == SchedulerType.CosineAnnealingLR:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=config.get("T_max", num_epochs),
                eta_min=config.get("eta_min", eta_min),
            )
        elif scheduler == SchedulerType.CosineAnnealingWarmRestart:  # noqa: RET505
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("T_0", 10),
                T_mult=config.get("T_mult", 1),
                eta_min=config.get("eta_min", eta_min),
            )
        return None

    def run_discrete_model_with_profile(
        self,
        profile: DiscreteProfile,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_supernet_checkpoint: bool = False,
    ) -> DiscreteTrainer:
        train_config = profile.get_trainer_config()
        searchspace_config = profile.get_searchspace_config(
            self.search_space_str.value, self.dataset_str.value
        )
        genotype_str = profile.get_genotype()

        return self.run_discrete_model(
            searchspace_config=searchspace_config,
            train_config=train_config,
            start_epoch=start_epoch,
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            use_supernet_checkpoint=use_supernet_checkpoint,
            genotype_str=genotype_str,
        )

    def get_discrete_model_from_genotype_str(
        self,
        search_space_str: str,
        genotype_str: str,
        searchspace_config: dict,
    ) -> torch.nn.Module:
        if search_space_str == ModelType.DARTS.value:
            searchspace_config["genotype"] = eval(genotype_str)
            if self.dataset_str.value in ("cifar10", "cifar100"):
                discrete_model = DARTSModel(**searchspace_config)
            elif self.dataset_str.value in ("imgnet16", "imgnet16_120"):
                discrete_model = DARTSImageNetModel(**searchspace_config)
            else:
                raise ValueError("undefined discrete model for this dataset.")
        else:
            raise ValueError("undefined discrete model for this search space.")

        return discrete_model

    def get_discrete_model_from_supernet(
        self,
    ) -> SearchSpace:
        # A) Use the experiment's self.search_space of the experiment.
        if hasattr(self, "search_space"):
            if self.search_space.arch_parameters:
                model = self.search_space.discretize()
                return model  # type: ignore
            raise ValueError("need to be a supernet to be able to get the discrete net")
        raise Exception("Need a searchspace to be able to fetch a discrete model")

    # logger load_genotype function handles for both supernet and discrete model
    def get_genotype_str_from_checkpoint(
        self,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_supernet_checkpoint: bool = False,
    ) -> str:
        if load_best_model or load_saved_model or (start_epoch > 0):
            genotype_str = self.logger.load_genotype(
                start_epoch,
                load_saved_model,
                load_best_model,
                use_supernet_checkpoint=use_supernet_checkpoint,
            )
            return genotype_str

        raise ValueError("is not a valid checkpoint.")

    def get_discrete_model(
        self,
        searchspace_config: dict,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_supernet_checkpoint: bool = False,
        use_expr_search_space: bool = False,
        genotype_str: str | None = None,
    ) -> tuple[torch.nn.Module, str]:
        # A) Use the experiment's self.search_space of the experiment.
        if use_expr_search_space:
            model = self.get_discrete_model_from_supernet()
            return model, model.get_genotype()  # type: ignore
        if sum([load_best_model, load_saved_model, (start_epoch > 0)]) == 1:
            genotype_str = self.get_genotype_str_from_checkpoint(
                start_epoch,
                load_saved_model,
                load_best_model,
                use_supernet_checkpoint,
            )
        elif sum(
            [
                load_best_model,
                load_saved_model,
                (start_epoch > 0),
                use_supernet_checkpoint,
            ],
        ) == 0 and hasattr(self, "search_space"):
            genotype_str = self.search_space.get_genotype().tostr()  # type: ignore

        elif genotype_str is None:
            raise ValueError("genotype cannot be empty")

        model = self.get_discrete_model_from_genotype_str(
            self.search_space_str.value,
            genotype_str,  # type: ignore
            searchspace_config,
        )
        return model, genotype_str  # type: ignore

    # refactor the name to train
    def run_discrete_model(
        self,
        searchspace_config: dict,
        train_config: dict | None = None,
        start_epoch: int = 0,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        use_supernet_checkpoint: bool = False,
        use_expr_search_space: bool = False,
        genotype_str: str | None = None,
    ) -> DiscreteTrainer:
        # should not care where the model comes from => genotype should be a
        # different function
        assert sum([load_best_model, load_saved_model, (start_epoch > 0)]) <= 1

        self.set_seed(self.seed)

        if load_saved_model or load_best_model or start_epoch > 0:
            last_run = False
            if not self.runtime:
                last_run = True

            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                dataset=str(self.dataset_str.value),
                seed=self.seed,
                runtime=self.runtime,
                use_supernet_checkpoint=use_supernet_checkpoint,
                last_run=last_run,
            )
        else:
            self.logger = Logger(
                log_dir="logs",
                exp_name=self.exp_name,
                search_space=self.search_space_str.value,
                dataset=str(self.dataset_str.value),
                seed=self.seed,
                runtime=None,
                use_supernet_checkpoint=use_supernet_checkpoint,
                last_run=False,
            )

        # different options to train a discrete model:
        # A) Use the experiment's self.search_space of the experiment.
        # B) From a supernet checkpoint, load, and discretize to get the model.
        # C) From a discerete checkpoint, load the model.
        # D) pass a genotype from the prompt to build a discrete model.
        # E) just use the default genotype which is set in the discrete_profile.

        model, genotype_str = self.get_discrete_model(
            searchspace_config=searchspace_config,
            start_epoch=start_epoch,
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            use_supernet_checkpoint=use_supernet_checkpoint,
            use_expr_search_space=use_expr_search_space,
            genotype_str=genotype_str,
        )
        model.to(device=DEVICE)
        # TODO: do i need this line?
        if use_supernet_checkpoint:
            start_epoch = 0
            load_saved_model = False
            load_best_model = False
            use_supernet_checkpoint = False
            self.logger.use_supernet_checkpoint = False
            self.logger.set_up_new_run()

        self.logger.save_genotype(genotype_str)

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(train_config.keys())  # type: ignore
        )
        trainer_arguments = Arguments(**train_config)  # type: ignore

        data = self._get_dataset(self.dataset_str)(
            root="datasets",
            cutout=trainer_arguments.cutout,  # type: ignore
            train_portion=trainer_arguments.train_portion,  # type: ignore
        )

        w_optimizer = self._get_optimizer(trainer_arguments.optim)(  # type: ignore
            model.parameters(),
            trainer_arguments.lr,  # type: ignore
            **trainer_arguments.optim_config,  # type: ignore
        )

        w_scheduler = self._get_scheduler(
            scheduler_str=trainer_arguments.scheduler,  # type: ignore
            optimizer=w_optimizer,
            num_epochs=trainer_arguments.epochs,  # type: ignore
            eta_min=trainer_arguments.learning_rate_min,  # type: ignore
            config=train_config.get("scheduler_config", {}),  # type: ignore
        )

        criterion = self._get_criterion(
            criterion_str=trainer_arguments.criterion  # type: ignore
        )

        trainer = DiscreteTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=trainer_arguments.batch_size,  # type: ignore
            use_data_parallel=trainer_arguments.use_data_parallel,  # type: ignore
            print_freq=trainer_arguments.print_freq,  # type: ignore
            drop_path_prob=trainer_arguments.drop_path_prob,  # type: ignore
            load_saved_model=load_saved_model,
            load_best_model=load_best_model,
            start_epoch=start_epoch,
            checkpointing_freq=trainer_arguments.checkpointing_freq,  # type: ignore
            epochs=trainer_arguments.epochs,  # type: ignore
        )

        trainer.train(
            epochs=trainer_arguments.epochs,  # type: ignore
            is_wandb_log=self.is_wandb_log,
        )

        trainer.test(is_wandb_log=self.is_wandb_log)

        return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning and training searched architectures", add_help=False
    )
    parser.add_argument(
        "--searchspace",
        default="darts",
        help="search space in (darts)",
        type=str,
    )
    parser.add_argument(
        "--sampler",
        default="darts",
        help="samplers in (darts, drnas)",
        type=str,
    )
    parser.add_argument(
        "--dropout",
        default=None,
        help="Dropout probability. 0 <= p < 1.",
        type=float,
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        default=False,
        help="Load the best model found from the previous run",
    )
    parser.add_argument(
        "--load_saved_model",
        action="store_true",
        default=False,
        help="Load the last saved model in the last run of training them",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        help="Specify the start epoch to continue the training of the model from the"
        + "previous run",
        type=int,
    )
    parser.add_argument(
        "--use_supernet_checkpoint",
        action="store_true",
        default=False,
        help="If you would like to load_best_model, load_saved_model, or start_epoch"
        + "from the supernets checkpoint pass True",
    )
    parser.add_argument(
        "--runtime",
        # default="",
        help="if you want to start from in a certain runtime",
        type=str,
    )

    args = parser.parse_args()
    IS_DEBUG_MODE = True
    is_wandb_log = IS_DEBUG_MODE is False

    searchspace = SearchSpaceType(args.searchspace)
    dataset = DatasetType(args.dataset)
    args.epochs = 3

    profile = DartsProfile(
        epochs=args.epochs,
        dropout=args.dropout,
    )

    config = profile.get_config()

    if args.runtime:
        assert check_date_format(args.runtime)

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=args.seed,
        is_wandb_log=is_wandb_log,
        debug_mode=IS_DEBUG_MODE,
        exp_name=args.exp_name,
        runtime=args.runtime,
    )

    # trainer = experiment.run_with_profile(
    #     profile,
    #     start_epoch=args.start_epoch,
    #     load_saved_model=args.load_saved_model,
    #     load_best_model=args.load_best_model,
    # )

    profile = DiscreteProfile()
    discret_trainer = experiment.run_discrete_model_with_profile(
        profile,
        start_epoch=args.start_epoch,
        load_saved_model=args.load_saved_model,
        load_best_model=args.load_best_model,
        use_supernet_checkpoint=args.use_supernet_checkpoint,
    )

    if is_wandb_log:
        wandb.finish()  # type: ignore
