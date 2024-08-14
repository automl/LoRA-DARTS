# Given a genotype, train it from scratch
from __future__ import annotations

import argparse
import json

import wandb

from confopt.profiles.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Genotype", add_help=False)

    parser.add_argument(
        "--searchspace",
        default="darts",
        help="choose the search space (darts, nb201, tnb101)",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )

    parser.add_argument(
        "--eval_epochs",
        default=600,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--genotype",
        help="genotype to train",
        type=str,
    )

    parser.add_argument(
        "--run_name",
        default="test",
        help="name of the run",
        type=str,
    )

    parser.add_argument(
        "--wandb_log", action="store_true", help="turn wandb logging on"
    )

    args = parser.parse_args()
    return args


def get_discrete_configuration(args: argparse.Namespace) -> DiscreteProfile:
    profile = DiscreteProfile(
        epochs=args.eval_epochs,
    )

    return profile


if __name__ == "__main__":
    args = read_args()

    assert args.searchspace in [
        "darts",
        "nb201",
    ], f"Does not support space of type {args.searchspace}"  # type: ignore
    assert args.dataset in [
        "cifar10",
        "cifar100",
        "imagenet",
    ], f"Does not support dataset of type {args.dataset}"  # type: ignore

    profile = get_discrete_configuration(args)
    profile.genotype = args.genotype

    print(f"Training {args.run_name} genotype: {args.genotype}")
    # Extra info for wandb tracking
    project_name = "LoRA_DARTS_Evaluation"
    run_name = args.run_name
    print(json.dumps(profile.get_trainer_config(), indent=2, default=str))
    # Experiment name for logging
    experiment_name = f"DISCRETE_{args.searchspace}_{args.dataset}_{args.run_name}"
    config = profile.get_trainer_config()
    config.update({"genotype": profile.get_genotype()})

    # instantiate wandb run
    if args.wandb_log:
        wandb.init(  # type: ignore
            name=experiment_name,
            project=project_name,
            config=config,
        )

    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        is_wandb_log=args.wandb_log,
        exp_name=experiment_name,
    )

    trainer = experiment.run_discrete_model_with_profile(profile)
