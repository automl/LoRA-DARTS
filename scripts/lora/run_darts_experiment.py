from __future__ import annotations

import argparse
import json

from confopt.profiles.profile_config import ProfileConfig
from confopt.profiles.profiles import DartsProfile, DRNASProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

dataset_size = {
    "cifar10": 10,
    "cifar100": 100,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LoRA WE/WS Experiment", add_help=False)

    parser.add_argument(
        "--searchspace",
        default="darts",
        help="choose the search space (darts)",
        type=str,
    )

    parser.add_argument(
        "--entangle_op_weights",
        action="store_true",
        default=False,
        help="Whether to use weight entanglement or not",
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )

    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--sampler",
        default="darts",
        help="Choose sampler from (darts, drnas)",
        type=str,
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use lora or not",
    )

    parser.add_argument(
        "--lora_warm_epochs",
        default=0,
        help="number of warm epochs for lora to run on",
        type=int,
    )

    parser.add_argument(
        "--lora_rank",
        default=0,
        help="rank for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_alpha",
        default=1,
        help="alpha multiplier for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_dropout",
        default=0,
        help="dropout value for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_merge_weights",
        action="store_true",
        default=False,
        help="merge lora weights with conv weights",
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--wandb_log", action="store_true", help="turn wandb logging on"
    )

    parser.add_argument(
        "--debug_mode", action="store_true", help="run experiment in debug mode"
    )

    args = parser.parse_args()
    return args


def get_configuration(
    profile_type: ProfileConfig, args: argparse.Namespace
) -> ProfileConfig:
    profile = profile_type(
        epochs=args.search_epochs,
        lora_rank=args.lora_rank,
        lora_warm_epochs=args.lora_warm_epochs,
        entangle_op_weights=args.entangle_op_weights,
        searchspace_str=args.searchspace,
        # lora_toggle_epochs=list(range(11, 100, 1)),
        # lora_toggle_probability=None,
        seed=args.seed,
    )
    return profile


def get_darts_configuration(args: argparse.Namespace) -> DartsProfile:
    return get_configuration(DartsProfile, args)


def get_drnas_configuration(args: argparse.Namespace) -> DRNASProfile:
    return get_configuration(DRNASProfile, args)


if __name__ == "__main__":
    args = read_args()

    assert args.searchspace in ["darts", "nb201"], f"Does not support space of type {args.searchspace}"  # type: ignore
    assert args.dataset in ["cifar10", "cifar100", "imagenet"], f"Soes not support dataset of type {args.dataset}"  # type: ignore

    if args.use_lora:
        assert args.lora_warm_epochs > 0, "argument --lora_warm_epochs should not be 0 when argument --use_lora is provided"  # type: ignore
        assert args.lora_rank > 0, "argument --lora_rank should be greater than 0"  # type: ignore

    assert args.sampler in ["darts", "drnas"], "This experiment supports only darts and drnas as samplers"  # type: ignore

    if args.sampler == "darts":
        profile = get_darts_configuration(args)
    elif args.sampler == "drnas":
        profile = get_drnas_configuration(args)

    searchspace_config = {
        "num_classes": dataset_size[args.dataset],
    }
    profile.set_searchspace_config(searchspace_config)
    profile.configure_lora_config(
        lora_dropout=args.lora_dropout,  # type: ignore
        merge_weights=args.lora_merge_weights,  # type: ignore
        lora_alpha=args.lora_alpha,  # type: ignore
    )

    # Extra info for wandb tracking
    project_name = "LoRA-DARTS"
    lora_or_vanilla = "lora" if args.use_lora else "vanilla"
    profile.configure_extra_config(
        {
            "project_name": project_name,
            "experiment_type": f"{lora_or_vanilla}",
        }
    )

    print(json.dumps(profile.get_config(), indent=2, default=str))

    # Experiment name for logging
    experiment_name = f"{args.sampler}_{lora_or_vanilla}"

    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        is_wandb_log=args.wandb_log,
        debug_mode=args.debug_mode,
        exp_name=experiment_name,
    )
    trainer = experiment.run_with_profile(profile, use_benchmark=True)
