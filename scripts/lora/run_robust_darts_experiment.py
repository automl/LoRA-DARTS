from __future__ import annotations

import argparse
import json

from confopt.profiles.profiles import DartsProfile, DRNASProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

searchspace = "robust_darts"
dataset_size = {
    "cifar10": 10,
    "cifar100": 100,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DRNAS Baseline run", add_help=False)

    parser.add_argument(
        "--space",
        default="s1",
        help="choose the robust darts searchspace from (s1, s2, s3, s4)",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )

    parser.add_argument(
        "--search_epochs",
        default=50,
        help="number of epochs to train the supernet",
        type=int,
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
        "--sampler",
        default="darts",
        help="Choose sampler from (darts, drnas)",
        type=str,
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


def get_darts_configuration(args: argparse.Namespace) -> DartsProfile:
    profile = DartsProfile(
        epochs=args.search_epochs,
        lora_rank=args.lora_rank,
        lora_warm_epochs=args.lora_warm_epochs,
    )
    return profile


def get_drnas_configuration(args: argparse.Namespace) -> DRNASProfile:
    profile = DRNASProfile(
        epochs=args.search_epochs,
        lora_rank=args.lora_rank,
        lora_warm_epochs=args.lora_warm_epochs,
    )
    return profile


if __name__ == "__main__":
    args = read_args()

    assert args.space in ["s1", "s2", "s3", "s4"], f"RobustDARTS does not support space of type {args.space}"  # type: ignore
    assert args.dataset in ["cifar10", "cifar100", "imagenet"], f"RobustDARTS does not support dataset of type {args.dataset}"  # type: ignore

    if args.use_lora:
        assert args.lora_warm_epochs > 0, "argument --lora_warm_epochs should not be 0 when argument --use_lora is provided"  # type: ignore
        assert args.lora_rank > 0, "argument --lora_rank should be greater than 0"  # type: ignore

    assert args.sampler in ["darts", "drnas"], "This experiment supports only darts and drnas as samplers"  # type: ignore

    if args.sampler == "darts":
        profile = get_darts_configuration(args)
    elif args.sampler == "drnas":
        profile = get_drnas_configuration(args)

    searchspace_config = {
        "num_classes": dataset_size[args.dataset],  # type: ignore
        "space": args.space,  # type: ignore
    }
    profile.set_searchspace_config(searchspace_config)
    profile.configure_lora_config(
        lora_dropout=args.lora_dropout,  # type: ignore
        merge_weights=args.lora_merge_weights,  # type: ignore
        lora_alpha=args.lora_alpha,  # type: ignore
    )

    # Extra info for wandb tracking
    project_name = "LoRA"
    lora_or_vanilla = "lora" if args.use_lora else "vanilla"
    profile.configure_extra_config(
        {
            "project_name": project_name,
            "searchspace_str": searchspace,
            "experiment_type": f"{lora_or_vanilla}",
            "seed": args.seed,  # type: ignore
            "sampler_type": args.sampler,  # type: ignore
        }
    )

    print(json.dumps(profile.get_config(), indent=2, default=str))

    # Experiment name for logging
    experiment_name = f"{searchspace}_{args.sampler}_{lora_or_vanilla}"

    experiment = Experiment(
        search_space=SearchSpaceType(searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        is_wandb_log=args.wandb_log,
        debug_mode=args.debug_mode,
        exp_name=experiment_name,
    )
    trainer = experiment.run_with_profile(profile, use_benchmark=True)
