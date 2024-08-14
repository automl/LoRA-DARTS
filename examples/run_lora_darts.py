from __future__ import annotations

import json

from confopt.profiles import DartsProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType("darts")
    dataset = DatasetType("cifar10")
    seed = 100

    profile = DartsProfile(
        sampler_sample_frequency="step",
        epochs=50,
        lora_rank=1,
        lora_warm_epochs=10,
    )

    config = profile.get_config()
    print(json.dumps(config, indent=2, default=str))
    IS_DEBUG_MODE = True # Set to False for a full run

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    search_trainer = experiment.run_with_profile(profile)
