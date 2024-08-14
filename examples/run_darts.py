from __future__ import annotations

from confopt.profiles import DartsProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

if __name__ == "__main__":
    searchspace = SearchSpaceType("darts")
    dataset = DatasetType("cifar10")
    seed = 100

    profile = DartsProfile(epochs=50)

    config = profile.get_config()
    print(config)
    IS_DEBUG_MODE = True # Set to False for a full run

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    experiment.run_with_profile(profile)
