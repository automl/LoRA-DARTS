#!/bin/bash

spaces=("darts")
samplers=("darts")
we=("true" "false")
rank=1

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for entanglement in "${we[@]}"; do
            echo scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement
            sbatch -J LoRA-${sampler}-${space}-WE-${entanglement}-100epochs scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement $rank
        done
    done
done

