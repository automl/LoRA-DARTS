#!/bin/bash

spaces=("darts")
samplers=("darts")
we=("true" "false")

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for entanglement in "${we[@]}"; do
            echo scripts/jobs/submit_non_lora_experiment.sh $space $sampler $entanglement
            sbatch -J NON_LoRA-${sampler}-${space}-WE-${entanglement} scripts/jobs/submit_non_lora_experiment.sh $space $sampler $entanglement
        done
    done
done
