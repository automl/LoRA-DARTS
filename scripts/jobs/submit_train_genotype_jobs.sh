#!/bin/bash

searchspace="darts"

run_names=("genotype_1")

for i in "${!run_names[@]}"; do
  run_name=${run_names[$i]}
  sbatch -J $run_name scripts/jobs/train_discrete_genotype.sh "$searchspace" "$run_name"
done
