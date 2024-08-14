#!/bin/bash
#SBATCH -p my_partition
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH -J robust_darts_lora # sets the job name
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate lora_darts

python scripts/lora/run_robust_darts_experiment.py --use_lora --lora_warm_epoch 10 --lora_rank 1 --lora_merge_weights --sampler "darts" --wandb_log --space "s2" --seed $SLURM_ARRAY_TASK_ID

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
