#!/bin/bash
#SBATCH -p my_partition
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH --time 1-00:00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 8
#SBATCH -J LoRA-DARTS-DARTS-WE # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <searchspace> <sampler> <weight-entanglement> <rank>"
    exit 1
fi

# Check if the searchspace is darts
if [ "$1" != "darts" ]; then
    echo "Error: searchspace must be 'darts'"
    exit 1
fi

# Check if the samplers are either darts or drnas
if [ "$2" != "darts" ] && [ "$2" != "drnas" ]; then
    echo "Error: optimizer must be either 'darts' or 'drnas'"
    exit 1
fi


# Check if the samplers are darts, drnas or gdas
searchspace=$1
sampler=$2
rank=$4

start=`date +%s`

source ~/.bashrc
conda activate lora_darts

export WANDB_MODE="offline"

if [ "$3" == "true" ]; then
    python scripts/lora/run_darts_experiment.py --search_epochs 100 --sampler $sampler --wandb_log --searchspace $searchspace --entangle_op_weights --seed $SLURM_ARRAY_TASK_ID --use_lora --lora_rank $rank --lora_warm_epochs 10
elif [ "$3" == "false" ]; then
    python scripts/lora/run_darts_experiment.py --search_epochs 100 --sampler $sampler --wandb_log --searchspace $searchspace --seed $SLURM_ARRAY_TASK_ID --use_lora --lora_rank $rank --lora_warm_epochs 10
else
    echo "Error: weight-entanglement must be 'true' or 'false'"
    exit 1
fi

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
