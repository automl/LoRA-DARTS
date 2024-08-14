#!/bin/bash
#SBATCH -p my_partition
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 9001-9003 # array size
#SBATCH --time 3-00:00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 2
#SBATCH -J TRAIN_GENOTYPE # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# Check if the samplers are darts, drnas or gdas
searchspace=$1
run_name=$2
genotype_file="scripts/train_discrete/genotypes_to_train/$run_name.txt"

# Check if the genotype file exists
if [[ ! -f "$genotype_file" ]]; then
  echo "Error: $genotype_file not found."
  exit 1
fi

# Read the genotype string from the file
genotype=$(<"$genotype_file")

echo "$genotype"
start=`date +%s`

source ~/.bashrc
conda activate lora_darts

# export WANDB_MODE="offline"
# echo "$genotype"
python scripts/train_discrete/train_genotype.py --wandb_log --searchspace "$searchspace" --genotype "$genotype"  --run_name "$run_name" --seed $SLURM_ARRAY_TASK_ID
