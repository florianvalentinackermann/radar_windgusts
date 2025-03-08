#!/bin/bash -l
#SBATCH --job-name="some_name"
#SBATCH --time=08:00:00
#SBATCH --output=output-%a.out
#SBATCH --error=error-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=postproc
#SBATCH --account=msclim
#SBATCH -a 0-100:1%10

conda activate yourenv

cd /your/code/folder

python -u daily_processing.py  -d $SLURM_ARRAY_TASK_ID
