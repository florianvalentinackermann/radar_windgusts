#!/bin/bash -l
#SBATCH --job-name="Testfile_trt_first_run"
#SBATCH --time=08:00:00
#SBATCH --output=slurm/output.out
#SBATCH --error=slurm/error.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=postproc
#SBATCH --account=msclim

conda activate testenv

cd /users/fackerma/newproject1/

python -u Testfile_trt_first_run.py
