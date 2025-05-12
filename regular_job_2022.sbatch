#!/bin/bash -l
#SBATCH --job-name="Model Setup"
#SBATCH --time=38:00:00
#SBATCH --output=slurm/output_b.out
#SBATCH --error=slurm/error_b.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=postproc
#SBATCH --account=msclim
#SBATCH --mail-user=florian.ackermann@students.unibe.ch  
#SBATCH --mail-type=START,END,FAIL  

conda activate testenv

cd /users/fackerma/newproject1/

python -u Model_Setup_3_b.py

