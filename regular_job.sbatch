#!/bin/bash -l
#SBATCH --job-name="Gust Marker"
#SBATCH --time=12:00:00
#SBATCH --output=slurm/output.out
#SBATCH --error=slurm/error.err
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

python -u Extracted_metainfo.py

