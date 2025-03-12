#!/bin/bash -l
#SBATCH --job-name="Wind daily"
#SBATCH --time=08:00:00
#SBATCH --output=slurm/output-%a.out
#SBATCH --error=slurm/error-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=postproc
#SBATCH --account=msclim
#SBATCH -a 0-9:1%10
#SBATCH --mail-user=florian.ackermann@students.unibe.ch  
#SBATCH --mail-type=START,END,FAIL  

conda activate testenv

cd /users/fackerma/newproject1/

python -u TRTProcessing_dailyjob.py  -d $SLURM_ARRAY_TASK_ID
