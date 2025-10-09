#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "res"   # job name
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=calc_res_out.out
#SBATCH --error=cal_res_error.txt


## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.11.6-gcc-13.2.0-fh6i4o3
module load mpich/4.1.2-gcc-11.3.1-ikiclyo

source /central/home/jyzhao/python_env/scalable_gps/bin/activate

# Comment out for test job submission
# python /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/preprocessing/03_calculate_res.py