#!/bin/bash

#SBATCH --time=1:00:00           
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)

#SBATCH -J "h_calc_tr"   # job name
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/hazard_calc_src/%j.out # STDOUT
#SBATCH --error=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/hazard_calc_src/%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source /home/jyzhao/python_env/scalable_gps/activate_env.sh

PREDICTION_DIR="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/prediction_results/4179_training_eqs_500_epochs_no_source"
USEMPI=True
OVERWRITE=True
PREDICTION_EQ_TYPE="training"

srun python /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/hazard_calc_src/calculate_hazard.py \
    --prediction_dir ${PREDICTION_DIR} \
    --use_MPI ${USEMPI} \
    --overwrite ${OVERWRITE} \
    --prediction_eq_type ${PREDICTION_EQ_TYPE}