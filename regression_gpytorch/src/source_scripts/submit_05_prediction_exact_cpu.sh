#!/bin/bash

#SBATCH --time=2:00:00           
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=8G   # memory per CPU core

#SBATCH -J "p_400_4179"   # job name
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/p_400_4179.cpu.out # STDOUT
#SBATCH --error=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/p_400_4179.cpu.err # STDERR

NUM_TRAIN_EQS=400
num_epochs=2500
NUM_CANDIDATE_EQS=4179
PREDICTION_BATCH_SIZE=512

OUTPUT_DIR="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/prediction_outputs_cpu/${NUM_TRAIN_EQS}_eqs"
mkdir -p ${OUTPUT_DIR}

exec >"${OUTPUT_DIR}/train_eq_${NUM_TRAIN_EQS}_${num_epochs}_conditional_${NUM_CANDIDATE_EQS}.out" 2>&1

source /home/jyzhao/python_env/gpytorch/activate_env.sh

python3 /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/05_prediction_exact_gp.py \
    --num_training_eqs ${NUM_TRAIN_EQS} \
    --num_epochs ${num_epochs} \
    --num_candidate_eqs ${NUM_CANDIDATE_EQS} \
    --prediction_batch_size ${PREDICTION_BATCH_SIZE}

rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/p_400_4179.cpu.out
rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/p_400_4179.cpu.err