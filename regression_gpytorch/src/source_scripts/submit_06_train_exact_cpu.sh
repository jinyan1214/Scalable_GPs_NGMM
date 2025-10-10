#!/bin/bash

#SBATCH --time=10:00:00           
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=32   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G   # memory per CPU core

#SBATCH -J "50_2500"   # job name number of training eqs and training iterations
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/50_2500.out # STDOUT
#SBATCH --error=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/50_2500.err # STDERR

NUM_TRAIN_EQS=50
NUM_TRAIN_ITER=1000

OUTPUT_DIR="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/training_outputs_exact/${NUM_TRAIN_EQS}_eqs"
mkdir -p ${OUTPUT_DIR}

exec >"${OUTPUT_DIR}/train_iter_${NUM_TRAIN_ITER}.out" 2>&1

source /home/jyzhao/python_env/gpytorch/activate_env.sh

python /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/06_training_exact_gp.py \
    --num_training_eqs ${NUM_TRAIN_EQS} \
    --num_training_iter ${NUM_TRAIN_ITER}

rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/50_2500.out
rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/50_2500.err