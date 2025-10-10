#!/bin/bash

#SBATCH --time=00:10:00           
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1      # (this example specifies 1 GPU's of any type)

#SBATCH -J "svgp_400_4179"   # job name
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/svgp_400_4179.out # STDOUT
#SBATCH --error=/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/svgp_400_4179.err # STDERR

NUM_TRAIN_EQS=400
num_epochs=2500

OUTPUT_DIR="/resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/output/prediction_outputs_svgp_cuda/job_out_${NUM_TRAIN_EQS}_eqs_${num_epochs}_epochs"
mkdir -p ${OUTPUT_DIR}

exec >"${OUTPUT_DIR}/train_eq_${NUM_TRAIN_EQS}_${num_epochs}.out" 2>&1

source /home/jyzhao/python_env/gpytorch/activate_env.sh

python3 /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/04_prediction_svgp.py \
    --num_training_eqs ${NUM_TRAIN_EQS} \
    --num_epochs ${num_epochs}

rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/svgp_400_4179.out
rm /resnick/groups/enceladus/jyzhao/Scalable_GPs_jz/regression_gpytorch/src/svgp_400_4179.err