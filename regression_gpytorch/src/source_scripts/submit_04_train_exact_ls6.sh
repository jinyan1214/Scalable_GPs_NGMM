#!/bin/bash

#SBATCH --time=10:00:00           
#SBATCH -N 1   # number of nodes
#SBATCH -n 1   # number of processor cores (i.e. tasks)
#SBATCH --partition=gpu-a100-small # other oprtions:gpu-h100, gpu-a100-small

#SBATCH -J "4179_mean"   # job name
#SBATCH --mail-user=jyzhao@caltech.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=all
#SBATCH -A DesignSafe-SimCenter

#SBATCH --output=/work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/output/training_outputs/4179_mean.out # STDOUT
#SBATCH --error=/work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/output/training_outputs/4179_mean.err # STDERR

NUM_TRAIN_EQS=4179
num_epochs=500
BETWEEN_KERNEL="True"
SOURCE_EFFECT="False"
NUM_RLZ="all"

OUTPUT_DIR="/work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/output/training_outputs_exact/${NUM_TRAIN_EQS}_eqs_BetweenKernel_${BETWEEN_KERNEL}_SourceEffect_${SOURCE_EFFECT}/${NUM_RLZ}_rlzs"
mkdir -p ${OUTPUT_DIR}

exec >"${OUTPUT_DIR}/train_epoch_${num_epochs}_ls6.out" 2>&1

source /work/07059/jyzhao/frontera/EPN_Sonoma/epn_sonoma_venv_ls6/bin/activate

python /work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/src/04_training_exactGP.py \
    --num_training_eqs ${NUM_TRAIN_EQS} \
    --num_rlzs ${NUM_RLZ} \
    --num_epochs ${num_epochs} \
    --between_kernel ${BETWEEN_KERNEL} \
    --source_effect ${SOURCE_EFFECT} \
    --overwrite True

rm /work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/src/4179_all.out
rm /work/07059/jyzhao/ls6/cybershake_ngmm/regression_gpytorch/src/4179_all.err