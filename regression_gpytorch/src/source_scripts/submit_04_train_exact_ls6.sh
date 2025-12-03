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

#SBATCH --output=/work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/src/4179_mean.out # STDOUT
#SBATCH --error=/work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/src/4179_mean.err # STDERR

NUM_TRAIN_EQS=4179
num_epochs=500
BETWEEN_KERNEL="True"
SOURCE_EFFECT="False"
NUM_RLZ="all"
TRAIN_ON_MEAN="True"

OUTPUT_DIR="/work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/output/training_outputs_exact/${NUM_TRAIN_EQS}_eqs_BetweenKernel_${BETWEEN_KERNEL}_SourceEffect_${SOURCE_EFFECT}/${NUM_RLZ}_rlzs"
if [ "${TRAIN_ON_MEAN}" = "True" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_train_on_mean"
fi
mkdir -p ${OUTPUT_DIR}

exec >"${OUTPUT_DIR}/train_epoch_${num_epochs}_ls6.out" 2>&1

source /work/07059/jyzhao/frontera/EPN_Sonoma/epn_sonoma_venv_ls6/bin/activate

python /work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/src/04_training_exactGP.py \
    --num_training_eqs ${NUM_TRAIN_EQS} \
    --num_rlzs ${NUM_RLZ} \
    --train_on_mean ${TRAIN_ON_MEAN} \
    --num_epochs ${num_epochs} \
    --between_kernel ${BETWEEN_KERNEL} \
    --source_effect ${SOURCE_EFFECT} \
    --overwrite True

rm /work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/src/4179_mean.out
rm /work/07059/jyzhao/ls6/cybershake_ngmm/Scalable_GPs_NGMM/regression_gpytorch/src/4179_mean.err