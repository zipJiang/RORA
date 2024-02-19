#!/bin/bash
#SBATCH --job-name=eval_all_decomps
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6


TARG=$1


# TODO: Need to actually implement a good commandline reader

REMOVAL_MODEL_TYPE=fasttext
DATASETNAME=strategyqa
REV_MODEL_TYPE=t5
RATIONALE_FORMAT=$2
NUM_NGRAMS=2
MIN_FREQ=1
MAX_TOKENS=10000
THRESHOLD=0.001
REMOVAL_EPOCHS=20
REMOVAL_BATCH_SIZE=256
GENERATION_EPOCHS=20
GENERATION_BATCH_SIZE=16
REV_EPOCHS=20
REV_BATCH_SIZE=16
IRM_COEFFICIENT=$3

conda run -p ./.env --no-capture-output source runs/configure.sh \
    --removal-model-type=${REMOVAL_MODEL_TYPE} \
    --dataset-name=${DATASETNAME} \
    --rationale-format=${RATIONALE_FORMAT} \
    --num-ngrams=${NUM_NGRAMS} \
    --min-freq=${MIN_FREQ} \
    --max-tokens=${MAX_TOKENS} \
    --threshold=${THRESHOLD} \
    --irm-coefficient=${IRM_COEFFICIENT} \
    --rev-model-type=${REV_MODEL_TYPE} \
    --removal-epochs=${REMOVAL_EPOCHS} \
    --removal-batch-size=${REMOVAL_BATCH_SIZE} \
    --generation-epochs=${GENERATION_EPOCHS} \
    --generation-batch-size=${GENERATION_BATCH_SIZE} \
    --rev-epochs=${REV_EPOCHS} \
    --rev-batch-size=${REV_BATCH_SIZE}

conda run -p ./.env --no-capture-output make $TARG