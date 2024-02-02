#!/bin/bash
#SBATCH --job-name=eval_all_decomps
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6-ir


TARG=$1


# TODO: Need to actually implement a good commandline reader


export REMOVAL_MODEL_TYPE=fasttext
export DATASETNAME=strategyqa
export REV_MODEL_TYPE=t5
export RATIONALE_FORMAT=$2
export NUM_NGRAMS=2
export MIN_FREQ=1
export MAX_TOKENS=10000
export THRESHOLD=0.001
export IRM_COEFFICIENT=1.0

conda run -p ./.env --no-capture-output make $TARG
