#!/bin/bash
#SBATCH --job-name=run-ecqa-eval
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6


TARG=$1


# TODO: Need to actually implement a good commandline reader


export REMOVAL_MODEL_TYPE=lstm
export DATASETNAME=ecqa
export REV_MODEL_TYPE=t5
export RATIONALE_FORMAT=$2
export NUM_NGRAMS=1
export MIN_FREQ=1
export MAX_TOKENS=10000
export THRESHOLD=0.001
export IRM_COEFFICIENT=10.0

conda run -p ./.env --no-capture-output make $TARG
