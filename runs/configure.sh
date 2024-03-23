#!/bin/bash

export DATA_DIR=data/
export MODEL_DIR=ckpt/
export CONFIG_DIR=configs/

# Customer variables


REMOVAL_MODEL_TYPE=
DATASETNAME=
REV_MODEL_TYPE=
RATIONALE_FORMAT=
NUM_NGRAMS=
MIN_FREQ=
MAX_TOKENS=
THRESHOLD=
IRM_COEFFICIENT=
REMOVAL_EPOCHS=
REMOVAL_BATCH_SIZE=
REV_EPOCHS=
REV_BATCH_SIZE=
GENERATION_EPOCHS=
GENERATION_BATCH_SIZE=
REV_LEARNING_RATE=


die () {
    echo >&2 "$@"
    exit 1
}


for i in "$@"; do
    case $i in
        --removal-model-type=*)
        REMOVAL_MODEL_TYPE="${i#*=}"
        # removal-model-type has to be either "fasttext" or "lstm"
        if [ "$REMOVAL_MODEL_TYPE" != "fasttext" ] && [ "$REMOVAL_MODEL_TYPE" != "lstm" ]; then
            die "removal-model-type has to be either 'fasttext' or 'lstm'"
        fi
        shift # past argument=value
        ;;
        -d=*|--dataset-name=*)
        DATASETNAME="${i#*=}"
        # datasetname has to be either "ecqa" or "strategyqa"
        if [ "$DATASETNAME" != "ecqa" ] && [ "$DATASETNAME" != "strategyqa" ]; then
            die "datasetname has to be either 'ecqa' or 'strategyqa'"
        fi
        shift # past argument=value
        ;;
        --rev-model-type=*)
        REV_MODEL_TYPE="${i#*=}"
        # rev-model-type has to be either "t5" or "deberta"
        if [ "$REV_MODEL_TYPE" != "t5" ] && [ "$REV_MODEL_TYPE" != "deberta" ]; then
            die "rev-model-type has to be either 't5' or 'deberta'"
        fi
        shift # past argument=value
        ;;
        --rationale-format=*)
        RATIONALE_FORMAT="${i#*=}"
        # rationale-format has to be one of "g", "gl", "gls", "gs", "l", "ls", "s", llama2", "gpt4", "gpt3", "flan"
        if [ "$RATIONALE_FORMAT" != "g" ] && [ "$RATIONALE_FORMAT" != "gl" ] && [ "$RATIONALE_FORMAT" != "gls" ] && [ "$RATIONALE_FORMAT" != "gs" ] && [ "$RATIONALE_FORMAT" != "l" ] && [ "$RATIONALE_FORMAT" != "ls" ] && [ "$RATIONALE_FORMAT" != "s" ] && [ "$RATIONALE_FORMAT" != "llama2" ] && [ "$RATIONALE_FORMAT" != "gpt4" ] && [ "$RATIONALE_FORMAT" != "gpt3" ] && [ "$RATIONALE_FORMAT" != "flan" ]; then
            die "rationale-format has to be one of 'g', 'gl', 'gls', 'gs', 'l', 'ls', 's', 'llama2', 'gpt4', 'gpt3', 'flan'"
        fi
        shift # past argument=value
        ;;
        --num-ngrams=*)
        NUM_NGRAMS="${i#*=}"
        # need to be either 1 or 2
        if [ "$NUM_NGRAMS" != "1" ] && [ "$NUM_NGRAMS" != "2" ]; then
            die "num-ngrams has to be either 1 or 2"
        fi
        shift # past argument=value
        ;;
        --min-freq=*)
        MIN_FREQ="${i#*=}"
        shift # past argument=value
        # has to be a number greater than 0
        if ! [[ "$MIN_FREQ" =~ ^[0-9]+$ ]] ; then
            die "min-freq has to be a number greater than 0"
        fi
        ;;
        --max-tokens=*)
        MAX_TOKENS="${i#*=}"
        # has to be a number greater and not equal to 0
        if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] ; then
            die "max-tokens has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --threshold=*)
        THRESHOLD="${i#*=}"
        # has to be a number greater than 0 (can be fractional)
        if ! [[ "$THRESHOLD" =~ ^[0-9]+[.]*[0-9]*$ ]] ; then
            die "threshold has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --irm-coefficient=*)
        IRM_COEFFICIENT="${i#*=}"
        # has to be a number greater than 0 (can be fractional)
        if ! [[ "$IRM_COEFFICIENT" =~ ^[0-9]+[.]*[0-9]*$ ]] ; then
            die "irm-coefficient has to be a number greater than 0"
        fi
        # if no fraction, then add .0 to the end
        if [[ "$IRM_COEFFICIENT" =~ ^[0-9]+$ ]] ; then
            IRM_COEFFICIENT="$IRM_COEFFICIENT.0"
        fi
        shift # past argument=value
        ;;
        --rev-epochs=*)
        REV_EPOCHS="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$REV_EPOCHS" =~ ^[0-9]+$ ]] ; then
            die "rev-epochs has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --removal-epochs=*)
        REMOVAL_EPOCHS="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$REMOVAL_EPOCHS" =~ ^[0-9]+$ ]] ; then
            die "removal-epochs has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --removal-batch-size=*)
        REMOVAL_BATCH_SIZE="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$REMOVAL_BATCH_SIZE" =~ ^[0-9]+$ ]] ; then
            die "removal-batch-size has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --rev-batch-size=*)
        REV_BATCH_SIZE="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$REV_BATCH_SIZE" =~ ^[0-9]+$ ]] ; then
            die "rev-batch-size has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --generation-epochs=*)
        GENERATION_EPOCHS="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$GENERATION_EPOCHS" =~ ^[0-9]+$ ]] ; then
            die "generation-epochs has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --generation-batch-size=*)
        GENERATION_BATCH_SIZE="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$GENERATION_BATCH_SIZE" =~ ^[0-9]+$ ]] ; then
            die "generation-batch-size has to be a number greater than 0"
        fi
        shift # past argument=value
        ;;
        --learning-rate=*)
        REV_LEARNING_RATE="${i#*=}"
        # has to be a number greater than 0
        if ! [[ "$REV_LEARNING_RATE" =~ ^[0-9]+[.]*[0-9]*$ ]] ; then
            die "learning-rate has to be a number greater than 0"
        fi
        # if no fraction, then add .0 to the end
        if [[ "$REV_LEARNING_RATE" =~ ^[0-9]+$ ]] ; then
            REV_LEARNING_RATE="$REV_LEARNING_RATE.0"
        fi
        shift # past argument=value
        ;;
        *)
              # unknown option
        ;;
    esac
done


# export all assigned values if they are not empty
# die if any of these values are empty

if [ -n "$REMOVAL_MODEL_TYPE" ]; then
    export REMOVAL_MODEL_TYPE
else
    die "removal-model-type is empty"
fi

if [ -n "$DATASETNAME" ]; then
    export DATASETNAME
else
    die "dataset-name is empty"
fi

if [ -n "$REV_MODEL_TYPE" ]; then
    export REV_MODEL_TYPE
else
    die "rev-model-type is empty"
fi

if [ -n "$RATIONALE_FORMAT" ]; then
    export RATIONALE_FORMAT
else
    die "rationale-format is empty"
fi

if [ -n "$NUM_NGRAMS" ]; then
    export NUM_NGRAMS
else
    die "num-ngrams is empty"
fi

if [ -n "$MIN_FREQ" ]; then
    export MIN_FREQ
else
    die "min-freq is empty"
fi

if [ -n "$MAX_TOKENS" ]; then
    export MAX_TOKENS
else
    die "max-tokens is empty"
fi

if [ -n "$THRESHOLD" ]; then
    export THRESHOLD
else
    die "threshold is empty"
fi

if [ -n "$IRM_COEFFICIENT" ]; then
    export IRM_COEFFICIENT
else
    die "irm-coefficient is empty"
fi

if [ -n "$REMOVAL_BATCH_SIZE" ]; then
    export REMOVAL_BATCH_SIZE
else
    die "removal-batch-size is empty"
fi

if [ -n "$REMOVAL_EPOCHS" ]; then
    export REMOVAL_EPOCHS
else
    die "removal-epochs is empty"
fi

if [ -n "$REV_EPOCHS" ]; then
    export REV_EPOCHS
else
    die "rev-epochs is empty"
fi

if [ -n "$REV_BATCH_SIZE" ]; then
    export REV_BATCH_SIZE
else
    die "rev-batch-size is empty"
fi

if [ -n "$GENERATION_EPOCHS" ]; then
    export GENERATION_EPOCHS
else
    die "generation-epochs is empty"
fi

if [ -n "$GENERATION_BATCH_SIZE" ]; then
    export GENERATION_BATCH_SIZE
else
    die "generation-batch-size is empty"
fi

if [ -n "$REV_LEARNING_RATE" ]; then
    export REV_LEARNING_RATE
else
    die "learning-rate is empty"
fi

# create and export the path variables to be used in the Makefile and configs.

export VOCAB_DIR=${DATA_DIR}${DATASETNAME}_vocabs/
export PYTHONPATH=$(pwd)

export VOCAB_FILE=${VOCAB_DIR}vocab_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}.pt
export REPORT_DIR=${DATA_DIR}reports/
export PLAIN_DATA_DIR=${DATA_DIR}${DATASETNAME}/plain/
export RAW_DATA_DIR=${DATA_DIR}${DATASETNAME}/raw/
export MODEL_GENERATED_RATIONALE_DIR=${DATA_DIR}generated_rationales/${DATASETNAME}/
export REMOVAL_DATA_DIR=${DATA_DIR}${DATASETNAME}/rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}/
export REMOVAL_MODEL=${MODEL_DIR}${DATASETNAME}_${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}/
export GENERATION_DATA_DIR=${DATA_DIR}${DATASETNAME}/generation_rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}_th=${THRESHOLD}/
export GENERATION_MODEL=${MODEL_DIR}${DATASETNAME}_generation_rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}_th=${THRESHOLD}/
export REV_DATA_DIR=${DATA_DIR}${DATASETNAME}/rev=${REV_MODEL_TYPE}_rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}_th=${THRESHOLD}/
export REV_MODEL=${MODEL_DIR}${DATASETNAME}_rev=${REV_MODEL_TYPE}_lr=${REV_LEARNING_RATE}_rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}_th=${THRESHOLD}_irm=${IRM_COEFFICIENT}/
export BASELINE_DATA_DIR=${DATA_DIR}${DATASETNAME}/baseline_${REV_MODEL_TYPE}/
export BASELINE_MODEL=${MODEL_DIR}${DATASETNAME}_baseline_${REV_MODEL_TYPE}_lr=${REV_LEARNING_RATE}/
# export RVB_DATA_DIR=${DATA_DIR}${DATASETNAME}/rvb/
# export RVB_MODEL=${MODEL_DIR}${DATASETNAME}_rvb/
# export RVR_DATA_DIR=${DATA_DIR}${DATASETNAME}/rvr_${RATIONALE_FORMAT}/
# export RVR_MODEL=${MODEL_DIR}${DATASETNAME}_rvr_${RATIONALE_FORMAT}/
export REPORT_FILE=${REPORT_DIR}${DATASETNAME}/rev=${REV_MODEL_TYPE}_lr=${REV_LEARNING_RATE}_rm=${REMOVAL_MODEL_TYPE}_format=${RATIONALE_FORMAT}_ng=${NUM_NGRAMS}_mf=${MIN_FREQ}_mt=${MAX_TOKENS}_th=${THRESHOLD}_irm=${IRM_COEFFICIENT}.json
export RVR_REPORT_FILE=${REPORT_DIR}${DATASETNAME}/rvr_${RATIONALE_FORMAT}.json

# CONFIG_PATHS
export REMOVAL_PREPROCESSING_CONFIG=${CONFIG_DIR}removal_configs/${DATASETNAME}_${REMOVAL_MODEL_TYPE}.yaml
export REMOVAL_TRAINING_CONFIG=${CONFIG_DIR}removal_training_configs/${DATASETNAME}_${REMOVAL_MODEL_TYPE}.yaml
export GENERATION_PREPROCESSING_CONFIG=${CONFIG_DIR}generation_configs/${DATASETNAME}.yaml
export GENERATION_TRAINING_CONFIG=${CONFIG_DIR}generation_training_configs/${DATASETNAME}.yaml
export REV_PREPROCESSING_CONFIG=${CONFIG_DIR}rev_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
export REV_TRAINING_CONFIG=${CONFIG_DIR}rev_training_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
export BASELINE_PREPROCESSING_CONFIG=${CONFIG_DIR}baseline_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
export BASELINE_TRAINING_CONFIG=${CONFIG_DIR}baseline_training_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
# export RVB_PREPROCESSING_CONFIG=${CONFIG_DIR}rvb_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
# export RVB_TRAINING_CONFIG=${CONFIG_DIR}rvb_training_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
# export RVR_PREPROCESSING_CONFIG=${CONFIG_DIR}rvr_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
# export RVR_TRAINING_CONFIG=${CONFIG_DIR}rvr_training_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
export TESTING_CONFIG=${CONFIG_DIR}testing_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml
export RVR_TESTING_CONFIG=${CONFIG_DIR}rvr_testing_configs/${DATASETNAME}_${REV_MODEL_TYPE}.yaml