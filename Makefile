DATASETNAME = strategyqa
DATA_DIR = data/
MODEL_DIR = ckpt/
CONFIG_DIR = configs/
VOCAB_DIR := $(DATA_DIR)$(DATASETNAME)_vocabs/
SUBDIRS = data ckpt steps
PYTHONPATH = $(shell pwd)
REV_MODEL_TYPE = bart
REMOVAL_MODEL_TYPE = fasttext
RATIONALE_TYPE = g
MIN_FREQ = 1
NUM_NGRAMS = 2
REV_EPOCHS = 20
REMOVAL_EPOCHS = 20
THRESHOLD = 0.001
MAX_TOKENS = 10000
IRM_COEFFICIENT = 10.0
REV_BATCH_SIZE = 32
UTIL_DIR = src/utils/

VOCAB_FILE := $(VOCAB_DIR)vocab_format=$(RATIONALE_TYPE)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS).pt
REMOVAL_TRAINING_CONFIG = $(DATASETNAME)_$(REMOVAL_MODEL_TYPE).yaml
REMOVAL_MODEL = $(DATA_DIR)$(DATASETNAME)/$(MODEL_DIR)$(DATASETNAME)_$(REMOVAL_MODEL_TYPE)_$(RATIONALE_TYPE)_$(MIN_FREQ)_$(THRESHOLD)
REV_MODEL = $(DATA_DIR)$(DATASETNAME)/$(MODEL_DIR)$(DATASETNAME)_$(REV_MODEL_TYPE)_$(RATIONALE_TYPE)_$(MIN_FREQ)

export PYTHONPATH

# ACTUAL STEPS

.SUFFIXES:

score_report.json : steps/eval_rev_with_model.py $(wildcard UTIL_DIR*.py) $(REV_MODEL) $(REMOVAL_MODEL) vocab data
	python3 steps/eval_rev_with_model.py --dataset-dir $(DATA_DIR)$(DATASETNAME) \
		--model-dir $(REV_MODEL) \
		--removal-model-dir $(REMOVAL_MODEL) \
		--rationale-format $(RATIONALE_TYPE) \
		--removal-threshold $(THRESHOLD) \
		--vocab-minimum-frequency $(MIN_FREQ)

$(REV_MODEL) : steps/train_irm_model.py $(wildcard UTIL_DIR*.py) $(REMOVAL_MODEL) vocab data
	python3 steps/trian_irm_model.py --task-name $(DATASETNAME) \
		--model-name $(REV_MODEL_TYPE) \
		--rationael-format $(RATIONALE_TYPE) \
		--epochs $(REV_EPOCHS) 
		--removal-threshold $(THRESHOLD) \
		--irm-coefficient $(IRM_COEFFICIENT) \
		--batch-size $(REV_BATCH_SIZE)

$(VOCAB_DIR) :
	mkdir -p $(VOCAB_DIR)


%.py : ;
%.yaml : ;


$(VOCAB_FILE) : scripts/generate_vocabs.py $(DATA_DIR)$(DATASETNAME) $(VOCAB_DIR)
	python3 scripts/generate_vocabs.py --dataset-dir $(DATA_DIR)$(DATASETNAME) \
		--rationale-format $(RATIONALE_TYPE) \
		--num-ngrams $(NUM_NGRAMS) \
		--min-freq $(MIN_FREQ) \
		--max-tokens $(MAX_TOKENS) \
		--rationale-only \
		--output-path $(VOCAB_FILE)

$(DATA_DIR)$(DATASETNAME) : _legacy/data/processed_datasets/$(DATASETNAME)
	rm -rf $(DATA_DIR)$(DATASETNAME)
	cp -r _legacy/data/processed_datasets/$(DATASETNAME) $(DATA_DIR)$(DATASETNAME)

$(MODEL_DIR)$(DATASETNAME)_fasttext_${RATIONALE_FORMAT} : steps/run_task.py $(VOCAB_FILE) $(DATA_DIR)$(DATASETNAME) ENV.env $(CONFIG_DIR)removal_training_configs/$(REMOVAL_TRAINING_CONFIG)
	python3 steps/run_task.py $(CONFIG_DIR)removal_training_configs/$(REMOVAL_TRAINING_CONFIG)

vocab : $(VOCAB_FILE) ;

data : $(DATA_DIR)$(DATASETNAME) ;

removal_model : $(MODEL_DIR)$(DATASETNAME)_fasttext_${RATIONALE_FORMAT} ;

ENV.env :
	touch ENV.env

.PHONY : clean