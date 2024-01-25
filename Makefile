DATA_DIR = data/
MODEL_DIR = ckpt/
CONFIG_DIR = configs/
VOCAB_DIR := $(DATA_DIR)$(DATASETNAME)_vocabs/
SUBDIRS = data ckpt steps
PYTHONPATH = $(shell pwd)
# REMOVAL_MODEL_TYPE = fasttext
# DATASETNAME = strategyqa
# REV_MODEL_TYPE = bart
# RATIONALE_FORMAT = g
# NUM_NGRAMS = 2
# MIN_FREQ = 1
# MAX_TOKENS = 10000
# THRESHOLD = 0.001
# IRM_COEFFICIENT = 10.0
REV_EPOCHS = 20
REMOVAL_EPOCHS = 20
REV_BATCH_SIZE = 32
UTIL_DIR = src/utils/

# SET VOCAB_FILE to fasttext if that's what you need.
VOCAB_FILE ?= $(VOCAB_DIR)vocab_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS).pt
REMOVAL_TRAINING_CONFIG = $(CONFIG_DIR)removal_training_configs/$(DATASETNAME)_$(REMOVAL_MODEL_TYPE).yaml
GENERATION_TRAINING_CONFIG = $(CONFIG_DIR)generation_training_configs/$(DATASETNAME)_t5.yaml
REV_TRAINING_CONFIG = $(CONFIG_DIR)rev_training_configs/$(DATASETNAME)_$(REV_MODEL_TYPE).yaml
REMOVAL_MODEL = $(MODEL_DIR)$(DATASETNAME)_$(REMOVAL_MODEL_TYPE)_$(RATIONALE_FORMAT)
GENERATION_MODEL = $(MODEL_DIR)$(DATASETNAME)_generation_$(RATIONALE_FORMAT)_$(MIN_FREQ)_$(MAX_TOKENS)_$(THRESHOLD)
REV_MODEL = $(MODEL_DIR)$(DATASETNAME)_rev_$(REV_MODEL_TYPE)_$(RATIONALE_FORMAT)_$(MIN_FREQ)_$(MAX_TOKENS)_$(THRESHOLD)_$(IRM_COEFFICIENT)

export PYTHONPATH

# ACTUAL STEPS

.SUFFIXES:

score_report.json : steps/eval_rev_with_model.py $(wildcard UTIL_DIR*.py) $(REV_MODEL) $(REMOVAL_MODEL) vocab data
	python3 steps/eval_rev_with_model.py --dataset-dir $(DATA_DIR)$(DATASETNAME) \
		--model-dir $(REV_MODEL) \
		--removal-model-dir $(REMOVAL_MODEL) \
		--rationale-format $(RATIONALE_FORMAT) \
		--removal-threshold $(THRESHOLD) \
		--vocab-minimum-frequency $(MIN_FREQ)

$(REV_MODEL) : steps/run_task.py $(REMOVAL_MODEL) $(GENERATION_MODEL) $(VOCAB_FILE) $(DATA_DIR)$(DATASETNAME)
	python3 steps/run_task.py $(REV_TRAINING_CONFIG)

$(VOCAB_DIR) :
	mkdir -p $(VOCAB_DIR)


%.py : ;
%.yaml : ;


$(VOCAB_FILE) : scripts/generate_vocabs.py $(DATA_DIR)$(DATASETNAME) $(VOCAB_DIR)
ifeq ($(VOCAB_FILE), fasttext)
	echo "Using fasttext, no need to generate vocab"
else
	python3 scripts/generate_vocabs.py --dataset-dir $(DATA_DIR)$(DATASETNAME) \
		--rationale-format $(RATIONALE_FORMAT) \
		--num-ngrams $(NUM_NGRAMS) \
		--min-freq $(MIN_FREQ) \
		--max-tokens $(MAX_TOKENS) \
		--rationale-only \
		--output-path $(VOCAB_FILE)
endif

$(DATA_DIR)$(DATASETNAME) : /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME)
	rm -rf $(DATA_DIR)$(DATASETNAME)
	cp -r /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME) $(DATA_DIR)$(DATASETNAME)

$(REMOVAL_MODEL) : steps/run_task.py $(VOCAB_FILE) $(DATA_DIR)$(DATASETNAME) ENV.env $(REMOVAL_TRAINING_CONFIG)
	rm -rf $(MODEL_DIR)$(DATASETNAME)_$(REMOVAL_MODEL_TYPE)_${RATIONALE_FORMAT}
	python3 steps/run_task.py $(REMOVAL_TRAINING_CONFIG)

$(GENERATION_MODEL) : steps/run_task.py $(VOCAB_FILE) $(DATA_DIR)$(DATASETNAME) ENV.env $(GENERATION_TRAINING_CONFIG) $(REMOVAL_MODEL)
	rm -rf $(GENERATION_MODEL)
	python3 steps/run_task.py $(GENERATION_TRAINING_CONFIG)

vocab : $(VOCAB_FILE) ;

data : $(DATA_DIR)$(DATASETNAME) ;

removal_model : $(REMOVAL_MODEL) ;

generation_model : $(GENERATION_MODEL) ;

rev_model : $(REV_MODEL) ;

ENV.env :
	touch ENV.env

.PHONY : clean