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
RAW_DATA_DIR = $(DATA_DIR)$(DATASETNAME)/raw
REMOVAL_DATA_DIR = $(DATA_DIR)$(DATASETNAME)/removal_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)/
REMOVAL_MODEL = $(MODEL_DIR)$(DATASETNAME)_$(REMOVAL_MODEL_TYPE)_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)/
GENERATION_DATA_DIR = $(DATA_DIR)$(DATASETNAME)/generation_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)_th=$(THRESHOLD)/
GENERATION_MODEL = $(MODEL_DIR)$(DATASETNAME)_generation_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)_th=$(THRESHOLD)/
REV_DATA_DIR = $(DATA_DIR)$(DATASETNAME)/rev_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)_th=$(THRESHOLD)/
REV_MODEL = $(MODEL_DIR)$(DATASETNAME)_$(REV_MODEL_TYPE)_format=$(RATIONALE_FORMAT)_ng=$(NUM_NGRAMS)_mf=$(MIN_FREQ)_mt=$(MAX_TOKENS)_th=$(THRESHOLD)_irm=$(IRM_COEFFICIENT)/
BASELINE_DATA_DIR = $(DATA_DIR)$(DATASETNAME)/baseline/
BASELINE_MODEL = $(MODEL_DIR)$(DATASETNAME)_baseline/

# CONFIG_PATHS
REMOVAL_PREPROCESSING_CONFIG = $(CONFIG_DIR)removal_configs/$(DATASETNAME)_$(REMOVAL_MODEL_TYPE).yaml
REMOVAL_TRAINING_CONFIG = $(CONFIG_DIR)removal_training_configs/$(DATASETNAME)_$(REMOVAL_MODEL_TYPE).yaml
GENERATION_PREPROCESSING_CONFIG = $(CONFIG_DIR)generation_configs/$(DATASETNAME)_t5.yaml
GENERATION_TRAINING_CONFIG = $(CONFIG_DIR)generation_training_configs/$(DATASETNAME)_t5.yaml
REV_PREPROCESSING_CONFIG = $(CONFIG_DIR)rev_configs/$(DATASETNAME)_$(REV_MODEL_TYPE).yaml
REV_TRAINING_CONFIG = $(CONFIG_DIR)rev_training_configs/$(DATASETNAME)_$(REV_MODEL_TYPE).yaml
BASELINE_PREPROCESSING_CONFIG = $(CONFIG_DIR)baseline_configs/$(DATASETNAME)_$(REV_MODEL_TYPE).yaml
BASELINE_TRAINING_CONFIG = $(CONFIG_DIR)baseline_training_configs/$(DATASETNAME)_$(REV_MODEL_TYPE).yaml

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

%.py : ;
%.yaml : ;


$(VOCAB_FILE) : scripts/generate_vocabs.py $(RAW_DATA_DIR)
	if [ ! -d $(VOCAB_DIR) ]; then mkdir -p $(VOCAB_DIR); fi
ifeq ($(VOCAB_FILE), fasttext)
	echo "Using fasttext, no need to generate vocab"
else
	python3 scripts/generate_vocabs.py --dataset-dir $(RAW_DATA_DIR) \
		--dataset-name $(DATASETNAME) \
		--rationale-format $(RATIONALE_FORMAT) \
		--num-ngrams $(NUM_NGRAMS) \
		--min-freq $(MIN_FREQ) \
		--max-tokens $(MAX_TOKENS) \
		--rationale-only \
		--output-path $(VOCAB_FILE)
endif

$(RAW_DATA_DIR) : /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME)
	if [ -d $(RAW_DATA_DIR) ]; then rm -r $(RAW_DATA_DIR); fi
	cp -r /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME) $(DATA_DIR)$(DATASETNAME)

$(REMOVAL_DATA_DIR) : steps/run_task.py src/tasks/preprocessing_tasks.py $(RAW_DATA_DIR) $(VOCAB_FILE) $(REMOVAL_PREPROCESSING_CONFIG)
	if [ -d $(REMOVAL_DATA_DIR) ]; then rm -r $(REMOVAL_DATA_DIR); fi
	mkdir -p $(REMOVAL_DATA_DIR)
	python3 steps/run_task.py $(REMOVAL_PREPROCESSING_CONFIG)

$(REMOVAL_MODEL) : steps/run_task.py src/tasks/training_tasks.py $(REMOVAL_DATA_DIR) $(VOCAB_FILE) $(REMOVAL_TRAINING_CONFIG)
	if [ -d $(REMOVAL_MODEL) ]; then rm -r $(REMOVAL_MODEL); fi
	python3 steps/run_task.py $(REMOVAL_TRAINING_CONFIG)

$(GENERATION_DATA_DIR) : steps/run_task.py src/tasks/preprocessing_tasks.py $(REMOVAL_MODEL) $(REMOVAL_DATA_DIR) $(VOCAB_FILE) $(GENERATION_PREPROCESSING_CONFIG)
	if [ -d $(GENERATION_DATA_DIR) ]; then rm -r $(GENERATION_DATA_DIR); fi	
	python3 steps/run_task.py $(GENERATION_PREPROCESSING_CONFIG)

$(GENERATION_MODEL) : steps/run_task.py src/tasks/training_tasks.py $(GENERATION_DATA_DIR) $(GENERATION_TRAINING_CONFIG)
	if [ -d $(GENERATION_MODEL) ]; then rm -r $(GENERATION_MODEL); fi
	python3 steps/run_task.py $(GENERATION_TRAINING_CONFIG)

$(REV_DATA_DIR) : steps/run_task.py src/tasks/preprocessing_tasks.py $(GENERATION_DATA_DIR) $(GENERATION_MODEL) $(REV_PREPROCESSING_CONFIG)
	if [ -d $(REV_DATA_DIR) ]; then rm -r $(REV_DATA_DIR); fi
	mkdir -p $(REV_DATA_DIR)
	python3 steps/run_task.py $(REV_PREPROCESSING_CONFIG)

$(REV_MODEL) : steps/run_task.py src/tasks/training_tasks.py $(REV_DATA_DIR) $(REV_TRAINING_CONFIG)
	if [ -d $(REV_MODEL) ]; then rm -r $(REV_MODEL); fi
	python3 steps/run_task.py $(REV_TRAINING_CONFIG)

$(BASELINE_DATA_DIR) : steps/run_task.py src/tasks/preprocessing_tasks.py $(RAW_DATA_DIR) $(BASELINE_PREPROCESSING_CONFIG)
	if [ -d $(BASELINE_DATA_DIR) ]; then rm -r $(BASELINE_DATA_DIR); fi
	mkdir -p $(BASELINE_DATA_DIR)
	python3 steps/run_task.py $(BASELINE_PREPROCESSING_CONFIG)

$(BASELINE_MODEL) : steps/run_task.py src/tasks/training_tasks.py $(BASELINE_DATA_DIR) $(BASELINE_TRAINING_CONFIG)
	if [ -d $(BASELINE_MODEL) ]; then rm -r $(BASELINE_MODEL); fi
	python3 steps/run_task.py $(BASELINE_TRAINING_CONFIG)

removal_dataset : $(REMOVAL_DATA_DIR)

removal_model: $(REMOVAL_MODEL)

generation_dataset: $(GENERATION_DATA_DIR)

generation_model : $(GENERATION_MODEL)

rev_dataset : $(REV_DATA_DIR)

rev_model : $(REV_MODEL)

baseline_dataset : $(BASELINE_DATA_DIR)

baseline_model : $(BASELINE_MODEL)

.PHONY : clean

clean :
	if [ -f $(VOCAB_FILE) ]; then rm $(VOCAB_FILE); fi
	if [ -d $(REMOVAL_DATA_DIR) ]; then rm -r $(REMOVAL_DATA_DIR); fi
	if [ -d $(REMOVAL_MODEL) ]; then rm -r $(REMOVAL_MODEL); fi
	if [ -d $(GENERATION_DATA_DIR) ]; then rm -r $(GENERATION_DATA_DIR); fi
	if [ -d $(GENERATION_MODEL) ]; then rm -r $(GENERATION_MODEL); fi
	if [ -d $(REV_DATA_DIR) ]; then rm -r $(REV_DATA_DIR); fi
	if [ -d $(REV_MODEL) ]; then rm -r $(REV_MODEL); fi
