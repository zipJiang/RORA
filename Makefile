.SUFFIXES:

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

$(REPORT_FILE) : $(REV_MODEL) $(BASELINE_DATA_DIR) $(TESTING_CONFIG) $(BASELINE_MODEL) $(REV_DATA_DIR)
	if [ ! -d $(REPORT_DIR)$(DATASETNAME) ]; then mkdir -p $(REPORT_DIR)$(DATASETNAME); fi
	python3 steps/run_task.py $(TESTING_CONFIG)

$(RVR_REPORT_FILE) : $(RVR_MODEL) $(RVB_DATA_DIR) $(RVR_TESTING_CONFIG) $(RVB_MODEL) $(RVR_DATA_DIR)
	if [ ! -d $(REPORT_DIR)$(DATASETNAME) ]; then mkdir -p $(REPORT_DIR)$(DATASETNAME); fi
	python3 steps/run_task.py $(RVR_TESTING_CONFIG)

$(PLAIN_DATA_DIR) : /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME)
	if [ -d $(PLAIN_DATA_DIR) ]; then rm -r $(PLAIN_DATA_DIR); fi
	cp -r /brtx/605-nvme2/zpjiang/REV-reimpl/_legacy/data/processed_datasets/$(DATASETNAME) $(PLAIN_DATA_DIR)

$(RAW_DATA_DIR) : $(PLAIN_DATA_DIR) scripts/attach_model_generated_rationales.py
	if [ -d $(RAW_DATA_DIR) ]; then rm -r $(RAW_DATA_DIR); fi
	python3 scripts/attach_model_generated_rationales.py \
		--dataset-name $(DATASETNAME) \
		--dataset-dir $(PLAIN_DATA_DIR) \
		--output-dir $(RAW_DATA_DIR) \
		--rationale-dir $(MODEL_GENERATED_RATIONALE_DIR)

$(REMOVAL_DATA_DIR) : $(RAW_DATA_DIR) $(VOCAB_FILE) $(REMOVAL_PREPROCESSING_CONFIG)
	if [ -d $(REMOVAL_DATA_DIR) ]; then rm -r $(REMOVAL_DATA_DIR); fi
	mkdir -p $(REMOVAL_DATA_DIR)
	python3 steps/run_task.py $(REMOVAL_PREPROCESSING_CONFIG)

$(REMOVAL_MODEL) : $(REMOVAL_DATA_DIR) $(VOCAB_FILE) $(REMOVAL_TRAINING_CONFIG)
	if [ -d $(REMOVAL_MODEL) ]; then rm -r $(REMOVAL_MODEL); fi
	python3 steps/run_task.py $(REMOVAL_TRAINING_CONFIG)

$(GENERATION_DATA_DIR) : $(REMOVAL_MODEL) $(REMOVAL_DATA_DIR) $(VOCAB_FILE) $(GENERATION_PREPROCESSING_CONFIG)
	if [ -d $(GENERATION_DATA_DIR) ]; then rm -r $(GENERATION_DATA_DIR); fi	
	python3 steps/run_task.py $(GENERATION_PREPROCESSING_CONFIG)

$(GENERATION_MODEL) : $(GENERATION_DATA_DIR) $(GENERATION_TRAINING_CONFIG)
	if [ -d $(GENERATION_MODEL) ]; then rm -r $(GENERATION_MODEL); fi
	python3 steps/run_task.py $(GENERATION_TRAINING_CONFIG)

$(REV_DATA_DIR) : $(GENERATION_DATA_DIR) $(GENERATION_MODEL) $(REV_PREPROCESSING_CONFIG)
	if [ -d $(REV_DATA_DIR) ]; then rm -r $(REV_DATA_DIR); fi
	mkdir -p $(REV_DATA_DIR)
	python3 steps/run_task.py $(REV_PREPROCESSING_CONFIG)

$(REV_MODEL) : $(REV_DATA_DIR) $(REV_TRAINING_CONFIG)
	if [ -d $(REV_MODEL) ]; then rm -r $(REV_MODEL); fi
	python3 steps/run_task.py $(REV_TRAINING_CONFIG)

$(BASELINE_DATA_DIR) : $(RAW_DATA_DIR) $(BASELINE_PREPROCESSING_CONFIG)
	if [ -d $(BASELINE_DATA_DIR) ]; then rm -r $(BASELINE_DATA_DIR); fi
	mkdir -p $(BASELINE_DATA_DIR)
	python3 steps/run_task.py $(BASELINE_PREPROCESSING_CONFIG)

$(BASELINE_MODEL) : $(BASELINE_DATA_DIR) $(BASELINE_TRAINING_CONFIG)
	if [ -d $(BASELINE_MODEL) ]; then rm -r $(BASELINE_MODEL); fi
	python3 steps/run_task.py $(BASELINE_TRAINING_CONFIG)

$(RVB_DATA_DIR) : $(RAW_DATA_DIR) $(RVB_PREPROCESSING_CONFIG)
	if [ -d $(RVB_DATA_DIR) ]; then rm -r $(RVB_DATA_DIR); fi
	python3 steps/run_task.py $(RVB_PREPROCESSING_CONFIG)

$(RVB_MODEL) : $(RVB_DATA_DIR) $(RVB_TRAINING_CONFIG)
	if [ -d $(RVB_MODEL) ]; then rm -r $(RVB_MODEL); fi
	python3 steps/run_task.py $(RVB_TRAINING_CONFIG)

$(RVR_DATA_DIR) : $(RAW_DATA_DIR) $(RVR_PREPROCESSING_CONFIG)
	if [ -d $(RVR_DATA_DIR) ]; then rm -r $(RVR_DATA_DIR); fi
	python3 steps/run_task.py $(RVR_PREPROCESSING_CONFIG)

$(RVR_MODEL) : $(RVR_DATA_DIR) $(RVR_TRAINING_CONFIG)
	if [ -d $(RVR_MODEL) ]; then rm -r $(RVR_MODEL); fi
	python3 steps/run_task.py $(RVR_TRAINING_CONFIG)

raw_dataset : $(RAW_DATA_DIR)

vocab_file : $(VOCAB_FILE)

removal_dataset : $(REMOVAL_DATA_DIR)

removal_model: $(REMOVAL_MODEL)

generation_dataset: $(GENERATION_DATA_DIR)

generation_model : $(GENERATION_MODEL)

rev_dataset : $(REV_DATA_DIR)

rev_model : $(REV_MODEL)

baseline_dataset : $(BASELINE_DATA_DIR)

baseline_model : $(BASELINE_MODEL)

rvb_dataset : $(RVB_DATA_DIR)

rvb_model : $(RVB_MODEL)

rvr_dataset : $(RVR_DATA_DIR)

rvr_model : $(RVR_MODEL)

rvr_report_file : $(RVR_REPORT_FILE)

report_file : $(REPORT_FILE)

.PHONY : clean

clean :
	if [ -d $(RAW_DATA_DIR) ]; then rm -r $(RAW_DATA_DIR); fi
	if [ -d $(PLAIN_DATA_DIR) ]; then rm -r $(PLAIN_DATA_DIR); fi
	if [ -f $(VOCAB_FILE) ]; then rm $(VOCAB_FILE); fi
	if [ -d $(REMOVAL_DATA_DIR) ]; then rm -r $(REMOVAL_DATA_DIR); fi
	if [ -d $(REMOVAL_MODEL) ]; then rm -r $(REMOVAL_MODEL); fi
	if [ -d $(GENERATION_DATA_DIR) ]; then rm -r $(GENERATION_DATA_DIR); fi
	if [ -d $(GENERATION_MODEL) ]; then rm -r $(GENERATION_MODEL); fi
	if [ -d $(REV_DATA_DIR) ]; then rm -r $(REV_DATA_DIR); fi
	if [ -d $(REV_MODEL) ]; then rm -r $(REV_MODEL); fi
	if [ -f $(REPORT_FILE) ]; then rm $(REPORT_FILE); fi
