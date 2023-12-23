# REV-Fast-and-Slow

## Latest Result with DeBERTa

Each row corresponds to the $\lambda$ of the IRM regularizations.

|       | g           | gl          | s           |
| ----- | ----------- | ----------- | ----------- |
| 0.    | $x$ - 0.327 | $x$ - 0.460 | $x$ - 0.663 |
| 10.   | $x$ - 0.474 | $x$ - 0.425 | $x$ - 0.631 |
| 100.  | $x$ - 0.676 | $x$ - 0.678 | $x$ - 0.684 |
| 1000. | $x$ - 0.692 | $x$ - 0.692 | $x$ - 0.691 |

## File Structure Description

```shellscript
steps/  // callable scripts corresponds to each step of REV score calculation.
src/   // source code of models, trainers, data collations etc. 
scripts/ // helper scripts to do examination, sanity check etc.
```

## Steps

Variables:
* `INPUT_DATA_PATH=/scratch/ylu130/data/strategyqa_dataset/strategyqa_train.json`
* `OUTPUT_DIRECTORY=Zhengping/strategyqa_custom_split`
* `PROCESSED_DATA_DIRECTORY=data/processed_datasets/strategyqa`
### Prepare Evaluation Models
1. Split datasets: `python scripts/prepare_strategy_qa.py --input-path={INPUT_DATA_PATH} --output-path={OUTPUT_DIRECTORY}`
2. Prepare huggingface dataset: `python steps/rationale_preprocessing.py --data-handle={OUTPUT_DIRECTORY} --split={SPLIT} --write-to={PROCESSED_DATA_DIRECTORY}`
3. Generate rationale variants: `python scripts/generate_vocabs.py --dataset-dir={PROCESSED_DATA_DIRECTORY} --rationale-format={RATIONALE_FORMAT}`
4. Train models: `python steps/train_rev_model.py --task-name {MODEL-DATASET} --rationale-format {RATIONALE_FORMAT}`
   1. Train all fasttext models: `bash bash/train_fasttext_models.sh`
   2. Train all t5 models: `bash bash/train_t5_models.sh`
### Masking Leaky tokens
5. Run IG and mask tokens: `python scripts/sample_masking.py --dataset-dir {PROCESSED_DATA_DIRECTORY} --rationale-format {RATIONALE_FORMAT} --minimum-frequency {MF} --write-to {OUTPUT_PATH}`
### Prepare IRM Finetuned Evaluation Model 
6. Train Generator: `python steps/train_generator.py --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD}`
7. Sample intervened rationale datapoint: `python scripts/sample_intervention_generation.py --model-dir {TRAINED_MODEL_SAV_DIR} --data-dir {PROCESSED_DATA_DIRECTORY}`
8. Train IRM: `python steps/train_irm_model.py --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD}`
### Final REV Evaluation
9. Evaluate: `python steps/eval_rev_with_model.py --dataset-dir {PROCESSED_DATA_DIRECTORY} --model-dir {EVALUATING_MODEL_DIR} --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD} --removal-model-dir {REMOVAL_MODEL_DIR}`

### Rationale Gneration
1. Train rationale generator: `python steps/train_rationale_generator.py --model-name {MODEL_NAME}`
2. Generate model rationales for strategyqa: `python scripts/generate_rationales.py --dataset-dir {OUTPUT_DIRECTORY} --model-name {MODEL_CHOICE} --num-sample {GENERATION_NUM} --demonstration-num {DEMONSTRATION_NUM} --output-dir {OUTPUT_DIR}`
