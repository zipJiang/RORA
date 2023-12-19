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

1. Split datasets: `python scripts/prepare_strategy_qa.py --input-path={INPUT_DATA_PATH} --output-path={OUTPUT_DIRECTORY}`
2. Prepare huggingface dataset: `python steps/rationale_preprocessing.py --data-handle={OUTPUT_DIRECTORY} --split={SPLIT} --write-to={PROCESSED_DATA_DIRECTORY}`
3. Generate rationale variants: `python scripts/generate_vocabs.py --dataset-dir={PROCESSED_DATA_DIRECTORY} --rationale-format={RATIONALE_FORMAT}`
4. Train rev models: `python steps/train_rev_model.py --task-name {MODEL-DATASET} --rationale-format {RATIONALE_FORMAT}`
   1. Train all fasttext models: `bash bash/train_fasttext_models.sh`
5. Run IG and mask tokens: `python scripts/sample_masking.py --threshold {THRESHOLD} --dataset-dir {PROCESSED_DATA_DIRECTORY} --rationale-format {RATIONALE_FORMAT} --minimum-frequency {MF} --write-to {OUTPUT_PATH}`