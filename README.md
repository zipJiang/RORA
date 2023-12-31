# REV-Fast-and-Slow

## Latest Result with DeBERTa

Each row corresponds to the $\lambda$ of the IRM regularizations.

|       | g           | gl          | s           |
| ----- | ----------- | ----------- | ----------- |
| 0.    | $x$ - 0.327 | $x$ - 0.460 | $x$ - 0.663 |
| 10.   | $x$ - 0.474 | $x$ - 0.425 | $x$ - 0.631 |
| 100.  | $x$ - 0.676 | $x$ - 0.678 | $x$ - 0.684 |
| 1000. | $x$ - 0.692 | $x$ - 0.692 | $x$ - 0.691 |

## Latest Result on Model rationale evaluation

$\lambda$ is the IRM regularizations coefficient and $\delta$ is the removal threshold

|T5-base  | g ($\lambda  = 100$, $\delta = 0.1$) | g($\lambda  = 10$, $\delta = 0.1$) |
| -----   | -----------------    | ----------------   |
| GPT-4   | $x$ - 0.441          | $x$ - 0.400        |
| GPT-3.5 | $x$ - 0.538          | $x$ - 0.810        |
| T5-large| $x$ - 0.705          | $x$ - 0.871        |
| GPT-2   | $x$ - 0.779          | $x$ - 1.086        |

## File Structure Description

```shellscript
steps/  // callable scripts corresponds to each step of REV score calculation.
src/   // source code of models, trainers, data collations etc. 
scripts/ // helper scripts to do examination, sanity check etc.
```

## Steps

Variable Examples:
```INPUT_DATA_PATH=/scratch/ylu130/data/strategyqa_dataset/strategyqa_train.json
OUTPUT_DIRECTORY=Zhengping/strategyqa_custom_split
OUTPUT_DIRECTORY2=Yining/generated_rationales/strategyqa
PROCESSED_DATA_DIRECTORY=data/processed_datasets/strategyqa
PROCESSED_DATA_DIRECTORY2=data/processed_datasets/strategyqa_model_rationale
DATA_NAME=gpt-4_demo=2_raw=True
```
### Prepare Synthetic Leaky Rationale
1. Split datasets: `python scripts/prepare_strategy_qa.py --input-path={INPUT_DATA_PATH} --output-path={OUTPUT_DIRECTORY}`
2. Prepare huggingface dataset: `python steps/rationale_preprocessing.py --data-handle={OUTPUT_DIRECTORY} --split={SPLIT} --write-to={PROCESSED_DATA_DIRECTORY}`
3. Generate rationale variants: `python scripts/generate_vocabs.py --dataset-dir={PROCESSED_DATA_DIRECTORY} --rationale-format={RATIONALE_FORMAT}`

### Prepare Base Models
1. Train models: `python steps/train_rev_model.py --task-name {TASK_NAME} --rationale-format {RATIONALE_FORMAT}`
   1. Train fasttext models used for leakage detection: `python steps/train_rev_model.py --task-name fasttext-{DATASET} --rationale-format {RATIONALE_FORMAT}`
   2. Train t5 models used for regular REV evaluation: `python steps/train_rev_model.py --task-name t5-{DATASET} --rationale-format {RATIONALE_FORMAT}`

### Detecting and Handling Leaky Parts
#### Detecting and Masking Leaky Tokens
1. Run IG and mask tokens: `python scripts/sample_masking.py --dataset-dir {PROCESSED_DATA_DIRECTORY} --rationale-format {RATIONALE_FORMAT} --minimum-frequency {MF} --write-to {OUTPUT_PATH}`

#### IRM Finetuning Evalaution Models 
1. Train Generator: `python steps/train_generator.py --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD}`
2. Sample intervened rationale datapoint: `python scripts/sample_intervention_generation.py --model-dir {TRAINED_MODEL_SAV_DIR} --data-dir {PROCESSED_DATA_DIRECTORY}`
3. Train IRM: `python steps/train_irm_model.py --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD}`
   
### Final REV Evaluation
1. Evaluate: `python steps/eval_rev_with_model.py --dataset-dir {PROCESSED_DATA_DIRECTORY} --model-dir {EVALUATING_MODEL_DIR} --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD} --removal-model-dir {REMOVAL_MODEL_DIR}`
   1. Use IRM finetuned model to evaluate: `python steps/eval_rev_with_model.py --dataset-dir {PROCESSED_DATA_DIRECTORY} --model-dir {EVALUATING_MODEL_DIR} --rationale-format {RATIONALE_FORMAT}`
   2. Use masked rationale to evaluate: `python steps/eval_rev_with_model.py  --dataset-dir {PROCESSED_DATA_DIRECTORY} --model-dir {EVALUATING_MODEL_DIR} --rationale-format {RATIONALE_FORMAT} --removal-threshold {THRESHOLD} --removal-model-dir {REMOVAL_MODEL_DIR}`

## Tests

### Evaluating Model Generated Rationale
1. Train rationale generator: `python steps/train_rationale_generator.py --model-name {MODEL_NAME}`
2. Generate model rationales for strategyqa: `python scripts/generate_rationales.py --dataset-dir {OUTPUT_DIRECTORY2} --model-name {MODEL_CHOICE} --num-sample {GENERATION_NUM} --demonstration-num {DEMONSTRATION_NUM} --output-dir {OUTPUT_DIR}`
3. Prepare huggingface dataset: `python steps/rationale_preprocessing.py --data-handle={OUTPUT_DIRECTORY2} --data-name={DATA_NAME} --split={SPLIT} --write-to={PROCESSED_DATA_DIRECTORY2}`
4. [Use IRM finetuned model to evaluate](#Final-REV-Evaluation)

### Evaluating Baselines
[Baseline Experiments](baselines/README.md)