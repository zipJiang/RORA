# RORA: Robust Free-Text Rationale Evaluation

## File Structure Description

```shellscript
steps/  // callable scripts corresponds to each step of REV score calculation.
src/   // source code of models, trainers, data collations etc. 
scripts/ // helper scripts to do examination, sanity check etc.
```

## Dataset Specification

We need to prepare the dataset in the following format:

### StrategyQA

```json
{
   "qid": "7fa631340ce8c42aba53",
   "term": "1980 United States presidential election",
   "description": "49th quadrennial presidential election in the United States",
   "question": "Were there greater landslides than 1980 United States presidential election?",
   "answer": true,
   "facts": [
      "A landslide refers to a competitor beating their opponent by a wide margin.",
      "Ronald Reagan defeated Jimmy carter in the 1980 United States presidential election by around 8 million votes.",
      "Franklin D. Roosevelt won the 1936 United States presidential election over Alf Landon by more than 11 million votes.",
      "In 1804 Thomas Jefferson received 162 (92%) of the electoral votes while Charles Cotesworth Pinckney received only 14 (8%)."
   ],
   "decomposition": [
      "By what votes margin did Ronald Reagan defeat Jimmy Carter in the 1980 US Presidential election?",
      "By how many votes was Franklin D. Roosevelt leading Alf Landon in the 1936 US Presidential election?",
      "How many more votes did Thomas Jefferson receive than Charles Cotesworth Pinckney in the 1804 United States presidential election?", "Are #2 and #3 greater individually than #1?"
   ],
   "vacuous_rationale": "There were greater landslides than 1980 United States presidential election."
}
```

### ECQA

We use the same split [here](https://huggingface.co/datasets/yangdong/ecqa).


The scripts expect data in the huggingface datasets format.

## Configurations

To run the code, you need to configure the environment by running the following command:

```bash

pip install -r requirements.txt
```

In case pwd is not in the PYTHONPATH, you need to add the path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Steps

1. Configure the environment

```bash
source runs/configure.sh \
    --removal-model-type=${REMOVAL_MODEL_TYPE} \
    --dataset-name=${DATASETNAME} \
    --rationale-format=${RATIONALE_FORMAT} \
    --num-ngrams=${NUM_NGRAMS} \
    --min-freq=${MIN_FREQ} \
    --max-tokens=${MAX_TOKENS} \
    --threshold=${THRESHOLD} \
    --irm-coefficient=${IRM_COEFFICIENT} \
    --rev-model-type=${REV_MODEL_TYPE} \
    --removal-epochs=${REMOVAL_EPOCHS} \
    --removal-batch-size=${REMOVAL_BATCH_SIZE} \
    --generation-epochs=${GENERATION_EPOCHS} \
    --generation-batch-size=${GENERATION_BATCH_SIZE} \
    --rev-epochs=${REV_EPOCHS} \
    --rev-batch-size=${REV_BATCH_SIZE} \
    --learning-rate=${REV_LEARNING_RATE}
```

By specifying the relevant parameters, the script will make sure all the relevant varaibles in the process got properly configured.

2. Create raw data by appending model-generated rationales

```bash
make raw_dataset
```

3. Create vocab files

Notice that we will use non-pretrained models to calculate attributions, so that we need to create our own vocab to avoid under-training.

```bash
make vocab_file
```

4. Create removal dataset that is used to train removal_model and calculate attributions

```bash
make removal_dataset
```

5. Train removal model

```bash
make removal_model
```

6. Generation dataset creation

Now using the calculation we generate the dataset that can be used to train the generation model.

```bash
make generation_dataset
```

7. Train generation model

```bash
make generation_model
```

8. Create the REV dataset

Now we have the model to generate counterfactual rationales for data, we can create the REV dataset.

```bash
make rev_dataset
```

9. Train REV model

```bash
make rev_model
```

10. Notice that we need a baseline model to calculate REV, so we need to train a baseline model (first we prepare dataset).

```bash
make baseline_dataset
```

11. Actually training the baseline model

```bash
make baseline_model
```

12. From these two models we are able to generate the report file.

```bash
make report_file
```

Notice that due to the nature of `Makefile`, the process will be executed in a pipeline manner, so that if you want to re-run the process, you need to clean the intermediate files.

```bash
make clean
```

And making the last step `report_file` will run the whole process.

## Citation

If you use this code, please cite the following paper:

```bibtex
@misc{jiang2024rora,
      title={RORA: Robust Free-Text Rationale Evaluation}, 
      author={Zhengping Jiang and Yining Lu and Hanjie Chen and Daniel Khashabi and Benjamin Van Durme and Anqi Liu},
      year={2024},
      eprint={2402.18678},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```