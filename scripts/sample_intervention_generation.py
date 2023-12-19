"""Get a model and sample and generate interventions
for the rationale datapoint.
"""
import click
import os
import transformers
import datasets
import torch
import json
from typing import Text
from torch.utils.data import DataLoader
from src.explainers.ig_explainer import IGExplainerFastText
from src.preprocessors.strategyqa_preprocessor import StrategyQAGlobalExplanationPreprocessor
from src.models.huggingface_wrapper_module import HuggingfaceWrapperModule
from src.models.fasttext import FastTextModule
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQAInfillingCollateFn,
    StrategyQANGramClassificationCollateFn
)
from src.utils.common import (
    formatting_t5_generation,
)

CACHE_DIR = '/scratch/ylu130/model-hf'

# TODO: This may subject to interface change
def parse_model_dir(model_dir: Text):
    basename = os.path.basename(model_dir if not model_dir.endswith("/") else model_dir[:-1])
    return {
        "taskname": basename.split("_")[0],
        "model_handle": basename.split("_")[1],
        "rationale_format": basename.split("_")[-2],
        "removal_threshold": float(basename.split("_")[-1]),
    }


@click.command()
@click.option("--model-dir", type=click.STRING, default="t5-base", help="Model to evaluate.")
@click.option("--data-dir", type=click.STRING, required=True, help="Data Directory")
@click.option("--num-samples", type=click.INT, default=None, help="Number of samples to generate (none for all).")
def main(
    model_dir,
    data_dir,
    num_samples
):
    """Running the trained model over
    the generation to get the interventions.
    """
    
    # TODO: generalize this to other tasks.
    model = HuggingfaceWrapperModule.load_from_dir(os.path.join(model_dir, "best_1"))
    hyperparams = parse_model_dir(model_dir)
    print(hyperparams)
    model.train(False)
    model.to('cuda:0')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_handle, cache_dir=CACHE_DIR)

    # preprocess to get attributions
    attribution_model_dir = "{ckpt}/fasttext-strategyqa_{rationale_format}_{vocab_minimum_frequency}/best_1/".format(
        ckpt="/scratch/ylu130/project/REV_reimpl/ckpt",
        vocab_minimum_frequency=1,
        **hyperparams
    )
    
    explainer = IGExplainerFastText(
        num_steps=20,
        max_input_length=256,
        model=FastTextModule.load_from_dir(attribution_model_dir),
        device="cuda:0",
    )
    explainer_vocab = torch.load("data/processed_datasets/strategyqa/vocab_format={rationale_format}_ng=2_mf={vocab_minimum_frequency}_mt=10000_r=1.pt".format(
        vocab_minimum_frequency=1,
        **hyperparams
    ))
    explainer_collate_fn = StrategyQANGramClassificationCollateFn(
        rationale_format=hyperparams["rationale_format"],
        vocab=explainer_vocab,
        max_input_length=256,
        nlp_model="en_core_web_sm",
        num_ngrams=2,
        rationale_only=True,
    )
    
    additional_preprocessor = StrategyQAGlobalExplanationPreprocessor(
        batch_size=1024,
        explainer=explainer,
        collate_fn=explainer_collate_fn,
    )
    # load the eval dataset.
    dataset = datasets.load_from_disk(os.path.join(data_dir, "validation"))
    dataset_train = datasets.load_from_disk(os.path.join(data_dir, "train"))
    
    dataset_train, train_features = additional_preprocessor(dataset_train)
    dataset, _ = additional_preprocessor(dataset, features=train_features)
    
    collate_fn = StrategyQAInfillingCollateFn(
        tokenizer=tokenizer,
        max_input_length=256,
        max_output_length=256,
        removal_threshold=hyperparams["removal_threshold"],
        rationale_format=hyperparams["rationale_format"],
    )
    counterfactual_collate_fn = StrategyQAInfillingCollateFn(
        tokenizer=tokenizer,
        max_input_length=256,
        max_output_length=256,
        removal_threshold=hyperparams["removal_threshold"],
        rationale_format=hyperparams["rationale_format"],
        intervention_on_label=True,
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False
    )
    counterfactual_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=counterfactual_collate_fn,
        shuffle=False
    )
    
    results = []
    
    num_samples = len(dataloader) if num_samples is None else num_samples
    
    with torch.no_grad():
        for batch, counterfactual_batch, bidx in zip(dataloader, counterfactual_dataloader, range(num_samples)):
            sequence_ids = model.generate(
                batch['input_ids'].to('cuda:0'),
                max_new_tokens=256,
                temperature=0.0
            )
            counterfactual_sequence_ids = model.generate(
                counterfactual_batch['input_ids'].to('cuda:0'),
                max_new_tokens=256,
                temperature=0.0
            )
            
            inputs = tokenizer.decode(batch['input_ids'][0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            labels = tokenizer.decode(batch['labels'][0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            counterfactual_inputs = tokenizer.decode(counterfactual_batch['input_ids'][0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            decoded = tokenizer.decode(sequence_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            counterfactual_decoded = tokenizer.decode(counterfactual_sequence_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            results.append({
                "bidx": bidx,
                "original": formatting_t5_generation(inputs, decoded),
                "counterfactual": formatting_t5_generation(counterfactual_inputs, counterfactual_decoded),
            })
            
    with open("data/examinations/counterfactuals/intervention_generation_{rationale_format}_{vocab_minimum_frequency}_{removal_threshold}.json".format(
        vocab_minimum_frequency=1,
        **hyperparams
    ), "w") as f:
        json.dump(results, f, indent=4)
            
            
if __name__ == "__main__":
    main()