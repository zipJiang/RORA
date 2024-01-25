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
from src.explainers.ig_explainer import IGExplainerLSTM
# from src.preprocessors.strategyqa_preprocessor import StrategyQAGlobalExplanationPreprocessor
from src.preprocessors import ECQAGlobalExplanationPreprocessor, ECQACounterfactualGenerationPreprocessor
from src.models.huggingface_wrapper_module import HuggingfaceWrapperModule
# from src.models.fasttext import FastTextModule
from src.models.lstm import BiEncodingLSTMModule
from src.optimizer_constructors.optimizer_constructor import RegistrableAdamWConstructor
# from src.collate_fns.strategyqa_collate_fn import (
#     StrategyQAInfillingCollateFn,
#     StrategyQANGramClassificationCollateFn
# )
from src.trainers import ECQATrainer
from src.collate_fns import (
    ECQAInfillingCollateFn,
    ECQALstmClassificationCollateFn,
    ECQAClassificationCollateFn
)
from src.utils.common import (
    formatting_t5_generation,
)


@click.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True, help="Model directory.")
@click.option("--vocab-path", type=click.Path(exists=True), required=True, help="Vocab path.")
@click.option("--data-dir", type=click.Path(exists=True), required=True, help="Rationale format.")
@click.option("--rationale-format", type=click.STRING, default="g", help="Rationale format.")
@click.option("--num-samples", type=click.INT, default=10, help="Number of samples to generate (none for all).")
@click.option("--output-path", type=click.Path(exists=False), required=True, help="Output path.")
def main(
    model_dir,
    vocab_path,
    data_dir,
    rationale_format,
    num_samples,
    output_path
):
    """Running the trained model over
    the generation to get the interventions.
    """
    generation_model = HuggingfaceWrapperModule.load_from_best(model_dir)
    
    dataset_train = datasets.load_from_disk(os.path.join(data_dir, "train"))
    dataset_eval = datasets.load_from_disk(os.path.join(data_dir, "validation"))
    
    attribution_preprocessor = ECQAGlobalExplanationPreprocessor(
        explainer=IGExplainerLSTM(
            num_steps=20,
            max_input_length=256,
            max_output_length=256,
            device=torch.device("cuda:0")
        ),
        collate_fn=ECQALstmClassificationCollateFn(
            rationale_format=rationale_format,
            vocab=torch.load(vocab_path),
            max_input_length=256,
            max_output_length=32,
            num_ngrams=1,
            rationale_only=True,
        ),
        batch_size=128,
    )
    
    dataset_train, features = attribution_preprocessor(dataset_train)
    dataset_eval, _ = attribution_preprocessor(dataset_eval, precomputed_attributions=features)
    
    generation_preprocessor = ECQACounterfactualGenerationPreprocessor(
        generation_model=generation_model,
        collate_fn=ECQAInfillingCollateFn(
            rationale_format=rationale_format,
            tokenizer=generation_model.tokenizer,
            max_input_length=256,
            max_output_length=32,
            removal_threshold=0.001,
            intervention_on_label=False
        ),
        batch_size=32,
    )
    
    dataset_eval = generation_preprocessor(dataset_eval)
    
    results = []

    for idx, item in enumerate(dataset_eval):
        if idx >= num_samples:
            break
        
        instance = {
            f"op{i}": {
                "answer": item[f"q_op{i}"],
                "rationale":  item[f"generated_rationale_op{i}"]
            } for i in range(1, 6)
        }
        
        results.append(instance)
        
    with open(output_path, 'w', encoding='utf-8') as file_:
        json.dump(results, file_, indent=4, sort_keys=True)
            
if __name__ == "__main__":
    main()