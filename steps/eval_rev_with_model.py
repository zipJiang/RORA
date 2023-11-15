"""Evaluate REV with a model.
"""
import click
import torch
import os
from typing import Any, Dict, List, Optional, Text
from functools import partial
import spacy
import datasets
import transformers
import tokenizations
from src.models.fasttext import FastTextModule
from src.models.huggingface_wrapper_module import HuggingfaceWrapperModule
from src.collate_fns.strategyqa_collate_fn import StrategyQANGramClassificationCollateFn
from src.explainers.ig_explainer import IGExplainerFastText
from src.utils.explain_utils import get_explanation_scores


@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-dir", type=click.Path(exists=True), help="Path to the model directory.")
@click.option("--rationale-format", type=click.Choice(["gl", "gs", "g", "n", "l", "s"]), help="Rationale format.", default="gl")
@click.option("--removal-vocab-path", type=click.Path(exists=True), help="Path to the vocab file.", default=None)
@click.option("--removal-threshold", type=click.FLOAT, default=0.1, help="Threshold for removing the rationale.", show_default=True)
@click.option("--removal-model-dir", type=click.Path(exists=True), help="Model used for removing the rationale.", default=None)
def main(
    dataset_dir,
    model_dir,
    rationale_format,
    removal_vocab_path,
    removal_threshold,
    removal_model_dir,
):
    """
    """

    # model: Optional[torch.nn.Module] = None
    # model_type: Optional[Text] = None
    # if model_dir.startswith("fasttext"):
    #     model = FastTextModule.load_from_dir(model_dir)
    #     model_type = "fasttext"
    # elif model_dir.startswith("t5"):
    #     # TODO: This might be generalized later to support other models.
    #     model = HuggingfaceWrapperModule.load_from_dir(model_dir)
    #     model_type = "t5"
        
    dataset = datasets.load_from_disk(dataset_dir)
    
    if removal_model_dir is not None:
        
        vocab = torch.load(removal_vocab_path)
        num_ngram = int(removal_vocab_path.split("ng=")[-1].split("_")[0])
        
        removal_model = FastTextModule.load_from_dir(removal_model_dir)
        
        collate_fn = StrategyQANGramClassificationCollateFn(
            rationale_format=rationale_format,
            vocab=vocab,
            max_input_length=256,
            nlp_model="en_core_web_sm",
            num_ngrams=num_ngram,
        )
        
        explainer = IGExplainerFastText(
            num_steps=20,
            max_input_length=256,
            model=removal_model,
            device="cuda:0"
        )
        
        dataset = dataset.map(
            partial(
                get_explanation_scores,
                collate_fn=collate_fn,
                explainer=explainer,
            ),
            batched=True,
            batch_size=1000
        )
        
        
if __name__ == '__main__':
    main()