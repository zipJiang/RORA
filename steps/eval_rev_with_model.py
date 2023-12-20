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
from src.models.huggingface_classifier import HuggingfaceClassifierModule
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQANGramClassificationCollateFn,
    StrategyQAGenerationCollateFn,
    StrategyQAEmbeddingClassificationCollateFn
)
from src.explainers.ig_explainer import IGExplainerFastText
from src.preprocessors.strategyqa_preprocessor import StrategyQAGlobalExplanationPreprocessor
from src.trainers.strategyqa_trainer import (
    StrategyQATrainer,
    StrategyQAIRMTrainer,
    StrategyQAClassificationIRMTrainer
)
from src.metrics.loss import AvgLoss
from src.metrics.accuracy import (
    GenerationAccuracyMetric,
    ClassificationAccuracy
)
from src.schedulers.linear_scheduler import LinearScheduler

CACHE_DIR = '/scratch/ylu130/model-hf'

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-dir", type=click.Path(exists=True), help="Path to the model directory.")
@click.option("--rationale-format", type=click.Choice(["gl", "gs", "g", "n", "l", "s"]), help="Rationale format.", default="gl")
@click.option("--removal-threshold", type=click.FLOAT, default=None, help="Threshold for removing the rationale.", show_default=True)
@click.option("--removal-model-dir", type=click.Path(exists=True), help="Model used for removing the rationale.", default=None)
@click.option("--vocab-minimum-frequency", type=click.INT, default=1, help="Minimum frequency for the vocabulary.", show_default=True)
def main(
    dataset_dir,
    model_dir,
    rationale_format,
    removal_threshold,
    removal_model_dir,
    vocab_minimum_frequency,
):
    """
    """

    # model: Optional[torch.nn.Module] = None
    train_dataset = datasets.load_from_disk(os.path.join(dataset_dir, "train"))
    dataset = datasets.load_from_disk(os.path.join(dataset_dir, "test"))
    
    if removal_model_dir is not None:
        num_ngrams = 2
        vocab = torch.load(f"data/processed_datasets/strategyqa/vocab_format={rationale_format}_ng={num_ngrams}_mf={vocab_minimum_frequency}_mt=10000_r=1.pt")
        
        removal_model = FastTextModule.load_from_dir(os.path.join(removal_model_dir, "best_1"))
        
        preprocess_collate_fn = StrategyQANGramClassificationCollateFn(
            rationale_format=rationale_format,
            vocab=vocab,
            max_input_length=256,
            nlp_model="en_core_web_sm",
            num_ngrams=num_ngrams,
        )
        
        explainer = IGExplainerFastText(
            num_steps=20,
            max_input_length=256,
            model=removal_model,
            device="cuda:0"
        )
        preprocessor = StrategyQAGlobalExplanationPreprocessor(
            explainer=explainer,
            collate_fn=preprocess_collate_fn,
            batch_size=1024,
        )

        train_dataset, train_features = preprocessor(train_dataset)
        dataset, _ = preprocessor(dataset, features=train_features)
    else:
        assert removal_threshold is None, "You must specify a removal model directory to use a removal threshold."
        
    # get the correct model
    if os.path.basename(model_dir if not model_dir.endswith("/") else model_dir[:-1]).split("_")[1].startswith("t5"):
        model_dir = os.path.join(model_dir, "best_1")
        model = HuggingfaceWrapperModule.load_from_dir(model_dir)
        model.eval()
        model.to("cuda:0")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_handle, cache_dir=CACHE_DIR)
        
        collate_fn = StrategyQAGenerationCollateFn(
            rationale_format=rationale_format,
            max_input_length=256,
            max_output_length=32,
            removal_threshold=removal_threshold,
            mask_by_delete=False,
            tokenizer=tokenizer,
        )
        

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1024,
            collate_fn=collate_fn,
        )
        
        trainer = StrategyQATrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0),
            device="cuda:0",
            metrics={
                "loss": AvgLoss(),
            },
            eval_metrics={
                "loss": AvgLoss(),
                "acc": GenerationAccuracyMetric(tokenizer=tokenizer),
            },
            main_metric="loss",
            save_dir=None,
        )
    
    else:
        model_dir = os.path.join(model_dir, "best_1")
        model = HuggingfaceClassifierModule.load_from_dir(model_dir)
        model.eval()
        model.to("cuda:0")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_handle, cache_dir=CACHE_DIR)

        collate_fn = StrategyQAEmbeddingClassificationCollateFn(
            rationale_format=rationale_format,
            max_input_length=256,
            tokenizer=tokenizer,
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1024,
            collate_fn=collate_fn,
        )
        
        trainer = StrategyQAClassificationIRMTrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0),
            device="cuda:0",
            metrics={
                "loss": AvgLoss(),
            },
            eval_metrics={
                "loss": AvgLoss(),
                "acc": ClassificationAccuracy()
            },
            main_metric="loss",
            irm_scheduler=LinearScheduler(
                start_val=0.0,
                end_val=0.0,
                num_steps=0
            ),
            save_dir=None,
        )
    
    eval_dict = trainer.evaluate(
        dataloader=dataloader,
        epoch=0
    )
        
    print(eval_dict)
        
if __name__ == '__main__':
    main()