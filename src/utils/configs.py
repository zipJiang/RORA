"""This module contains the configurations for getting
inputs for the models.
"""
from typing import Text, Dict, Any, List, Optional
import torch
import datasets
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    AutoTokenizer,
    AdamW
)
from src.models.huggingface_wrapper_module import (
    HuggingfaceWrapperModule
)
from src.models.fasttext import FastTextModule
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQAEmbeddingClassificationCollateFn,
    StrategyQAGenerationCollateFn,
    StrategyQANGramClassificationCollateFn
)
from src.trainers.strategyqa_trainer import StrategyQATrainer
from src.trainers.trainer import Trainer
from src.metrics.accuracy import (
    GenerationAccuracyMetric,
    ClassificationAccuracy
)
from src.metrics.loss import AvgLoss
from src.explainers.ig_explainer import IGExplainerFastText
from src.utils.explain_utils import get_explanation_scores


def get_params(
    task_name: Text,
    rationale_format: Text,
    removal_threshold: Optional[float] = None,
) -> Dict[Text, Any]:
    """Take a experiment handle nad generate the params to
    initialize a trainer.
    """
    
    if task_name == "t5-strategyqa":
        model_name = "t5-base"
        learning_rate = 1e-4
        model = HuggingfaceWrapperModule(
            model_handle=model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        dataset_train = datasets.load_from_disk(
            "data/processed_datasets/strategyqa/train"
        )
        dataset_eval = datasets.load_from_disk(
            "data/processed_datasets/strategyqa/validation"
        )
        
        attribution_model_dir = f"ckpt/fasttext-strategyqa_{rationale_format}/best_1/"
        
        explainer = IGExplainerFastText(
            num_steps=20,
            max_input_length=256,
            model=FastTextModule.load_from_dir(attribution_model_dir),
            device="cuda:0",
        )
        explainer_vocab = torch.load(f"data/processed_datasets/strategyqa/vocab_format={rationale_format}_ng=2_mf=1_mt=10000.pt")
        explainer_collate_fn = StrategyQANGramClassificationCollateFn(
            rationale_format=rationale_format,
            vocab=explainer_vocab,
            max_input_length=256,
            nlp_model="en_core_web_sm",
            num_ngrams=2,
        )
        
        dataset_train = dataset_train.map(
            partial(
                get_explanation_scores,
                collate_fn=explainer_collate_fn,
                explainer=explainer,
            ),
            batched=True
        )
        
        dataset_eval = dataset_eval.map(
            partial(
                get_explanation_scores,
                collate_fn=explainer_collate_fn,
                explainer=explainer,
            ),
            batched=True
        )

        collate_fn = StrategyQAGenerationCollateFn(
            rationale_format=rationale_format,
            tokenizer=tokenizer,
            removal_threshold=removal_threshold,
        )
        
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        trainer = StrategyQATrainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=learning_rate,
            ),
            metrics={
                "loss": AvgLoss(),
            },
            eval_metrics={
                "accuracy": GenerationAccuracyMetric(tokenizer=tokenizer),
                "loss": AvgLoss(),  # Notice that this is used to evaluate the logits for rev (best achievable)
            },
            main_metric="loss",
            save_dir=f"ckpt/{task_name}_{rationale_format}_{removal_threshold if removal_threshold is not None else 'none'}",
            direction='-',
            save_top_k=1,
            device="cuda:0",
        )
        
        return {
            "trainer": trainer,
            "dataloader_train": dataloader_train,
            "dataloader_eval": dataloader_eval,
        }
        
    elif task_name == "fasttext-strategyqa":
        num_ngrams = 2
        vocab = torch.load(f"data/processed_datasets/strategyqa/vocab_format={rationale_format}_ng={num_ngrams}_mf=1_mt=10000.pt")
        
        model = FastTextModule(
            vocab_size=len(vocab),
            embedding_dim=20,
            output_dim=2,
            pad_idx=vocab['<pad>'],
        )
        
        dataloader_train = DataLoader(
            dataset=datasets.load_from_disk(
                "data/processed_datasets/strategyqa/train"
            ),
            batch_size=256,
            shuffle=True,
            collate_fn=StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
                vocab=vocab,
                max_input_length=256,
                nlp_model="en_core_web_sm",
                num_ngrams=num_ngrams,
            )
        )
        
        dataloader_eval = DataLoader(
            dataset=datasets.load_from_disk(
                "data/processed_datasets/strategyqa/validation"
            ),
            batch_size=256,
            shuffle=False,
            collate_fn=StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
                vocab=vocab,
                max_input_length=256,
                nlp_model="en_core_web_sm",
                num_ngrams=num_ngrams,
            )
        )
        
        trainer = Trainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=1e-2,
            ),
            metrics={
                "accuracy": ClassificationAccuracy(),
                "loss": AvgLoss(),
            },
            eval_metrics={
                "accuracy": ClassificationAccuracy(),
                "loss": AvgLoss(),
            },
            main_metric="loss",
            direction='-',
            save_top_k=1,
            device="cuda:0",
            save_dir=f"ckpt/{task_name}_{rationale_format}",
        )
        
        return {
            "trainer": trainer,
            "dataloader_train": dataloader_train,
            "dataloader_eval": dataloader_eval,
        }