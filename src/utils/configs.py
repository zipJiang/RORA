"""This module contains the configurations for getting
inputs for the models.
"""
from typing import Text, Dict, Any, List, Optional
import torch
import numpy as np
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
from src.models.huggingface_classifier import (
    HuggingfaceClassifierModule
)
from src.models.fasttext import FastTextModule
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQAGenerationCollateFn,
    StrategyQANGramClassificationCollateFn,
    StrategyQAInfillingCollateFn,
    StrategyQAIRMCollateFn,
    StrategyQAEmbeddingClassificationCollateFn,
    StrategyQAIRMEmbeddingClassificationCollateFn
)
from src.trainers.strategyqa_trainer import (
    StrategyQATrainer,
    StrategyQAInfillTrainer,
    StrategyQAIRMTrainer,
    StrategyQAClassificationIRMTrainer
)
from src.trainers.trainer import Trainer
from src.metrics.accuracy import (
    GenerationAccuracyMetric,
    ClassificationAccuracy
)
from src.metrics.loss import AvgLoss
from src.metrics.stats_extractor import StatsExtractor
from src.explainers.ig_explainer import IGExplainerFastText
from src.preprocessors.strategyqa_preprocessor import (
    StrategyQALocalExplanationPreprocessor,
    StrategyQAGlobalExplanationPreprocessor,
    StrategyQACounterfactualGenerationPreprocessor
)
from src.schedulers.linear_scheduler import LinearScheduler


def get_params(
    task_name: Text,
    rationale_format: Text,
    vocab_minimum_frequency: int = 1,
    removal_threshold: Optional[float] = None,
    mask_by_delete: bool = False,
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
            "data/strategyqa/train"
        )
        dataset_eval = datasets.load_from_disk(
            "data/strategyqa/validation"
        )

        if rationale_format != "n":
            attribution_model_dir = f"ckpt/fasttext-strategyqa_{rationale_format}_{vocab_minimum_frequency}/best_1/"
            
            explainer = IGExplainerFastText(
                num_steps=20,
                max_input_length=256,
                model=FastTextModule.load_from_dir(attribution_model_dir),
                device="cuda:0",
            )
            explainer_vocab = torch.load(f"data/strategyqa_vocabs/vocab_format={rationale_format}_ng=2_mf={vocab_minimum_frequency}_mt=10000_r=1.pt")
            explainer_collate_fn = StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
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
            
            dataset_train, train_features = additional_preprocessor(dataset_train)
            dataset_eval, _ = additional_preprocessor(dataset_eval, features=train_features)

        collate_fn = StrategyQAGenerationCollateFn(
            rationale_format=rationale_format,
            tokenizer=tokenizer,
            removal_threshold=removal_threshold,
            mask_by_delete=mask_by_delete,
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
            save_dir=f"ckpt/{task_name}_{rationale_format}_{vocab_minimum_frequency}_{removal_threshold if removal_threshold is not None else 'none'}_{'delete' if mask_by_delete else 'mask'}",
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
        vocab = torch.load(f"data/strategyqa_vocabs/vocab_format={rationale_format}_ng={num_ngrams}_mf={vocab_minimum_frequency}_mt=10000_r=1.pt")
        
        model = FastTextModule(
            vocab_size=len(vocab),
            embedding_dim=20,
            output_dim=2,
            pad_idx=vocab['<pad>'],
        )
        
        dataloader_train = DataLoader(
            dataset=datasets.load_from_disk(
                "data/strategyqa/train"
            ),
            batch_size=256,
            shuffle=True,
            collate_fn=StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
                vocab=vocab,
                max_input_length=256,
                nlp_model="en_core_web_sm",
                num_ngrams=num_ngrams,
                rationale_only=True,
            )
        )
        
        dataloader_eval = DataLoader(
            dataset=datasets.load_from_disk(
                "data/strategyqa/validation"
            ),
            batch_size=256,
            shuffle=False,
            collate_fn=StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
                vocab=vocab,
                max_input_length=256,
                nlp_model="en_core_web_sm",
                num_ngrams=num_ngrams,
                rationale_only=True
            )
        )
        
        trainer = Trainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=1e-1,
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
        
        
def get_generation_params(
    task_name: Text,
    model_name: Text,
    rationale_format: Text,
    removal_threshold: float,
    batch_size: int,
    minimum_frequency: int
):
    """Get parameters that are specific to generation models.
    """
    
    learning_rate = 1e-4
    
    if task_name == 'strategyqa':
        model = HuggingfaceWrapperModule(
            model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        
        num_ngram = 2
        
        preprocessor = StrategyQAGlobalExplanationPreprocessor(
            explainer=IGExplainerFastText(
                num_steps=20,
                max_input_length=256,
                model=FastTextModule.load_from_dir(f"ckpt/fasttext-strategyqa_{rationale_format}_{minimum_frequency}/best_1/"),
                device="cuda:0",
            ),
            collate_fn=StrategyQANGramClassificationCollateFn(
                rationale_format=rationale_format,
                vocab=torch.load(f"data/strategyqa/vocab_format={rationale_format}_ng={num_ngram}_mf={minimum_frequency}_mt=10000_r=1.pt"),
                max_input_length=256,
                nlp_model="en_core_web_sm",
                num_ngrams=2,
                rationale_only=True,
            ),
            batch_size=batch_size
        )
        
        dataset_train = datasets.load_from_disk(
            "data/strategyqa/train"
        )
        dataset_train, train_features = preprocessor(dataset_train)
        dataset_eval = datasets.load_from_disk(
            "data/strategyqa/validation",
        )
        dataset_eval, _ = preprocessor(dataset_eval, features=train_features)
        
        collate_fn = StrategyQAInfillingCollateFn(
            rationale_format=rationale_format,
            tokenizer=tokenizer,
            max_input_length=256,
            max_output_length=256,
            removal_threshold=removal_threshold,
        )
        
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        trainer = StrategyQAInfillTrainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=learning_rate,
            ),
            metrics={
                "loss": AvgLoss()
            },
            eval_metrics={
                "loss": AvgLoss(),
            },
            main_metric="loss",
            save_dir=f"ckpt/generation/{task_name}_{model_name}_{rationale_format}_{removal_threshold if removal_threshold is not None else 'none'}",
            direction='-',
            save_top_k=1,
            device="cuda:0",
        )
        
    return {
        "trainer": trainer,
        "dataloader_train": dataloader_train,
        "dataloader_eval": dataloader_eval,
    }
    
    
def get_irm_params(
    task_name: Text,
    model_name: Text,
    generation_model_name: Text,
    rationale_format: Text,
    removal_threshold: float,
    batch_size: int,
    minimum_frequency: int,
    warmup_epochs: int,
    irm_coefficient: float,
):
    """
    """
    if task_name != "strategyqa":
        raise ValueError("IRM is only implemented for strategyqa.")

    learning_rate = 1e-4

    dataset_train = datasets.load_from_disk(
        "data/strategyqa/train"
    )
    dataset_eval = datasets.load_from_disk(
        "data/strategyqa/validation"
    )

    if rationale_format == "n":
        raise ValueError("IRM is only implemented for rationales (can't do baseline).")

    attribution_model_dir = f"ckpt/fasttext-strategyqa_{rationale_format}_{minimum_frequency}/best_1/"
    
    explainer = IGExplainerFastText(
        num_steps=20,
        max_input_length=256,
        model=FastTextModule.load_from_dir(attribution_model_dir),
        device="cuda:0",
    )
    explainer_vocab = torch.load(f"data/strategyqa/vocab_format={rationale_format}_ng=2_mf={minimum_frequency}_mt=10000_r=1.pt")
    explainer_collate_fn = StrategyQANGramClassificationCollateFn(
        rationale_format=rationale_format,
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
    
    dataset_train, train_features = additional_preprocessor(dataset_train)
    dataset_eval, _ = additional_preprocessor(dataset_eval, features=train_features)
    
    generation_model = HuggingfaceWrapperModule.load_from_dir(
        f"ckpt/generation/strategyqa_t5-base_{rationale_format}_{removal_threshold}/best_1"
    )
    generation_tokenizer = AutoTokenizer.from_pretrained(
        generation_model_name
    )
    
    counterfactual_preprocessor = StrategyQACounterfactualGenerationPreprocessor(
        tokenizer=generation_tokenizer,
        generation_model=generation_model,
        collate_fn_base=StrategyQAInfillingCollateFn(
            rationale_format=rationale_format,
            tokenizer=generation_tokenizer,
            max_input_length=256,
            max_output_length=256,
            removal_threshold=removal_threshold,
            intervention_on_label=False
        ),
        collate_fn_counterfactual=StrategyQAInfillingCollateFn(
            rationale_format=rationale_format,
            tokenizer=generation_tokenizer,
            max_input_length=256,
            max_output_length=256,
            removal_threshold=removal_threshold,
            intervention_on_label=True
        ),
        batch_size=256,
        device="cuda:0",
    )
    
    dataset_train = counterfactual_preprocessor(dataset_train)
    
    if model_name.startswith("t5"):
        model = HuggingfaceWrapperModule(
            model_handle=model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
    
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=StrategyQAIRMCollateFn(
                tokenizer=tokenizer,
                max_input_length=256,
                max_output_length=256,
                rationale_format=rationale_format,
            )
        )
        
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=StrategyQAGenerationCollateFn(
                rationale_format=rationale_format,
                max_input_length=256,
                max_output_length=256,
                tokenizer=tokenizer,
                removal_threshold=removal_threshold,
            )
        )
        
        trainer = StrategyQAIRMTrainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=learning_rate,
            ),
            metrics={
                "loss": AvgLoss(),
                "factual_loss": StatsExtractor(
                    indexing_path="environment::factual.loss",
                    reduction=lambda x: np.array(x).mean().item()
                ),
                "counterfactual_loss": StatsExtractor(
                    indexing_path="environment::counterfactual.loss",
                    reduction=lambda x: np.array(x).mean().item()
                ),
                "factual_grad": StatsExtractor(
                    indexing_path="environment::factual.reg",
                    reduction=lambda x: np.array(x).tolist()
                ),
                "counterfactual_grad": StatsExtractor(
                    indexing_path="environment::counterfactual.reg",
                    reduction=lambda x: np.array(x).tolist()
                ),
                "accuracy": ClassificationAccuracy(),
            },
            eval_metrics={
                "accuracy": ClassificationAccuracy(),
                "loss": AvgLoss(),  # Notice that this is used to evaluate the logits for rev (best achievable)
            },
            main_metric="loss",
            save_dir=f"ckpt/irm/{task_name}_{model_name.replace('/', '::')}_{rationale_format}_{removal_threshold if removal_threshold is not None else 'none'}_{irm_coefficient}",
            device="cuda:0",
            irm_scheduler=LinearScheduler(
                start_val=0.0,
                end_val=irm_coefficient,
                num_steps=len(dataloader_train) * warmup_epochs,
            ),
            warmup_epochs=warmup_epochs,
            direction='-',
            save_top_k=1,
        )
    else:
        learning_rate = 5e-6
        model = HuggingfaceClassifierModule(
            model_handle=model_name,
            num_labels=2,  # 2 is the default for strategyqa
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=StrategyQAIRMEmbeddingClassificationCollateFn(
                tokenizer=tokenizer,
                max_input_length=256,
                rationale_format=rationale_format,
            ),
        )
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=StrategyQAEmbeddingClassificationCollateFn(
                rationale_format=rationale_format,
                tokenizer=tokenizer,
                max_input_length=256,
            )
        )
        
        trainer = StrategyQAClassificationIRMTrainer(
            model=model,
            optimizer=AdamW(
                params=model.parameters(),
                lr=learning_rate,
            ),
            metrics={
                "loss": AvgLoss(),
                "factual_loss": StatsExtractor(
                    indexing_path="environment::factual.loss",
                    reduction=lambda x: np.array(x).mean().item()
                ),
                "counterfactual_loss": StatsExtractor(
                    indexing_path="environment::counterfactual.loss",
                    reduction=lambda x: np.array(x).mean().item()
                ),
                # "factual_grad": StatsExtractor(
                #     indexing_path="environment::factual.reg",
                #     reduction=lambda x: np.array(x).tolist()
                # ),
                # "counterfactual_grad": StatsExtractor(
                #     indexing_path="environment::counterfactual.reg",
                #     reduction=lambda x: np.array(x).tolist()
                # ),
                "accuracy": ClassificationAccuracy(),
            },
            eval_metrics={
                "accuracy": ClassificationAccuracy(),
                "loss": AvgLoss(),  # Notice that this is used to evaluate the logits for rev (best achievable)
            },
            main_metric="loss",
            save_dir=f"ckpt/irm/{task_name}_{model_name.replace('/', '::')}_{rationale_format}_{removal_threshold if removal_threshold is not None else 'none'}_{irm_coefficient}",
            device="cuda:0",
            irm_scheduler=LinearScheduler(
                start_val=0.0,
                end_val=irm_coefficient,
                num_steps=len(dataloader_train) * warmup_epochs,
            ),
            warmup_epochs=warmup_epochs,
            direction='-',
            save_top_k=1,
        )
    
    return {
        "trainer": trainer,
        "dataloader_train": dataloader_train,
        "dataloader_eval": dataloader_eval,
    }