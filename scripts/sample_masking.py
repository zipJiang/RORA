"""Run the preprocessor for StrategyQA (Potentially other dataset),
and manually examine whether there are noticeable artifacts that can be
utilized.
"""
import os
import json
import click
from transformers import AutoTokenizer
import random
import datasets
import torch
from torch.utils.data import DataLoader
from src.explainers.ig_explainer import (
    IGExplainerFastText,
    IGExplainerLSTM
)
from src.preprocessors.strategyqa_preprocessor import StrategyQAGlobalExplanationPreprocessor
from src.preprocessors.ecqa_preprocessor import ECQAGlobalExplanationPreprocessor
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQANGramClassificationCollateFn,
    StrategyQAGenerationCollateFn,
    StrategyQAInfillingCollateFn
)
from src.collate_fns.ecqa_collate_fn import (
    ECQALstmClassificationCollateFn,
    ECQAGenerationCollateFn,
    ECQAInfillingCollateFn
)
from src.models.fasttext import (
    FastTextModule,
    BiEncodingFastTextModule
)
from src.models.lstm import BiEncodingLSTMModule
from src.utils.common import (
    get_embedding,
    get_vocab
)


@click.command()
@click.option('--dataset-dir', type=click.Path(exists=True), help='Path to the dataset directory.')
@click.option('--rationale-format', type=click.Choice(['g', 'gl', 's']), default='g', help='The format of the rationale.')
@click.option('--removal-threshold', type=float, default=0.05, help='The threshold for removing words.')
def main(
    dataset_dir,
    rationale_format,
    removal_threshold
):
    """
    """
    
    collate_fn = StrategyQAGenerationCollateFn(
        rationale_format=rationale_format,
        tokenizer=AutoTokenizer.from_pretrained('t5-base'),
        max_input_length=512,
        removal_threshold=removal_threshold,
        mask_by_delete=False
    )
    
    dataset = datasets.load_from_disk(dataset_dir)
    
    for row in dataset:
        print('---' * 20)
        print(row['question'])
        print(collate_fn.rationale_templating(row))
        print(collate_fn.remove_spurious(collate_fn.rationale_templating(row), attributions=row['attributions']))
        print('---' * 20)
        
        
if __name__ == '__main__':
    main()