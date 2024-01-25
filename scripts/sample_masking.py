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
    ECQAClassificationCollateFn,
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
@click.option("--threshold", type=click.FLOAT, default=0.1, help="The threshold to use.")
@click.option("--dataset-dir", type=click.Path(exists=True), help="The dataset directory.")
@click.option("--rationale-format", type=click.Choice(['g', 'l', 's', 'gls', 'gs', 'ls', 'gl', 'n']), help="The rationale format to use.")
@click.option("--num-samples", type=click.INT, default=5, help="The number of samples to use.")
@click.option("--seed", type=click.INT, default=42, help="The random seed to use.")
@click.option("--minimum-frequency", type=click.INT, default=10, help="The minimum frequency of a token to be included in the vocabulary.")
@click.option("--write-to", type=click.Path(exists=False), help="The path to write the output to.")
@click.option("--mask-by-delete", is_flag=True, default=False, help="Whether to mask by delete.")
def main(
    threshold,
    dataset_dir,
    rationale_format,
    num_samples,
    seed,
    minimum_frequency,
    write_to,
    mask_by_delete
):
    """
    """
    
    train_dataset = datasets.load_from_disk(
        os.path.join(dataset_dir, 'train')
    )
    validation_dataset = datasets.load_from_disk(
        os.path.join(dataset_dir, 'validation')
    )
    
    dbnm = os.path.basename(dataset_dir[:-1] if dataset_dir.endswith("/") else os.path.basename(dataset_dir))
    print(dbnm)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    if dbnm.startswith("strategyqa"):
        explainer = IGExplainerFastText(
            num_steps=20,
            max_input_length=256,
            model=FastTextModule.load_from_dir(f"ckpt/fasttext-strategyqa_{rationale_format}/best_1/"),
            device="cuda:0",
        )
    
        dataset, features = StrategyQAGlobalExplanationPreprocessor(
            explainer=explainer,
            collate_fn=StrategyQANGramClassificationCollateFn(
                vocab=torch.load(f"data/processed_datasets/strategyqa/vocab_format={rationale_format}_ng=2_mf={minimum_frequency}_mt=10000_r=1.pt"),
                rationale_format=rationale_format,
                max_input_length=256,
                num_ngrams=2,
                rationale_only=True,
            ),
            batch_size=1024
        )(validation_dataset, feature_calculation_dataset=train_dataset)
        
        collate_fn = StrategyQAInfillingCollateFn(
            tokenizer=tokenizer,
            removal_threshold=threshold,
            rationale_format=rationale_format,
            max_input_length=256,
            max_output_length=32,
        )
        
        subset = random.Random(seed).choices(
            [item for item in dataset],
            k=num_samples
        )
        

        with open(write_to, 'w', encoding='utf-8') as f:
            outputs = [{
                "rationales": {
                    "original": collate_fn.rationale_templating(item),
                    "-0.1": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.1, mask_by_delete=mask_by_delete),
                    "-0.05": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.05, mask_by_delete=mask_by_delete),
                    "-0.01": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.01, mask_by_delete=mask_by_delete),
                },
                "targets": {
                    "original": collate_fn.non_removal_templating(item),
                    "-0.1": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.1, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                    "-0.05": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.05, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                    "-0.01": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.01, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                },
                "question": item["question"],
                "label": item['answer']
            } for item in subset]
                
            json.dump(outputs, f, indent=2)
            
    elif dbnm.startswith("ecqa"):
        explainer = IGExplainerLSTM(
            num_steps=100,
            max_input_length=256,
            max_output_length=32,
            # model=BiEncodingFastTextModule.load_from_dir(f"ckpt/ecqa_fasttext_{rationale_format}/best_1/"),
            model=BiEncodingLSTMModule.load_from_dir(f"ckpt/ecqa_lstm_{rationale_format}/best_1/"),
            device="cuda:0",
        )
        
        dataset, features = ECQAGlobalExplanationPreprocessor(
            explainer=explainer,
            collate_fn=ECQALstmClassificationCollateFn(
                rationale_format=rationale_format,
                max_input_length=256,
                nlp_model="en_core_web_sm",
                vocab=get_vocab(f"data/ecqa_vocabs/vocab_format={rationale_format}_ng=1_mf={minimum_frequency}_mt=10000.pt"),
                rationale_only=True,
            ),
            batch_size=32
        )(validation_dataset, feature_calculation_dataset=train_dataset)
        
        collate_fn = ECQAInfillingCollateFn(
            tokenizer=tokenizer,
            removal_threshold=threshold,
            rationale_format=rationale_format,
            max_input_length=256,
            max_output_length=32,
        )
        
        subset = random.Random(seed).choices(
            [item for item in dataset],
            k=num_samples
        )
        

        with open(write_to, 'w', encoding='utf-8') as f:
            outputs = [{
                "rationales": {
                    "original": collate_fn.rationale_templating(item),
                    "-0.1": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.1, mask_by_delete=mask_by_delete),
                    "-0.05": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.05, mask_by_delete=mask_by_delete),
                    "-0.01": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.01, mask_by_delete=mask_by_delete),
                    "-0.001": collate_fn.remove_spurious(collate_fn.rationale_templating(item), attributions=item['attributions'], removal_threshold=0.001, mask_by_delete=mask_by_delete),
                },
                # "targets": {
                #     "original": collate_fn.non_removal_templating(item),
                #     "-0.1": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.1, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                #     "-0.05": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.05, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                #     "-0.01": collate_fn.retain_spurious(collate_fn.non_removal_templating(item), attributions=item['attributions'], removal_threshold=0.01, offsets=len(collate_fn.non_removal_no_rationale_templating(item))),
                # },
                "question": item["q_text"],
                "label": item['q_ans']
            } for item in subset]
                
            json.dump(outputs, f, indent=2)
        
        
if __name__ == '__main__':
    main()