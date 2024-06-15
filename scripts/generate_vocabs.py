"""Generate vocabulary files for a given dataset.
"""
import click
import torchtext
import spacy
import torch
import os
import datasets
from typing import List, Dict, Any, Iterator, Text
from torch.utils.data import DataLoader
from src.collate_fns.strategyqa_collate_fn import (
    StrategyQANGramClassificationCollateFn,
    generate_no_more_than_ngrams,
    __TEMPLATES__,
    __LABEL_TO_LEAKY_RATIONALE__
)
from src.collate_fns.ecqa_collate_fn import __QUESTION_TEMPLATES__

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--task-format", type=click.Choice(['strategyqa', 'ecqa']), default="strategyqa", help="The task format to use.")
@click.option("--rationale-format", type=click.Choice(["gl", "gs", "g", "n", "l", "s", "gls", "ls", "ss", "gsl"]), help="The rationale format to use.")
@click.option("--num-ngrams", type=click.INT, default=2, help="The number of ngrams to generate.")
@click.option("--min-freq", type=click.INT, default=1, help="The minimum frequency of a token to be included in the vocabulary.")
@click.option("--max-tokens", type=click.INT, default=10000, help="The maximum number of tokens to include in the vocabulary.")
@click.option("--rationale-only", is_flag=True, show_default=True, default=False, help="Whether to only use the rationale for vocab generation.")
def main(
    dataset_dir,
    task_format,
    rationale_format,
    num_ngrams,
    min_freq,
    max_tokens,
    rationale_only
):
    """Run vocab generation for a given dataset.
    """
    
    # TODO: add support for other dataset as well
    
    dataset = datasets.load_from_disk(
        os.path.join(dataset_dir, 'train')
    )
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def vocab_creation_iterator_strategyqa() -> Iterator[List[Text]]:
        """Given a dataset (strategyqa), create an iterator
        that yields a list of text.
        """
        
        for item in dataset:
            sentence = __TEMPLATES__[rationale_format].format(
                # question=item['question'],
                gold_rationale=' '.join(item['facts']),
                base_rationale=item['vacuous_rationale'],
                leaky_rationale=__LABEL_TO_LEAKY_RATIONALE__[item['answer']]
            )
            
            if not rationale_only:
                sentence = f"question: {item['question']} rationale: {sentence}"
            
            yield generate_no_more_than_ngrams(
                [token.text for token in nlp(sentence)],
                num_ngrams
            )
    
    def vocab_creation_iterator_ecqa() -> Iterator[List[Text]]:
        """Given a dataset (ecqa), create an iterator
        that yields a list of text.
        """
        for item in dataset:
            sentence = __TEMPLATES__[rationale_format].format(
                gold_rationale="{pos} {neg}".format(
                            pos = item['taskA_pos'],
                            neg = item['taskA_neg']
                        ),
                base_rationale=item[f'vacuous_rationale_{item["label"]}'],
                leaky_rationale=f"The answer is {item['q_ans']}"
            )   
            
            question = __QUESTION_TEMPLATES__.format(
                question=item['q_text'],
                op1=item['q_op1'],
                op2=item['q_op2'],
                op3=item['q_op3'],
                op4=item['q_op4'],
                op5=item['q_op5'],
            )

            if not rationale_only:
                sentence = f"question: {question} rationale: {sentence}"
            
            yield generate_no_more_than_ngrams(
                [token.text for token in nlp(sentence)],
                num_ngrams
            )
            
    vocab = torchtext.vocab.build_vocab_from_iterator(
        vocab_creation_iterator_strategyqa() if task_format == 'strategyqa' else vocab_creation_iterator_ecqa(),
        min_freq=min_freq,
        max_tokens=max_tokens,
        specials=['<pad>', '<unk>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    torch.save(
        vocab, os.path.join(dataset_dir, f'vocab_format={rationale_format}_ng={num_ngrams}_mf={min_freq}_mt={max_tokens}_r={1 if rationale_only else 0}.pt')
    )
    
    
if __name__ == '__main__':
    main()
