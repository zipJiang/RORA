"""Generate vocabulary files for a given dataset.
"""
import re
import click
import torchtext
import spacy
import torch
import os
import datasets
from typing import List, Dict, Any, Iterator, Text
from torch.utils.data import DataLoader
from src.utils.templating import __TEMPLATES__
from src.collate_fns.strategyqa_collate_fn import (
    generate_no_more_than_ngrams,
    __LABEL_TO_LEAKY_RATIONALE__
)
from src.collate_fns.ecqa_collate_fn import (
    retrieve_vacuous
)


@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--rationale-format", type=click.Choice(['g', 'l', 's', 'gls', 'gs', 'ls', 'gl', 'n']), help="The rationale format to use.")
@click.option("--num-ngrams", type=click.INT, default=2, help="The number of ngrams to generate.")
@click.option("--min-freq", type=click.INT, default=1, help="The minimum frequency of a token to be included in the vocabulary.")
@click.option("--max-tokens", type=click.INT, default=10000, help="The maximum number of tokens to include in the vocabulary.")
@click.option("--rationale-only", is_flag=True, show_default=True, default=False, help="Whether to only use the rationale for vocab generation.")
@click.option("--output-path", type=click.Path(), help="The path to save the vocab file.")
def main(
    dataset_dir,
    rationale_format,
    num_ngrams,
    min_freq,
    max_tokens,
    rationale_only,
    output_path
):
    """Run vocab generation for a given dataset.
    """
    
    # TODO: add support for other dataset as well
    
    dataset = datasets.load_from_disk(
        os.path.join(dataset_dir, 'train')
    )
    
    dataset_name = os.path.basename(dataset_dir if dataset_dir[-1] != '/' else dataset_dir[:-1])
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def strategyqa_vocab_creation_iterator() -> Iterator[List[Text]]:
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
            
    def ecqa_vocab_creation_iterator() -> Iterator[List[Text]]:
        """Given a dataset (ecqa), create an iterator
        that yields a list of text.
        """
        
        for item in dataset:
            sentence = __TEMPLATES__[rationale_format].format(
                gold_rationale=item['taskB'],
                base_rationale=retrieve_vacuous(item),
                leaky_rationale=f"The answer is: {item['q_ans']}."
            ) + ' ' + ' '.join([item[f'q_op{i}'] for i in range(1, 6)])
            
            if not rationale_only:
                sentence = f"question: {item['q_text']} rationale: {sentence}"
                
            sentence = re.sub(r'\s+', ' ', sentence).strip()

            yield generate_no_more_than_ngrams(
                [token.text for token in nlp(sentence)],
                num_ngrams
            ) if num_ngrams > 1 else [token.text for token in nlp(sentence)]
            
    vocab_creation_iterators = {
        "strategyqa": strategyqa_vocab_creation_iterator,
        "ecqa": ecqa_vocab_creation_iterator
    }

    vocab = torchtext.vocab.build_vocab_from_iterator(
        vocab_creation_iterators[dataset_name](),
        min_freq=min_freq,
        max_tokens=max_tokens,
        specials=['<pad>', '<unk>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    torch.save(
        # vocab, os.path.join(dataset_dir, f'vocab_format={rationale_format}_ng={num_ngrams}_mf={min_freq}_mt={max_tokens}_r={1 if rationale_only else 0}.pt')
        vocab, output_path
    )
    
    
if __name__ == '__main__':
    main()
