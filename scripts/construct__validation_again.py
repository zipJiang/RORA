"""
"""
import os
import datasets
from typing import List, Dict, Text, Any
import transformers
from src.utils.common import dict_of_list_to_list_of_dict
from src.collate_fns import ECQAGenerationCollateFn


def main():
    """
    """
    eval_collate_fn = ECQAGenerationCollateFn(
        rationale_format="gl",
        max_input_length=512,
        max_output_length=32,
        tokenizer=transformers.AutoTokenizer.from_pretrained("t5-base"),
    )
    
    def _eval_collate_fn_wrapper(examples: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        return eval_collate_fn(dict_of_list_to_list_of_dict(examples))
    
    for format in ['g', 's', 'l']:
        for split in ['validation', 'test']:
            dataset = datasets.load_from_disk(f"data/ecqa/generation_format={format}_ng=1_mf=1_mt=10000_th=0.001/{split}")
            dataset = dataset.map(
                lambda _: {},
                remove_columns=list(filter(lambda x: x.startswith("_"), dataset.column_names)),
                load_from_cache_file=False,
                batched=True,
            ).map(
                _eval_collate_fn_wrapper,
                batched=True,
                load_from_cache_file=False,
                batch_size=128
            )
            
            dataset.save_to_disk(f"data/ecqa/rev_format={format}_ng=1_mf=1_mt=10000_th=0.001/{split}")
    
    
if __name__ == '__main__':
    main()