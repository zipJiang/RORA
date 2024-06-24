"""Script used to attach the generated rationales to the model's predictions.
"""
import click
from typing import Text, List, Dict, Any
import os
import datasets
import json
from glob import glob


__INDEXING_FIELDS__ = {
    "strategyqa": {
        "rationale": "question",
        "dataset": "question",
    },
    "ecqa": {
        "rationale": "id",
        "dataset": "q_no",
    }
}


__RATIONALE_PARSER__ = {
    "strategyqa": lambda item: ' '.join(item["facts"]),
    "ecqa": lambda item: item["abstractive_explanation"]
}


__MODELS_TO_TAKE__ = {
    "strategyqa": {"flan", "gpt3", "gpt4", "llama2"},
    "ecqa": {"cose"}
}


@click.command()
@click.option('--dataset-name', type=click.Choice(["strategyqa", "ecqa"]), help='Name of the dataset to use.')
@click.option('--dataset-dir', type=click.Path(exists=True), help='Path to the dataset directory.')
@click.option('--output-dir', type=click.Path(exists=False), help='Path to the output directory.')
@click.option('--rationale-dir', type=click.Path(exists=True), help='Path to the rationale directory.')
def main(
    dataset_name,
    dataset_dir,
    output_dir,
    rationale_dir,
):
    """Take the rationale from the rationale directory and attach it to the model's predictions.
    """
    
    def _parse_model_string(model_str: Text) -> Text:
        """Take a model string and generate the model name.
        """
        
        name_map = {
            "flan-t5-large": "flan",
            "gpt-3.5-turbo": "gpt3",
            "gpt-4": "gpt4",
            "gpt2": "gpt2",
            "Llama-2-7b-hf": "llama2",
            "t5-large": "t5",
            "cose": "cose"
        }

        return name_map[model_str.split("/")[-1].split("_")[0]]
    
    # first load all the rationales
    
    all_model_rationales = {}

    for model_string in glob(os.path.join(rationale_dir, "*")):
        model_name = _parse_model_string(model_string)
        if model_name not in __MODELS_TO_TAKE__[dataset_name]:
            continue
        
        all_model_rationales[model_name] = {}
        
        for split_name in ["train", "validation", "test"]:
            with open(os.path.join(model_string, f"{split_name}.jsonl"), 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f]
                for item in items:
                    # assert item['question'] not in all_model_rationales[model_name], f"Duplicate question found: {item['question']}"
                    # We already know that in strategyqa the question is NOT UNIQUE, but in training set
                    # this only happens once, so we can ignore it without too much impact.
                    all_model_rationales[model_name][item[__INDEXING_FIELDS__[dataset_name]["rationale"]]] = __RATIONALE_PARSER__[dataset_name](item)
                    
    def _attach_rationale(example):
        query = example[__INDEXING_FIELDS__[dataset_name]["dataset"]]
        
        return_dict = {}
        for model_name, model_rationales in all_model_rationales.items():
            try:
                return_dict[f"{model_name}_rationale"] = model_rationales[query]
            except KeyError:
                print('-' * 20)
                print(example)
                print(f"Query: {query}")
                print()
                print('-' * 20)
                return_dict[f"{model_name}_rationale"] = example['vacuous_rationale']
            
        return return_dict
                
    # load the dataset and try to start mapping
    for split_name in ["train", "validation", "test"]:
        dataset = datasets.load_from_disk(os.path.join(dataset_dir, split_name))
        dataset = dataset.map(
            _attach_rationale,
            batched=False,
        )
        print(dataset[0])
        dataset.save_to_disk(os.path.join(output_dir, split_name))

            
if __name__ == '__main__':
    main()