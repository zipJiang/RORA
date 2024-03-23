"""This script is used to generate human batches from the
StrategyQA dataset (test). We'll include all the model types
to be used in the human annotation batch.
"""
import click
import datasets
import json
import csv
import sys
from random import shuffle


__MODEL_TYPES__ = [
    'gpt4',
    'gpt3',
    'llama2',
    'flan'
]


__NUMBER_OF_ITEMS_IN_PILOT__ = 29


@click.command()
@click.option('--data-path', help='Path to the StrategyQA dataset')
@click.option('--pilot-output-path', help='Path to the pilot output directory')
@click.option('--output-path', help='Path to the output directory')
def main(
    data_path,
    pilot_output_path,
    output_path
):
    """
    """
    
    dataset = datasets.load_from_disk(data_path)
    
    # first extract dataset
    items = [dict(item) for item in dataset]

    pilot_batch_items = items[:__NUMBER_OF_ITEMS_IN_PILOT__]
    remaining_items = items[__NUMBER_OF_ITEMS_IN_PILOT__:]
    
    _template = lambda x: "True" if x else "False"
    
    with open(pilot_output_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'payload'])
        writer.writeheader()
        
        to_write = []

        for idx, item in enumerate(pilot_batch_items):
            for midx, model_type in enumerate(__MODEL_TYPES__):
                to_write.append({
                    'id': f"pilot_{idx}_{midx}",
                    'payload': json.dumps({
                        "question": item["question"],
                        "answer": _template(item["answer"]),
                        "modelType": model_type,
                        "explanation": item[f"{model_type}_rationale"],
                        "qid": item["qid"]
                    })
                })
                
        shuffle(to_write)
        writer.writerows(to_write)
                
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'payload'])
        writer.writeheader()

        to_write = []
        for idx, item in enumerate(remaining_items):
            for midx, model_type in enumerate(__MODEL_TYPES__):
                to_write.append({
                    'id': f"item_{idx}_{midx}",
                    'payload': json.dumps({
                        "question": item["question"],
                        "answer": _template(item["answer"]),
                        "modelType": model_type,
                        "explanation": item[f"{model_type}_rationale"],
                        "qid": item["qid"]
                    })
                })
                
        shuffle(to_write)
        writer.writerows(to_write)
                
                
if __name__ == '__main__':
    main()