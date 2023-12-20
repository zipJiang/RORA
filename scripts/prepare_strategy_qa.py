"""Take a strategy QA datatset and prepare it to work with huggingface
dataset.
"""
import json
import click
import os
import random

@click.command()
@click.option("--input-path", type=click.Path(exists=True, dir_okay=False), help="Path to the input dataset.")
@click.option("--output-path", type=click.Path(exists=False, dir_okay=True), help="Path to the output directory.")
@click.option("--split-ratio", type=float, default=0.2, help="Ratio of the dataset to use for validation (and test).")
@click.option("--seed", type=int, default=42, help="Random seed to use for splitting the dataset.")
def main(
    input_path,
    output_path,
    split_ratio,
    seed
):
    """
    """
    
    # validate split_ratio is valid
    assert 2 * split_ratio < 1, "split_ratio is too large."
    
    with open(input_path, "r", encoding='utf-8') as file_:
        data = json.load(file_)

    random_obj = random.Random(seed)
    
    random_obj.shuffle(data)
    
    # calculate the number of examples to use for validation and test
    num_examples = int(len(data) * split_ratio)
    
    validation_data = data[:num_examples]
    test_data = data[num_examples:2*num_examples]
    train_data = data[2*num_examples:]
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, "train.jsonl"), "w", encoding='utf-8') as file_:
        for example in train_data:
            # remove the evidenc column
            example = {
                key: value for key, value in example.items() if key != "evidence"
            }
            file_.write(json.dumps(example, ensure_ascii=False) + "\n")

    # did the same for validation and test
    with open(os.path.join(output_path, "validation.jsonl"), "w", encoding='utf-8') as file_:
        for example in validation_data:
            # remove the evidenc column
            example = {
                key: value for key, value in example.items() if key != "evidence"
            }
            file_.write(json.dumps(example, ensure_ascii=False) + "\n")
            
    with open(os.path.join(output_path, "test.jsonl"), "w", encoding='utf-8') as file_:
        for example in test_data:
            # remove the evidenc column
            example = {
                key: value for key, value in example.items() if key != "evidence"
            }
            file_.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            
if __name__ == '__main__':
    main()