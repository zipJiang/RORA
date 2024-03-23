"""We generate rationales for cose to be included in
the cose dataset.
"""
from datasets import load_dataset
import click
import json
import os


@click.command()
@click.option("--write-to", type=click.Path(exists=False), help="Path to write the rationales to (dataset-dir).")
def main(
    write_to
):
    """
    """
    
    dataset = load_dataset("cos_e", "v1.11")
    
    os.makedirs(write_to, exist_ok=True)
    for split in ["train", "validation"]:
        with open(os.path.join(write_to, f"{split}.jsonl"), "w", encoding='utf-8') as file_:
            for row in dataset[split]:
                file_.write(json.dumps(row) + "\n")
                
    # make up for the pseudo test.jsonl
    with open(os.path.join(write_to, "test.jsonl"), "w", encoding='utf-8') as file_:
        pass
                

if __name__ == "__main__":
    main()