"""Take a dataset and dump the content.
"""
import json
import os
import datasets
import click


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), help="Path to the input directory.")
@click.option("--output-dir", type=click.Path(), help="Path to the output directory.")
def main(
    input_dir: str,
    output_dir: str
):
    """
    """

    os.makedirs(output_dir, exist_ok=False)

    for split in ["train", "validation", "test"]:
        data = datasets.load_from_disk(
            os.path.join(input_dir, split)
        )

        with open(os.path.join(output_dir, f"{split}.jsonl"), "w", encoding='utf-8') as f:
            for item in data:
                item = {k: v for k, v in item.items() if not k.startswith("_")}
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()