"""
"""
import datasets
import click
import os
import shutil


@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
def main(
    dataset_dir
):
    
    for name in ['validation', 'test']:
        dataset = datasets.load_from_disk(os.path.join(dataset_dir, name))
        dataset = dataset.map(
            lambda example: {'_input_ids': example['input_ids'], '_attention_mask': example['attention_mask'], '_labels': example['labels']},
            remove_columns=['input_ids', 'attention_mask', 'labels'],
            batched=True,
        )
        print(dataset[0])
        shutil.rmtree(os.path.join(dataset_dir, name), ignore_errors=True)
        dataset.save_to_disk(os.path.join(dataset_dir, "_" + name))
        
if __name__ == '__main__':
    main()