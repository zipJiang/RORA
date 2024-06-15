"""Take a processed ECQA dataset and prepare it as a huggingface dataset for simulation experiment
"""
import json
import click
import os
import datasets

from src.preprocessors.ecqa_preprocessor import ECQASimulationPreprocessor
from src.preprocessors.strategyqa_preprocessor import StrategyQAVacuousRationalePreprocessor

@click.command()
@click.option("--data-handle", type=click.STRING, default="yangdong/ecqa")
@click.option("--split", type=click.Choice(["train", "validation", "test"]), help="Which split to use.", default='validation')
@click.option("--write-to", type=click.Path(exists=False, file_okay=True), help="Path to write the output to (will be subdir-ed by data_name (not none) and split).")
def main(
    data_handle,
    split,
    write_to
):
    """Simulate ECQA task to StrategyQA task with fixed labels (True or False)
    """

    dataset = datasets.load_dataset(data_handle, split=split)
    
    simulation_preprocessor = ECQASimulationPreprocessor()
    dataset = simulation_preprocessor(dataset)

    params = {
        "batch_size": 128,
        "temperature": 0.0,
        "num_return_sequences": 1,
        "num_beam_groups": 1,
        "num_beams": 1,
    }
    vacuous_preprocessor = StrategyQAVacuousRationalePreprocessor(**params)
    dataset = vacuous_preprocessor(dataset)
    
    dataset = dataset.remove_columns("question")
    dataset = dataset.rename_column("full_question", "question")

    os.makedirs(write_to, exist_ok=True)
    dataset.save_to_disk(os.path.join(write_to, split))
    
if __name__ == '__main__':
    main()