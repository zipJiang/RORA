"""This script takes a step to convert data 
into a rationale dataset that have the target rationale
processed into a single field.
"""
import click
import os
import datasets
from src.preprocessors.ecqa_preprocessor import ECQAVacuousRationalePreprocessor
from src.preprocessors.strategyqa_preprocessor import StrategyQAVacuousRationalePreprocessor


__REGISTRY__ = {
    "yangdong/ecqa": [
        {
            "cls": ECQAVacuousRationalePreprocessor,
            "params": {
                "batch_size": 128,
                "temperature": 0.0,
                "num_return_sequences": 1,
                "num_beam_groups": 1,
                "num_beams": 1,
            }
        }
    ],
    "Zhengping/strategyqa_custom_split": [
        {
            "cls": StrategyQAVacuousRationalePreprocessor,
            "params": {
                "batch_size": 128,
                "temperature": 0.0,
                "num_return_sequences": 1,
                "num_beam_groups": 1,
                "num_beams": 1,
            }
        }
    ], 
    # for model generated rationale datasets, the generated vacuous rationales
    # will not be used in the following experiments, as only `g` rationale will be used
    "Yining/generated_rationales/strategyqa": [
        {
            "cls": StrategyQAVacuousRationalePreprocessor,
            "params": {
                "batch_size": 128,
                "temperature": 0.0,
                "num_return_sequences": 1,
                "num_beam_groups": 1,
                "num_beams": 1,
            }
        }
    ],
    "Yining/generated_rationales/ecqa": [
        {
            "cls": StrategyQAVacuousRationalePreprocessor,
            "params": {
                "batch_size": 128,
                "temperature": 0.0,
                "num_return_sequences": 1,
                "num_beam_groups": 1,
                "num_beams": 1,
            }
        }
    ],    
}

@click.command()
@click.option("--data-handle", type=click.STRING, default="esnli. (will be subdir-ed by data_name if it is not None)")
@click.option("--data-name", type=click.STRING, default=None, help="Customized dataset to use. e.g. 'gpt-4_demo=2_raw=True'")
@click.option("--split", type=click.Choice(["train", "validation", "test"]), help="Which split to use.", default='validation')
@click.option("--write-to", type=click.Path(exists=False, file_okay=True), help="Path to write the output to (will be subdir-ed by data_name (not none) and split).")
def main(
    data_handle,
    data_name,
    split,
    write_to
):
    """
    """

    data_path = os.path.join(data_handle, data_name) if data_name is not None else data_handle
    write_to = os.path.join(write_to, data_name) if data_name is not None else write_to

    dataset = datasets.load_dataset(data_path, split=split)
    
    for preprocessor_config in __REGISTRY__[data_handle]:
        preprocessor = preprocessor_config["cls"](**preprocessor_config["params"])
        dataset = preprocessor(dataset)
    
    os.makedirs(write_to, exist_ok=True)
    
    dataset.save_to_disk(os.path.join(write_to, split))
    
if __name__ == '__main__':
    main()