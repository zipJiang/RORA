"""Generate ratioanles for a given dataset.
"""
import click
import os
from src.utils.configs import get_inference_params

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-name",  type=click.Choice(["gpt-4", "gpt-3.5-turbo", "t5-large", "gpt2"]), help="The model to use.")
@click.option("--num-sample", type=click.INT, default=-1, help="The number of samples to generate.")
@click.option("--demonstration-num", type=click.INT, default=2, help="The number of demonstrations to help generate.")
@click.option("--output-dir", type=click.Path(exists=True), help="The output directory to save rationales.")
def main(
    dataset_dir,
    model_name,
    num_sample,
    demonstration_num,
    output_dir
):
    """Run rationale generation for a given dataset.
    """

    params = get_inference_params(dataset_dir=dataset_dir,
                                  model_name=model_name,
                                  num_sample=num_sample,
                                  demonstration_num=demonstration_num)
    
    generator = params['generator']
    dataloader = params['dataloader']

    output_path = os.path.join(output_dir, f"{model_name}_demo={demonstration_num}.jsonl")
    generator.inference(dataloader, output_dir=output_path)

if __name__ == '__main__':
    main()