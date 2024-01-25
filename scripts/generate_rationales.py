"""Generate ratioanles for a given dataset.
"""
import click
import os
from src.utils.configs import get_inference_params

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), help="Path to the dataset directory.")
@click.option("--model-name",  type=click.Choice(["gpt-4", "gpt-3.5-turbo", "t5-large", "gpt2", "google/flan-t5-large", "meta-llama/Llama-2-7b-hf"]), help="The model to use.")
@click.option("--num-sample", type=click.INT, default=-1, help="The number of samples to generate.")
@click.option("--demonstration-num", type=click.INT, default=2, help="The number of demonstrations to help generate.")
@click.option("--output-dir", type=click.Path(exists=True), help="The output directory to save rationales.")
@click.option("--batch-size", type=click.INT, default=64, help="The batch size to use.")
@click.option("--use-raw-model", is_flag=True, default=False, help="Use the raw model without finetuning.")
@click.option("--split", type=click.Choice(["train", "validation", "test"]), default="test", help="the split to generate rationales for.")
def main(
    dataset_dir,
    model_name,
    num_sample,
    demonstration_num,
    output_dir,
    batch_size,
    use_raw_model,
    split
):
    """Run rationale generation for a given dataset.
    """

    params = get_inference_params(dataset_dir=dataset_dir,
                                  model_name=model_name,
                                  num_sample=num_sample,
                                  demonstration_num=demonstration_num,
                                  batch_size=batch_size,
                                  use_raw_model=use_raw_model,
                                  split=split)
    
    generator = params['generator']
    dataloader = params['dataloader']

    model_name = model_name.split("/")[1] if "/" in model_name else model_name
    output_path = os.path.join(output_dir, "{model_name}_demo={demonstration_num}_raw={use_raw}".format(
        model_name=model_name,
        demonstration_num=demonstration_num,
        use_raw = use_raw_model
        ), f"{split}.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generator.inference(dataloader, output_dir=output_path)

if __name__ == '__main__':
    main()