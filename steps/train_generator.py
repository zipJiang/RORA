"""Training a generator model for T-5 question modification.
"""
import click
from src.utils.configs import get_generation_params
from src.trainers.trainer import Trainer
from torch.utils.data import DataLoader


@click.command()
@click.option("--task-name", type=click.STRING, default="strategyqa", help="Task to train on.")
@click.option("--model-name", type=click.STRING, default="t5-base", help="Model to train.")
@click.option("--data-name", type=click.STRING, default=None, help="Data to train on.")
@click.option("--rationale-format", type=click.Choice(["g", "l", "s", "gs", "ls", "gl", "gls", "n"]), default="g")
@click.option("--epochs", type=click.INT, default=20)
@click.option("--removal-threshold", type=click.FLOAT, default=None)
@click.option("--batch-size", type=click.INT, default=8)
@click.option("--minimum-frequency", type=click.INT, default=1)
def main(
    task_name,
    model_name,
    data_name,
    rationale_format,
    epochs,
    removal_threshold,
    batch_size,
    minimum_frequency,
):
    """Training T-5 for mask refilling.
    """
    
    params = get_generation_params(
        task_name=task_name,
        model_name=model_name,
        data_name=data_name,
        rationale_format=rationale_format,
        removal_threshold=removal_threshold,
        batch_size=batch_size,
        minimum_frequency=minimum_frequency,
    )
    
    trainer: Trainer = params['trainer']
    dataloader_train: DataLoader = params['dataloader_train']
    dataloader_eval: DataLoader = params['dataloader_eval']
    
    trainer.train(
        dataloader=dataloader_train,
        eval_dataloader=dataloader_eval,
        epochs=epochs,
        patience=3,
    )
    
    
if __name__ == '__main__':
    main()