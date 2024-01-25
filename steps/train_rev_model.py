"""Train a REV model with the given input types.
"""
import click
from torch.utils.data import DataLoader
from src.utils.configs import get_params
from src.trainers.trainer import Trainer


@click.command()
@click.option("--task-name", type=click.STRING, default="t5-strategyqa")
@click.option("--rationale-format", type=click.Choice(["gl", "gs", "g", "n", "l", "s", "gls", "ls", "ss"]), default="g")
@click.option("--epochs", type=click.INT, default=20)
@click.option("--removal-threshold", type=click.FLOAT, default=None)
@click.option("--delete", is_flag=True, default=False)
@click.option("--rationale-only", is_flag=True, default=False)
def main(
    task_name,
    rationale_format,
    epochs,
    removal_threshold,
    delete,
    rationale_only,
):
    """
    """
    # TODO: support other tasks as well.
    params = get_params(
        task=task_name,
        rationale_format=rationale_format,
        removal_threshold=removal_threshold,
        mask_by_delete=delete,
        rationale_only=rationale_only
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