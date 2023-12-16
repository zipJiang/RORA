"""This is similar to train_rev_model.py, but we use the IRM loss instead of the NLL.
"""
import click
from torch.utils.data import DataLoader
from src.utils.configs import get_irm_params
from src.trainers.trainer import Trainer


@click.command()
@click.option("--task-name", type=click.STRING, default="t5-strategyqa")
@click.option("--model-name", type=click.STRING, default="t5-base")
@click.option("--rationale-format", type=click.Choice(["g", "l", "s", "gs", "ls", "gl", "gls", "n"]), default="g")
@click.option("--epochs", type=click.INT, default=20)
@click.option("--removal-threshold", type=click.FLOAT, default=None)
@click.option("--irm-coefficient", type=click.FLOAT, default=1e2)
@click.option("--batch-size", type=click.INT, default=24)
def main(
    task_name,
    model_name,
    rationale_format,
    epochs,
    removal_threshold,
    irm_coefficient,
    batch_size,
):
    """
    """
    # TODO: support other tasks as well.
    params = get_irm_params(
        task_name=task_name,
        model_name=model_name,
        generation_model_name="t5-base",
        rationale_format=rationale_format,
        removal_threshold=removal_threshold,
        batch_size=batch_size,
        minimum_frequency=1,
        warmup_epochs=2,
        irm_coefficient=irm_coefficient,
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