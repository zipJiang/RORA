"""Training a generator model for T-5 and GPT-2 rationale generation.
"""
import click
from src.utils.configs import get_rationalize_params
from src.trainers.trainer import Trainer
from torch.utils.data import DataLoader


@click.command()
@click.option("--task-name", type=click.STRING, default="strategyqa", help="Task to train on.")
@click.option("--model-name", type=click.STRING, default="t5-base", help="Model to train.")
@click.option("--epochs", type=click.INT, default=20)
@click.option("--batch-size", type=click.INT, default=8)
@click.option("--learning-rate", type=click.FLOAT, default=1e-4)

def main(
    task_name,
    model_name,
    epochs,
    batch_size,
    learning_rate,
):
    """Training T-5 and GPT-2 for rationale generation.
    """
    
    params = get_rationalize_params(
        task_name=task_name,
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_wandb=False,
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