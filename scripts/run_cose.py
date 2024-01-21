import click
import os
import datasets
from src.preprocessors.ecqa_preprocessor import COSESimulationPreprocessor

@click.command()
@click.option("--exp-name", type=click.Choice(["preprocess", "simulation_preprocess", "prepare_vocab", "rev_train", "rev_eval"]), default="simulation_preprocess")

def main(
    exp_name
):
    """
    
    """
    if exp_name == "preprocess":
        train_data = datasets.load_dataset("cos_e", "v1.11", split="train")
        validation_data = datasets.load_dataset("cos_e", "v1.11", split="validation")
        os.makedirs("data/processed_datasets/cose/train", exist_ok=True)
        os.makedirs("data/processed_datasets/cose/validation", exist_ok=True)
        train_data.save_to_disk("data/processed_datasets/cose/train")
        validation_data.save_to_disk("data/processed_datasets/cose/validation")

    elif exp_name == "simulation_preprocess":
        train_data = datasets.load_dataset("cos_e", "v1.11", split="train")
        validation_data = datasets.load_dataset("cos_e", "v1.11", split="validation")
        processor = COSESimulationPreprocessor()
        train_data = processor(train_data)
        validation_data = processor(validation_data)
        train_data = train_data.map(lambda x: {"vacuous_rationale": None})
        validation_data = validation_data.map(lambda x: {"vacuous_rationale": None})
        os.makedirs("data/processed_datasets/cose_simulation/train", exist_ok=True)
        os.makedirs("data/processed_datasets/cose_simulation/validation", exist_ok=True)
        train_data.save_to_disk("data/processed_datasets/cose_simulation/train")
        validation_data.save_to_disk("data/processed_datasets/cose_simulation/validation")
        
    elif exp_name == "prepare_vocab":
        os.system(f"python scripts/generate_vocabs.py "
                  f"--dataset-dir data/processed_datasets/cose_simulation "
                  f"--rationale-format g "
                  f"--rationale-only")
    elif exp_name == "rev_train":
        # train detector
        os.system(f"python steps/train_rev_model.py "
                  f"--task-name fasttext-cose_simulation " 
                  f"--rationale-format g")
        # train generator
        os.system(f"python steps/train_generator.py " 
                  f"--task-name cose_simulation "
                  f"--rationale-format g " 
                  f"--removal-threshold 0.1")
        # train irm
        os.system(f"python steps/train_irm_model.py "
                  f"--task-name cose_simulation "
                  f"--rationale-format g "
                  f"--removal-threshold 0.1 "
                  f"--irm-coefficient 10.0")
        os.system(f"python steps/train_irm_model.py "
                  f"--task-name cose_simulation "
                  f"--rationale-format g "
                  f"--removal-threshold 0.1 "
                  f"--irm-coefficient 100.0")
    elif exp_name == "rev_eval":
        os.system(f"python steps/eval_rev_with_model.py "
                  f"--task-name cose_simulation "
                  f"--dataset-dir data/processed_datasets/cose_simulation "
                  f"--model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/cose_simulation_t5-base_g_0.1_100.0 "
                  f"--rationale-format g")
        os.system(f"python steps/eval_rev_with_model.py "
                  f"--task-name cose_simulation "
                  f"--dataset-dir data/processed_datasets/cose_simulation "
                  f"--model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/cose_simulation_t5-base_g_0.1_10.0 "
                  f"--rationale-format g")
if __name__ == "__main__":
    main()
