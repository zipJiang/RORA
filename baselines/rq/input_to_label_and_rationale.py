"""
Fine-tunes a model to jointly generate labels + rationales given input.
Partially based on https://github.com/huggingface/transformers/tree/7cb203fae4e7964e9e99400b375d660ebce765ee/examples/language-modeling/run_language_modeling.py (Huggingface Transformers v2.9.1)
See Huggingface repository for licensing agreement.

Code formatted using https://github.com/psf/black
"""

import logging
import math
import os

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from feature_conversion_methods import input_to_explanation_plus_label
from modeling_t5 import T5ForConditionalGeneration as CustomT5ForConditionalGeneration

from trainer import Trainer
from custom_args import (
    DataTrainingArguments,
    ModelArguments,
    compute_metrics,
    compare_models_with_noise,
)
import torch
import nlp
import datasets
import git
import time
from datetime import datetime
import sys
import json
import random

random.seed(10)

logger = logging.getLogger(__name__)

CACHE_DIR = '/scratch/ylu130/model-hf'

class SequenceCollator:
    def __init__(self, pad_token):
        self.pad_token_mapping = {
            "labels": -100,
            "attention_mask": 0,
            "decoder_attention_mask": 0,
            "input_ids": pad_token,
        }
        self.columns = [
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_attention_mask",
        ]

    def collate_batch(self, examples):
        # batch inputs for training
        batch = {}
        for key in examples[0].keys():
            if key in self.columns:
                tmp_list = []
                for item in examples:
                    tmp_list.append(item[key])

                # pad lists to max length
                if isinstance(tmp_list[0], list):
                    max_length = max(map(len, tmp_list))
                    tmp_list = [
                        el + [self.pad_token_mapping[key]] * (max_length - len(el))
                        for el in tmp_list
                    ]

                batch[key] = torch.tensor(tmp_list, dtype=torch.long)
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    og_start_time = time.time()

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not training_args.do_train:
        if (not model_args.pretrained_model_file) and (
            not data_args.generations_filepath
        ):
            raise Exception(
                "if not training a model from scratch, must specify a trained model to load for evaluation or generations in a file to evaluate"
            )

    if data_args.roar_drop_percent is not None:
        # assert a proportion is specified
        assert data_args.roar_drop_percent > 0 and data_args.roar_drop_percent < 1

    # make sure only one dataset split pick if manually specifying evaluation file
    if data_args.generations_filepath is not None:
        training_args.do_train = False
        training_args.do_eval = False
        if "train" in data_args.generations_filepath:
            data_args.train_predict = True
            data_args.test_predict = False
            data_args.dev_predict = False
        elif "test" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = True
            data_args.dev_predict = False
        elif "validation" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = False
            data_args.dev_predict = True

    if model_args.save_gradients:
        assert model_args.gradient_method in {
            "raw",
            "times_input",
            "smoothgrad",
            "smoothgrad_squared",
            "integrated",
        }
        if model_args.gradient_method in {"smoothgrad", "smoothgrad_squared"}:
            assert model_args.smoothgrad_stdev > 0 and model_args.smoothgrad_stdev < 1
        if model_args.gradient_method in {
            "smoothgrad",
            "smoothgrad_squared",
            "integrated",
        }:
            assert model_args.nsamples > 0
        assert model_args.combination_method in {"l1", "sum"}

    # create a new directory if fine-tuning an existing checkpoint or training/evaluating a HF pretrained model
    # do not do this when re-evaluating a pretrained_model_file
    if training_args.do_train or (
        not model_args.pretrained_model_file and not data_args.generations_filepath
    ):
        # create a save directory and a logfile
        save_path = training_args.output_dir
        training_args.output_dir = os.path.join(
            save_path, f"{datetime.now().strftime('%m%d%y_%H%M%S')}_{data_args.rationale_format}"
        )
        training_args.logging_dir = training_args.output_dir
        assert os.path.exists(save_path)
        assert not os.path.exists(training_args.output_dir)
        os.makedirs(training_args.output_dir)

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        handlers = [
            logging.FileHandler(os.path.join(training_args.output_dir, "logger.log")),
            logging.StreamHandler(),
        ]
    else:
        # don't overwrite existing logfile or create new directory
        training_args.output_dir = model_args.pretrained_model_file
        handlers = [logging.StreamHandler()]

    if data_args.encoder_noise_variance is not None:
        # must be in evaluation mode
        assert not training_args.do_train
        assert model_args.pretrained_model_file is not None
        assert data_args.test_predict or data_args.dev_predict
        assert 40 > data_args.encoder_noise_variance > 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Save path: %s" % training_args.output_dir)

    # get git hash and branch where deployed
    repo = git.Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha
    git_branch = repo.active_branch.name
    logger.info("Git branch: %s" % git_branch)
    logger.info("Git hash: %s" % git_hash)

    assert data_args.task_name in {"cos_e", "esnli", 'strategyqa'}

    # set gradient accumulation steps to always use batch size == 64
    if 64 % training_args.per_device_train_batch_size != 0:
        raise Exception(
            "Batch size is not a divisor of 64, resulting in inconsistent gradient-accumulation behavior"
        )
    training_args.gradient_accumulation_steps = int(
        64 / training_args.per_device_train_batch_size
    )

    if training_args.do_train:
        # write command and args to file
        with open(
            os.path.join(training_args.output_dir, "commandline_args.txt"), "w"
        ) as f:
            f.write("Git branch: " + git_branch + "\n")
            f.write("Git hash: " + git_hash + "\n")
            f.write("Command:\n")
            f.write("\n".join(sys.argv[1:]))
            f.write("Training args:\n")
            # make training_args dict writeable
            tmp = training_args.__dict__
            tmp.pop("__cached__setup_devices", None)
            tmp.pop("evaluation_strategy", None)
            tmp.pop("lr_scheduler_type", None)
            tmp.pop("logging_strategy", None)
            tmp.pop("save_strategy", None)
            json.dump(tmp, f, indent=2)
            f.write("Data args:\n")
            json.dump(data_args.__dict__, f, indent=2)
            f.write("Model args:\n")
            json.dump(model_args.__dict__, f, indent=2)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    logger.info("Loading pretrained tokenizer...")
    if model_args.pretrained_model_file:
        # load pretrained tokenizer from directory
        tokenizer = T5Tokenizer.from_pretrained(model_args.pretrained_model_file)
    else:
        # load pretrained tokenizer from Huggingface
        tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir=CACHE_DIR)

    if (data_args.generations_filepath is None) or (
        data_args.encoder_noise_variance is not None
        and data_args.noised_generations_filepath is None
    ):
        if model_args.pretrained_model_file:
            # load pretrained model from directory at best checkpoint
            ckpts = [
                name
                for name in os.listdir(model_args.pretrained_model_file)
                if PREFIX_CHECKPOINT_DIR in name
            ]
            if len(ckpts) != 1:
                raise Exception(
                    "more than 1 checkpoint file stored in pretrained path. revisit save directory"
                )
            model_load_path = os.path.join(model_args.pretrained_model_file, ckpts[0])
            if (
                data_args.encoder_noise_variance is not None
                or model_args.save_gradients
            ):
                # initialize custom model
                model = CustomT5ForConditionalGeneration.from_pretrained(
                    model_load_path
                )
            else:
                model = T5ForConditionalGeneration.from_pretrained(model_load_path)
            if model_args.dropout_rate:
                raise Exception(
                    "can't update/specify dropout currently when load pretrained model from directory"
                )

        else:
            # load pretrained model from HuggingFace
            logger.info("Loading pretrained model")
            if model_args.dropout_rate:
                model = T5ForConditionalGeneration.from_pretrained(
                    "t5-base", dropout_rate=model_args.dropout_rate, cache_dir=CACHE_DIR
                )
            else:
                model = T5ForConditionalGeneration.from_pretrained("t5-base", cache_dir=CACHE_DIR)

        model.resize_token_embeddings(len(tokenizer))
        model = model.to(training_args.device)
    else:
        model = None

    # load (new) cos-e version
    if data_args.task_name == "cos_e":
        assert data_args.version_name in {"v1.11", "v1.0"}
        version_arg = data_args.version_name
    else:
        version_arg = None

    # Get datasets
    if data_args.task_name == "strategyqa":
        data_dir = 'data/processed_datasets/strategyqa'
        train = datasets.load_from_disk(os.path.join(data_dir, 'train'))
        validation = datasets.load_from_disk(os.path.join(data_dir, 'validation'))
        test = datasets.load_from_disk(os.path.join(data_dir, 'test'))

        dataset = datasets.DatasetDict({
            'train': train,
            'validation': validation,
            'test': test
        })
    else:
        dataset = nlp.load_dataset(data_args.task_name, version_arg)

    # Apply method, and format dataset to torch.Tensor outputs
    for split in dataset.keys():
        if data_args.gradients_filepath is not None:
            # load gradients object
            with open(
                data_args.gradients_filepath.replace("train", split)
            ) as json_file:
                gradients = json.load(json_file)

            # collect list of valid indices (some CoS-E v1.11 train instances were thrown out due to bad decoding)
            valid_inxs = []
            for i in range(len(dataset[split])):
                if "instance_%d" % (i + 1) in gradients["label"]:
                    valid_inxs.append(i)

        # apply independently to each example
        dataset[split] = dataset[split].map(
            lambda i, x: input_to_explanation_plus_label(
                i,
                x,
                tokenizer,
                datasource=data_args.task_name,
                expl_only=model_args.rationale_only,
                label_only=model_args.label_only,
                gradients=gradients
                if data_args.gradients_filepath is not None
                else None,
                threshold=data_args.roar_drop_percent,
                rationale_format=data_args.rationale_format,
            ),
            # had some replicability issues with batch/cache set to True
            batched=False,
            with_indices=True,
            load_from_cache_file=False,
        )

        if data_args.gradients_filepath is not None:
            # subset down to correct ids
            dataset[split] = dataset[split].select(valid_inxs)
            if split == "train":
                train_valid_inxs = len(valid_inxs)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"] if data_args.task_name in ["esnli", 'strategyqa'] else None

    if data_args.task_name == "esnli":
        assert len(train_dataset) == 549367
        assert len(eval_dataset) == 9842
        assert len(test_dataset) == 9824
    elif data_args.task_name == "cos_e":
        if data_args.version_name == "v1.11":
            if data_args.gradients_filepath is not None:
                assert len(train_dataset) == train_valid_inxs
            else:
                assert len(train_dataset) == 9741
            assert len(eval_dataset) == 1221
        elif data_args.version_name == "v1.0":
            assert len(train_dataset) == 7610
            assert len(eval_dataset) == 950
        assert test_dataset is None

    logger.info("****LOG****")
    logger.info(len(train_dataset))
    logger.info(len(eval_dataset))
    if data_args.task_name in ["esnli", "strategyqa"]:
        logger.info(len(test_dataset))

    if data_args.generations_filepath is None:
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_args=data_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
            data_collator=SequenceCollator(pad_token=tokenizer.pad_token_id),
            tokenizer=tokenizer,
            device=training_args.device,
        )

    # Training
    if training_args.do_train:
        start_time = time.time()
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory
        tokenizer.save_pretrained(training_args.output_dir)
        train_time = time.time() - start_time
        model = trainer.model

    # Evaluation
    results = {}
    if model is not None:
        model.eval()

    if training_args.do_eval:
        start_time = time.time()

        logger.info("*** Evaluate on train set***")
        train_output = trainer.evaluate(train_dataset)
        perplexity = math.exp(train_output["eval_loss"])
        results["perplexity_train"] = perplexity

        logger.info("*** Evaluate on dev set***")
        eval_output = trainer.evaluate(eval_dataset)
        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity_validation"] = perplexity

        if data_args.task_name in ["esnli", 'strategyqa']:
            # also evaluate on test
            logger.info("*** Evaluate on test set***")
            test_output = trainer.evaluate(test_dataset)
            logger.info("test loss @ best dev epoch: %0.4f" % test_output["eval_loss"])
            perplexity = math.exp(test_output["eval_loss"])
            results["perplexity_test"] = perplexity

        eval_time = time.time() - start_time

    if data_args.generations_filepath is None:
        # only save to checkpoint folder if one exists (i.e. trained a model or loaded a model you previously trained)
        if training_args.do_train or model_args.pretrained_model_file:
            ckpts = [
                name
                for name in os.listdir(training_args.output_dir)
                if PREFIX_CHECKPOINT_DIR in name
            ]
            if len(ckpts) != 1:
                raise Exception(
                    "more than 1 checkpoint file stored in pretrained path. revisit save directory"
                )
            save_path = os.path.join(training_args.output_dir, ckpts[0])
        # else save to main directory (i.e. evaluating existing HF pretrained model)
        else:
            save_path = training_args.output_dir
    else:
        save_path = os.path.dirname(data_args.generations_filepath)

    # store predictions
    start_time = time.time()
    if data_args.train_predict:
        logger.info("*** Predict on train set***")
        if data_args.generations_filepath is not None:
            assert "train" in data_args.generations_filepath
        if data_args.encoder_noise_variance is not None:
            if data_args.noised_generations_filepath is not None:
                assert "train" in data_args.noised_generations_filepath
            logger.info("*** Predict on train set***")
            (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            ) = compare_models_with_noise(
                data_args.encoder_noise_variance,
                save_path,
                train_dataset,
                model,
                tokenizer,
                "train",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                clean_generations_file=data_args.generations_filepath,
                noised_generations_file=data_args.noised_generations_filepath,
            )
            (
                results["train_acc"],
                results["train_skip_count"],
                results["train_acc_noised"],
                results["train_label_flips"],
                results["train_expl_flips"],
                results["train_both_flips"],
            ) = (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            )
            if acc:
                logger.info("Train Accuracy: %f" % acc)
                logger.info("Noised Train Accuracy: %f" % noised_acc)
                logger.info("Train label flips %f" % label_flips)
                logger.info("Train explanation flips %f" % explanation_flips)
                logger.info("Train both flips %f" % both_flips)
            logger.info("Train # instances skipped %d" % skip_count)
        elif model_args.save_gradients:
            trainer.compute_gradients(
                save_path,
                train_dataset,
                "train",
                data_args.task_name,
                model_args,
            )
        else:
            acc = compute_metrics(
                save_path,
                train_dataset,
                model,
                tokenizer,
                "train",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                generations_file=data_args.generations_filepath,
            )
            results["train_acc"] = acc
            if acc != "n/a":
                logger.info("Train Accuracy: %f" % acc)

    if data_args.test_predict and data_args.task_name in ["esnli", 'strategyqa']:
        logger.info("*** Predict on test set***")
        if data_args.generations_filepath is not None:
            assert "test" in data_args.generations_filepath
        if data_args.encoder_noise_variance is not None:
            if data_args.noised_generations_filepath is not None:
                assert "test" in data_args.noised_generations_filepath
            (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            ) = compare_models_with_noise(
                data_args.encoder_noise_variance,
                save_path,
                test_dataset,
                model,
                tokenizer,
                "test",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                clean_generations_file=data_args.generations_filepath,
                noised_generations_file=data_args.noised_generations_filepath,
            )
            (
                results["test_acc"],
                results["test_skip_count"],
                results["test_acc_noised"],
                results["test_label_flips"],
                results["test_expl_flips"],
                results["test_both_flips"],
            ) = (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            )
            if acc:
                logger.info("Test Accuracy: %f" % acc)
                logger.info("Noised Test Accuracy: %f" % noised_acc)
                logger.info("Test label flips %f" % label_flips)
                logger.info("Test explanation flips %f" % explanation_flips)
                logger.info("Test both flips %f" % both_flips)
            logger.info("Test # instances skipped %d" % skip_count)
        elif model_args.save_gradients:
            trainer.compute_gradients(
                save_path,
                test_dataset,
                "test",
                data_args.task_name,
                model_args,
            )
        else:
            acc = compute_metrics(
                save_path,
                test_dataset,
                model,
                tokenizer,
                "test",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                generations_file=data_args.generations_filepath,
            )
            results["test_acc"] = acc
            if acc != "n/a":
                logger.info("Test Accuracy: %f" % acc)

    if data_args.dev_predict:
        logger.info("*** Predict on dev set***")
        if data_args.generations_filepath is not None:
            assert "validation" in data_args.generations_filepath
        if data_args.encoder_noise_variance is not None:
            if data_args.noised_generations_filepath is not None:
                assert "validation" in data_args.noised_generations_filepath
            logger.info("*** Predict on dev set***")
            (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            ) = compare_models_with_noise(
                data_args.encoder_noise_variance,
                save_path,
                eval_dataset,
                model,
                tokenizer,
                "validation",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                clean_generations_file=data_args.generations_filepath,
                noised_generations_file=data_args.noised_generations_filepath,
            )
            (
                results["dev_acc"],
                results["dev_skip_count"],
                results["dev_acc_noised"],
                results["dev_label_flips"],
                results["dev_expl_flips"],
                results["dev_both_flips"],
            ) = (
                acc,
                skip_count,
                noised_acc,
                label_flips,
                explanation_flips,
                both_flips,
            )
            if acc:
                logger.info("Dev Accuracy: %f" % acc)
                logger.info("Noised Dev Accuracy: %f" % noised_acc)
                logger.info("Dev label flips %f" % label_flips)
                logger.info("Dev explanation flips %f" % explanation_flips)
                logger.info("Dev both flips %f" % both_flips)
            logger.info("Dev # instances skipped %d" % skip_count)
        elif model_args.save_gradients:
            trainer.compute_gradients(
                save_path,
                eval_dataset,
                "validation",
                data_args.task_name,
                model_args,
            )
        else:
            acc = compute_metrics(
                save_path,
                eval_dataset,
                model,
                tokenizer,
                "validation",
                data_args.task_name,
                training_args.device,
                rationale_only=model_args.rationale_only,
                label_only=model_args.label_only,
                generations_file=data_args.generations_filepath,
            )
            results["dev_acc"] = acc
            if acc != "n/a":
                logger.info("Dev Accuracy: %f" % acc)

    if not model_args.save_gradients:
        if data_args.generations_filepath is None:
            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results_lm.txt"
            )
        else:
            output_eval_file = os.path.join(
                os.path.dirname(os.path.dirname(data_args.generations_filepath)),
                "eval_results_lm.txt",
            )
        with open(output_eval_file, "a+") as writer:
            for key in sorted(results.keys()):
                if results[key] is not None:
                    logger.info("  %s = %s", key, str(results[key]))
                    writer.write("%s = %s\n" % (key, str(results[key])))

    predict_time = time.time() - start_time

    # final logs
    logger.info("Git branch: %s" % git_branch)
    logger.info("Git hash: %s" % git_hash)
    logger.info("Save path: %s" % training_args.output_dir)
    if training_args.do_train:
        logger.info("total train time: %.4f hours" % (train_time / 60.0 / 60.0))
    if training_args.do_eval:
        logger.info("total eval time: %.4f hours" % (eval_time / 60.0 / 60.0))
    if (
        data_args.train_predict
        or data_args.dev_predict
        or (data_args.test_predict and data_args.task_name in ["esnli", "strategyqa"])
    ):
        logger.info("total predict time: %.4f hours" % (predict_time / 60.0 / 60.0))
    logger.info(
        "TOTAL SCRIPT TIME: %.4f hours" % ((time.time() - og_start_time) / 60.0 / 60.0)
    )

if __name__ == "__main__":
    main()
