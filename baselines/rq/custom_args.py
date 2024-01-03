"""
Custom arguments and evaluation helper functions.
Partially based on https://github.com/huggingface/transformers/tree/7cb203fae4e7964e9e99400b375d660ebce765ee/examples/language-modeling/run_language_modeling.py (Huggingface Transformers v2.9.1)
See Huggingface repository for licensing agreement.

Code formatted using https://github.com/psf/black
"""

from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
import os

__LABEL_TO_ANSWER__ = {
    True: "yes",
    False: "no"
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    rationale_only: bool = field(
        default=False,
        metadata={
            "help": "Only produce rationales and not labels (first model in pipeline)"
        },
    )
    label_only: bool = field(
        default=False,
        metadata={"help": "Only produce labels and not rationales (I-->O baseline)"},
    )
    include_input: bool = field(
        default=False,
        metadata={"help": "Append input to second model in pipeline"},
    )
    use_dev_real_expls: bool = field(
        default=False,
        metadata={
            "help": "Use this flag for test case where we want to test on gold-label predictions rather than generations"
        },
    )
    save_gradients: bool = field(
        default=False,
        metadata={
            "help": "Use this flag to compute gradient attributions for feature importance agreement (and save them to a JSON file)"
        },
    )
    pretrained_model_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pass a pretrained model save_path to re-load for evaluation"
        },
    )
    predictions_model_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pass a file where can find predictions from generation model for the dev set (first model in pipeline)"
        },
    )
    dropout_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": "Specify a dropout rate, if don't want to use default in transformers/configuration_t5.py"
        },
    )
    gradient_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the gradient method to use; must be specified if save_gradients is on. Choices are 'raw', 'times_input', 'smoothgrad', 'smoothgrad_squared', or 'integrated'."
        },
    )
    combination_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "The method to use to combine dimensions of the embedding vector during gradient attribution. Must be specified if save_gradients is on. Choices are 'sum' or 'l1'."
        },
    )
    smoothgrad_stdev: Optional[float] = field(
        default=None,
        metadata={"help": "Standard Deviation for the Smoothgrad Gaussian"},
    )
    nsamples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of Samples for the Smoothgrad or Integrated Gradients approximation"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on."})
    rationale_format: str = field(
        default=None,
        metadata={"help": "The format of the rationales to use for training."},
    )
    early_stopping_threshold: int = field(
        default=10,
        metadata={"help": "The number of patience epochs for early stopping."},
    )
    train_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for train set and save"}
    )
    test_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for test set and save"}
    )
    dev_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for dev set and save"}
    )
    version_name: Optional[str] = field(
        default="v1.11", metadata={"help": "Version of CoS-E to load"}
    )
    generations_filepath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-generated (clean) model generations for evaluation. Can be used on its own or paired with the `noised_generations_filepath` argument to evaluate pairs of clean & noisy generations."
        },
    )
    encoder_noise_variance: Optional[int] = field(
        default=None,
        metadata={
            "help": "Pass value for variance on noise Gaussian added to inputs at test-time"
        },
    )
    noised_generations_filepath: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-generated noisy model generations (fixed with random seed)"
        },
    )
    gradients_filepath: str = field(
        default=None,
        metadata={"help": "Path to gradient attributions for ROAR"},
    )
    roar_drop_percent: float = field(
        default=None,
        metadata={"help": "A float indicating the proportion of top tokens to drop"},
    )


def compute_metrics(
    save_path,
    dataset,
    model,
    tokenizer,
    split,
    task,
    device,
    rationale_only=False,
    label_only=False,
    generations_file=None,
):
    fname = os.path.join(save_path, "%s_generations.txt" % split)
    analysis_file = os.path.join(save_path, "%s_posthoc_analysis.txt" % split)
    if os.path.isfile(fname):
        fname = fname.split(".txt")[0] + "_1.txt"
    if os.path.isfile(analysis_file):
        analysis_file = analysis_file.split(".txt")[0] + "_1.txt"

    if generations_file is None:
        generations_list = []
        with open(fname, "w") as w:
            for i, element in tqdm(enumerate(dataset), total=len(dataset)):
                inpt_tensor = torch.tensor(
                    element["question_encoding"], device=device
                ).reshape(1, -1)
                # to improve performance, set the min length to 100 tokens
                out = model.generate(
                    inpt_tensor,
                    max_length=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                words = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
                # write out all generated tokens (strip newlines)
                words = words.replace("\n", " ").strip()
                w.write(words + "\n")
                generations_list.append(words)
    else:
        # load from file
        with open(generations_file, "r") as f:
            generations_list = f.readlines()
        analysis_file = os.devnull

    if rationale_only:
        return parse_wt5_no_label(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )
    elif label_only:
        return parse_wt5_label_only(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )
    else:
        return parse_wt5_output(
            analysis_file, generations_list, dataset, task, tokenizer.eos_token
        )


def parse_wt5_output(f, generations_list, dataset, task, eos_token):
    acc = []
    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):
            pred_l = line.split("explanation:")[0].strip()
            if len(line.split("explanation:")) > 1:
                pred_e = line.split("explanation:")[1].strip()
                if eos_token in pred_e:
                    pred_e = pred_e.split(eos_token)[0].strip()
                # also split on extra id token (which tends to appear as a delimiter frequently)
                pred_e = pred_e.split("<extra_id")[0].strip()
            else:
                pred_e = ""

            if task == "cos_e":
                gold_l = gold["answer"]
                gold_e1 = gold["abstractive_explanation"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_l = gold["label"]
                gold_e1 = gold["explanation_1"]
                gold_e2 = gold["explanation_2"]
                # convert to string
                if gold_l == 0:
                    gold_l = "entailment"
                elif gold_l == 1:
                    gold_l = "neutral"
                elif gold_l == 2:
                    gold_l = "contradiction"
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
            elif task == "strategyqa":
                gold_l = __LABEL_TO_ANSWER__[gold['answer']]
                gold_e1 = ' '.join(gold['facts'])
                g.write(gold['question'] + "\n")

            if task == "esnli":
                g.write(
                    "Correct: " + gold_l + " | " + gold_e1 + " [SEP] " + gold_e2 + "\n"
                )
            elif task == "cos_e":
                g.write("Correct: " + gold_l + " | " + gold_e1 + "\n")
            elif task == "strategyqa":
                g.write("Correct: " + gold_l + " | " + gold_e1 + "\n")

            g.write("Predicted: " + pred_l + " | " + pred_e + "\n")

            # calculate metrics
            met = gold_l == pred_l
            acc.append(met)
            g.write("Label Considered Correct: " + str(met) + "\n")
            g.write("\n")

    assert len(acc) == len(generations_list)
    return sum(acc) / len(acc) * 100


def parse_wt5_label_only(f, generations_list, dataset, task, eos_token):
    acc = []
    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):
            if eos_token not in line:
                # split on period or extra id token (which tends to appear as a delimiter frequently)
                pred_l = line.split(".")[0].split("<extra_id")[0].strip()
            else:
                # split on EOS token or extra id token
                pred_l = line.split(eos_token)[0].split("<extra_id")[0].strip()

            if task == "cos_e":
                gold_l = gold["answer"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_l = gold["label"]
                # convert to string
                if gold_l == 0:
                    gold_l = "entailment"
                elif gold_l == 1:
                    gold_l = "neutral"
                elif gold_l == 2:
                    gold_l = "contradiction"
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
            elif task == "strategyqa":
                gold_l = __LABEL_TO_ANSWER__[gold['answer']]
                g.write(gold['question'] + "\n")

            g.write("Correct: " + gold_l + " | " + "\n")
            g.write("Predicted: " + pred_l + " | " + "\n")

            # calculate metrics
            met = gold_l.lower() == pred_l.lower()

            acc.append(met)
            g.write("Considered Correct: " + str(met) + "\n")
            g.write("\n")

    assert len(acc) == len(generations_list)
    return sum(acc) / len(acc) * 100


def parse_wt5_no_label(f, generations_list, dataset, task, eos_token):
    with open(f, "w") as g:
        for i, (line, gold) in tqdm(
            enumerate(zip(generations_list, dataset)), total=len(dataset)
        ):
            if len(line.split("explanation:")) > 1:
                pred_e = line.split("explanation:")[1].strip()
                if eos_token in pred_e:
                    pred_e = pred_e.split(eos_token)[0].strip()
                # also split on extra id token (which tends to appear as a delimiter frequently)
                pred_e = pred_e.split("<extra_id")[0].strip()
            else:
                pred_e = ""

            if task == "cos_e":
                gold_e1 = gold["abstractive_explanation"]
                g.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_e1 = gold["explanation_1"]
                gold_e2 = gold["explanation_2"]
                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
            elif task == "strategyqa":
                gold_e1 = ' '.join(gold['facts'])
                g.write(gold['question'] + "\n")

            if task == "esnli":
                g.write("Correct: | " + gold_e1 + " [SEP] " + gold_e2 + "\n")
            elif task == "cos_e":
                g.write("Correct: | " + gold_e1 + "\n")
            elif task == "strategyqa":
                g.write("Correct: | " + gold_e1 + "\n")

            g.write("Predicted: " + " | " + pred_e + "\n")
            g.write("\n")

    return "n/a"

def compare_models_with_noise(
    encoder_noise_variance,
    save_path,
    dataset,
    model,
    tokenizer,
    split,
    task,
    device,
    rationale_only=False,
    label_only=False,
    clean_generations_file=None,
    noised_generations_file=None,
):
    assert encoder_noise_variance > 0

    # arg checks
    if label_only or rationale_only:
        raise Exception("only for 2-headed end-to-end model")

    # make folder with specified file info
    if noised_generations_file is None:
        new_folder = os.path.join(
            save_path,
            "encoder_noise_variance_%s_split_%s" % (str(encoder_noise_variance), split),
        )
        assert not os.path.exists(new_folder)
        os.makedirs(new_folder)
    else:
        # use current path
        new_folder = os.path.split(noised_generations_file)[0]

    fname = os.path.join(new_folder, "%s_generations.txt" % split)
    second_fname = os.path.join(new_folder, "%s_noisy_generations.txt" % split)
    joint_analysis_file = os.path.join(
        new_folder, "%s_noisy_vs_clean_posthoc_analysis.txt" % split
    )
    noisy_analysis_file = os.path.join(
        new_folder, "%s_noisy_posthoc_analysis.txt" % split
    )
    clean_analysis_file = os.path.join(
        new_folder, "%s_clean_posthoc_analysis.txt" % split
    )
    for filename in [
        fname,
        second_fname,
        joint_analysis_file,
        noisy_analysis_file,
        clean_analysis_file,
    ]:
        if os.path.isfile(filename):
            filename = filename.split(".txt")[0] + "_1.txt"
            if os.path.isfile(filename):
                raise Exception("both files already exist")

    if clean_generations_file is None:
        # actually produce clean generations and write to file
        generations_list = []
        with open(fname, "w") as w:
            for i, element in tqdm(enumerate(dataset), total=len(dataset)):
                inpt_tensor = torch.tensor(
                    element["question_encoding"], device=device
                ).reshape(1, -1)
                out = model.generate(
                    input_ids=inpt_tensor,
                    max_length=100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    variance=None,
                )
                words = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
                # every generation should have an EOS token, if model is properly trained (have validated on E-SNLI test set)
                words = (
                    words.replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
                )
                w.write(words + "\n")
                generations_list.append(words)
    else:
        # load from file
        with open(clean_generations_file, "r") as f:
            lines = f.readlines()
            # strip newlines & EOS token (if exists)
        generations_list = [
            l.replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
            for l in lines
        ]

    if noised_generations_file is None:
        # add noise, and do it again
        comparable_generations = []
        with open(second_fname, "w") as w:
            for i, element in tqdm(enumerate(dataset), total=len(dataset)):
                inpt_tensor = torch.tensor(
                    element["question_encoding"], device=device
                ).reshape(1, -1)
                out = model.generate(
                    input_ids=inpt_tensor,
                    max_length=100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    variance=encoder_noise_variance,
                )
                words = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
                # every generation should have an EOS token, if model is properly trained (have validated on E-SNLI test set)
                # however, noise often breaks this assumption in extreme cases
                words = (
                    words.replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
                )
                w.write(words + "\n")
                comparable_generations.append(words)
    else:
        # load from file
        with open(noised_generations_file, "r") as f:
            lines = f.readlines()
            # strip newlines & EOS token (if exists)
        comparable_generations = [
            l.replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
            for l in lines
        ]

    # assert both for same dataset-split
    assert len(comparable_generations) == len(generations_list)

    return parse_noise_output(
        joint_analysis_file,
        clean_analysis_file,
        noisy_analysis_file,
        generations_list,
        comparable_generations,
        dataset,
        task,
        tokenizer.eos_token,
    )


def parse_noise_output(
    f, c, n, generations_list, comparable_generations, dataset, task, eos_token
):
    acc = []
    noised_acc = []
    label_flips = []
    explanation_flips = []
    both_flips = []
    skip_count = 0
    no_format = 0
    with open(f, "w") as joint, open(c, "w") as clean, open(n, "w") as noisy:
        for i, (line, noised, gold) in tqdm(
            enumerate(zip(generations_list, comparable_generations, dataset)),
            total=len(dataset),
        ):
            try:
                pred_l = line.split("explanation:")[0].strip()
                if len(line.split("explanation:")) > 1:
                    pred_e = line.split("explanation:")[1].strip()
                    if eos_token in pred_e:
                        pred_e = pred_e.split(eos_token)[0].strip()
                else:
                    pred_e = ""

                if len(noised.split("explanation:")) > 1:
                    pred_l_noised = noised.split("explanation:")[0].strip()
                    pred_e_noised = noised.split("explanation:")[1].strip()
                else:
                    no_format += 1
                    pred_l_noised = ""
                    pred_e_noised = noised
                if eos_token in pred_e_noised:
                    pred_e_noised = pred_e_noised.split(eos_token)[0].strip()
            except:
                print(
                    "Line couldn't be processed (most likely due to format issue): ",
                    line,
                )
                skip_count += 1
                continue

            if task == "cos_e":
                gold_l = gold["answer"]
                gold_e1 = gold["abstractive_explanation"]
                joint.write(gold["question"] + "\n")
                clean.write(gold["question"] + "\n")
                noisy.write(gold["question"] + "\n")
            elif task == "esnli":
                gold_l = gold["label"]
                gold_e1 = gold["explanation_1"]
                gold_e2 = gold["explanation_2"]
                # convert to string
                if gold_l == 0:
                    gold_l = "entailment"
                elif gold_l == 1:
                    gold_l = "neutral"
                elif gold_l == 2:
                    gold_l = "contradiction"
                joint.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
                clean.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
                noisy.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
            else:
                raise Exception("unknown task")

            if task == "esnli":
                joint.write(
                    "Correct: " + gold_l + " | " + gold_e1 + " [SEP] " + gold_e2 + "\n"
                )
                clean.write(
                    "Correct: " + gold_l + " | " + gold_e1 + " [SEP] " + gold_e2 + "\n"
                )
                noisy.write(
                    "Correct: " + gold_l + " | " + gold_e1 + " [SEP] " + gold_e2 + "\n"
                )
            elif task == "cos_e":
                joint.write("Correct: " + gold_l + " | " + gold_e1 + "\n")
                clean.write("Correct: " + gold_l + " | " + gold_e1 + "\n")
                noisy.write("Correct: " + gold_l + " | " + gold_e1 + "\n")

            joint.write("Predicted: " + pred_l + " | " + pred_e + "\n")
            joint.write(
                "Noisy predicted: " + pred_l_noised + " | " + pred_e_noised + "\n"
            )

            # only write predicted explanations for rationale-to-label model
            clean.write("Predicted explanation: " + pred_e + "\n")
            noisy.write("Predicted explanation: " + pred_e_noised + "\n")
            clean.write("Predicted label: " + pred_l + "\n")
            noisy.write("Predicted label: " + pred_l_noised + "\n")
            clean.write("\n")
            noisy.write("\n")

            # calculate metrics
            met = gold_l == pred_l
            met_noised = gold_l == pred_l_noised

            acc.append(met)
            noised_acc.append(met_noised)
            tl = pred_l.lower() != pred_l_noised.lower()
            te = pred_e.lower() != pred_e_noised.lower()
            label_flips.append(tl)
            explanation_flips.append(te)
            both_flips.append(tl and te)
            joint.write("Label flipped: " + str(tl) + "\n")
            joint.write("Explanation flipped: " + str(te) + "\n")
            joint.write("\n")

    print("% No format:", str(no_format / len(acc) * 100))

    return (
        sum(acc) / len(acc) * 100,
        skip_count,
        sum(noised_acc) / len(noised_acc) * 100,
        sum(label_flips) / len(label_flips) * 100,
        sum(explanation_flips) / len(explanation_flips) * 100,
        sum(both_flips) / len(both_flips) * 100,
    )
