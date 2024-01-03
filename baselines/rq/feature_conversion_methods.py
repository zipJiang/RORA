""" 
Helper preprocessing functions. 

Code formatted using https://github.com/psf/black
"""
import random
import numpy as np

random.seed(10)

__TEMPLATES__ = {
    "g": "{gold_rationale}",
    "s": "{base_rationale}",
    "l": "{leaky_rationale}",
    "gs": "{gold_rationale} {base_rationale}",
    "ls": "{leaky_rationale} {base_rationale}",
    "gl": "{gold_rationale} {leaky_rationale}",
    "gls": "{gold_rationale} {leaky_rationale} {base_rationale}",
    "n": ""
}


__LABEL_TO_ANSWER__ = {
    True: "yes",
    False: "no"
}



__LABEL_TO_LEAKY_RATIONALE__ = {
    True: f"The answer is {__LABEL_TO_ANSWER__[True]}",
    False: f"The answer is {__LABEL_TO_ANSWER__[False]}"
}


def cose_explanation_to_label(
    example,
    index,
    tokenizer,
    pred_only=False,
    predictions_file=None,
    include_input=False,
    rationale_format=None
):
    # Format:
    # if include_input:
    # Input: "cos_e question: [question] choice: [choice_0] choice: [choice_1] choice: [choice_2] explanation: [abstractive_explanation]"
    # if not include_input:
    # Input: "cos_e choice: [choice_0] choice: [choice_1] choice: [choice_2] explanation: [abstractive_explanation]"
    # Output: "[answer]"

    if pred_only:
        abstr_expl = predictions_file[index]
    else:
        abstr_expl = example["abstractive_explanation"]

    if include_input:
        question = example["question"]
        input_string = (
            f"cos_e question: {question} choice: "
            + " choice: ".join(example["choices"])
            + f" explanation: {abstr_expl}"
        )
    else:
        input_string = (
            f"cos_e choice: "
            + " choice: ".join(example["choices"])
            + f" explanation: {abstr_expl}"
        )

    answer_string = example["answer"]

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings


def esnli_explanation_to_label(
    example,
    index,
    tokenizer,
    pred_only=False,
    predictions_file=None,
    include_input=False,
    rationale_format=None
):
    # Format:
    # if include_input:
    # Input: "nli hypothesis: [hypothesis] premise: [premise] explanation: [abstractive_explanation]"
    # if not include_input:
    # Input: "nli explanation: [abstractive_explanation]"
    # Output: "[answer]"

    hypothesis = example["hypothesis"]
    premise = example["premise"]

    if pred_only:
        abstr_expl = predictions_file[index]
    else:
        abstr_expl = example["explanation_1"]

    if include_input:
        input_string = (
            f"nli hypothesis: {hypothesis} premise: {premise} explanation: {abstr_expl}"
        )
    else:
        input_string = f"nli explanation: {abstr_expl}"

    if example["label"] == 0:
        answer_string = "entailment"
    elif example["label"] == 1:
        answer_string = "neutral"
    elif example["label"] == 2:
        answer_string = "contradiction"

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings

def strategyqa_explanation_to_label(
    example,
    index,
    tokenizer,
    pred_only=False,
    predictions_file=None,
    include_input=False,
    rationale_format=None
):
    assert rationale_format is not None, "rationale format must be specified for strategyqa"

    if pred_only:
        expl = predictions_file[index]
    else:
        template = __TEMPLATES__[rationale_format]
        expl = template.format(
                gold_rationale=' '.join(example['facts']),
                base_rationale=example['vacuous_rationale'],
                leaky_rationale=__LABEL_TO_LEAKY_RATIONALE__[example['answer']]
            )

    if include_input:
        question = example["question"]
        input_string = (
            f"strategyqa question: {question}"
            + f" explanation: {expl}"
        )
    else:
        input_string = (
            + f"explanation: {expl}"
        )

    answer_string = __LABEL_TO_ANSWER__[example["answer"]]

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings

def input_to_explanation_plus_label(
    example,
    index,
    tokenizer,
    datasource=None,
    expl_only=False,
    label_only=False,
    gradients=None,
    threshold=None,
    rationale_format=None,
):
    # CoS-E Format:
    # Input: "explain cos_e question: [question] choice: [choice_0] choice: [choice_1] choice: [choice_2]"

    # e-SNLI Format:
    # Input: "explain nli hypothesis: [hypothesis] premise: [premise]"

    # Output: "[answer] explanation: [abstractive_explanation]"
    # Explanation-only output: "None explanation: [abstractive_explanation]"
    # Label-only output: "[answer]"

    assert datasource in {"cos_e", "esnli", 'strategyqa'}
    if datasource == 'strategyqa' and not label_only:
        assert rationale_format is not None, "rationale format must be specified for strategyqa"

    if datasource == "cos_e":
        input_string, answer_string = cose_wt5_format(
            example, expl_only=expl_only, label_only=label_only
        )
    elif datasource == "esnli":
        input_string, answer_string = esnli_wt5_format(
            example, expl_only=expl_only, label_only=label_only
        )
    elif datasource == "strategyqa":
        input_string, answer_string = strategyqa_wt5_format(
            example, expl_only=expl_only, label_only=label_only, rationale_format=rationale_format
        )

    # tokenizer takes care of model-specific special tokens
    encodings = tokenizer.encode_plus(
        input_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    if threshold is not None:
        # compute top-k tokens
        k_length = round(len(encodings["input_ids"]) * threshold)
        if k_length > 0:
            if gradients is not None:
                # select tokens to drop from ranked gradient file
                # if gradient object doesn't exist, throw out
                try:
                    grads = gradients["label"]["instance_%d" % (index + 1)][
                        "grad_input_2"
                    ]
                except:
                    # index doesn't exist in gradients file because couldn't produce gradients due to bad decoding
                    return {}
                assert len(encodings["input_ids"]) == len(grads)
                # select tokens to drop based on absolute-value rank-importance
                top_token_inxs = np.argsort(np.abs(grads))[-k_length:]
            else:
                # select random tokens to drop for baseline
                top_token_inxs = random.sample(
                    [i for i in range(len(encodings["input_ids"]))], k_length
                )
        else:
            top_token_inxs = []
        assert len(top_token_inxs) == k_length

        # first determine which top-tokens are spans
        # replace these tokens with special "sentinel"/pad tokens
        # there are 100 extra_ids from positions 32000-32099 in the tokenizer by default
        tmp_lst = []
        tmp_attns = []
        curr = 1000
        curr_extra_id = 32100
        for i, (item, att) in enumerate(
            zip(encodings["input_ids"], encodings["attention_mask"])
        ):
            if i in top_token_inxs:
                if i != curr + 1:
                    if curr_extra_id - 1 < 32000:
                        raise Exception("too small id value")
                    tmp_lst.append(curr_extra_id - 1)
                    tmp_attns.append(att)
                    curr_extra_id = curr_extra_id - 1
                curr = i
            else:
                tmp_lst.append(item)
                tmp_attns.append(att)

        assert len(tmp_lst) == len(tmp_attns)

        encodings["input_ids"] = tmp_lst
        encodings["attention_mask"] = tmp_attns

    # note even with "labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string + tokenizer.eos_token,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]

    encodings["question_encoding"] = encodings["input_ids"]

    return encodings


def cose_wt5_format(item, expl_only=False, label_only=False):
    question = item["question"]
    answer = item["answer"]
    abstr_expl = item["abstractive_explanation"]

    input_string = f"explain cos_e question: {question} choice: " + " choice: ".join(
        item["choices"]
    )

    if expl_only:
        answer_string = f"None explanation: {abstr_expl}"
    elif label_only:
        answer_string = f"{answer}"
    else:
        answer_string = f"{answer} explanation: {abstr_expl}"

    return input_string, answer_string


def esnli_wt5_format(item, expl_only=False, label_only=False):
    premise = item["premise"]
    hypothesis = item["hypothesis"]
    if item["label"] == 0:
        answer = "entailment"
    elif item["label"] == 1:
        answer = "neutral"
    elif item["label"] == 2:
        answer = "contradiction"
    abstr_expl = item["explanation_1"]

    input_string = f"explain nli hypothesis: {hypothesis} premise: {premise}"
    if expl_only:
        answer_string = f"None explanation: {abstr_expl}"
    elif label_only:
        answer_string = f"{answer}"
    else:
        answer_string = f"{answer} explanation: {abstr_expl}"

    return input_string, answer_string

def strategyqa_wt5_format(item, expl_only=False, label_only=False, rationale_format=None):

    question = item['question']
    answer = __LABEL_TO_ANSWER__[item['answer']]
    input_string = f"explain strategyqa question: {question}"

    if label_only:
        answer_string = f"{answer}"
    else:
        template = __TEMPLATES__[rationale_format]
        expl = template.format(
                gold_rationale=' '.join(item['facts']),
                base_rationale=item['vacuous_rationale'],
                leaky_rationale=__LABEL_TO_LEAKY_RATIONALE__[item['answer']]
            )

        if expl_only:
            answer_string = f"explanation: {expl}"
        else:
            answer_string = f"{answer} explanation: {expl}"

    return input_string, answer_string


