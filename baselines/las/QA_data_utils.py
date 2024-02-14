import os
import csv
import argparse
import logging
import json
import time
import torch
import torch.nn.functional as F
import datasets
import numpy as np
import pandas as pd
from utils import removeNonAscii, isNaN

__LABEL_TO_ANSWER__ = {
    True: "0",
    False: "1"
}

__TEMPLATES__ = {
    "ground_truth": "{gold_rationale}",
    "s": "{base_rationale}",
    "l": "{leaky_rationale}",
    "gs": "{gold_rationale} {base_rationale}",
    "ls": "{leaky_rationale} {base_rationale}",
    "gl": "{gold_rationale} {leaky_rationale}",
    "gls": "{gold_rationale} {leaky_rationale} {base_rationale}",
    "gsl": "{gold_rationale} {base_rationale} {leaky_rationale}",
    "n": ""
}

__LABEL_TO_LEAKY_RATIONALE__ = {
    True: f"The answer is {__LABEL_TO_ANSWER__[True]}",
    False: f"The answer is {__LABEL_TO_ANSWER__[False]}"
}

class QAExample(object):
    '''used for training models with CQA data'''
    def __init__(self,
                 qa_id,
                 question,
                 explanation,
                 choices,
                 label = None,
                 num_choices = 3):
        self.cqa_id = qa_id
        self.question = question
        self.explanation = explanation
        self.label = int(label)
        self.choices = choices

        if num_choices == 2:
            self.choices_str = f'The choices are {self.choices[0]} and {self.choices[1]}.'
        elif num_choices == 3:
            self.choices_str = f'The choices are {self.choices[0]}, {self.choices[1]}, and {self.choices[2]}.'
        elif num_choices == 5:
            self.choices_str = f'The choices are {self.choices[0]}, {self.choices[1]}, {self.choices[2]}, {self.choices[3]}, and {self.choices[4]}.'
        else:
            raise NotImplementedError

        # self.choices_str = f'The choices are {self.choices[0]}, {self.choices[1]}, and {self.choices[2]}.' \
        #                     if self.version == '1.0' \
        #                     else \
        #                    f'The choices are {self.choices[0]}, {self.choices[1]}, {self.choices[2]}, {self.choices[3]}, and {self.choices[4]}.'

        self.explanation_list = [explanation] \
                                if not isinstance(explanation, list) \
                                else \
                                explanation
            
    def __str__(self):
        return self.__repr__()

    def __repr__(self):

        list_ = [f"question: {self.question}"] + \
            [f"choice {d}: {exp}" for d,exp in enumerate(self.choices)] + \
            [f"explanation: {self.explanation}"]

        if self.label is not None:
            list_.append(f"label: {self.label}")

        return "\n".join(list_)
    
def convert_strategyqa_jsonl_to_csv(data, data_dir, target_data_dir, split, target_split):
    assert data == 'StrategyQA' or data == 'StrategyQAModel' or data == 'ECQAModel', "only works for StrategyQA"
    jsonl_file = os.path.join(data_dir, f'{split}.jsonl')
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    labels = [__LABEL_TO_ANSWER__[d['answer']] for d in data]
    questions = [d['question'] for d in data]
    id = [d['qid'] for d in data] if 'qid' in data[0].keys() else [i for i in range(len(data))]
    human_exp = [' '.join(d['facts']) for d in data]
    choice_0 = ['yes'] * len(data)
    choice_1 = ['no'] * len(data)
    df = pd.DataFrame({'id': id, 'question': questions, 'choice_0': choice_0, 'choice_1': choice_1, 'human_exp': human_exp, 'label': labels})
    df.to_csv(os.path.join(target_data_dir, f'{target_split}.csv'), index = False)

def convert_ecqa_dataset_to_csv(data, data_dir, target_data_dir, split, target_split):
    if "generated_rationales" not in data_dir:
        data = datasets.load_from_disk(os.path.join('data/processed_datasets/ecqa', split))
        id = [d['q_no'] for d in data] if 'q_no' in data[0].keys() else [i for i in range(len(data))]
        choice_0 = [d['q_op1'] for d in data]
        choice_1 = [d['q_op2'] for d in data]
        choice_2 = [d['q_op3'] for d in data]
        choice_3 = [d['q_op4'] for d in data]
        choice_4 = [d['q_op5'] for d in data]
        gold_answer = [d['q_ans'] for d in data]
        # compare gold_answer to choices and get label
        labels = []
        for i in range(len(data)):
            if gold_answer[i] == choice_0[i]:
                labels.append(0)
            elif gold_answer[i] == choice_1[i]:
                labels.append(1)
            elif gold_answer[i] == choice_2[i]:
                labels.append(2)
            elif gold_answer[i] == choice_3[i]:
                labels.append(3)
            elif gold_answer[i] == choice_4[i]:
                labels.append(4)
            else:
                raise ValueError
        questions = [d['q_text'] for d in data]
        human_exp = [f"{d['taskA_pos']} {d['taskA_neg']}" for d in data]
    else:
        jsonl_file = os.path.join(data_dir, f'{split}.jsonl')
        with open(jsonl_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
        id, questions, choice_0, choice_1, choice_2, choice_3, choice_4, labels, gold_answer, human_exp = [], [], [], [], [], [], [], [], [], []
        for i, example in enumerate(data):
            question = example['question']
            options = question.split(" Options: ")[1].split(", ")
            assert len(options) == 5
            question = question.split(" Options: ")[0]
            answer = example['answer']
            exp = ' '.join(example['facts'])

            id.append(i)
            questions.append(question)
            choice_0.append(options[0])
            choice_1.append(options[1])
            choice_2.append(options[2])
            choice_3.append(options[3])
            choice_4.append(options[4])
            gold_answer.append(answer)
            human_exp.append(exp)

        for i in range(len(data)):
            if gold_answer[i] == choice_0[i]:
                labels.append(0)
            elif gold_answer[i] == choice_1[i]:
                labels.append(1)
            elif gold_answer[i] == choice_2[i]:
                labels.append(2)
            elif gold_answer[i] == choice_3[i]:
                labels.append(3)
            elif gold_answer[i] == choice_4[i]:
                labels.append(4)
            else:
                raise ValueError

    df = pd.DataFrame({'id': id, 
                       'question': questions, 
                       'choice_0': choice_0, 
                       'choice_1': choice_1, 
                       'choice_2': choice_2,
                       'choice_3': choice_3,
                       'choice_4': choice_4,
                       'human_exp': human_exp,
                       'label': labels,
                       'gold_answer': gold_answer})

    df.to_csv(os.path.join(target_data_dir, f'{target_split}.csv'), index = False)

def convert_cose_dataset_to_csv(data, data_dir, target_data_dir, split, target_split):
    data = datasets.load_from_disk(os.path.join(data_dir, split))
    id = [d['id'] for d in data]
    question = [d['question'] for d in data]
    choice_0 = [d['choices'][0] for d in data]
    choice_1 = [d['choices'][1] for d in data]
    choice_2 = [d['choices'][2] for d in data]
    choice_3 = [d['choices'][3] for d in data]
    choice_4 = [d['choices'][4] for d in data]
    gold_answer = [d['answer'] for d in data]
    # compare gold_answer to choices and get label
    labels = []
    for i in range(len(data)):
        if gold_answer[i] == choice_0[i]:
            labels.append(0)
        elif gold_answer[i] == choice_1[i]:
            labels.append(1)
        elif gold_answer[i] == choice_2[i]:
            labels.append(2)
        elif gold_answer[i] == choice_3[i]:
            labels.append(3)
        elif gold_answer[i] == choice_4[i]:
            labels.append(4)
        else:
            raise ValueError
    human_exp = [d['abstractive_explanation'] for d in data]
    df = pd.DataFrame({'id': id,
                          'question': question,
                          'choice_0': choice_0,
                          'choice_1': choice_1,
                          'choice_2': choice_2,
                          'choice_3': choice_3,
                          'choice_4': choice_4,
                          'human_exp': human_exp,
                          'label': labels,
                          'gold_answer': gold_answer})
    df.to_csv(os.path.join(target_data_dir, f'{target_split}.csv'), index = False)

def load_vacuous_rationale_strategyqa(split):
    data = datasets.load_from_disk(os.path.join('/home/ylu130/workspace/REV-reimpl/data/processed_datasets/strategyqa', split))
    vacuous_rationale = [d['vacuous_rationale'] for d in data]

    return vacuous_rationale

def load_vacuous_ecqa(split):
    data = datasets.load_from_disk(os.path.join('/home/ylu130/workspace/REV-reimpl/data/processed_datasets/ecqa', split))
    vacuous_rationale = [d[f'vacuous_rationale_{d["label"]}'] for d in data]

    return vacuous_rationale

def read_ecqa(args, input_file, explanations_to_use, version, 
            labels_to_use = 'label', filter_explanations = None, split=None):

    df = pd.read_csv(input_file)
    df = df.applymap(removeNonAscii)
    n = len(df) if not args.small_data else args.small_size
    num_choices = 5
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)
    # simulate_rationalized is used to pull out the predicted explanation when simulating a CAGE-Ra model
    simulate_rationalized = (args.condition_on_explanations and not args.multi_explanation and 'st.ra' in (labels_to_use.lower() if isinstance(labels_to_use, str) else '' ))

    ids = df['id']
    questions = df['question']
    choice_cols = [f'choice_{i}' for i in range(num_choices)]
    choices = df[choice_cols]    
    labels = df[labels_to_use] if labels_to_use is not None else [0] * n
    print("using labels: %s" % labels_to_use)    

    if explanations_to_use != 'ground_truth' and explanations_to_use not in df.columns:
        vacuous_rationales = load_vacuous_ecqa(split)
        template = __TEMPLATES__[explanations_to_use]

        explanations = [template.format(
                        gold_rationale=df.loc[i, 'human_exp'],
                        base_rationale=vacuous_rationales[i],
                        leaky_rationale=f"The answer is {df.loc[i, 'gold_answer']}")
                    for i in range(len(df))]
        df[explanations_to_use] = explanations
        df.to_csv(input_file, index=False)
    else:
        exp_cols = explanations_to_use if explanations_to_use != 'ground_truth' else 'human_exp'
        explanations = df[exp_cols]
    
    print(f"getting explanations from {explanations_to_use}")

    # pick out the predicted explanations, according to the task model's prediction 
    if simulate_rationalized:
        print("picking out predicted explanations")
        explanations = [explanations.loc[i,exp_cols[label]] for i, label in enumerate(labels)]

    examples = [QAExample(qa_id = ids[i],
                        question = questions[i],
                        choices = choices.iloc[i].tolist(),
                        explanation = explanations[i],
                        label = labels[i],
                        num_choices = num_choices) 
               for i in range(n)]

    # filter pre-specified bad explanations (e.g. bad explanations in v1.1 data). see https://github.com/salesforce/cos-e/issues/2
    if filter_explanations is not None:
        examples = [ex for ex in examples if not ex.explanation in filter_explanations]

    return examples

def read_strategyqa(args, input_file, explanations_to_use, version, 
            labels_to_use = 'label', filter_explanations = None, split=None):

    df = pd.read_csv(input_file)
    df = df.applymap(removeNonAscii)
    n = len(df) if not args.small_data else args.small_size
    num_choices = 2 if version == '1.0' else 5
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)
    # simulate_rationalized is used to pull out the predicted explanation when simulating a CAGE-Ra model
    simulate_rationalized = (args.condition_on_explanations and not args.multi_explanation and 'st.ra' in (labels_to_use.lower() if isinstance(labels_to_use, str) else '' ))

    ids = df['id']
    questions = df['question']
    choice_cols = [f'choice_{i}' for i in range(num_choices)]
    choices = df[choice_cols]    
    labels = df[labels_to_use] if labels_to_use is not None else [0] * n
    print("using labels: %s" % labels_to_use)    

    if explanations_to_use != 'ground_truth' and explanations_to_use not in df.columns:
        vacuous_rationales = load_vacuous_rationale_strategyqa(split)
        template = __TEMPLATES__[explanations_to_use]
        explanations = [template.format(
                        gold_rationale=df.loc[i, 'human_exp'],
                        base_rationale=vacuous_rationales[i],
                        leaky_rationale=__LABEL_TO_LEAKY_RATIONALE__[df.loc[i, 'label']])
                    for i in range(len(df))]
        df[explanations_to_use] = explanations
        df.to_csv(input_file, index=False)
    else:
        exp_cols = explanations_to_use if explanations_to_use != 'ground_truth' else 'human_exp'
        explanations = df[exp_cols]
    
    print(f"getting explanations from {explanations_to_use}")

    # pick out the predicted explanations, according to the task model's prediction 
    if simulate_rationalized:
        print("picking out predicted explanations")
        explanations = [explanations.loc[i,exp_cols[label]] for i, label in enumerate(labels)]

    examples = [QAExample(qa_id = ids[i],
                        question = questions[i],
                        choices = choices.iloc[i].tolist(),
                        explanation = explanations[i],
                        label = labels[i],
                        num_choices = num_choices) 
               for i in range(n)]

    # filter pre-specified bad explanations (e.g. bad explanations in v1.1 data). see https://github.com/salesforce/cos-e/issues/2
    if filter_explanations is not None:
        examples = [ex for ex in examples if not ex.explanation in filter_explanations]

    return examples

def read_CQA(args, input_file, explanations_to_use, version, 
            labels_to_use = 'label', filter_explanations = None):

    df = pd.read_csv(input_file)
    df = df.applymap(removeNonAscii)
    n = len(df) if not args.small_data else args.small_size
    num_choices = 3 if version == '1.0' else 5
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)
    # simulate_rationalized is used to pull out the predicted explanation when simulating a CAGE-Ra model
    simulate_rationalized = (args.condition_on_explanations and not args.multi_explanation and 'st.ra' in (labels_to_use.lower() if isinstance(labels_to_use, str) else '' ))

    # if test data, make sure explanations_to_use isn't ground_truth or oracle
    if 'test' in input_file and (explanations_to_use == 'ground_truth' or explanations_to_use == 'oracle'):
        explanations_to_use = 'None'

    ids = df['id']
    questions = df['question']
    choice_cols = [f'choice_{i}' for i in range(num_choices)]
    choices = df[choice_cols]    
    labels = df[labels_to_use] if labels_to_use is not None else [0] * n
    print("using labels: %s" % labels_to_use)    

    if explanations_to_use == 'None':
        explanations = [''] * n
    else:
        exp_cols = explanations_to_use        
        try:
            explanations = df[exp_cols]
            print(f"getting explanations from {explanations_to_use}")
        except:            
            if explanations_to_use == 'ground_truth':
                exp_cols = 'human_exp' if 'human_exp' in df.columns else 'human_expl_open-ended'
            elif explanations_to_use == 'oracle':
                exp_cols = 'human_exp' if 'human_exp' in df.columns else 'human_expl_open-ended'
            elif explanations_to_use == 'gpt':
                exp_cols = 'gpt'
            elif explanations_to_use == 'gpt2':
                exp_cols = 'gpt2'
            elif explanations_to_use == 'multi_gpt2':
                exp_cols = [f'gpt2_exps_{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5':
                exp_cols = 't5-single-exp'
            elif explanations_to_use == 'MT_t5':
                exp_cols = 't5-MT-single-exp'
            elif explanations_to_use == 'multi_t5':
                exp_cols = [f't5-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5':
                exp_cols = [f't5-MT-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5_pred':
                exp_cols = 't5-MT-multi-exp-pred'   
            elif explanations_to_use == 'bert_cage':
                exp_cols = 'bert-cage-single-exp'
            elif explanations_to_use == 'bert_multi_cage':
                exp_cols = [f'bert-cage-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5-agent-re':
                exp_cols = 't5-agent-re-exp'
            elif explanations_to_use == 't5-agent-ra':
                exp_cols = 't5-agent-ra-exp'
            # ST (task or simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' not in explanations_to_use:
                exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            # MT (simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' in explanations_to_use:
                exp_cols = [f't5-MT-multi-exp-pred-seed{args.seed}' for i in range(num_choices)]
            print(f"getting explanations from {exp_cols}")
            explanations = df[exp_cols]

    # pick out the predicted explanations, according to the task model's prediction 
    if simulate_rationalized:
        print("picking out predicted explanations")
        explanations = [explanations.loc[i,exp_cols[label]] for i, label in enumerate(labels)]

    examples = [QAExample(qa_id = ids[i],
                        question = questions[i],
                        choices = choices.iloc[i].tolist(),
                        explanation = explanations.iloc[i].tolist() if multi_exp else explanations[i],
                        label = labels[i],
                        num_choices = num_choices) 
               for i in range(n)]

    # filter pre-specified bad explanations (e.g. bad explanations in v1.1 data). see https://github.com/salesforce/cos-e/issues/2
    if filter_explanations is not None:
        examples = [ex for ex in examples if not ex.explanation in filter_explanations]

    return examples

def get_tensors_for_T5_split(args, examples, tokenizer, max_seq_length : int, condition_on_explanations : bool, multi_explanation : bool,
                             spliced_explanation_len = None, explanations_only = False):
    """
    Converts a list of QAExamples into features for use with T5.

    Spliced explanation len is used in 2-agent setup, where input_ids are spliced into with sampled explanations from a model. (need to leave enough room for this)

    Format:
        Sequence 1: "[task/explain]: What is the answer to this question? The choices are choice0, choice1, choice2."
        Task Sequence 2: "The answer is: {answer}"
        Exp. Sequence 2: "The answer is {choice} because {explanation}"

    Note:
        tensor_ids serves as input_ids to model.forward
        tensors_labels serves as lm_labels to model.forward
                
    Returns: list of tensors
        
    """
    input_padding_id = tokenizer.pad_token_id   
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    task_prefix_ids = tokenizer.encode("task:", add_special_tokens = False)
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens = False)

    return_data = []

    for example_index, example in enumerate(examples):

        # per-question variables
        question_str = example.question
        choices_str = example.choices_str
        answer_str = example.choices[example.label]
        explanation_str = example.explanation
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'
        choice_label = example.label
        task_input_ids_list = []
        task_output_ids_list = []
        task_output_labels_list = []
        explanation_context_ids_list = []

        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing before words (rather than adding all the ids)
        input_str = f"[CLS] {question_str} {choices_str} [SEP]" 
        if spliced_explanation_len is not None:
            cap_length = max_seq_length-len(task_prefix_ids)-spliced_explanation_len
        else:
            cap_length = max_seq_length-len(task_prefix_ids)

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length 
            question_tokens = tokenizer.encode(question_str)
            keep_up_to = len(question_tokens) - over_by - 1  # leaves buffer question mark below
            new_question_tokens = question_tokens[:keep_up_to]
            question_str = tokenizer.decode(new_question_tokens) + '?'
            # print("Trimmed a question by %d tokens" % (len(question_tokens) - len(new_question_tokens)))
            # print("OLD:", tokenizer.decode(question_tokens))
            # print("NEW:", question_str)
            # print()

        # in explanations only, remove the question
        if explanations_only:
            question_str = ""

        # get string formats
        if not condition_on_explanations:
            input_str = f"[CLS] {question_str} {choices_str} [SEP]" 
        if condition_on_explanations and not multi_explanation:
            input_str = f"[CLS] {question_str} {choices_str} [SEP] My commonsense tells me {explanation_str}"
        elif condition_on_explanations and multi_explanation:
            # make task_input_ids in answer loop below
            input_str = ""
        task_answer_str = f"The answer is: {answer_str}"
        explanation_output_str = f"The answer is {answer_str} because {explanation_str}" \
                                    if multi_explanation \
                                    else \
                                 f"My commonsense tells me that {explanation_str}"

        # get token_ids 
        _input_ids = tokenizer.encode(input_str, add_special_tokens = False)
        task_input_ids = task_prefix_ids + _input_ids 
        explanation_input_ids = explanation_prefix_ids + _input_ids
        explanation_only_ids = tokenizer.encode(example.explanation, add_special_tokens = False)
        _task_answer_ids = tokenizer.encode(task_answer_str, add_special_tokens = False)
        _explanation_output_ids = tokenizer.encode(explanation_output_str, add_special_tokens = False) + [eos_token_id]

        _truncate_seq_pair(task_input_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_input_ids, [], max_seq_length)
        _truncate_seq_pair(_explanation_output_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_only_ids, [], max_seq_length)
    
        for choice_index, choice in enumerate(example.choices):

            if condition_on_explanations and multi_explanation:                
                if len(example.explanation_list) > 1:
                    explanation_str = example.explanation_list[choice_index]            
                else:
                    explanation_str = ''
                task_input_str = f"[CLS] {question_str} {choices_str} [SEP] The answer is {choice} because {explanation_str}"
                task_input_ids = task_prefix_ids + tokenizer.encode(task_input_str, add_special_tokens = False)
                _truncate_seq_pair(task_input_ids, [], max_seq_length)
                ids_padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
                task_input_ids += ids_padding
                task_input_ids_list.append(task_input_ids)

            task_output_str = f"The answer is: {choice}"    
            _task_output_ids = tokenizer.encode(task_output_str, add_special_tokens = False)    
            ids_padding = [input_padding_id] * (max_seq_length - len(_task_output_ids))
            labels_padding = [label_padding_id] * (max_seq_length - len(_task_output_ids))
            task_output_ids = _task_output_ids + ids_padding
            task_output_labels = _task_output_ids + labels_padding
            task_output_ids_list.append(task_output_ids)
            task_output_labels_list.append(task_output_labels)

            explanation_context_str = f"The answer is {choice} because" \
                                        if multi_explanation \
                                        else \
                                      f"My commonsense tells me that"
            explanation_context_ids = tokenizer.encode(explanation_context_str, add_special_tokens = False)    
            if choice == answer_str: 
                context_len = len(explanation_context_ids)
            explanation_context_ids += [input_padding_id] * (max_seq_length - len(explanation_context_ids))
            _truncate_seq_pair(explanation_context_ids, [], max_seq_length)
            explanation_context_ids_list.append(explanation_context_ids)
            
        # pad up to the max sequence len. NOTE input_padding_id goes on inputs to either the encoder or decoder. label_padding_id goes on lm_labels for decode
        padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
        task_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_input_ids))
        explanation_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_only_ids))
        explanation_only_ids += padding

        # store explanation_len for dropout/masking purposes
        explanation_len = len([e for e in explanation_context_ids if e != input_padding_id]) + len([e for e in explanation_only_ids if e != input_padding_id]) 
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_task_answer_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_task_answer_ids))
        task_answer_ids = _task_answer_ids + ids_padding
        task_answer_labels = _task_answer_ids + labels_padding
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_explanation_output_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_explanation_output_ids))
        explanation_output_ids = _explanation_output_ids + ids_padding
        explanation_output_labels = _explanation_output_ids + labels_padding
        explanation_output_labels[:context_len] = [label_padding_id]*context_len # no LM loss on the explanation_context_str 
        
        # make into tensors and accumulate
        task_input_ids = torch.tensor(task_input_ids if len(task_input_ids_list) < 1 else task_input_ids_list, dtype = torch.long)
        task_input_masks = (task_input_ids!=input_padding_id).float()
        task_answer_ids = torch.tensor(task_answer_ids, dtype = torch.long)
        task_answer_masks = (task_answer_ids!=input_padding_id).float()
        task_answer_labels = torch.tensor(task_answer_labels, dtype = torch.long)
        task_output_ids = torch.tensor(task_output_ids_list, dtype = torch.long)
        task_output_masks = (task_output_ids!=input_padding_id).float()
        task_output_labels = torch.tensor(task_output_labels_list, dtype = torch.long)
        explanation_input_ids = torch.tensor(explanation_input_ids, dtype = torch.long)
        explanation_input_masks = (explanation_input_ids!=input_padding_id).float()        
        explanation_output_ids = torch.tensor(explanation_output_ids, dtype = torch.long)
        explanation_output_masks = (explanation_output_ids!=input_padding_id).float()
        explanation_output_labels = torch.tensor(explanation_output_labels, dtype = torch.long)
        explanation_context_ids = torch.tensor(explanation_context_ids_list, dtype = torch.long)
        task_choice_label = torch.tensor(choice_label, dtype = torch.long)
        explanation_only_ids = torch.tensor(explanation_only_ids, dtype = torch.long)
        explanation_len = torch.tensor(explanation_len).long()
        
        data_point = [task_input_ids, task_input_masks, 
                      task_answer_ids, task_answer_masks, task_answer_labels,
                      task_output_ids, task_output_masks, task_output_labels, task_choice_label,
                      explanation_input_ids, explanation_input_masks,
                      explanation_output_ids, explanation_output_masks, explanation_output_labels,
                      explanation_context_ids, explanation_only_ids, explanation_len]
        return_data.append(data_point)

    # now reshape list of lists of tensors to list of tensors
    n_cols = len(return_data[0])
    return_data = [torch.stack([data_point[j] for data_point in return_data], dim=0) for j in range(n_cols)]

    return return_data



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()