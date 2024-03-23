"""ECQA Collate functions that runs on ECQA.
"""
import re
import torchtext
from overrides import overrides
import torch
from typing import Dict, Text, Any, Optional, List
from transformers import PreTrainedTokenizer
from .collate_fn import (
    CollateFn,
    VocabularizerMixin,
    SpuriousRemovalMixin
)
from ..utils.templating import __TEMPLATES__


def retrieve_vacuous(item) -> Text:
    options = [item[f"q_op{i}"] for i in range(1, 6)]
    try:
        index = options.index(item['q_ans'])
    except ValueError:
        raise ValueError(f"Answer not found in options: {item['q_ans']}")
    
    return item[f"vacuous_rationale_op{index + 1}"]
        

@CollateFn.register("ecqa-collate-fn")
class ECQACollateFn(CollateFn):
    
    __TEMPLATES__: Dict[Text, Text] = __TEMPLATES__
    
    def __init__(
        self,
        rationale_format: Text,
        rationale_only: Optional[bool] = False
    ):
        super().__init__(rationale_format=rationale_format)
        self.rationale_only = rationale_only

    def question_templating(self, item: Dict[Text, Any]) -> Text:
        """Create the question template from the item.
        """
        return item['q_text']

    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Create the rationle template from the item.
        """
        
        # we need to retrieve the vacuous rationale
        # according to the rationale_format
        gold_rationale = item['taskB']
        base_rationale = retrieve_vacuous(item)
        leaky_rationale = f"The answer is: option {self.get_label_index(item) + 1}"
        option_rationale = ' '.join(f"option {i} : " + item[f'q_op{i}'] for i in range(1, 6))
        
        # we need to retrieve all the candidate rationales
        candidates = {key: value for key, value in item.items() if key.endswith("_rationale")}
        
        return self.__TEMPLATES__[self.rationale_format].format(
            gold_rationale=gold_rationale,
            base_rationale=base_rationale,
            leaky_rationale=leaky_rationale,
            option_rationale=option_rationale,
            **candidates
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        if self.rationale_only:
            return self.rationale_templating(item)
        return f"question: {self.question_templating(item)} rationale: {self.rationale_templating(item)}"
    
    def get_label_index(
        self,
        item: Dict[Text, Any]
    ) -> int:
        """We will get items that are relevant to the label.
        providing the original item in the dataset.
        """

        return [item[f'q_op{i}'] for i in range(1, 6)].index(item['q_ans'])
        
        
@CollateFn.register("ecqa-lstm-classification-collate-fn")
class ECQALstmClassificationCollateFn(VocabularizerMixin, ECQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        vocab: torchtext.vocab.Vocab,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        nlp_model: Optional[Text] = "en_core_web_sm",
        num_ngrams: Optional[int] = 2,
        pad_token: Optional[Text] = '<pad>',
        rationale_only: Optional[bool] = False,
        included_keys: Optional[List[Text]] = None
    ):
        super().__init__(
            vocab=vocab,
            nlp_model=nlp_model,
            rationale_format=rationale_format
        )
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_ngrams = num_ngrams
        self.pad_token = pad_token
        self.pad_token_id = self.vocab[self.pad_token]
        self.rationale_only = rationale_only
        self.included_keys = included_keys

    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """This will be very similar to the collate function in the
        StrategyQACase, but we need to add the rationales from
        the ECQA dataset.
        """
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) if not self.rationale_only else self.rationale_templating(item) for item in x
        ]

        tknzd: List[List[Dict[Text, Any]]] = self.sequential_tokenize(input_strs)
        
        lengths = self.get_lengths(tknzd, max_length=self.max_input_length)
        
        # also need to vocabularize the rationales
        # use the same encoding to allow batched processing
        all_choices: List[List[Dict[Text, Any]]] = [
            self.sequential_tokenize(
            [
                f"option {i} : " + item[f'q_op{i}'] for i in range(1, 6)
            ]) for item in x
        ]
        
        choice_lengths = [self.get_lengths(choices, max_length=self.max_output_length) for choices in all_choices]
        
        kwargs = {}
        if self.included_keys is not None:
            kwargs = {k: [item[k] for item in x] for k in self.included_keys}
        
        return {
            '_input_ids': torch.tensor(self.vocabularize_and_pad(tknzd, self.max_input_length), dtype=torch.int64),
            "_lengths": torch.tensor(lengths, dtype=torch.int64),
            '_choices': torch.tensor([self.vocabularize_and_pad(choices, self.max_output_length) for choices in all_choices], dtype=torch.int64),
            "_choices_lengths": torch.tensor(choice_lengths, dtype=torch.int64),
            '_labels': torch.tensor(
                [
                    self.get_label_index(item) for item in x
                ],
                dtype=torch.int64
            ),
            # "kwargs": kwargs
            "tokenized_inputs": tknzd,
            "tokenized_choices": all_choices,
        }
        
        
@CollateFn.register("ecqa-infilling-collate-fn")
class ECQAInfillingCollateFn(SpuriousRemovalMixin, ECQACollateFn):
    """Unlike StrategyQA Case, we need to use a different
    collate function for generative training, because in generation
    we only need to provide the input and the target for the
    gold answer.
    """
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 256,
        removal_threshold: Optional[float] = None,
        intervention_on_label: Optional[bool] = False
    ):
        super().__init__(
            rationale_format=rationale_format,
            removal_threshold=removal_threshold,
            mask_by_delete=False
        )
        self.special_token_pattern = re.compile(r"<extra_id_\d+>")
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.intervention_on_label = intervention_on_label
        
    @overrides
    def question_templating(self, item: Dict[Text, Any]) -> Text:
        """Create the question template from the item.
        """
        return item['q_text'] + ' ' + ' '.join([f"op{i}: " + item[f'q_op{i}'] for i in range(1, 6)])
        
    def dynamic_templating(self, item: Dict[Text, Any], op_index: Optional[int] = None) -> Text:
        """The difference here is that we foreground the answer
        field to input.
        """
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=item[f'q_op{op_index}'] if op_index is not None else item['q_ans'],
            question=self.question_templating(item),
            rationale=self.remove_spurious(self.rationale_templating(item), attributions=item['attributions'])
        )
    
    def dynamic_non_removal_templating(self, item: Dict[Text, Any], op_index: Optional[int] = None) -> Text:
        """We need a non-removal templating for the generation,
        so that we can compare the output with the original.
        """
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=item[f'q_op{op_index}'] if op_index is not None else item['q_ans'],
            question=self.question_templating(item),
            rationale=self.rationale_templating(item)
        )
        
    def dynamic_get_offsets(self, item: Dict[Text, Any], op_index: Optional[int] = None) -> int:
        """Calculate the offset by templating non-removal-no-rationale.
        """
        
        non_removal_no_rationale_template = "answer: {answer} question: {question} rationale: ".format(
            answer=item[f"q_op{op_index}"] if op_index is not None else item['q_ans'],
            question=self.question_templating(item),
        )

        return len(non_removal_no_rationale_template)
    
    def dynamic_label_templating(self, item: Dict[Text, Any], op_index: Optional[int] = None) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        
        return self.retain_spurious(
            self.dynamic_non_removal_templating(item, op_index),
            attributions=item["attributions"],
            offsets=self.dynamic_get_offsets(item, op_index)
        )
    
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        _flatten = lambda x: [item for sublist in x for item in sublist]
        
        # construct prompt and target
        if self.intervention_on_label:
            input_strs: List[Text] = _flatten(
                [
                    [self.dynamic_templating(item, op_index=i) for i in range(1, 6)] for item in x
                ]
            )
        else:
            input_strs: List[Text] = [
                self.dynamic_templating(item) for item in x
            ]
        
        input_outputs = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_outputs.input_ids
        attention_mask = input_outputs.attention_mask
        
        labels = self.tokenizer(
            _flatten(
                [
                    [self.dynamic_label_templating(item, i) for i in range(1, 6)] for item in x
                ]
            ) if self.intervention_on_label else [
                self.dynamic_label_templating(item) for item in x
            ],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            '_input_ids': input_ids if not self.intervention_on_label else input_ids.view(-1, 5, self.max_input_length),
            "_attention_mask": attention_mask if not self.intervention_on_label else attention_mask.view(-1, 5, self.max_input_length),
            '_labels': labels if not self.intervention_on_label else labels.view(-1, 5, self.max_output_length),
        }
        
        
@CollateFn.register("ecqa-generation-collate-fn")
class ECQAGenerationCollateFn(ECQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        intervention_on_label: Optional[bool] = False,
        rationale_only: Optional[bool] = False
    ):
        """
        """
        # print(super().__init__)
        super().__init__(rationale_format=rationale_format, rationale_only=rationale_only)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.intervention_on_label = intervention_on_label
        
    def option_templating(self, item: Dict[Text, Any]) -> Text:
        """Generate the option template.
        """
        return ' '.join([f"({i}) " + item[f'q_op{i}'] for i in range(1, 6)])
    

    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        if self.rationale_only:
            return f"{self.rationale_templating(item)} answer: "
        return f"question: {self.question_templating(item)} options: {self.option_templating(item)} rationale: {self.rationale_templating(item)} answer: "
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        if not self.intervention_on_label:
            # construct prompt and target
            input_strs: List[Text] = [
                self.templating(item) for item in x
            ]
            
            input_outputs = self.tokenizer(
                input_strs,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = input_outputs.input_ids
            attention_mask = input_outputs.attention_mask
            
            labels = self.tokenizer(
                [
                    item["q_ans"] for item in x
                ],
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids

            labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id

        else:
            
            def _flatten(lst):
                return [item for sublist in lst for item in sublist]
            
            def _reorder(item, index):
                # reorder the options to get the correct answer
                # to the first place.
                return [item[f'q_op{index + 1}']] + [item[f'q_op{i}'] for i in range(1, 6) if i != index + 1]

            input_strs: List[Text] = _flatten([
                [self.templating(item)] * 5 for item in x
            ])
            
            input_outputs = self.tokenizer(
                input_strs,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = input_outputs.input_ids
            attention_mask = input_outputs.attention_mask
            
            labels = self.tokenizer(
                _flatten(
                    [
                        _reorder(item, self.get_label_index(item)) for item in x
                    ]
                ),
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
        return {
            '_input_ids': input_ids.view(-1, 5, self.max_input_length) if self.intervention_on_label else input_ids,
            "_attention_mask": attention_mask.view(-1, 5, self.max_input_length) if self.intervention_on_label else attention_mask,
            '_labels': labels.view(-1, 5, self.max_output_length) if self.intervention_on_label else labels,
        }
        
        
@CollateFn.register("ecqa-irm-collate-fn")
class ECQAIRMCollateFn(ECQAGenerationCollateFn):
    """The problem here is that we try to make sure
    that the we always have all output in the same
    batch.
    
    As an additional benefit, we're able to keep
    using the generation collate function.
    """
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
    ):
        """rationale_format is for compatibility with other collate functions
        and logging purposes.
        """
        
        super().__init__(
            rationale_format=rationale_format,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        
    def dynamic_templating(
        self,
        item: Dict[Text, Any],
        op_index: int
    ) -> Text:
        """Given an op_index, generate the rationale template
        that will be used for the (pseudo) IRM task.
        """
        
        rationale = item[f'generated_rationale_op{op_index}']
        return f"question: {self.question_templating(item)} options: {self.option_templating(item)} rationale: {rationale} answer: "
    
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        _flatten = lambda lst: [item for sublist in lst for item in sublist]
        
        input_strs: List[Text] = _flatten(
            [
                [self.dynamic_templating(item, op_index=i) for i in range(1, 6)] for item in x
            ]
        )
        
        input_outputs = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_outputs.input_ids
        attention_mask = input_outputs.attention_mask
        
        labels = self.tokenizer(
            _flatten(
                [
                    [item["q_ans"] for _ in range(1, 6)] for item in x
                ],
            ),
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids # [batch_size * 5, max_output_length]
        
        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            '_input_ids': input_ids.view(-1, 5, self.max_input_length),
            "_attention_mask": attention_mask.view(-1, 5, self.max_input_length),
            '_labels': labels.view(-1, 5, self.max_output_length),
        }