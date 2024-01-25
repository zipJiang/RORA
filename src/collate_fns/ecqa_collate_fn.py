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
from ..utils.ngrams import generate_no_more_than_ngrams
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
    
    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)

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
        
        return self.__TEMPLATES__[self.rationale_format].format(
            gold_rationale=gold_rationale,
            base_rationale=base_rationale,
            leaky_rationale=leaky_rationale
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {self.question_templating(item)} rationale: {self.rationale_templating(item)}"
    
    def get_label_index(
        self,
        item: Dict[Text, Any]
    ) -> int:
        """We will get items that are relevant to the label.
        providing the original item in the dataset.
        """

        return [item[f'q_op{i}'] for i in range(1, 6)].index(item['q_ans'])
    
    
@CollateFn.register("ecqa-ngram-classification-collate-fn")
class ECQAClassificationCollateFn(VocabularizerMixin, ECQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        vocab: torchtext.vocab.Vocab,
        max_input_length: Optional[int] = 256,
        nlp_model: Optional[Text] = "en_core_web_sm",
        num_ngrams: Optional[int] = 2,
        pad_token: Optional[Text] = 'korpaljo',
        rationale_only: Optional[bool] = False,
        included_keys: Optional[List[Text]] = None
    ):
        super().__init__(
            vocab=vocab,
            nlp_model=nlp_model,
            rationale_format=rationale_format
        )
        self.max_input_length = max_input_length
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

        tknzd = self.vocabularize(input_strs, max_length=self.max_input_length)
        
        # also need to vocabularize the rationales
        # use the same encoding to allow batched processing
        choices = [self.vocabularize([f"option {i} : " + item[f'q_op{i}'] for i in range(1, 6)], max_length=self.max_input_length) for item in x]
        
        input_ids = [[tknzd_single, *choice] for tknzd_single, choice in zip(tknzd, choices)]
        
        kwargs = {}
        if self.included_keys is not None:
            kwargs = {k: [item[k] for item in x] for k in self.included_keys}
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64).view(-1, self.max_input_length),
            'labels': torch.tensor(
                [
                    self.get_label_index(item) for item in x
                ],
                dtype=torch.int64
            ),
            "kwargs": kwargs
        }
        
        
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

        tknzd = self.sequential_vocabularize(input_strs, max_length=self.max_input_length)
        lengths = self.get_lengths(input_strs, max_length=self.max_input_length)
        
        # also need to vocabularize the rationales
        # use the same encoding to allow batched processing
        choices = [
            self.sequential_vocabularize(
            [
                f"option {i} : " + item[f'q_op{i}'] for i in range(1, 6)
            ], max_length=self.max_output_length) for item in x
        ]
        
        choice_lengths = [self.get_lengths([f"option {i} : " + item[f'q_op{i}'] for i in range(1, 6)], max_length=self.max_output_length) for item in x]
        
        kwargs = {}
        if self.included_keys is not None:
            kwargs = {k: [item[k] for item in x] for k in self.included_keys}
        
        return {
            'input_ids': torch.tensor(tknzd, dtype=torch.int64).view(-1, self.max_input_length),
            "lengths": torch.tensor(lengths, dtype=torch.int64).view(-1),
            'choices': torch.tensor(choices, dtype=torch.int64).view(-1, self.max_output_length),
            "choices_lengths": torch.tensor(choice_lengths, dtype=torch.int64).view(-1),
            'labels': torch.tensor(
                [
                    self.get_label_index(item) for item in x
                ],
                dtype=torch.int64
            ),
            "kwargs": kwargs
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
            padding=True,
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
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
        }
        
        
@CollateFn.register("ecqa-generation-collate-fn")
class ECQAGenerationCollateFn(ECQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def option_templating(self, item: Dict[Text, Any]) -> Text:
        """Generate the option template.
        """
        return ' '.join([f"({i}) " + item[f'q_op{i}'] for i in range(1, 6)])
    

    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {self.question_templating(item)} options: {self.option_templating(item)} rationale: {self.rationale_templating(item)} answer: "
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) for item in x
        ]
        
        input_outputs = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding=True,
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
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
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
            padding=True,
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
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids # [batch_size * 5, max_output_length]
        
        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
        }