from typing import Dict, Any, Text, Optional, List, Tuple
import torch
import spacy
from transformers import PreTrainedTokenizer
import torchtext
import re
from .collate_fn import (
    CollateFn,
    VocabularizerMixin,
    SpuriousRemovalMixin
)
from ..utils.templating import __TEMPLATES__
from ..utils.ngrams import generate_no_more_than_ngrams
from overrides import overrides
from registrable import Lazy


__LABEL_TO_ANSWER__ = {
    True: "yes",
    False: "no"
}



__LABEL_TO_LEAKY_RATIONALE__ = {
    True: f"The answer is {__LABEL_TO_ANSWER__[True]}",
    False: f"The answer is {__LABEL_TO_ANSWER__[False]}"
}

@CollateFn.register("strategyqa-collate-fn")
class StrategyQACollateFn(CollateFn):
    
    __TEMPLATES__ = __TEMPLATES__
    __LABEL_TO_ANSWER__ = __LABEL_TO_ANSWER__
    __LABEL_TO_LEAKY_RATIONALE__ = __LABEL_TO_LEAKY_RATIONALE__
    
    def __init__(
        self,
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        template = self.__TEMPLATES__[self.rationale_format]
        
        # supply all rationale candidates
        candidates = {key: value for key, value in item.items() if key.endswith("_rationale")}
        
        return template.format(
            gold_rationale=' '.join(item['facts']),
            base_rationale=item['vacuous_rationale'],
            leaky_rationale=self.__LABEL_TO_LEAKY_RATIONALE__[item['answer']],
            **candidates
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {item['question']} rationale: {self.rationale_templating(item)}"
    
    
# TODO: Now masking is implemented on the GenerationCollateFn,
# But we only need that at the InfillingCollateFn.
@CollateFn.register("strategyqa-generation-collate-fn")
class StrategyQAGenerationCollateFn(StrategyQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        # removal_threshold: Optional[float] = None,
        # mask_by_delete: Optional[bool] = False,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        return "question: {question} rationale: {rationale}".format(
            question=item['question'],
            rationale=self.rationale_templating(item)
        )
        
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_outputs.input_ids
        attention_mask = input_outputs.attention_mask
        
        labels = self.tokenizer(
            [
                self.__LABEL_TO_ANSWER__[item['answer']] for item in x
            ],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        neg_labels = self.tokenizer(
            [
                self.__LABEL_TO_ANSWER__[not item['answer']] for item in x
            ],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        ).input_ids
        
        neg_labels[neg_labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            '_input_ids': input_ids,
            "_attention_mask": attention_mask,
            '_labels': labels,
            "_neg_labels": neg_labels
        }
    
    
@CollateFn.register("strategyqa-irm-collate-fn")
class StrategyQAIRMCollateFn(StrategyQACollateFn):
    """A collate function used to generate IRM.
    training data (basically don't have to process the rationale again).
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        rationale_format: Optional[Text] = "",
    ):
        """rationale_format is for compatibility with other collate functions
        and logging purposes.
        """
        
        super().__init__(
            rationale_format=rationale_format,
        )
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def factual_templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        return "question: {question} rationale: {rationale}".format(
            question=item['question'],
            # rationale=item['factual_rationale']
            rationale=self.rationale_templating(item)
        )
        
    def counterfactual_templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        return "question: {question} rationale: {rationale}".format(
            question=item['question'],
            rationale=item['counterfactual_rationale']
        )
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        result_dict = {}
        for env in ["factual", "counterfactual"]:
            input_strs: List[Text] = [
                getattr(self, f"{env}_templating")(item) for item in x
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
                [
                    self.__LABEL_TO_ANSWER__[item['answer']] for item in x
                ],
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids

            labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            
            neg_labels = self.tokenizer(
                [
                    self.__LABEL_TO_ANSWER__[not item['answer']] for item in x
                ],
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            neg_labels[neg_labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
            result_dict[env] = {
                '_input_ids': input_ids,
                "_attention_mask": attention_mask,
                '_labels': labels,
                "_neg_labels": neg_labels
            }
            
        # combine the result_dict for different environments
        return_dict = {
            key: torch.stack([result_dict[env][key] for env in ['factual', 'counterfactual']], axis=1)
            for key in result_dict['factual'].keys()
        }
        
        return return_dict
    
    
@CollateFn.register("strategyqa-infilling-collate-fn")
class StrategyQAInfillingCollateFn(
    SpuriousRemovalMixin,
    StrategyQAGenerationCollateFn
):
    """This is one of the generation tasks, that requires
    a different masking.
    """
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 256,
        removal_threshold: Optional[float] = None,
        intervention_on_label: Optional[bool] = False,
    ):
        super().__init__(
            rationale_format=rationale_format,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            removal_threshold=removal_threshold,
            mask_by_delete=False
        )
        self.special_token_pattern = re.compile(r"<extra_id_\d+>")
        self._intervention_on_label = intervention_on_label
        
    @property
    def intervention_on_label(self) -> bool:
        return self._intervention_on_label
    
    @intervention_on_label.setter
    def intervention_on_label(self, value: bool):
        self._intervention_on_label = value
    
    @overrides
    def templating(self, item: Dict[Text, Any], intervention: bool = False) -> Text:
        """The difference here is that we foreground the answer
        field to input.
        """
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not intervention else not item['answer']],
            question=item['question'],
            rationale=self.remove_spurious(self.rationale_templating(item), attributions=item['attributions'])
        )
    
    def non_removal_templating(self, item: Dict[Text, Any], intervention: bool = False) -> Text:
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not intervention else not item['answer']],
            question=item['question'],
            rationale=self.rationale_templating(item)
        )

    def non_removal_no_rationale_templating(self, item: Dict[Text, Any], intervention: bool = False) -> Text:
        return "answer: {answer} question: {question} rationale: ".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not intervention else not item['answer']],
            question=item['question'],
        )

    def label_templating(self, item: Dict[Text, Any], intervention: bool = False) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        return self.retain_spurious(
            self.non_removal_templating(item, intervention=intervention),
            attributions=item["attributions"],
            offsets=len(self.non_removal_no_rationale_templating(item, intervention=intervention))
        )
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        if not self.intervention_on_label:
            input_strs: List[Text] = [
                self.templating(item) for item in x
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
                [
                    self.label_templating(item) for item in x
                ],
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            
            return {
                '_input_ids': input_ids,
                "_attention_mask": attention_mask,
                '_labels': labels,
            }

        else:
            # the new semantics of intervention on label will
            # now be having both the counterfactual and the factual.
            
            def _flatten(x):
                return [item for sublist in x for item in sublist]
            
            input_strs: List[Text] = _flatten(
                [
                    [
                        self.templating(item, intervention=False),
                        self.templating(item, intervention=True)
                    ] for item in x
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
                        [
                            self.label_templating(item, intervention=False),
                            self.label_templating(item, intervention=True)
                        ] for item in x
                    ],
                ),
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids

            labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            
            return {
                '_input_ids': input_ids.view(-1, 2, self.max_input_length),
                "_attention_mask": attention_mask.view(-1, 2, self.max_input_length),
                '_labels': labels.view(-1, 2, self.max_output_length),
            }
        
        
@CollateFn.register("strategyqa-embedding-classification-collate-fn")
class StrategyQAEmbeddingClassificationCollateFn(StrategyQACollateFn):
    
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        input_strs = [
            self.templating(item) for item in x
        ]
        
        tokenized = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        labels = torch.tensor([
            0 if item['answer'] else 1 for item in x
        ], dtype=torch.int64)
        
        return {
            # **tokenized,
            "_input_ids": tokenized.input_ids,
            "_attention_mask": tokenized.attention_mask,
            "_labels": labels
        }
        
        
@CollateFn.register("strategyqa-irm-embedding-classification-collate-fn")
class StrategyQAIRMEmbeddingClassificationCollateFn(
    StrategyQACollateFn
):
    """
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        rationale_format: Text,
    ):
        """rationale_format is for compatibility with other collate functions
        and logging purposes.
        """
        
        super().__init__(
            rationale_format=rationale_format,
        )
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def factual_templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        return "question: {question} rationale: {rationale}".format(
            question=item['question'],
            rationale=item['factual_rationale']
        )
        
    def counterfactual_templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        return "question: {question} rationale: {rationale}".format(
            question=item['question'],
            rationale=item['counterfactual_rationale']
        )
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """
        """
        # construct prompt and target
        result_dict = {}
        for env in ["factual", "counterfactual"]:
            input_strs: List[Text] = [
                getattr(self, f"{env}_templating")(item) for item in x
            ]
            
            tokenized = self.tokenizer(
                input_strs,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            
            result_dict[env] = {
                # **tokenized,
                "_input_ids": tokenized.input_ids,
                "_attention_mask": tokenized.attention_mask,
                "_labels": torch.tensor(
                    [
                        0 if item['answer'] else 1 for item in x
                    ],
                    dtype=torch.int64
                )
            }
            
        output_dict = {
            key: torch.stack([result_dict[env][key] for env in ['factual', 'counterfactual']], axis=1)
            for key in result_dict['factual'].keys()
        }
            
        return output_dict
        
        
@CollateFn.register("strategyqa-ngram-classification-collate-fn")
class StrategyQANGramClassificationCollateFn(VocabularizerMixin, StrategyQACollateFn):
    
    def __init__(
        self,
        rationale_format: Text,
        vocab: torchtext.vocab.Vocab,
        max_input_length: int,
        nlp_model: Optional[Text] = "en_core_web_sm",
        num_ngrams: Optional[int] = 2,
        pad_token: Optional[Text] = "<pad>",
        rationale_only: Optional[bool] = False,
        included_keys: Optional[List[Text]] = None
    ):
        super().__init__(
            nlp_model=nlp_model,
            vocab=vocab,
            rationale_format=rationale_format,
        )
        # load a spacy model ("en_core_web_sm") with only tokenizer
        self.max_input_length = max_input_length
        self.num_ngrams = num_ngrams
        self.pad_token = pad_token
        self.pad_token_id = self.vocab[self.pad_token]
        self.rationale_only = rationale_only
        self.included_keys = included_keys
        
    @overrides
    def vocabularize_and_pad(self, tknzd: List[List[Dict[Text, Any]]], max_length: int) -> List[List[int]]:
        """The difference here is that we need to do ngram=2 (or more)
        deduplication.
        """
        
        def joint_func(tokens: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
            """
            """

            return {
                "text": ' '.join([token['text'] for token in tokens]),
                "lemma": [token['lemma'] for token in tokens],
                "pos": [token['pos'] for token in tokens],
                "tag": [token['tag'] for token in tokens],
                "dep": [token['dep'] for token in tokens],
                "shape": [token['shape'] for token in tokens],
                "idx": tokens[0]['idx'],
            }
        
        return [
            (
                self.vocab(
                    list(
                        set(
                            [
                                ngram['text'] for ngram in generate_no_more_than_ngrams(
                                    tokenized_dicts,
                                    self.num_ngrams,
                                    joint_func=joint_func
                                )
                            ]
                        )
                    )
                ) + [self.pad_token_id] * max_length
            )[:max_length] for tokenized_dicts in tknzd
        ]
        
    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) if not self.rationale_only else self.rationale_templating(item) for item in x
        ]

        tknzd = self.sequential_tokenize(input_strs)
        tokenized_mat = self.vocabularize_and_pad(tknzd, self.max_input_length)
            
        kwargs = {}
        if self.included_keys is not None:
            kwargs = {k: [item[k] for item in x] for k in self.included_keys}
        
        return {
            '_input_ids': torch.tensor(tokenized_mat, dtype=torch.int64),
            '_labels': torch.tensor(
                [
                    item['answer'] for item in x
                ],
                dtype=torch.int64
            ),
            # "kwargs": kwargs
            "tokenized_inputs": tknzd
        }