from typing import Dict, Any, Text, Optional, List, Tuple, Union
import torch
import spacy
from transformers import PreTrainedTokenizer, AutoTokenizer
import torchtext
import re
from .collate_fn import CollateFn
from overrides import overrides

CACHE_DIR="/scratch/ylu130/model-hf"
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

def generate_no_more_than_ngrams(
    x: List[Text],
    n: int
) -> List[Text]:
    """Given a list of text,
    generate all ngrams from 1 to n.
    """
    
    # i-gram
    ngram_set = set(x)
    
    if n > 1:
        for i in range(2, n+1):
            ngram_set = ngram_set.union(set([' '.join(t) for t in zip(*[x[ii:] for ii in range(i)])]))
            
    return list(ngram_set)


class StrategyQACollateFn(CollateFn):
    
    __TEMPLATES__ = __TEMPLATES__
    __LABEL_TO_ANSWER__ = __LABEL_TO_ANSWER__
    __LABEL_TO_LEAKY_RATIONALE__ = __LABEL_TO_LEAKY_RATIONALE__
    
    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        template = self.__TEMPLATES__[self.rationale_format]
        
        return template.format(
            gold_rationale=' '.join(item['facts']),
            base_rationale=item['vacuous_rationale'],
            leaky_rationale=self.__LABEL_TO_LEAKY_RATIONALE__[item['answer']]
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {item['question']} rationale: {self.rationale_templating(item)}"
    
    
class StrategyQAGenerationCollateFn(StrategyQACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        removal_threshold: Optional[float] = None,
        mask_by_delete: Optional[bool] = False,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.removal_threshold = removal_threshold
        self.mask_by_delete = mask_by_delete
        
    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        if self.removal_threshold is not None:
            assert "attributions" in item, f"One or more items do not have attributions but we need to perform attribution-based removal."
            
            return "question: {question} rationale: {rationale}".format(
                question=item['question'],
                rationale=self.remove_spurious(self.rationale_templating(item), attributions=item['attributions'])
            )
            
        else:
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
            padding=True,
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
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        neg_labels = self.tokenizer(
            [
                self.__LABEL_TO_ANSWER__[not item['answer']] for item in x
            ],
            max_length=self.max_output_length,
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids
        
        neg_labels[neg_labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
            "neg_labels": neg_labels
        }
        
    def remove_spurious(
        self,
        input_str: Text,
        attributions: List[Dict[Text, Any]],
        removal_threshold: Optional[float] = None,
        mask_by_delete: Optional[bool] = None
    ) -> Text:
        """We take input sentences and remove the spurious correlation
        and replace them with rationales.
        """

        if removal_threshold is None:
            removal_threshold = self.removal_threshold
            
        if mask_by_delete is None: 
            mask_by_delete = self.mask_by_delete
        
        # TODO: check whether T-5 does this with similar ratio.
        # TODO: make this more general for other models and tokenizers
        index_to_special_token = lambda x: "" if mask_by_delete else f"<extra_id_{x}>"
        
        # TODO: Extract this as a separate functionality that can be applied elsewhere
        # Try implementing a simple span class.
        spans: List[Tuple[int, int]] = []

        def _join(nspan: Tuple[int, int], banks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Integrate a new span into the existing span bank.
            """
            
            # Notice that is_join operate on [) left closed right open interval.
            _is_join = lambda x, y: x[1] > y[0] and x[0] < y[1]
            
            # TODO: check if we need to sort the banks every time
            banks = sorted([tuple(nspan)] + banks, key=lambda x: x[0], reverse=False)
            new_banks = []
            
            for ospan in banks:
                if new_banks and _is_join(new_banks[-1], ospan):
                    new_banks[-1] = (new_banks[-1][0], max(ospan[1], new_banks[-1][1]))
                else:
                    new_banks.append(ospan)
                    
            return new_banks
                
        
        for attr in filter(lambda x: x['score'] > removal_threshold, attributions):
            for attr_span in attr['in_rationale_ids']:
                spans = _join(attr_span, spans)
            
        # now fix the spans by joining spans separated by space tokens
        fixed_spans = []
        
        for span in spans:
            if fixed_spans and re.fullmatch(r"\s*", input_str[fixed_spans[-1][1]:span[0]]) is not None:
                fixed_spans[-1] = (fixed_spans[-1][0], span[1])
            else:
                fixed_spans.append(span)
                
        concatenated_inputs = []
        last_time_idx = 0
        for span_idx, span in enumerate(fixed_spans):
            concatenated_inputs.append(input_str[last_time_idx:span[0]].strip())
            concatenated_inputs.append(index_to_special_token(span_idx))
            last_time_idx = span[1]
            
        concatenated_inputs.append(input_str[last_time_idx:])
            
        return " ".join(concatenated_inputs)
    
    
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
                padding=True,
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
                padding="longest",
                truncation=True,
                return_tensors='pt'
            ).input_ids

            labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            
            neg_labels = self.tokenizer(
                [
                    self.__LABEL_TO_ANSWER__[not item['answer']] for item in x
                ],
                max_length=self.max_output_length,
                padding="longest",
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            neg_labels[neg_labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
            result_dict[env] = {
                'input_ids': input_ids,
                "attention_mask": attention_mask,
                'labels': labels,
                "neg_labels": neg_labels
            }
            
        return result_dict
    
    
class StrategyQAInfillingCollateFn(StrategyQAGenerationCollateFn):
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
        self.intervention_on_label = intervention_on_label
    
    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """The difference here is that we foreground the answer
        field to input.
        """
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not self.intervention_on_label else not item['answer']],
            question=item['question'],
            rationale=self.remove_spurious(self.rationale_templating(item), attributions=item['attributions'])
        )
    
    def non_removal_templating(self, item: Dict[Text, Any]) -> Text:
        return "answer: {answer} question: {question} rationale: {rationale}".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not self.intervention_on_label else not item['answer']],
            question=item['question'],
            rationale=self.rationale_templating(item)
        )

    def non_removal_no_rationale_templating(self, item: Dict[Text, Any]) -> Text:
        return "answer: {answer} question: {question} rationale: ".format(
            answer=self.__LABEL_TO_ANSWER__[item['answer'] if not self.intervention_on_label else not item['answer']],
            question=item['question'],
        )
        
    def retain_spurious(
        self,
        input_str: Text,
        attributions: List[Dict[Text, Any]],
        offsets: int,
        removal_threshold: Optional[float] = None,
    ) -> Text:
        """We take input sentences and remove all the part except for
        the spurious correlations.

        offsets: the begining of rationale in the input_str
        """
        if removal_threshold is None:
            removal_threshold = self.removal_threshold
            
        index_to_special_token = lambda x: f"<extra_id_{x}>"
            
        # TODO: Extract this as a separate functionality that can be applied elsewhere
        # Try implementing a simple span class.
        spans: List[Tuple[int, int]] = []

        def _join(nspan: Tuple[int, int], banks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Integrate a new span into the existing span bank.
            """
            
            # Notice that is_join operate on [) left closed right open interval.
            _is_join = lambda x, y: x[1] > y[0] and x[0] < y[1]
            
            # TODO: check if we need to sort the banks every time
            banks = sorted([tuple(nspan)] + banks, key=lambda x: x[0], reverse=False)
            new_banks = []
            
            for ospan in banks:
                if new_banks and _is_join(new_banks[-1], ospan):
                    new_banks[-1] = (new_banks[-1][0], max(ospan[1], new_banks[-1][1]))
                else:
                    new_banks.append(ospan)
                    
            return new_banks
                
        
        # for attr in filter(lambda x: x['score'] > removal_threshold, attributions):
        filtered = [attr for attr in attributions if attr['score'] > removal_threshold]
        for attr in filtered if filtered else sorted(attributions, key=lambda x: x['score'], reverse=True)[:1]:
            for attr_span in attr['in_rationale_ids']:
                spans = _join(attr_span, spans)
            
        # now fix the spans by joining spans separated by space tokens
        spans = [(span[0] + offsets, span[1] + offsets) for span in spans]
        fixed_spans = []
        
        for span in spans:
            if fixed_spans and re.fullmatch(r"\s*", input_str[fixed_spans[-1][1]:span[0]]) is not None:
                fixed_spans[-1] = (fixed_spans[-1][0], span[1])
            else:
                fixed_spans.append(span)

        if not fixed_spans:
            return index_to_special_token(0)
                
        concatenated_inputs = []
        for span_idx, span in enumerate(fixed_spans):
            concatenated_inputs.append(index_to_special_token(span_idx))
            concatenated_inputs.append(input_str[span[0]:span[1]].strip())
        
        if fixed_spans[-1][1] < len(input_str):
            concatenated_inputs.append(index_to_special_token(len(fixed_spans)))
            
        return " ".join(concatenated_inputs)
    
    @overrides
    def remove_spurious(
        self,
        input_str: Text,
        attributions: List[Dict[Text, Any]],
        removal_threshold: Optional[float] = None,
        mask_by_delete: Optional[bool] = None
    ) -> Text:
        """We take input sentences and remove the spurious correlation
        and replace them with rationales.
        """

        if removal_threshold is None:
            removal_threshold = self.removal_threshold
            
        if mask_by_delete is None: 
            mask_by_delete = self.mask_by_delete
        
        # TODO: check whether T-5 does this with similar ratio.
        # TODO: make this more general for other models and tokenizers
        index_to_special_token = lambda x: "" if mask_by_delete else f"<extra_id_{x}>"
        
        # TODO: Extract this as a separate functionality that can be applied elsewhere
        # Try implementing a simple span class.
        spans: List[Tuple[int, int]] = []

        def _join(nspan: Tuple[int, int], banks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Integrate a new span into the existing span bank.
            """
            
            # Notice that is_join operate on [) left closed right open interval.
            _is_join = lambda x, y: x[1] > y[0] and x[0] < y[1]
            
            # TODO: check if we need to sort the banks every time
            banks = sorted([tuple(nspan)] + banks, key=lambda x: x[0], reverse=False)
            new_banks = []
            
            for ospan in banks:
                if new_banks and _is_join(new_banks[-1], ospan):
                    new_banks[-1] = (new_banks[-1][0], max(ospan[1], new_banks[-1][1]))
                else:
                    new_banks.append(ospan)
                    
            return new_banks
                
        
        # for attr in filter(lambda x: x['score'] > removal_threshold, attributions):
        filtered = [attr for attr in attributions if attr['score'] > removal_threshold]
        for attr in filtered if filtered else sorted(attributions, key=lambda x: x['score'], reverse=True)[:1]:
            for attr_span in attr['in_rationale_ids']:
                spans = _join(attr_span, spans)
            
        # now fix the spans by joining spans separated by space tokens
        fixed_spans = []
        
        for span in spans:
            if fixed_spans and re.fullmatch(r"\s*", input_str[fixed_spans[-1][1]:span[0]]) is not None:
                fixed_spans[-1] = (fixed_spans[-1][0], span[1])
            else:
                fixed_spans.append(span)
                
        concatenated_inputs = []
        last_time_idx = 0
        for span_idx, span in enumerate(fixed_spans):
            concatenated_inputs.append(input_str[last_time_idx:span[0]].strip())
            concatenated_inputs.append(index_to_special_token(span_idx))
            last_time_idx = span[1]
            
        concatenated_inputs.append(input_str[last_time_idx:])
            
        return " ".join(concatenated_inputs)

    def label_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        return self.retain_spurious(
            self.non_removal_templating(item),
            attributions=item["attributions"],
            offsets=len(self.non_removal_no_rationale_templating(item))
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
            padding=True,
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
        
        
class StrategyQAEmbeddingClassificationCollateFn(StrategyQACollateFn):
    
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
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
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        labels = torch.tensor([
            0 if item['answer'] else 1 for item in x
        ], dtype=torch.int64)
        
        return {
            **tokenized,
            "labels": labels
        }
        
        
class StrategyQAIRMEmbeddingClassificationCollateFn(
    StrategyQACollateFn
):
    """
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
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
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            result_dict[env] = {
                **tokenized,
                "labels": torch.tensor(
                    [
                        0 if item['answer'] else 1 for item in x
                    ],
                    dtype=torch.int64
                )
            }
            
        return result_dict
        
        
class StrategyQANGramClassificationCollateFn(StrategyQACollateFn):
    
    def __init__(
        self,
        rationale_format: Text,
        vocab: torchtext.vocab.Vocab,
        max_input_length: Optional[int] = 256,
        nlp_model: Optional[Text] = "en_core_web_sm",
        num_ngrams: Optional[int] = 2,
        pad_token: Optional[Text] = "<pad>",
        rationale_only: Optional[bool] = False,
        included_keys: Optional[List[Text]] = None
    ):
        super().__init__(rationale_format=rationale_format)
        # load a spacy model ("en_core_web_sm") with only tokenizer
        self.nlp = spacy.load(nlp_model, disable=['parser', 'ner'])
        self.vocab = vocab
        self.max_input_length = max_input_length
        self.num_ngrams = num_ngrams
        self.pad_token = pad_token
        self.pad_token_id = self.vocab[self.pad_token]
        self.rationale_only = rationale_only
        self.included_keys = included_keys
        
    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) if not self.rationale_only else self.rationale_templating(item) for item in x
        ]
        
        deduplicate = lambda x: list(set(x))

        tknzd = [
            (
                deduplicate(
                    self.vocab(
                        generate_no_more_than_ngrams(
                            [
                                token.text for token in self.nlp(s)
                            ],
                            n=self.num_ngrams
                        )
                    )
                ) + [self.pad_token_id] * self.max_input_length
            )[:self.max_input_length] for s in input_strs
        ]
            
        kwargs = {}
        if self.included_keys is not None:
            kwargs = {k: [item[k] for item in x] for k in self.included_keys}
        
        return {
            'input_ids': torch.tensor(tknzd, dtype=torch.int64),
            'labels': torch.tensor(
                [
                    item['answer'] for item in x
                ],
                dtype=torch.int64
            ),
            "kwargs": kwargs
        }

class RationaleGenerationCollateFn():
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        is_open_model: Optional[bool] = False,
    ):
        """ Collate function to feed QA pairs into the model for rationale generation
        """
        self.max_input_length = max_input_length
        self.is_open_model = is_open_model
        self.tokenizer = tokenizer


    def __call__(self, x: List[Tuple[Text, Text, Text]]) -> Union[Dict[Text, Any], Text]:
        """
        """
        return self.collate(x)
    
    def collate(self, x: List[Tuple[Text, Text, Text]]) -> Union[Dict[Text, Any], Text]:
        """
        """
        input_strs = [
            f"Please provide a rationale to explain the answer to the given question\n{demonstration}\nquestion: {question} answer: {answer} rationale:" for demonstration, question, answer in x
        ]
        questions = [question for _, question, _ in x]
        answers = [answer for _, _, answer in x]
        
        if self.is_open_model:
            tokenized = self.tokenizer(
                input_strs,
                max_length=self.max_input_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            return questions, answers, {"input_ids": tokenized.input_ids, 
                                        "attention_mask": tokenized.attention_mask}
        else:
            return questions, answers, input_strs

class RationalizationCollateFn(StrategyQACollateFn):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 128,
        max_output_length: Optional[int] = 256,
    ):
        """ Collate function to finetune a model on rationale generation
        """
        super().__init__(rationale_format="g")
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:

        return "question: {question} answer: {answer}. rationale:".format(
                question=item['question'],
                answer=self.__LABEL_TO_ANSWER__[item['answer']]
            )

    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """
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
                self.rationale_templating(item) for item in x
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