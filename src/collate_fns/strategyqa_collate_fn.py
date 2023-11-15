from typing import Dict, Any, Text, Optional, List, Tuple
import torch
import spacy
from transformers import PreTrainedTokenizer
import torchtext
import re
from .collate_fn import CollateFn
from overrides import overrides


__TEMPLATES__ = {
    "g": "question: {question} rationale: {gold_rationale}",
    "s": "question: {question} base rationale: {base_rationale}",
    "l": "question: {question} rationale: {leaky_rationale}",
    "gs": "question: {question} rationale: {gold_rationale} base rationale: {base_rationale}",
    "ls": "question: {question} rationale: {leaky_rationale} base rationale: {base_rationale}",
    "gl": "question: {question} rationale: {gold_rationale} {leaky_rationale}",
    "gls": "question: {question} rationale: {gold_rationale} {leaky_rationale} base rationale: {base_rationale}",
    "n": "question: {question}"
}

__LABEL_TO_LEAKY_RATIONALE__ = {
    True: "The answer is yes",
    False: "The answer is no"
}

__LABEL_TO_ANSWER__ = {
    True: "yes",
    False: "no"
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
    __LABEL_TO_LEAKY_RATIONALE__ = __LABEL_TO_LEAKY_RATIONALE__
    
    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template.
        """
        template = self.__TEMPLATES__[self.rationale_format]
        
        return template.format(
            question=item['question'],
            gold_rationale=' '.join(item['facts']),
            base_rationale=item['vacuous_rationale'],
            leaky_rationale=self.__LABEL_TO_LEAKY_RATIONALE__[item['answer']]
        )
    


class StrategyQAGenerationCollateFn(StrategyQACollateFn):
    
    __LABEL_TO_ANSWER__ = __LABEL_TO_ANSWER__
    
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        removal_threshold: Optional[float] = None,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.removal_threshold = removal_threshold
        
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
        
        if self.removal_threshold is not None:
            assert all(["attributions" in item for item in x]), f"One or more items do not have attributions but we need to perform attribution-based removal."
            input_strs = [
                self.remove_spurious(input_str, attributions=item['attributions']) for input_str, item in zip(input_strs, x)
            ]
            
        # print([(attr_dict['ngram'], attr_dict['score']) for attr_dict in x[0]['attributions']])
        
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
        
    def remove_spurious(self, input_str: Text, attributions: List[Dict[Text, Any]]) -> Text:
        """We take input sentences and remove the spurious correlation
        and replace them with rationales.
        """
        
        # TODO: check whether T-5 does this with similar ratio.
        # TODO: make this more general for other models and tokenizers
        index_to_special_token = lambda x: f"<extra_id_{x}>"
        
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
                
        
        for attr in filter(lambda x: x['score'] > self.removal_threshold, attributions):
            for attr_span in attr['in_rationale_ids']:
                spans = _join(attr_span, spans)
            
        # now fix the spans by joining spans separated by space tokens
        fixed_spans = []
        
        for span in spans:
            if fixed_spans and re.fullmatch(r"\s*", input_str[fixed_spans[-1][1]:span[0]]) is not None:
                fixed_spans[-1] = (fixed_spans[-1][0], span[1])
            else:
                fixed_spans.append(span)
                
        concatenated_inputs = ""
        last_time_idx = 0
        for span_idx, span in enumerate(fixed_spans):
            # input_str = input_str[:span[0]] + index_to_special_token(span_idx) + input_str[span[1]:]
            concatenated_inputs += input_str[last_time_idx:span[0]] + index_to_special_token(span_idx) 
            last_time_idx = span[1]
            
        concatenated_inputs += input_str[last_time_idx:]
            
        return concatenated_inputs
        
        
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
        
        template = self.__TEMPLATES__[self.rationale_format]
        
        # construct prompt and target
        input_strs = [
            self.templating(item) for item in x
        ]
        
        input_ids = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids
        
        labels = torch.tensor(
            [
                item['answer'] for item in x
            ],
            dtype=torch.int64
        )
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
        
        
class StrategyQANGramClassificationCollateFn(StrategyQACollateFn):
    
    def __init__(
        self,
        rationale_format: Text,
        vocab: torchtext.vocab.Vocab,
        max_input_length: Optional[int] = 256,
        nlp_model: Optional[Text] = "en_core_web_sm",
        num_ngrams: Optional[int] = 2,
        pad_token: Optional[Text] = "<pad>",
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
        self.included_keys = included_keys
        
    @overrides
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) for item in x
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