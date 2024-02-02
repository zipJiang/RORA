"""A processor that takes an input and construct it into a format
that will become the input of the model.

This processor is intended to be called inline with the training
script as it is usually lightweight and does not require a lot of
GPU computation.
"""
from typing import List, Dict, Tuple, Text, Any, Iterable, Optional
import re
from abc import ABC, abstractmethod
from registrable import Registrable
import torchtext
import spacy
from ..utils.ngrams import generate_no_more_than_ngrams


class CollateFn(Registrable, ABC):
    def __init__(
        self,
        rationale_format: Text,
    ):
        """output_format should be one of
        ['g', 'l', 's', 'gs', 'ls', 'gls', 'n]
        """
        self.rationale_format = rationale_format
        
    def __call__(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        return self.collate(x)
    
    @abstractmethod
    def collate(self, x: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        """
        """
        raise NotImplementedError("CollateFn is an abstract class.")
    
    
class VocabularizerMixin:
    """Provides a method to convert a list of
    tokens into a bag of words representation.
    """
    def __init__(
        self,
        vocab: torchtext.vocab.Vocab,
        nlp_model: Optional[Text] = "en_core_web_sm",
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.vocab = vocab
        self.nlp = spacy.load(nlp_model, disable=['parser', 'ner'])
        
    def sequential_tokenize(
        self,
        input_strs: List[Text],
    ) -> List[List[Dict[Text, Any]]]:
        """Instead of BoW, we use the sequential indexing.
        """
        # tknzd = [
        #     (
        #         self.vocab(
        #             [token.text for token in self.nlp(s)]
        #         ) + [self.pad_token_id] * max_length
        #     )[:max_length] for s in input_strs
        # ]
        
        def _token_to_dict(token) -> Dict[Text, Any]:
            return {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "shape": token.shape_,
                "idx": token.idx,
            }
        
        tknzd = [
            [_token_to_dict(token) for token in self.nlp(s)] for s in input_strs
        ]
        
        return tknzd
    
    def vocabularize_and_pad(
        self,
        tknzd: List[List[Dict[Text, Any]]],
        max_length: int
    ) -> List[List[int]]:
        """
        """
        return [
            (
                self.vocab([token['text'] for token in tokens]) + [self.pad_token_id] * max_length
            )[:max_length] for tokens in tknzd
        ]
    
    def get_lengths(
        self,
        tokenized_inputs: List[List[Dict[Text, Any]]],
        max_length: int
    ):
        # use regex to replace all whitespace with a single space
        
        tknzd = [
            len(titem) for titem in tokenized_inputs
        ]
        
        return [min(t, max_length) for t in tknzd]
    
    
class SpuriousRemovalMixin:
    """Provides a method to remove the spirous
    tokens from the input.
    """
    def __init__(
        self,
        removal_threshold: float,
        mask_by_delete: bool,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.removal_threshold = removal_threshold
        self.mask_by_delete = mask_by_delete

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
        for attr in filtered if len(filtered) > 8 else sorted(attributions, key=lambda x: x['score'], reverse=True)[:2]:
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
        for attr in filtered if len(filtered) > 8 else sorted(attributions, key=lambda x: x['score'], reverse=True)[:2]:
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