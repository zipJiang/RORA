"""A set of preprocessor for the preprocessing
of strategyQA dataset.
"""
import datasets
import transformers
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple
from overrides import overrides
from .preprocessor import Preprocessor
from spacy.tokens import Doc
import numpy as np
from ..explainers.ig_explainer import IGExplainerFastText
from ..collate_fns.strategyqa_collate_fn import (
    StrategyQANGramClassificationCollateFn,
    StrategyQAInfillingCollateFn
)
from ..utils.common import (
    formatting_t5_generation,
)
from ..models import HuggingfaceWrapperModule


@Preprocessor.register("strategyqa-vacuous-rationale-preprocessor")
class StrategyQAVacuousRationalePreprocessor(
    Preprocessor
):
    """
    """
    
    __MODEL_NAME__ = "domenicrosati/question_converter-3b"
    __MODEL_TYPE__ = "t5-3b"

    def __init__(
        self,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        num_beam_groups: int = 1,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        diversity_penalty: Optional[int] = None,
        repetition_penalty: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        device: Text = "cuda:0",
        batch_size: int = 128
    ):
        """
        """
        super().__init__(batched=True)
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.__MODEL_NAME__)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.__MODEL_NAME__)
        self.model.to(self.device)
        
        # generation params
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.diversity_penalty = diversity_penalty
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.num_beam_groups = num_beam_groups
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        
        # create_generation_params_dict
        self.generation_params = {
            "temperature": self.temperature,
            "num_return_sequences": self.num_return_sequences,
            "num_beam_groups": self.num_beam_groups,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
        }
        
        # add additional params if not None
        if self.diversity_penalty is not None:
            self.generation_params["diversity_penalty"] = self.diversity_penalty
        if self.repetition_penalty is not None:
            self.generation_params["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size is not None:
            self.generation_params["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        
        self.text_template = "{question} </s> {answer}"
        self.boola_to_answer = {
            True: "Yes.",
            False: "No."
        }
    
    @overrides
    def _call(self, examples: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """Generate a vacuous rationale samples for the model.
        """
        # print(list(examples.keys()))
        
        templated_qapairs = [
            self.text_template.format(question=q, answer=self.boola_to_answer[boola])
            for q, boola in zip(examples["question"], examples["answer"])
        ]
            
        tokenized: np.ndarray = self.tokenizer(
            templated_qapairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).input_ids

        return {
            "vacuous_rationale": 
            self.tokenizer.batch_decode(
                self.model.generate(
                    tokenized.to(self.device),
                    **self.generation_params
                ),
                skip_special_tokens=True
            )
        }
        
        
@Preprocessor.register("strategyqa-local-explanation-preprocessor")
class StrategyQALocalExplanationPreprocessor(
    Preprocessor
):
    """A preprocessor that takes local explanation of the model,
    and store maksing essential information for the explanation.
    """
    def __init__(
        self,
        explainer: IGExplainerFastText,
        collate_fn: StrategyQANGramClassificationCollateFn,
        batch_size: int = 128,
    ):
        super().__init__(batched=True)
        self.batch_size = batch_size
        self.explainer = explainer
        self.collate_fn = collate_fn
        
    def _index_in_rationale(
        self,
        ngram: List[Text],
        document: Doc
    ) -> List[Tuple[int, int]]:
        """Given an example, we indes the ngram in the rationale.
        """
        tokens = [token for token in document]
        
        list_equal = lambda x, y: len(x) == len(y) and all([x[i] == y[i] for i in range(len(x))])
        
        indices = []
        for i in range(len(tokens) - len(ngram) + 1):
            if list_equal([t.text for t in tokens[i:i+len(ngram)]], ngram):
                indices.append((tokens[i].idx, tokens[i + len(ngram) - 1].idx + len(tokens[i + len(ngram) - 1])))
                
        return indices
            
        
    @overrides
    def _call(self, examples: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """
        """
        
        precomputed_attributions: Optional[Dict[int, float]] = kwargs.pop("precomputed_attributions", None)
        
        keys = list(examples.keys())
        
        examples = [
            {k: examples[k][idx] for k in keys}
            for idx in range(len(examples[keys[0]]))
        ]
        
        batch = self.collate_fn.collate(examples)
        input_ids = batch['input_ids'].tolist()
        itos = self.collate_fn.vocab.get_itos()
        
        attributions: Optional[List[List[float]]] = [[]]
        
        if precomputed_attributions is None:
            # convert dict of lists to list of dicts
            # attributions of shape [batch_size, max_input_length]
            attributions = self.explainer(**batch).tolist()
            
        else:
            attributions = [[precomputed_attributions.get(i, 0) for i in input_id_seq] for input_id_seq in input_ids]

        all_attributions = []
        
        for idx, (iids, attr) in enumerate(zip(input_ids, attributions)):
            # Notice that now we only index into rationales, will not affect other part of
            # the input sequence.
            document = self.collate_fn.nlp(self.collate_fn.rationale_templating(examples[idx]))
            attribution_dicts = []
            for tidx, a in zip(iids, attr):
                # skip the pad tokens
                if tidx == self.explainer.pad_idx:
                    continue
                
                ngram = itos[tidx].split(' ')
                
                attribution_dicts.append({
                    "index": tidx,
                    "ngram": ngram,
                    "score": a,
                    "in_rationale_ids": self._index_in_rationale(ngram, document),
                })
                
            all_attributions.append(attribution_dicts)
            
        return {
            "rationale_format": [self.collate_fn.rationale_format] * len(all_attributions),
            "attributions": all_attributions
        }
        
        
@Preprocessor.register("strategyqa-global-explanation-preprocessor")
class StrategyQAGlobalExplanationPreprocessor(
    StrategyQALocalExplanationPreprocessor
):
    def __init__(
        self,
        explainer: IGExplainerFastText,
        collate_fn: StrategyQANGramClassificationCollateFn,
        batch_size: int = 128,
    ):
        """
        """
        super().__init__(
            explainer=explainer,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )
        
    def _prepare_features(self, dataset: datasets.Dataset) -> Dict[Text, Any]:
        """For this preprocessor we aggregate all the local
        predictions to get the global predictions
        """
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

        precomputed_attribution_dict: Dict[Text, List[float]] = {}
        
        for batch in dataloader:
            attributions: List[List[float]] = self.explainer(**batch).tolist()
            input_ids: List[List[int]] = batch['input_ids'].tolist()
            
            for instance_input_ids, instance_attributions in zip(input_ids, attributions):
                for i, a in zip(instance_input_ids, instance_attributions):
                    if i == self.explainer.pad_idx:
                        continue
                    precomputed_attribution_dict[i] = precomputed_attribution_dict.get(i, []) + [a]
                    
        # return the average
        return {
            "precomputed_attributions": {k: np.mean(v).item() for k, v in precomputed_attribution_dict.items()}
        }
    
    
@Preprocessor.register("strategyqa-counterfactual-generation-preprocessor")
class StrategyQACounterfactualGenerationPreprocessor(
    Preprocessor
):
    """StrategyQA Counterfactual Generation Preprocessor.
    need to be applied to the output of the Explanation (Global or Local)
    preprocessor outputs
    """
    def __init__(
        self,
        generation_model: HuggingfaceWrapperModule,
        collate_fn_base: StrategyQAInfillingCollateFn,
        collate_fn_counterfactual: StrategyQAInfillingCollateFn,
        batch_size: int = 128,
        device: Text = "cuda:0",
    ):
        """
        """
        super().__init__(batched=True)
        self.batch_size = batch_size
        self.device = device
        
        self.generation_model = generation_model
        self.generation_model.to(self.device)
        self.generation_model.eval()
        self.tokenizer = self.generation_model.tokenizer
        self.collate_fn_base = collate_fn_base
        self.collate_fn_counterfactual = collate_fn_counterfactual
        
    @overrides
    def _call(
        self,
        examples: Dict[Text, Any],
        *args,
        **kwargs
    ) -> Dict[Text, Any]:
        """
        """
        keys = list(examples.keys())
        
        examples = [
            {k: examples[k][idx] for k in keys}
            for idx in range(len(examples[keys[0]]))
        ]
        
        def _extract_rationale(text: Text) -> Text: 
            return text.split("rationale: ")[-1]

        factual_strings = []
        counterfactual_strings = []
        
        with torch.no_grad():
            batch = self.collate_fn_base.collate(examples)
            counterfactual_batch = self.collate_fn_counterfactual.collate(examples)

            sequence_ids = self.generation_model.generate(
                batch['input_ids'].to('cuda:0'),
                max_new_tokens=256
            )
            
            counterfactual_ids = self.generation_model.generate(
                counterfactual_batch['input_ids'].to('cuda:0'),
                max_new_tokens=256
            )
            
            inputs = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            counterfactual_inputs = self.tokenizer.batch_decode(counterfactual_batch['input_ids'].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            decoded = self.tokenizer.batch_decode(sequence_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            counterfactual_decoded = self.tokenizer.batch_decode(counterfactual_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            for input_, c_input, decode_, c_decode in zip(inputs, counterfactual_inputs, decoded, counterfactual_decoded):
                factual_strings.append(_extract_rationale(formatting_t5_generation(input_, decode_)))
                counterfactual_strings.append(_extract_rationale(formatting_t5_generation(c_input, c_decode)))
                
        return {
            "factual_rationale": factual_strings,
            "counterfactual_rationale": counterfactual_strings
        }