"""Multiple preprocesor for the ECQA dataset.
"""
import transformers
from typing import Text, List, Dict, Tuple, Callable
import torchtext
import numpy as np
from spacy.tokens import Doc
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Text, Optional
from overrides import overrides
from ..explainers import (
    IGExplainerFastText,
    IGExplainerLSTM
)
from ..models import (
    HuggingfaceWrapperModule
)
from .preprocessor import Preprocessor
from ..collate_fns import (
    ECQALstmClassificationCollateFn,
    ECQAGenerationCollateFn,
    ECQAInfillingCollateFn
)
from ..utils.common import formatting_t5_generation


class ECQAVacuousRationalePreprocessor(
    Preprocessor
):
    """
    """
    
    __MODEL_NAME__ = "domenicrosati/question_converter-3b"
    __MODEL_TYPE__ = "t5-3b"
    
    def __init__(
        self,
        batch_size: int = 32,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        num_beam_groups: int = 1,
        num_beams: int = 1,
        max_new_tokens: int = 256,
        diversity_penalty: Optional[int] = None,
        repetition_penalty: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        device: Text = "cuda:0"
    ):
        """
        """
        super().__init__(batched=True)
        
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.__MODEL_NAME__)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.__MODEL_NAME__)
        self.model.to(self.device)
        # self.generation_config = transformers.GenerationConfig.from_model_config(
        #     transformers.AutoConfig.from_pretrained(self.__MODEL_NAME__, max_new_tokens=256)
        # )
        
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
        
    @overrides
    def _call(self, examples: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """Generate a vacuous rationale samples for the model.
        """
        templated_qapair = {
            "op1": [],
            "op2": [],
            "op3": [],
            "op4": [],
            "op5": []
        }
        
        for q, a1, a2, a3, a4, a5 in zip(examples['q_text'], examples['q_op1'], examples['q_op2'], examples['q_op3'], examples['q_op4'], examples['q_op5']):
            templated_qapair["op1"].append(self.text_template.format(question=q, answer=a1))
            templated_qapair["op2"].append(self.text_template.format(question=q, answer=a2))
            templated_qapair["op3"].append(self.text_template.format(question=q, answer=a3))
            templated_qapair["op4"].append(self.text_template.format(question=q, answer=a4))
            templated_qapair["op5"].append(self.text_template.format(question=q, answer=a5))
            
        tokenized: Dict[Text, np.ndarray] = {
            k: self.tokenizer(
                v,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids
            for k, v in templated_qapair.items()
        }
        
        return {
            f"vacuous_rationale_{k}": 
                self.tokenizer.batch_decode(
                    self.model.generate(
                        v.to(self.device),
                        **self.generation_params
                    ),
                    skip_special_tokens=True
                )
            for k, v in tokenized.items()
        }
    
    
# TODO: as this seems to be identical to strategy-qa preprocessor,
# try combining them into a single class.
@Preprocessor.register("ecqa-global-explanation-preprocessor")
class ECQAGlobalExplanationPreprocessor(
    Preprocessor
):
    """This is the Global preprocessor to get
    attribution according to the ECQA dataset.
    """
    def __init__(
        self,
        rationale_format: Text,
        explainer: IGExplainerLSTM,
        vocab: torchtext.vocab.Vocab,
        collate_fn: Callable,
        batch_size: int = 128,
    ):
        """
        """
        super().__init__(batched=True)
        self.rationale_format = rationale_format
        self.vocab = vocab
        self.explainer = explainer
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        
    def _index_in_rationale(
        self,
        ngram: List[Text],
        document: List[Dict[Text, Any]],
    ) -> List[Tuple[int, int]]:
        """Given an example, we indes the ngram in the rationale.
        Notice that this time we'll no longer index with normalized
        tokens.
        """
        list_equal = lambda x, y: len(x) == len(y) and all([x[i] == y[i] for i in range(len(x))])
        
        indices = []
        for i in range(len(document) - len(ngram) + 1):
            # print("-" * 20)
            # print([token['text'] for token in document[i:i+len(ngram)]])
            # print(ngram)
            # print("-" * 20)
            if list_equal([token['text'] for token in document[i:i+len(ngram)]], ngram):
                indices.append((document[i]['idx'], document[i + len(ngram) - 1]['idx'] + len(document[i + len(ngram) - 1]['text'])))
                
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
        
        batch = self.collate_fn(examples)
        input_ids = batch['input_ids'].tolist()
        # itos = self.collate_fn.vocab.get_itos()
        itos = self.vocab.get_itos()
        
        attributions: Optional[List[List[float]]] = [[]]
        
        if precomputed_attributions is None:
            # convert dict of lists to list of dicts
            # attributions of shape [batch_size, max_input_length]
            attributions = self.explainer(**batch).tolist()
            
        else:
            attributions = [[precomputed_attributions.get(i, 0) for i in input_id_seq] for input_id_seq in input_ids]

        all_attributions = []
        
        for idx, (iids, attr) in enumerate(zip(input_ids, attributions)):
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
                    "in_rationale_ids": self._index_in_rationale(ngram, examples[idx]['tokenized_inputs']),
                })
                
            all_attributions.append(attribution_dicts)
            
        return {
            "rationale_format": [self.rationale_format] * len(all_attributions),
            "attributions": all_attributions
        }
        
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
        
        for batch in tqdm(dataloader):
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
    
    
@Preprocessor.register("ecqa-counterfactual-generation-preprocessor")
class ECQACounterfactualGenerationPreprocessor(
    Preprocessor
):
    """StrategyQA Counterfactual Generation Preprocessor.
    need to be applied to the output of the Explanation (Global or Local)
    preprocessor outputs
    """
    def __init__(
        self,
        generation_model: HuggingfaceWrapperModule,
        collate_fn: Callable,
        batch_size: int = 32,
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
        self.collate_fn = collate_fn
        
    @overrides
    def _call(
        self,
        examples: Dict[Text, Any],
        *args,
        **kwargs
    ) -> Dict[Text, Any]:
        """
        """
        
        batch = self.collate_fn(examples)
        
        def _extract_rationale(text: Text) -> Text: 
            return text.split("rationale: ")[-1]
        
        return_dict = {
            f"generated_rationale_op{i}": [] for i in range(1, 6)
        }

        with torch.no_grad():

            sequence_ids = self.generation_model.generate(
                batch['input_ids'].to('cuda:0'),
                max_new_tokens=256
            )
            
            inputs = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            decoded = self.tokenizer.batch_decode(sequence_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            assert len(inputs) == len(decoded), "Input and decoded should have the same length."
            
            for chunk_start_idx in range(0, len(inputs), 5):
                chunked_inputs = inputs[chunk_start_idx:chunk_start_idx+5]
                chunked_decoded = decoded[chunk_start_idx:chunk_start_idx+5]

                # debugging info
                for idx, (input_, decode_) in enumerate(zip(chunked_inputs, chunked_decoded)):
                    # print(formatting_t5_generation(input_, decode_))
                    return_dict[f"generated_rationale_op{idx + 1}"].append(
                        _extract_rationale(
                            formatting_t5_generation(input_, decode_)
                        )
                    )
                
        return return_dict