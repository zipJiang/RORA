"""Multiple preprocesor for the ECQA dataset.
"""
import transformers
import numpy as np
from typing import Dict, Any, Text, Optional
from overrides import overrides
from .preprocessor import Preprocessor


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
    def _call(self, examples: Dict[Text, Any]) -> Dict[Text, Any]:
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