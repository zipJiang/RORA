"""A set of preprocessor for the preprocessing
of strategyQA dataset.
"""
import transformers
import numpy as np
from typing import Dict, Any, Text, Optional
from overrides import overrides
from .preprocessor import Preprocessor


class StrategyQAVacuousRationalePreprocessor(
    Preprocessor
):
    """
    """
    
    __MODEL_NAME__ = "domenicrosati/question_converter-3b"
    __MODEL_TYPE__ = "t5-3b"

    def __init__(
        self,
        batch_size: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        num_beam_groups: int = 1,
        num_beams: int = 1,
        max_new_tokens: int = 128,
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
    def _call(self, examples: Dict[Text, Any]) -> Dict[Text, Any]:
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