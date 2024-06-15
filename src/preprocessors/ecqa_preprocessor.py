"""Multiple preprocesor for the ECQA dataset.
"""
import transformers
import numpy as np
from typing import Dict, Any, Text, Optional, Union
from overrides import overrides
import datasets
from .preprocessor import Preprocessor, PreprocessorOutput

__QUESTION_TEMPLATES__ = "{question} Options: {op1}, {op2}, {op3}, {op4}, {op5}"


CACHE_DIR="/scratch/ylu130/model-hf"

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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.__MODEL_NAME__, cache_dir=CACHE_DIR)
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.__MODEL_NAME__, cache_dir=CACHE_DIR)
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
        
        labels = []
        for a, a1, a2, a3, a4, a5 in zip(examples['q_ans'], examples['q_op1'], examples['q_op2'], examples['q_op3'], examples['q_op4'], examples['q_op5']):
            if a == a1:
                correct_answer_id = "op1"
            elif a == a2:
                correct_answer_id = "op2"
            elif a == a3:
                correct_answer_id = "op3"
            elif a == a4:
                correct_answer_id = "op4"
            elif a == a5:
                correct_answer_id = "op5"
            else:
                raise ValueError("Answer not found in the options.")
            labels.append(correct_answer_id)

        return {
                **{f"vacuous_rationale_{k}": 
                        self.tokenizer.batch_decode(
                            self.model.generate(
                                v.to(self.device),
                                **self.generation_params
                            ),
                            skip_special_tokens=True
                        )
                    for k, v in tokenized.items()
                }, 
                "label": labels
            }

class ECQASimulationPreprocessor(Preprocessor):
    """
    """
    __QUESTION_TEMPLATES__ = __QUESTION_TEMPLATES__

    def __init__(
        self,
        batch_size: int = 32
    ):
        
        super().__init__(batched=False)
        self.batch_size = batch_size

        self.question_template = "{question}. Is the answer {answer} correct?"
        self.question_to_converter_template = "Is the answer {answer} true for the question: {question}"
    
    @overrides
    def __call__(
        self,
        dataset: datasets.Dataset,
        **kwargs
    ) -> Union[PreprocessorOutput, datasets.Dataset]:
        
        # create a new dataset that use _call to process the dataset
        # every _call will return a dict of lists

        question_list, rationale_list, label_list, questions_to_converter_list = [], [], [], []
        for i in range(len(dataset)):
            output = self._call(dataset[i])
            question_list.extend(output["questions"])
            rationale_list.extend(output["rationales"])
            label_list.extend(output["labels"])
            if "questions_to_converter" in output:
                questions_to_converter_list.extend(output["questions_to_converter"])

        if len(questions_to_converter_list) > 0:
            return datasets.Dataset.from_dict({
                "full_question": question_list,
                "facts": rationale_list,
                "answer": label_list,
                "question": questions_to_converter_list
            })
        else:
            return datasets.Dataset.from_dict({
                "question": question_list,
                "facts": rationale_list,
                "answer": label_list
            })


    @overrides
    def _call(self, example: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """Convert multiple choice question to 
        a binary True/False question
        """
        # process original ecqa dataset
        if "q_op1" in example:
            questions_to_converter = [] # feed to the question converter model
            questions, labels, rationales = [], [], []

            q, o1, o2, o3, o4, o5, a, pos, neg  = example['q_text'], example['q_op1'], example['q_op2'], example['q_op3'], example['q_op4'], example['q_op5'], example['q_ans'], example['taskA_pos'], example['taskA_neg']

            question = self.__QUESTION_TEMPLATES__.format(
                question=q,
                op1=o1,
                op2=o2,
                op3=o3,
                op4=o4,
                op5=o5,
            )
            for op in [o1, o2, o3, o4, o5]:
                questions.append(
                    self.question_template.format(
                        question=question,
                        answer=op
                    )
                )
                labels.append(op==a)
                rationales.append([f"{pos} {neg}"])
                questions_to_converter.append(
                    self.question_to_converter_template.format(
                        question=q,
                        answer=op
                    )
                )

            return {
                "questions": questions,
                "rationales": rationales,
                "labels": labels,
                "questions_to_converter": questions_to_converter
            }
        # process model-generated rationale dataset
        else:
            questions, labels, rationales = [], [], []
            question = example['question']
            options = question.split(" Options: ")[1].split(", ")
            assert len(options) == 5
            answer = example['answer']
            for op in options:
                questions.append(
                    self.question_template.format(
                        question=question,
                        answer=op
                    )
                )
                labels.append(op==answer)
                rationales.append(example['facts'])

            return {
                "questions": questions,
                "rationales": rationales,
                "labels": labels
            }

class COSESimulationPreprocessor(Preprocessor):
    """
    """
    __QUESTION_TEMPLATES__ = __QUESTION_TEMPLATES__

    def __init__(
        self,
        batch_size: int = 32
    ):
        
        super().__init__(batched=False)
        self.batch_size = batch_size

        self.question_template = "{question}. Is the answer {answer} correct?"
    
    @overrides
    def __call__(
        self,
        dataset: datasets.Dataset,
        **kwargs
    ) -> Union[PreprocessorOutput, datasets.Dataset]:
        
        # create a new dataset that use _call to process the dataset
        # every _call will return a dict of lists

        question_list, rationale_list, label_list = [], [], []
        for i in range(len(dataset)):
            output = self._call(dataset[i])
            question_list.extend(output["questions"])
            rationale_list.extend(output["rationales"])
            label_list.extend(output["labels"])

        return datasets.Dataset.from_dict({
            "question": question_list,
            "facts": rationale_list,
            "answer": label_list
        })


    @overrides
    def _call(self, example: Dict[Text, Any], *args, **kwargs) -> Dict[Text, Any]:
        """Convert multiple choice question to 
        a binary True/False question
        """

        questions, labels, rationales = [], [], []
        q, o1, o2, o3, o4, o5, a, r  = example['question'], example['choices'][0], example["choices"][1], example["choices"][2], example["choices"][3], example["choices"][4], example['answer'], example['abstractive_explanation']
        question = self.__QUESTION_TEMPLATES__.format(
            question=q,
            op1=o1,
            op2=o2,
            op3=o3,
            op4=o4,
            op5=o5,
        )
        for op in [o1, o2, o3, o4, o5]:
            questions.append(
                self.question_template.format(
                    question=question,
                    answer=op
                )
            )
            labels.append(op==a)
            rationales.append([r])

        return {
            "questions": questions,
            "rationales": rationales,
            "labels": labels,
        }