from typing import Dict, Any, Text, Optional, List, Tuple, Union
import torch
from transformers import PreTrainedTokenizer
from .collate_fn import CollateFn
from .strategyqa_collate_fn import __TEMPLATES__
from overrides import overrides

CACHE_DIR="/scratch/ylu130/model-hf"

__QUESTION_TEMPLATES__ = "{question} Options: {op1}, {op2}, {op3}, {op4}, {op5}"

class ECQACollateFn(CollateFn):
    
    __TEMPLATES__ = __TEMPLATES__
    __QUESTION_TEMPLATES__ = __QUESTION_TEMPLATES__

    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        template = self.__TEMPLATES__[self.rationale_format]

        return template.format(
            gold_rationale="{pos} {neg}".format(
                    pos = item['taskA_pos'],
                    neg = item['taskA_neg']
                ),
            base_rationale=item[f'vacuous_rationale_{item["label"]}'],
            leaky_rationale=f"The answer is {item['q_ans']}"
        )

    def question_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """

        return self.__QUESTION_TEMPLATES__.format(
            question=item['q_text'],
            op1=item['q_op1'],
            op2=item['q_op2'],
            op3=item['q_op3'],
            op4=item['q_op4'],
            op5=item['q_op5'],
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {self.question_templating(item)} rationale: {self.rationale_templating(item)}"
    
    
class ECQAQARationalizationCollateFn(ECQACollateFn):

    def __init__(
        self,
        model_name: Text,
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
        # whether to do language modeling or seq2seq modeling
        self.is_lm = model_name.startswith("gpt")

    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        if self.is_lm:
            return "question: {question} answer: {answer}. rationale: {rationale}".format(
                    question=self.question_templating(item),
                    answer=item['q_ans'],
                    rationale=self.rationale_templating(item)
                )
        else:
            return "question: {question} answer: {answer}. rationale:".format(
                    question=self.question_templating(item),
                    answer=item['q_ans']
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
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        input_ids = input_outputs.input_ids
        attention_mask = input_outputs.attention_mask
        
        if not self.is_lm:
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
        else:
            # prepare for language modeling
            q_id = self.tokenizer(' rationale', return_tensors='pt')['input_ids'][0][0]
            q_idxs = (input_ids == q_id).nonzero()
            for idx, attn_mask in enumerate(attention_mask):
                attn_mask[q_idxs[idx][1]:] = 0
            temp_labels = []
            for idx, input_id in enumerate(input_ids):
                label = input_id.clone()
                label[:q_idxs[idx][1]] = self.tokenizer.pad_token_id
                temp_labels.append(label)
            labels = torch.stack(temp_labels)

        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
        }
    
    
class ECQARationaleGenerationCollateFn():
    
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
            f"Please provide a rationale to explain the correct option to the given question. Also, provide an explanation for each wrong option.\n{demonstration}\nquestion: {question} answer: {answer} rationale:" for demonstration, question, answer in x
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
