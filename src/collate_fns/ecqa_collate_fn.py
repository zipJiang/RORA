from typing import Dict, Any, Text, Optional, List, Tuple, Union
import torch
from transformers import PreTrainedTokenizer
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


class ECQACollateFn(CollateFn):
    
    __TEMPLATES__ = __TEMPLATES__

    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        template = self.__TEMPLATES__[self.rationale_format]

        return template.format(
            gold_rationale=item['taskB'],
            base_rationale=item[f'vacuous_rationale_{item["label"]}'],
            leaky_rationale=f"The answer is {item['q_ans']}"
        )

    def question_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        template = "{question} Options: {op1} {op2} {op3} {op4} {op5}"

        return template.format(
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
    
    
        
        
    