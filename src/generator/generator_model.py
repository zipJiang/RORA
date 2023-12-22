"""Open- and closed- source model for generating rationales
"""
import os
import openai
import backoff 
import json
import transformers
from overrides import overrides
from ..models.model import Model

CACHE_DIR="/scratch/ylu130/model-hf"
completion_tokens = prompt_tokens = 0

api_key = os.environ["OPENAI_API_KEY"]
if api_key == None or api_key == "":
    raise Exception("OPENAI_API_KEY not found")
else:
    openai.api_key = api_key


class APIModel:
    def __init__(self, 
                 model: str,
                 temperature: float = 1,
                 max_tokens: int = 256,
                 top_p: float = 1,
                 n: int = 1,):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.n = n
        
    @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
    def chatcompletions_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return openai.Completion.create(**kwargs)

    def chatgpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.chatcompletions_with_backoff(model=self.model, 
                                                messages=self.input, 
                                                temperature=self.temperature, 
                                                max_tokens=self.max_tokens, 
                                                n=self.n,
                                                top_p=self.top_p)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
        return outputs

    def completiongpt(self) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        res = self.completions_with_backoff(model=self.model, 
                                            messages=self.input, 
                                            temperature=self.temperature, 
                                            max_tokens=self.max_tokens, 
                                            n=self.n, 
                                            top_p=self.top_p)
        outputs.extend([choice["text"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
        return outputs

    @staticmethod
    def gpt_usage(model="gpt-4"):
        global completion_tokens, prompt_tokens
        if model == "gpt-4":
            cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif model == "gpt-3.5-turbo":
            cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
        elif "davinci" in model:
            cost = completion_tokens / 1000 * 0.02 + prompt_tokens / 1000 * 0.02
        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


    def __call__(self, input) -> list:
        self.input = input
        if "davinci" in self.model:
            return self.completiongpt()
        else:
            self.input = [{"role": "user", "content": self.input}]
            return self.chatgpt()

class OpenModel(Model):
    def __init__(self,
                 model_handle: str):
        super().__init__()
        self.model_handle = model_handle
        if "gpt" in model_handle:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_handle, cache_dir=CACHE_DIR)
        elif "t5" in model_handle:
            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_handle, cache_dir=CACHE_DIR)

    @overrides
    def save_to_dir(self, path: str):
        """Save the model to the given path.
        """
        self.model.save_pretrained(path)
        
        # save the model_handle as well.
        with open(os.path.join(path, 'custom.config'), 'w', encoding='utf-8') as file_:
            json.dump({
                'model_handle': self.model_handle
            }, file_, indent=4, sort_keys=True)

    @classmethod
    def load_from_dir(cls, path: str) -> "OpenModel":
        """Load the model from the given path.
        """
        with open(os.path.join(path, 'custom.config'), 'r', encoding='utf-8') as file_:
            config = json.load(file_)
            
        model_handle = config['model_handle']
        model = cls(model_handle=model_handle)
        if "gpt" in model_handle:
            model.model = transformers.AutoModelForCausalLM.from_pretrained(path, cache_dir=CACHE_DIR)
        elif "t5" in model_handle:
            model.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(path, cache_dir=CACHE_DIR)
        return model