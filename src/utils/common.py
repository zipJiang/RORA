"""Common utils that can be shared across different tasks.
"""
from typing import Text
import torch
import torchtext
import re


__PATTERN__ = re.compile(r"(<extra_id_(\d+)>)")
__MASK_SPLIT__ = r"<FIXED_MASK_SPLIT>"
__SPECIAL_TOKEN_REMOVAL__ = re.compile(r"(<pad>|<s>|</s>|<unk>)")


def formatting_t5_generation(
    input_sequence: Text,
    output_sequence: Text,
) -> Text:
    """Generate a formatted string for the T5 generation,
    given the masked part.
    """
    inputs = __PATTERN__.sub(__MASK_SPLIT__, __SPECIAL_TOKEN_REMOVAL__.sub("", input_sequence)).split(__MASK_SPLIT__)
    outputs = __PATTERN__.sub(__MASK_SPLIT__, __SPECIAL_TOKEN_REMOVAL__.sub("", output_sequence)).split(__MASK_SPLIT__)
    
    inputs = inputs[1:] if inputs[0] == "" else inputs
    outputs = outputs[1:] if outputs[0] == "" else outputs
    
    results = []
    
    for inp, oup in zip(inputs, outputs):
        results.append(inp)
        results.append(oup)
        
    return "".join(results)


def move_to_device(obj, device):
    """
    """
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(v, device) for v in obj])
    elif isinstance(obj, set):
        return set([move_to_device(v, device) for v in obj])
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj
    
    
def get_vocab(handle: Text) -> torchtext.vocab.Vocab:
    if handle == "fasttext":
        fasttext_vectors = torchtext.vocab.FastText()
        vocab = torchtext.vocab.vocab(fasttext_vectors.stoi)
        vocab.set_default_index(len(vocab) - 1)  # This is sort of a hack to make sure that the <unk> token is the last token in the vocab.
        
        return vocab
    
    return torch.load(handle)

def get_embedding(handle: Text) -> torch.nn.Module:
    if handle == "fasttext":
        return torch.nn.Embedding.from_pretrained(torchtext.vocab.FastText().vectors, padding_idx=-1)
    else:
        raise ValueError(f"Unknown embedding handle: {handle}")
    
    
def dict_of_list_to_list_of_dict(d: dict) -> list:
    """Convert a dictionary of lists to a list of dictionaries.
    """
    keys = d.keys()
    return [dict(zip(keys, vals)) for vals in zip(*d.values())]

def list_of_dict_to_dict_of_list(l: list) -> dict:
    """Convert a list of dictionaries to a dictionary of lists.
    """
    keys = l[0].keys()
    return {key: [d[key] for d in l] for key in keys}