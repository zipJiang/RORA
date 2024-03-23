"""Isolating Out NGrams from the StrategyQA Data
Preparation.
"""
from typing import List, Text
from typing import Callable, TypeVar, Iterator


T_ = TypeVar('T_')


def generate_no_more_than_ngrams(
    x: List[T_],
    n: int,
    joint_func: Callable[[List[T_]], T_] = lambda x: ' '.join(x)
) -> Iterator[T_]:
    """Given a list of text,
    generate all ngrams from 1 to n.
    """
    
    if n > 1:
        for i in range(2, n+1):
            yield from [joint_func(t) for t in zip(*[x[ii:] for ii in range(i)])]
    else:
        yield from x