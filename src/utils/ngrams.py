"""Isolating Out NGrams from the StrategyQA Data
Preparation.
"""
from typing import List, Text


def generate_no_more_than_ngrams(
    x: List[Text],
    n: int
) -> List[Text]:
    """Given a list of text,
    generate all ngrams from 1 to n.
    """
    
    # i-gram
    ngram_set = set(x)
    
    if n > 1:
        for i in range(2, n+1):
            ngram_set = ngram_set.union(set([' '.join(t) for t in zip(*[x[ii:] for ii in range(i)])]))
            
    return list(ngram_set)