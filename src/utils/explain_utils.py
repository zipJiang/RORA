"""Using the explainer to explain the model's predictions.
"""
from spacy.tokens import Doc
import numpy as np
from src.explainers.ig_explainer import IGExplainerFastText
from src.collate_fns.strategyqa_collate_fn import StrategyQANGramClassificationCollateFn
from typing import Any, Dict, List, Optional, Text, Tuple


def get_explanation_scores(
    examples: Dict[Text, List[Text]],
    collate_fn: StrategyQANGramClassificationCollateFn,
    explainer: IGExplainerFastText,
) -> Dict[Text, List[Text]]:
    """Run the explanation model to get the explanation scores.
    """
    
    # convert dict of lists to list of dicts
    keys = list(examples.keys())
    
    examples = [
        {k: examples[k][idx] for k in keys}
        for idx in range(len(examples[keys[0]]))
    ]
    
    batch = collate_fn.collate(examples)
    
    # attributions of shape [batch_size, max_input_length]
    attributions: np.ndarray = explainer(**batch).tolist()
    
    input_ids = batch['input_ids'].tolist()
    itos = collate_fn.vocab.get_itos()

    def _index_in_rationale(
        ngram: List[Text],
        document: Doc
    ) -> List[Tuple[int, int]]:
        """Given an example, we indes the ngram in the rationale.
        """
        tokens = [token for token in document]
        
        list_equal = lambda x, y: len(x) == len(y) and all([x[i] == y[i] for i in range(len(x))])
        
        indices = []
        for i in range(len(tokens) - len(ngram) + 1):
            if list_equal([t.text for t in tokens[i:i+len(ngram)]], ngram):
                indices.append((tokens[i].idx, tokens[i + len(ngram) - 1].idx + len(tokens[i + len(ngram) - 1])))
                
        return indices
    
    all_attributions = []
    
    for idx, (iids, attr) in enumerate(zip(input_ids, attributions)):
        document = collate_fn.nlp(collate_fn.templating(examples[idx]))
        attribution_dicts = []
        for tidx, a in zip(iids, attr):
            # skip the pad tokens
            if tidx == explainer.pad_idx:
                continue
            
            ngram = itos[tidx].split(' ')
            
            attribution_dicts.append({
                "index": tidx,
                "ngram": ngram,
                "score": a,
                "in_rationale_ids": _index_in_rationale(ngram, document),
            })
            
        all_attributions.append(attribution_dicts)
        
    return {
        "rationale_format": [collate_fn.rationale_format] * len(all_attributions),
        "attributions": all_attributions
    }