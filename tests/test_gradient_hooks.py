"""This function is used to test wqhether the hooks
as we expected.
"""
import torch
import torchtext
import unittest
from torchtext.vocab import Vocab
from src.models.fasttext import FastTextModule
from src.explainers.ig_explainer import IGExplainerFastText
from src.collate_fns.strategyqa_collate_fn import StrategyQANGramClassificationCollateFn


class TestGradientHooks(unittest.TestCase):
    def setUp(self):
        self.model_path = "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_gl/best_1/"
        self.model = FastTextModule.load_from_dir(
            self.model_path
        )
        self.vocab: torchtext.vocab.Vocab = torch.load("data/processed_datasets/strategyqa/vocab_format=gl_ng=2_mf=1_mt=10000.pt")
        self.collate_fn = StrategyQANGramClassificationCollateFn(
            rationale_format="gl",
            vocab=self.vocab,
            max_input_length=32,
            nlp_model="en_core_web_sm",
            num_ngrams=2,
        )
        
        self.input_batch = [
            {
                "question": "What is a good answer?",
                "facts": [
                    "This is a fact.",
                    "Those are not."
                ],
                "vacuous_rationale": "Yes.",
                "answer": True
            },
        ]
        
        self.explainer = IGExplainerFastText(
            model=self.model,
            num_steps=5,
            max_input_length=32,
            device="cuda:0"
        )
        
        self.multiplicity = 1
        
    def test_gradient_hook(self):
        """Test whether the gradient can be
        successfully extracted by the hook.
        """
        
        batch = self.collate_fn(self.input_batch)
        
        # attributions of shape [batch_size, max_input_length]
        attributions = self.explainer(**batch)
        attributions = attributions.tolist()
        # input_ids of shape [batch_size, max_input_length]
        itos = self.vocab.get_itos()
        ngrams = [[itos[tki] for tki in bow if tki != self.model.pad_idx] for bow in batch['input_ids'].cpu()]
        
        for ngram_list, attr_list in zip(ngrams, attributions):
            for ngram, attr in zip(ngram_list, attr_list):
                print(ngram, attr)