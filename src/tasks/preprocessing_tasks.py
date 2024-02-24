"""Now we are separating the preprocessing tasks from the main
piepline. This is to make the pipeline more modular and presumably
faster to run.
"""
from typing import Text, Dict, Any, List
import os
import torch
from overrides import overrides
from registrable import Lazy
import datasets
from glob import glob
from transformers import AutoTokenizer
from ..models import Model
from ..explainers import Explainer
from .task import Task
from ..collate_fns import (
    CollateFn,
)
from ..preprocessors import (
    Preprocessor,
    ECQACounterfactualGenerationPreprocessor,
    ECQAGlobalExplanationPreprocessor
)
from ..utils.common import (
    get_vocab,
    dict_of_list_to_list_of_dict,
    list_of_dict_to_dict_of_list
)


@Task.register("preprocessing-removal")
class PreprocessRemovalTask(Task):
    """This task is responsible for creating a new dataset
    """
    def __init__(
        self,
        data_dir: Text,
        output_dir: Text,
        vocab_path: Text,
        collate_fn: Lazy[CollateFn],
    ):
        """This task takes a dataset and prepare the data
        for removal training.
        """
        super().__init__()
        self.vocab = get_vocab(vocab_path)
        self.collate_fn = collate_fn.construct(
            vocab=self.vocab
        )
        self.data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
        self.output_dir = output_dir
        
    @overrides
    def run(self):
        """
        """
        
        def _processing_func(batch: Dict[Text, Any]) -> Dict[Text, Any]:
            batch = dict_of_list_to_list_of_dict(batch)
            return self.collate_fn(batch)

        for subdir in glob(self.data_dir + "*"):
            dataset = datasets.load_from_disk(subdir)
            dataset = dataset.map(
                _processing_func,
                batched=True,
                load_from_cache_file=False
            )

            # save to disk
            dataset.save_to_disk(
                os.path.join(
                    self.output_dir,
                    os.path.basename(subdir)
                )
            )
            
            
@Task.register("preprocessing-generation")
class PreprocessGenerationTask(Task):
    def __init__(
        self,
        data_dir: Text,
        output_dir: Text,
        vocab_path: Text,
        model_name: Text,
        batch_size: int,
        attribution_model: Model,
        explainer: Lazy[Explainer],
        explainer_preprocessor: Lazy[Preprocessor],
        generation_collate_fn: Lazy[CollateFn],
    ):
        """
        """
        super().__init__()
        self.data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.vocab = get_vocab(vocab_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.attribution_model = attribution_model
        self.explainer = explainer.construct(
            model=self.attribution_model,
        )
        
        # define the quick and easy collate_fn
        self.preprocessor: Preprocessor = explainer_preprocessor.construct(
            explainer=self.explainer,
            collate_fn=lambda x: {k[1:]: torch.tensor(v) for k, v in list_of_dict_to_dict_of_list(x).items() if k.startswith("_")},
            vocab=self.vocab,
        )
        
        self.generation_collate_fn = generation_collate_fn.construct(
            tokenizer=self.tokenizer
        )
        
    @overrides
    def run(self):
        """
        """
        
        train_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "train"))
        val_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "validation"))
        test_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "test"))
        
        train_dataset, features = self.preprocessor(train_dataset)
        val_dataset, _ = self.preprocessor(val_dataset, features=features)
        test_dataset, _ = self.preprocessor(test_dataset, features=features)
        
        # now after this run, clean-up all the fields with _ prefix,
        # which is not neede anymore as it is the model inputs
        train_dataset = train_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), train_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            lambda x: self.generation_collate_fn(dict_of_list_to_list_of_dict(x)),
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        val_dataset = val_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), val_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            lambda x: self.generation_collate_fn(dict_of_list_to_list_of_dict(x)),
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        test_dataset = test_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), test_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            lambda x: self.generation_collate_fn(dict_of_list_to_list_of_dict(x)),
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        
        train_dataset.save_to_disk(os.path.join(self.output_dir, "train"))
        val_dataset.save_to_disk(os.path.join(self.output_dir, "validation"))
        test_dataset.save_to_disk(os.path.join(self.output_dir, "test"))
        
        
@Task.register("preprocessing-baseline")
class PreprocessingBaselineTask(Task):
    def __init__(
        self,
        data_dir: Text,
        output_dir: Text,
        batch_size: int,
        model_name: Text,
        collate_fn: Lazy[CollateFn],
    ):
        """Take a generated dataset and prepare it for IRM training.
        """
        
        self.batch_size = batch_size
        self.data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
        self.output_dir = output_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collate_fn = collate_fn.construct(
            tokenizer=self.tokenizer
        )
        
    def run(self):
        """
        """
        train_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "train"))
        val_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "validation"))
        test_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "test"))
        
        
        # now we run the generation collate_fn to tokenize all the pairs
        def _collate_fn_wrapper(examples: List[Dict[Text, Any]]) -> Dict[Text, Any]:
            return self.collate_fn(dict_of_list_to_list_of_dict(examples))
        
        # def _eval_collate_fn_wrapper(examples: List[Dict[Text, Any]]) -> Dict[Text, Any]:
        #     return self.eval_collate_fn(dict_of_list_to_list_of_dict(examples))
        
        train_dataset = train_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), val_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            _collate_fn_wrapper,
            batched=True,
            batch_size=self.batch_size,
            load_from_cache_file=False,
        )

        val_dataset = val_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), val_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            _collate_fn_wrapper,
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        test_dataset = test_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), test_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            _collate_fn_wrapper,
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )

        train_dataset.save_to_disk(os.path.join(self.output_dir, "train"))
        val_dataset.save_to_disk(os.path.join(self.output_dir, "validation"))
        test_dataset.save_to_disk(os.path.join(self.output_dir, "test"))

        
@Task.register("preprocessing-irm")
class PreprocessingIRMTask(Task):
    def __init__(
        self,
        data_dir: Text,
        output_dir: Text,
        batch_size: int,
        generation_model: Model,
        generation_collate_fn: Lazy[CollateFn],
        model_name: Text,
        counterfactual_preprocessor: Lazy[Preprocessor],
        collate_fn: Lazy[CollateFn],
        eval_collate_fn: Lazy[CollateFn],
    ):
        """Take a generated dataset and prepare it for IRM training.
        """
        
        self.batch_size = batch_size
        self.data_dir = data_dir + '/' if data_dir[-1] != '/' else data_dir
        self.output_dir = output_dir
        self.generation_model = generation_model
        self.generation_collate_fn = generation_collate_fn.construct(
            tokenizer=self.generation_model.tokenizer,
            intervention_on_label=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collate_fn = collate_fn.construct(
            tokenizer=self.tokenizer
        )
        self.eval_collate_fn = eval_collate_fn.construct(
            tokenizer=self.tokenizer
        )
        
        def _collate_fn(
            examples: Dict[Text, Any]
        ) -> Dict[Text, Any]:
            """This collate_fn is used to collate the counterfactual generation data,
            and will be used to do the counterfactual generation.
            """
            _flatten = lambda x: x.view(-1, x.size(-1)) if x.dim() > 2 else x.flatten()
            
            return {k[1:]: _flatten(torch.tensor(v)) for k, v in examples.items() if k.startswith("_")}
        
        self.counterfactual_preprocessor = counterfactual_preprocessor.construct(
            generation_model=self.generation_model,  # add this one on to show the dependency (tokenizer)
            collate_fn=_collate_fn
        )
        
    def run(self):
        """
        """
        train_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "train"))
        val_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "validation"))
        test_dataset = datasets.load_from_disk(os.path.join(self.data_dir, "test"))
        
        train_dataset = train_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), train_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            lambda x: self.generation_collate_fn(dict_of_list_to_list_of_dict(x)),
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        train_dataset = self.counterfactual_preprocessor(train_dataset)
        
        # now we run the generation collate_fn to tokenize all the pairs
        def _collate_fn_wrapper(examples: List[Dict[Text, Any]]) -> Dict[Text, Any]:
            return self.collate_fn(dict_of_list_to_list_of_dict(examples))
        
        def _eval_collate_fn_wrapper(examples: List[Dict[Text, Any]]) -> Dict[Text, Any]:
            return self.eval_collate_fn(dict_of_list_to_list_of_dict(examples))
        
        train_dataset = train_dataset.map(
            _collate_fn_wrapper,
            batched=True,
            batch_size=self.batch_size,
            load_from_cache_file=False,
        )

        val_dataset = val_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), val_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            _eval_collate_fn_wrapper,
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )
        
        test_dataset = test_dataset.map(
            lambda _: {},
            remove_columns=list(filter(lambda x: x.startswith("_"), test_dataset.column_names)),
            load_from_cache_file=False,
            batched=True,
        ).map(
            _eval_collate_fn_wrapper,
            batched=True,
            load_from_cache_file=False,
            batch_size=self.batch_size
        )

        train_dataset.save_to_disk(os.path.join(self.output_dir, "train"))
        val_dataset.save_to_disk(os.path.join(self.output_dir, "validation"))
        test_dataset.save_to_disk(os.path.join(self.output_dir, "test"))