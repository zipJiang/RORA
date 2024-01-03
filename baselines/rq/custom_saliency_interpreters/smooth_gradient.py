import math
from typing import Dict, Any

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import (
    SaliencyInterpreter,
)
from allennlp.predictors import Predictor
from allennlp.nn import util


@SaliencyInterpreter.register("smooth_gradient")
class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)

    Registered as a `SaliencyInterpreter` with name "smooth_gradient".
    """

    def __init__(self, predictor: Predictor, stdev, num_samples) -> None:
        super().__init__(predictor)
        # Hyperparameters
        self.stdev = stdev
        self.num_samples = num_samples

    def saliency_interpret_from_json(self, inputs: JsonDict, squared, l1) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        instances_with_grads["label"] = dict()
        instances_with_grads["explanation"] = dict()
        instances_with_grads["full"] = dict()
        for idx, instance in enumerate(labeled_instances):
            if not instance:
                print("skipping index: ", idx)
                continue

            # do once for label, and once for explanation
            for term in ["label", "explanation", "full"]:
                tmp_instance = instance.copy()

                # create decoder_input_ids object such that decoding occurs as normal based on predicted sequence, even as we change the labels used to compute loss
                tmp_instance["decoder_input_ids"] = self.predictor._model._shift_right(
                    tmp_instance["full_predictions"]
                )
                tmp_instance["labels"] = tmp_instance["%s_predictions" % term]
                del tmp_instance["label_predictions"]
                del tmp_instance["explanation_predictions"]
                del tmp_instance["full_predictions"]

                # Run smoothgrad
                grads = self._smooth_grads(tmp_instance, squared=squared)

                # Normalize results
                for key, grad in grads.items():
                    # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
                    # Fine for now, but should fix for consistency.

                    # The [0] here is undo-ing the batching that happens in get_gradients.
                    # we remove absolute value and L1-normalization
                    if l1:
                        embedding_grad = numpy.linalg.norm(grad[0], ord=1, axis=1)
                    else:
                        embedding_grad = numpy.sum(grad[0], axis=1)
                    normalized_grad = [e for e in embedding_grad]
                    grads[key] = normalized_grad

                instances_with_grads[term]["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, stdev: float):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """

        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape).to(output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        embedding_layer = self.predictor._model.shared
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(
        self, instance: Instance, squared=False
    ) -> Dict[str, numpy.ndarray]:
        total_gradients: Dict[str, Any] = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            grads = self.predictor.get_gradients(instance)[0]
            handle.remove()

            # Sum gradients
            if total_gradients == {}:
                total_gradients = grads
            else:
                for key in grads.keys():
                    if squared:
                        total_gradients[key] += numpy.square(grads[key])
                    else:
                        total_gradients[key] += grads[key]

        # Average the gradients
        for key in total_gradients.keys():
            total_gradients[key] /= self.num_samples

        return total_gradients
