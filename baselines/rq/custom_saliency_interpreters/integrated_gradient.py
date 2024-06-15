import math
from typing import List, Dict, Any

import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import (
    SaliencyInterpreter,
)
from allennlp.nn import util


@SaliencyInterpreter.register("integrated_gradient")
class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)

    Registered as a `SaliencyInterpreter` with name "integrated_gradient".
    """

    def saliency_interpret_from_json(self, inputs: JsonDict, nsamples, l1) -> JsonDict:
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

                # Run integrated gradients
                grads = self._integrate_gradients(tmp_instance, nsamples)

                # Normalize results
                for key, grad in grads.items():
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

    def _register_forward_hook(self, alpha: int, embeddings_list: List):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.

        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = self.predictor._model.shared
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(
        self, instance: Instance, steps
    ) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given [`Instance`](../../data/instance.md)
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use [nsamples] terms in the summation approximation of the integral in integrated grad
        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self.predictor.get_gradients(instance)[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads
