import math

from typing import List
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import (
    SaliencyInterpreter,
)
from allennlp.nn import util


@SaliencyInterpreter.register("simple_gradient")
class SimpleGradient(SaliencyInterpreter):
    """
    Registered as a `SaliencyInterpreter` with name "simple_gradient".
    """

    def saliency_interpret_from_json(
        self, inputs: JsonDict, multiply, l1
    ) -> JsonDict:  # JsonDict is just Dict[str, Any]
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        labeled_instances = self.predictor.json_to_labeled_instances(
            inputs
        )  # JsonDict -> List[Instance]. Also adds predictions from running model.forward()
        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list: List[numpy.ndarray] = []

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

                # Hook used for saving embeddings
                handle = self._register_forward_hook(embeddings_list)
                grads = self.predictor.get_gradients(tmp_instance)[0]
                handle.remove()

                # Gradients come back in the reverse order that they were sent into the network
                embeddings_list.reverse()
                for key, grad in grads.items():
                    # Get number at the end of every gradient key (they look like grad_input_[int], we're getting this [int] part and subtracting 1 for zero-based indexing).
                    # This is then used as an index into the reversed input array to match up the gradient and its respective embedding.
                    input_idx = int(key[-1]) - 1
                    # The [0] here is undo-ing the batching that happens in get_gradients.
                    # we remove absolute value and L1-normalization
                    if multiply:
                        item = (
                            grad[0] * embeddings_list[input_idx]
                        )  # elementwise multiply
                    else:
                        item = grad[0]
                    if l1:
                        emb_grad = numpy.linalg.norm(item, ord=1, axis=1)
                    else:
                        emb_grad = numpy.sum(item, axis=1)
                    normalized_grad = [e for e in emb_grad]
                    grads[key] = normalized_grad

                instances_with_grads[term]["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, embeddings_list: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

        embedding_layer = self.predictor._model.shared
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle
