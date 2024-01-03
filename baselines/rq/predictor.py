from tqdm import tqdm
from torch import backends
import torch

"""
Modified from https://github.com/allenai/allennlp/blob/master/allennlp/predictors/predictor.py (v1.3.0)
"""


class Predictor:
    def __init__(
        self, model, dataset_reader, tokenizer, device, frozen: bool = True
    ) -> None:
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self._tokenizer = tokenizer
        self.device = device
        self.cuda_device = next(self._model.named_parameters())[1].get_device()

    def json_to_labeled_instances(self, inputs):  # JsonDict -> List[Instance]
        """
        Converts incoming json to a [`Instance`](../data/instance.md),
        runs the model on the newly created instance, and adds labels to the
        `Instance`s given by the model's output.

        # Returns

        `List[instance]`
            A list of `Instance`'s.
        """
        outputs = []

        for i, inp in enumerate(tqdm(inputs, desc="Computing Generations")):
            for k, v in inp.items():
                inp[k] = v.to(self.device)

            decoded = self._model.generate(
                input_ids=inp["input_ids"],
                max_length=100,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            assert decoded[0, 0] == self._model.config.decoder_start_token_id
            # strip <bos> token which will be re-added later
            decoded = decoded[:, 1:]

            # get break point
            if (
                self._tokenizer.encode("explanation:")[0] not in decoded
                or self._tokenizer.encode("explanation:")[-1] not in decoded
            ):
                print(
                    "No separator token: ",
                    self._tokenizer.decode(decoded.squeeze(0).tolist()),
                )
                inp = {}
            else:
                first_break = (
                    decoded.squeeze(0)
                    .tolist()
                    .index(self._tokenizer.encode("explanation:")[0])
                )
                inp["label_predictions"] = torch.tensor(
                    [
                        decoded.squeeze(0).tolist()[:first_break]
                        + [-100 for i in range(first_break, len(decoded.squeeze(0)))]
                    ],
                    dtype=decoded.dtype,
                ).to(self.device)

                second_break = (
                    decoded.squeeze(0)
                    .tolist()
                    .index(self._tokenizer.encode("explanation:")[-1])
                )
                # EOS token already cut
                if self._tokenizer.eos_token_id in decoded:
                    end_pos = (
                        decoded.squeeze(0).tolist().index(self._tokenizer.eos_token_id)
                    )
                    inp["explanation_predictions"] = torch.tensor(
                        [
                            [-100 for i in range(second_break + 1)]
                            + decoded.squeeze(0).tolist()[second_break + 1 : end_pos]
                            + [-100 for i in range(end_pos, len(decoded.squeeze(0)))]
                        ],
                        dtype=decoded.dtype,
                    ).to(self.device)
                else:
                    inp["explanation_predictions"] = torch.tensor(
                        [
                            [-100 for i in range(second_break + 1)]
                            + decoded.squeeze(0).tolist()[second_break + 1 :]
                        ],
                        dtype=decoded.dtype,
                    ).to(self.device)

                inp["full_predictions"] = decoded
                assert (
                    inp["full_predictions"].shape
                    == inp["explanation_predictions"].shape
                    == inp["label_predictions"].shape
                )

                inp["decoder_attention_mask"] = torch.ones(decoded.shape).to(
                    self.device
                )
            outputs.append(inp)

        return outputs

    def get_gradients(
        self, instance
    ):  # List[Instance] -> Tuple[Dict[str, Any], Dict[str, Any]]
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset_tensor_dict = instance
        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.forward(**dataset_tensor_dict)
            loss = outputs[0]
            self._model.zero_grad()
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, outputs

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        [`BasicTextFieldEmbedder`](../modules/text_field_embedders/basic_text_field_embedder.md)
        class. Used to save the gradients of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = self._model.shared
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks
