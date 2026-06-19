#!/usr/bin/env python
# pylint: disable=no-member,line-too-long,too-many-arguments,unused-argument

from typing import List, Tuple, Type, TypeVar, Union

import omegaconf
import torch

from elvef.models.decoder import Decoder, create_decoder
from elvef.models.define_activation import ActivationType, create_activation
from elvef.models.encoder import Encoder, create_encoder

T = TypeVar("T", bound="Predictor")


class Predictor(torch.nn.Module):
    """Top-level predictor module

    Attributes:
        encoder (Encoder): encoder
        decoder (Decoder): decoder
        last_activation_function (ActivationType):
            last activation function to convert logits to posterior probabilities
        segment_voting_rule (str): merge pred_probs estimated by multiple segments
            using 'mean' or 'max'. Defaults to 'mean'.

    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        last_activation_function: ActivationType,
    ) -> None:
        """init

        Args:
            encoder (Encoder): encoder
            decoder (Decoder): decoder
            last_activation_function (ActivationType):
                last activation function to convert logits to posterior probabilities
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.last_activation_function_module = last_activation_function
        self.segment_voting_rule = "mean"

    @classmethod
    def build_model(
        cls: Type[T],
        output_dim: int,
        config: omegaconf.dictconfig.DictConfig,
        is_test: bool,
        can_use_cuda: bool,
        **kwargs,
    ) -> T:
        """create Predictor instance

        Args:
            output_dim (int): output dimension
            config (omegaconf.dictconfig.DictConfig): configuration
            is_test (bool): test mode if True
            can_use_cuda (bool): use cuda if True

        Returns:
            Predictor: predictor instance

        """
        last_activation_function_name = str(
            config["model"]["last_activation_function_name"]
        )
        encoder = create_encoder(config, is_test, can_use_cuda)
        decoder = create_decoder(encoder.get_output_dim(), output_dim, config)

        return cls(
            encoder,
            decoder,
            create_activation(last_activation_function_name),
        )

    def get_output_dim(self) -> int:
        """return output dimension"""
        return self.decoder.get_output_dim()

    def forward(
        self,
        xs_pad: torch.Tensor,
        output_labels: Union[torch.Tensor, None] = None,
        loss_weights: Union[torch.Tensor, None] = None,
        hospital_labels: Union[torch.Tensor, None] = None,
        sample_names: Union[List[str], None] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """compute logits

        Note:
            - shape:
              B: batch size, C: channel size, T: sequence length
              N: test-time augmentation axis, P: auscultation location
              O: output dimension

        Args:
            xs_pad (torch.Tensor): input data of shape (B, C, T), (B, N, C, T),
                or (B, N, P, C, T).
            output_labels (Union[torch.Tensor,  None]): output labels of shape (B, O).
                Defaults to None.
            loss_weights (Union[torch.Tensor, None]): loss weights.
                Defaults to None.
            sample_names (Union[List[str], None]): sample names (B,).
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None]]: logits and output labels

        """
        # is_mil = xs_pad.dim() == 4
        # if is_mil:
        #     batch, n_expand, channels, seqlen = xs_pad.size()
        #     xs_pad = xs_pad.view(-1, channels, seqlen)

        encoder_output = self.encoder(xs_pad)

        logits, labels = self.decoder(
            encoder_output, output_labels, loss_weights, sample_names
        )
        return logits, labels

    def predict(
        self,
        xs_pad: torch.Tensor,
        n_expand: int = -1,
        sample_names: Union[List[str], None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute posterior probabilities

        Note:
            - tensor shape:
              - B: batch size
              - N: expansion size
              - C: channel size
              - T: sequence length
              - O: output dimension
            - xs_pad is (B, C, T) if n_expand is -1. (BxN, C, T) otherwise.

        Raises:
            NotImplementedError : Unknown segment_voting_rule

        Args:
            xs_pad (torch.Tensor): input data of shape (B, C, T) or (B, P, C, T).
            n_expand (int): expansion size derived from test-time augmentation.
                Defaults to -1.
            sample_names (Union[List[str], None]): sample names (B,).
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logits and posterior probabilities.
                The tensor shape is (B, O).
        """
        logits, _ = self.forward(
            xs_pad,
            None,
            None,
            sample_names=sample_names,
        )
        pred_prob = self.last_activation_function_module(logits)

        if n_expand > -1:
            if self.segment_voting_rule == "mean":
                logits = torch.mean(logits.view(-1, n_expand, logits.size(-1)), 1)
                pred_prob = torch.mean(
                    pred_prob.view(-1, n_expand, pred_prob.size(-1)), 1
                )
            elif self.segment_voting_rule == "max":
                logits = torch.max(logits.view(-1, n_expand, logits.size(-1)), 1)[0]
                pred_prob = torch.max(
                    pred_prob.view(-1, n_expand, pred_prob.size(-1)), 1
                )[0]
            else:
                raise NotImplementedError("Unknown segment_voting_rule")

        return logits, pred_prob

    def get_model_state(
        self, module_names: Tuple[str, ...], state_name: str
    ) -> Union[torch.Tensor, None]:
        """get internal state

        Args:
            module_names (Tuple[str, ...]): module name
            state_name (str): state name

        Raises:
            ValueError: no module name

        Returns:
            Union[torch.Tensor, None]: retrieved state
        """
        state: Union[torch.Tensor, None] = None

        if len(module_names) >= 2:
            if module_names[0:2] == ("Predictor", "Encoder"):
                # move to Encoder
                pass

            elif module_names[0:2] == ("Predictor", "Decoder"):
                # move to Decoder
                state = self.decoder.get_model_state(("Decoder",), state_name)  # type: ignore

        elif len(module_names) == 1 and module_names[0] == ("Predictor"):
            # search local state
            pass

        else:
            raise ValueError("No module name")

        return state


def create_predictor_child(
    output_dim: int,
    config: omegaconf.dictconfig.DictConfig,
    is_test: bool,
    can_use_cuda: bool,
) -> Predictor:
    """create Predictor instance

    Args:
        output_dim (int): output dimension
        config (omegaconf.dictconfig.DictConfig): configuration
        is_test (bool): test mode if True
        can_use_cuda (bool): use cuda if True

    Returns:
        Predictor: predictor instance
    """

    last_activation_function_name = str(
        config["model"]["last_activation_function_name"]
    )
    encoder = create_encoder(config, is_test, can_use_cuda)
    decoder = create_decoder(encoder.get_output_dim(), output_dim, config)

    return Predictor(
        encoder,
        decoder,
        create_activation(last_activation_function_name),
    )
