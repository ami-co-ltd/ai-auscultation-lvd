#!/usr/bin/env python
# pylint: disable=no-else-return

from logging import getLogger
from typing import Union

import torch
from omegaconf import DictConfig

from elvef.models.attention_decoder import AttentionDecoder

logger = getLogger(__name__)


DecoderType = torch.nn.Module


class Decoder(torch.nn.Module):
    """Decoder to generate logits and posterior probability

    Attributes:
        decoder_module (DecoderType): decoder module
        loss_weights (Union[torch.Tensor, None]): loss weights
    """

    def __init__(self, decoder_module: DecoderType) -> None:
        """init

        Args:
            decoder_module (DecoderType): decoder module
        """
        super().__init__()
        self.decoder_module = decoder_module
        self.loss_weights: Union[torch.Tensor, None] = None

    def get_output_dim(self) -> int:
        """return output dimension"""
        return self.decoder_module.get_output_dim()  # type: ignore[operator]

    def forward(
        self,
        xs_pad: torch.Tensor,
        output_labels: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
        sample_names: Union[list[str], None] = None,
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """forward path of decoder module

        Note:
            B: batch size
            C: channel size
            T: sequence length
            O: output dimension

        Args:
            xs_pad (torch.Tensor): input tensor of shape (B, C, T)
            output_labels (Union[torch.Tensor,  None]): output labels of shape (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights
            sample_names (Union[list[str], None]): sample names (B,).
                Defaults to None.

        Returns:
            tuple[torch.Tensor, Union[torch.Tensor, None]]: logits and output labels

        """
        # Different decoders accept different arguments.
        kwargs: dict[str, list[str]] = {}
        if isinstance(self.decoder_module, AttentionDecoder):
            kwargs["sample_names"] = sample_names  # type: ignore

        xs_pad, output_labels, self.loss_weights = self.decoder_module(
            xs_pad, output_labels, loss_weights, **kwargs
        )
        return xs_pad, output_labels


def create_decoder(input_dim: int, output_dim: int, config: DictConfig) -> Decoder:
    """create Decoder instance

    Args:
        input_dim (int): input dimension for decoder module
        output_dim (int): output dimension
        config (DictConfig): configuration

    Raises:
        ValueError: failed to create decoder_module

    Returns:
        Decoder: decoder instance

    """
    decoder: Decoder

    if config.model.decoder.decoder_type == "attention_pooling":
        decoder = Decoder(
            AttentionDecoder.build_from_yaml(input_dim, output_dim, config, extra_dim=0)
        )
    else:
        raise NotImplementedError

    return decoder
