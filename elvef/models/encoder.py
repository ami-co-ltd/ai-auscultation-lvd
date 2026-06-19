#!/usr/bin/env python
# pylint: disable=no-else-return,line-too-long,unused-argument

import omegaconf
import torch

from elvef.models.base_extractor import BaseExtractor
from elvef.models.effnet_encoder import create_effnet1d

# from heartsound_train.utils.dataclasses.models.encoder import EncoderParam

EncoderType = torch.nn.Module


class Encoder(torch.nn.Module):
    """Encoder to generate hidden vectors

    Note:
        - use encoder_module for future updates, e.g.,
          - multi-branch/multi-channel encoder
          - concat of hand-crafted features

    Attributes:
        encoder_module (EncoderType): encoder module

    """

    def __init__(
        self,
        encoder_module: EncoderType,
    ) -> None:
        """init

        Args:
            encoder_module (EncoderType): encoder module
        """
        super().__init__()
        self.encoder_module = encoder_module

    def get_output_dim(self) -> int:
        """get hidden vector dimension of encoder output

        Returns:
            int: dimension

        """
        return self.encoder_module.get_output_dim()  # type: ignore[operator]

    def forward(self, xs_pad: torch.Tensor) -> torch.Tensor:
        """forward path of encoder module

        Args:
            xs_pad (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        """
        return self.encoder_module(xs_pad)


def create_encoder(
    config: omegaconf.dictconfig.DictConfig,
    is_test: bool,
    can_use_cuda: bool,
    input_channels: int = 1,
    input_seqlen: int = int(5.0 * 2000),
) -> Encoder:
    """create Encoder instance

    Args:
        config (omegaconf.dictconfig.DictConfig): configuration
        is_test (bool): test mode if True
        can_use_cuda (bool): use cuda if True.

    Raises:
        ValueError: Got unknown encoder type.

    Returns:
        Encoder: encoder instance
    """

    extractor = BaseExtractor()

    encoder = Encoder(create_effnet1d(input_channels, input_seqlen, config, extractor))

    return encoder
