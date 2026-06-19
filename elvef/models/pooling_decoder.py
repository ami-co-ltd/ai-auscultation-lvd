#!/usr/bin/env python
# pylint: disable=line-too-long

from logging import getLogger
from typing import Any

import omegaconf
import torch

from elvef.utils.dataclasses.models.mlp import MlpParam

logger = getLogger(__name__)


class PoolingDecoder(torch.nn.Module):
    """PoolingDecoder

    Attributes:
        layers (torch.nn.modules.container.ModuleList):
            torch module list
        output_dim (int): output dimension

    """

    @staticmethod
    def help():
        """help"""
        logger.warning("")

    @staticmethod
    def create_pooling_decoder_arguments(
        input_dim: int,
        output_dim: int,
        config: omegaconf.dictconfig.DictConfig,
    ) -> MlpParam:
        """create arguments for PoolingDecoder

        Args:
            input_dim (int): input dimension of decoder (output dim of encoder)
            output_dim (int): output dimension of decoder (#labels)
            config (omegaconf.dictconfig.DictConfig): arguments

        Raises:
            ValueError: Failed to parse PoolingDecoder

        Returns:
            MlpParam: parameters
        """
        try:
            hidden_dim = int(config["model"]["decoder"]["hidden_dim"])
            n_layers = int(config["model"]["decoder"]["n_layers"])
            activation_function_name = str(
                config["model"]["decoder"]["activation_function_name"]
            )

        except ValueError as exit_after_usage:
            PoolingDecoder.help()
            raise ValueError("Failed to parse PoolingDecoder") from exit_after_usage

        return MlpParam(
            input_dim, hidden_dim, output_dim, n_layers, activation_function_name
        )

    def forward(self, *args, **kwargs) -> Any:
        """forward path"""
        raise NotImplementedError
