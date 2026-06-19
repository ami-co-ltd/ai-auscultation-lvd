#!/usr/bin/env python
# pylint: disable=line-too-long,too-many-arguments

from logging import getLogger
from typing import Any, List, Tuple, Union

import omegaconf
import torch
from torch import nn

from elvef.models.mixup import Mixup, create_mixup_module
from elvef.models.mlp import MixupMlp, Mlp
from elvef.models.pooling_decoder import PoolingDecoder
from elvef.utils.dataclasses.models.attention import MultiHeadAttentionPoolingParam
from elvef.utils.dataclasses.models.mlp import MlpParam

logger = getLogger(__name__)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling layer.

    Args:
        param (MultiHeadAttentionPoolingParam): parameter
    """

    def __init__(self, param: MultiHeadAttentionPoolingParam) -> None:
        """Construct an MultiHeadedAttentionPooling object.

        Args:
            param (MultiHeadAttentionPoolingParam): parameter

        """
        super().__init__()
        self.n_head = param.n_head
        self.linear_head = nn.Linear(param.n_feat, param.n_head)
        self.dropout = nn.Dropout(p=param.dropout_rate)

    @classmethod
    def build_from_arguments(
        cls,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
    ):
        """create MultiheadAttentionPooling instance

        Args:
            n_head (int): The number of heads
            n_feat (int): The number of features. Output dimension of encoder.
            dropout_rate (float): Dropout rate.

        Returns:
            MultiHeadAttentionPooling: created instance

        """
        param = MultiHeadAttentionPoolingParam(n_head, n_feat, dropout_rate)
        return cls(param)

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot product attention.

        Note:
            - B: batch size
            - T: sequence length of query
            - C: dimension of input
            - H: the number of heads

        Args:
            input_data (torch.Tensor): input data of shape (B, C, T)

        Returns:
            torch.Tensor: Output tensor (B, HxC)
        """
        n_batch = input_data.size(0)
        input_data = input_data.transpose(1, 2)  # -> (B, T, C)

        # compute attention weights
        # (B, T, C) -> (B, T, H)
        weights = self.linear_head(input_data).view(n_batch, -1, self.n_head)
        weights = weights.transpose(1, 2)  # -> (B, H, T)
        weights = self.dropout(torch.softmax(weights, dim=-1))

        # apply weights
        # (B, H, T) x (B, T, C)
        output = torch.matmul(weights, input_data).view(n_batch, -1)
        return output


class AttentionDecoder(torch.nn.Module):
    """AttentionPoolingDecoder

    Attributes:
        attention (MultiHeadAttentionPooling): attention module
        layers (torch.nn.modules.container.ModuleList):
            torch module list
        output_dim (int): output dimension of Mlp
        feature_fusions (list[Any]): modules for feature fusion

    """

    def __init__(
        self,
        attention_module: MultiHeadAttentionPooling,
        mlp_param: MlpParam,
        mixup_module: Union[Mixup, None] = None,
    ) -> None:
        """init

        Args:
            attention (MultiHeadAttentionPooling): attention module
            mlp_param (MlpParam): parameters for Mlp
            mixup_module (Union[Mixup, None]): mixup module. Defaults to None.
        """
        super().__init__()

        self.attention_module = attention_module

        self.layers = torch.nn.ModuleList()
        if mixup_module is None:
            self.layers += [Mlp(mlp_param)]
        else:
            self.layers += [MixupMlp(mlp_param, mixup_module)]
        self.output_dim = mlp_param.output_dim

        self.feature_fusions: list[Any] = []

    @staticmethod
    def help():
        """help"""
        logger.warning("")

    @classmethod
    def build_from_yaml(
        cls,
        input_dim: int,
        output_dim: int,
        config: omegaconf.dictconfig.DictConfig,
        extra_dim: int = 0,
    ):
        """create AttentionDecoder instance

        Args:
            input_dim (int): input dimension of AttentionPoolingDecoder.
                Output dimension of encoder.
            output_dim (int): output dimension. The number of symbols.
            config (omegaconf.dictconfig.DictConfig): config
            extra_dim (int): extra feature dimension. This will be >0 when
                the feature fusion is activated.

        Returns:
            AttentionDecoder: created instance

        """
        # 1. create attention module
        n_head = int(config["model"]["decoder"]["attention"]["n_headers"])
        dropout_rate = float(config["model"]["decoder"]["attention"]["dropout_rate"])
        attention_module = MultiHeadAttentionPooling.build_from_arguments(
            n_head, input_dim, dropout_rate
        )

        # 2. create mlp module
        input_dim = input_dim * n_head  # output dim of attention
        mlp_param = PoolingDecoder.create_pooling_decoder_arguments(
            input_dim + extra_dim, output_dim, config
        )
        mixup_module = create_mixup_module(config)
        return cls(attention_module, mlp_param, mixup_module)

    def get_output_dim(self) -> int:
        """return output dimension"""
        return self.output_dim

    def forward(
        self,
        xs_pad: torch.Tensor,
        output_labels: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
        sample_names: Union[List[str], None] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """forward path

        Note:
            - tensor shape:
                - B: batch size
                - C: channel size
                - T: sequence length
                - O: output dimension

        Args:
            xs_pad (torch.Tensor): input tensor of shape (B, C, T)
            output_labels (Union[torch.Tensor,  None]): output labels of shape (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights of shape (B, O)
            sample_names (Union[List[str], None]): sample names (B,).
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                logits and output labels

        """
        # 1. remove time-axes
        xs_pad = self.attention_module(xs_pad)

        # 2. apply feature fusion if specified
        for module in self.feature_fusions:
            xs_pad = module(xs_pad, sample_names)

        # 3. mlp
        for layer in self.layers:
            if isinstance(layer, MixupMlp):
                xs_pad, output_labels, loss_weights = layer.forward_impl(
                    xs_pad, output_labels, loss_weights
                )
            else:
                xs_pad = layer(xs_pad)

        return xs_pad, output_labels, loss_weights
