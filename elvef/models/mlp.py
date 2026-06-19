#!/usr/bin/env python
# pylint: disable=line-too-long

from functools import singledispatch
from typing import Tuple, Union

import torch
from torch import nn

from elvef.models.define_activation import create_activation
from elvef.models.mixup import Mixup
from elvef.utils.dataclasses.models.mlp import MlpParam


class Mlp(torch.nn.Module):
    """Simple MLP class

    Attributes:
        layers (torch.nn.modules.container.ModuleList):
            torch module list

    """

    def __init__(self, param: MlpParam) -> None:
        """init

        Args:
            param (MlpParam): parameters for Mlp class
        """
        super().__init__()
        activation_function = create_activation(param.activation_function_name)

        self.layers = torch.nn.ModuleList()
        self.layers += [
            nn.Linear(param.input_dim, param.hidden_dim),
            nn.BatchNorm1d(param.hidden_dim),
            activation_function,
        ]

        for _ in range(param.n_layers):
            self.layers += [
                nn.Linear(param.hidden_dim, param.hidden_dim),
                nn.BatchNorm1d(param.hidden_dim),
                activation_function,
            ]

        self.layers += [nn.Linear(param.hidden_dim, param.output_dim)]

    def forward(self, xs_pad: torch.Tensor) -> torch.Tensor:
        """Propagate

        Note:
            - B: batch size
            - H, H': hidden dimension
            - O: output dimension

        Args:
            xs_pad (torch.Tensor): input tensor of shape (B, H)

        Returns:
            torch.Tensor: hidden vector of shape (B, H')
        """
        # for i in range(len(self.layers)): xs_pad = self.layers[i](xs_pad)
        for layer in self.layers:
            xs_pad = layer(xs_pad)

        return xs_pad


class MixupMlp(Mlp):
    """MLP w/ Mixup module

    Attributes:
        layers (torch.nn.modules.container.ModuleList):
            torch module list

    """

    def __init__(self, mlp_param: MlpParam, mixup_module: Mixup) -> None:
        """init

        Args:
            mlp_param (MlpParam): parameters for Mlp class
            mixup_module (Mixup): mixup module
        """
        super().__init__(mlp_param)
        self.mixup_module = mixup_module

    def forward_impl(
        self,
        xs_pad: torch.Tensor,
        output_labels: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """forward method

        Note:
            - tensor shape:
                - B: batch size
                - H, H': hidden dimension
                - O: output dimension
            - module list example:
                - 0: Linear(in_features=3, out_features=3, bias=True)
                - 1: BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                - 2: Mish()
                - 3: Linear(in_features=3, out_features=3, bias=True)
                - 4: BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                - 5: Mish()
                - 6: Linear(in_features=3, out_features=3, bias=True)
                - 7: BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                - 8: Mish()
                - 9: Linear(in_features=3, out_features=4, bias=True)

        Args:
            xs_pad (torch.Tensor): input tensor of shape (B, H)
            output_labels (Union[torch.Tensor, None]): output labels of shape (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights of shape (B, O)

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                hidden vector of shape (B, H'), output labels, and loss weights
        """
        for i_layer, layer in enumerate(self.layers):
            xs_pad = layer(xs_pad)

            # apply mixup
            if self.training and i_layer == self.mixup_module.mixup_param.layer_index:
                xs_pad, output_labels, loss_weights = self.mixup_module.forward(
                    xs_pad, output_labels, loss_weights
                )

        return xs_pad, output_labels, loss_weights


@singledispatch
def create_mlp(mlp_param: MlpParam) -> Mlp:
    """create Mlp instance

    Args:
        mlp_param (MlpParam): parameters

    Returns:
        Mlp: instance
    """
    return Mlp(mlp_param)


def create_mixup_mlp(mlp_param: MlpParam, mixup_module: Mixup) -> MixupMlp:
    """create MixupMlp instance

    Args:
        mlp_param (MlpParam): parameters
        mixup_module (Mixup): mixup module

    Returns:
        MixupMlp: instance
    """
    return MixupMlp(mlp_param, mixup_module)


@create_mlp.register
def _(  # type: ignore
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    activation_function_name: str,
) -> Mlp:
    """create MLP instance

    Args:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        output_dim (int): output dimension
        n_layers (int): number of layers of hidden
        activation_function_name (str): activation function name

    Returns:
        MLP: instance
    """
    return Mlp(
        MlpParam(input_dim, hidden_dim, output_dim, n_layers, activation_function_name)
    )
