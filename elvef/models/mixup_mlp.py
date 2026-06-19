#!/usr/bin/env python
# pylint: disable=too-many-arguments,too-many-instance-attributes

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn


class Mixup:
    """class to apply mixup"""

    def __init__(self, alpha: float) -> None:
        """init

        Args:
            alpha (float): alpha

        """
        self.alpha = alpha
        self.ignore_id: int = -1

    def get_coeff(self, batch_size: int = -1) -> torch.Tensor:
        """get interpolation coefficients

        Args:
            batch_size (int): batch size. Defaults to -1.

        Returns:
            torch.Tensor: interpolation coefficient. The returned tensor shape
                is (batch_size,) if batch_size > 0.

        """
        if batch_size > 0:
            coeff = np.random.beta(self.alpha, self.alpha, size=(batch_size,))
            coeff = np.maximum(coeff, 1.0 - coeff).astype(np.float32)
            return torch.from_numpy(coeff).view(-1, 1)

        coeff_f = (
            float(np.random.beta(self.alpha, self.alpha)) if self.alpha > 0.0 else 1.0
        )
        coeff_f = max(coeff_f, 1.0 - coeff_f)  # for loss weight
        return torch.tensor(coeff_f)

    def forward(
        self,
        input_data: torch.Tensor,
        output_labels: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
        update_loss_weights: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """forward path

        Note:
            - B: batch size
            - O: output dimension

        Args:
            input_data (torch.Tensor): input data of shape (B, *)
            output_labels (Union[torch.Tensor, None]): output data of shape (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights of shape (B, O)
            update_loss_weights (bool): interpolate loss weights if True. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
                input, output, and loss weights after mixup
        """
        if output_labels is None:
            return input_data, output_labels, loss_weights

        # 1. interpolation coefficient
        batch_size: int = input_data.size(0)
        device = input_data.device
        dtype = input_data.dtype
        coeff = self.get_coeff(batch_size).to(device).float()
        swap_indices = torch.randperm(batch_size).to(device)

        # 2. mixup
        # - input data
        mixed_input_data = (
            coeff * input_data + (1.0 - coeff) * input_data[swap_indices, :]
        )

        # - output data
        output_labels_a = output_labels
        output_labels_b = output_labels[swap_indices]

        empty_ids = torch.logical_or(
            output_labels_a == self.ignore_id,
            output_labels_b == self.ignore_id,
        )
        mixed_output_labels = coeff * output_labels_a + (1.0 - coeff) * output_labels_b
        mixed_output_labels[empty_ids] = self.ignore_id

        # - loss weights
        mixed_loss_weights: Union[torch.Tensor, None] = None
        if loss_weights is None:
            mixed_loss_weights = None
        else:
            if update_loss_weights:
                mixed_loss_weights = (
                    coeff * loss_weights + (1.0 - coeff) * loss_weights[swap_indices]
                )
            else:
                mixed_loss_weights = loss_weights
            mixed_loss_weights.data[empty_ids] = self.ignore_id  # type: ignore
            mixed_loss_weights = mixed_loss_weights.to(device, dtype=dtype)  # type: ignore

        return (
            mixed_input_data.to(device, dtype=dtype),
            mixed_output_labels.to(device, dtype=dtype),
            mixed_loss_weights,
        )


class MixupMlp(torch.nn.Module):
    """Mixup + MLP"""

    def __init__(
        self,
        hidden_units: List[int],
        mixup_layer_num: int,
        act_fn_name: str = "gelu",
        alpha: float = 0.2,
        dropout_rate: float = 0.1,
        is_last: bool = True,
        disable_last_layer_bias: bool = False,
    ) -> None:
        """init"""
        super().__init__()

        self.hidden_units = hidden_units
        self.mixup_layer_num = mixup_layer_num
        self.mixup_layer = Mixup(alpha)
        self.layers = torch.nn.ModuleList()

        if act_fn_name != "gelu":
            raise ValueError("Not supported")
        self.act_fn = torch.nn.GELU()
        self.dropout_rate = dropout_rate
        self.is_last = is_last

        self.create_layers(disable_last_layer_bias)

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int = 1,
        n_layers: int = 4,
        mixup_layer_num: int = 5,
        alpha: float = 0.2,
        dropout_rate: float = 0.1,
        is_last: bool = True,
        hidden_dim: int = -1,
        disable_last_layer_bias: bool = False,
    ):
        """create a default MLP module"""
        if hidden_dim < 0:
            hidden_units = [input_dim] * n_layers + [output_dim]
        else:
            hidden_units = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        return cls(
            hidden_units,
            mixup_layer_num,
            alpha=alpha,
            dropout_rate=dropout_rate,
            is_last=is_last,
            disable_last_layer_bias=disable_last_layer_bias,
        )

    def create_layers(self, disable_last_layer_bias: bool) -> None:
        """create MLP"""
        for idx in range(len(self.hidden_units) - 1):
            current_hidden = self.hidden_units[idx]
            next_hidden = self.hidden_units[idx + 1]
            is_last = (idx == len(self.hidden_units) - 2) and self.is_last

            if is_last:
                self.layers += [
                    nn.Linear(
                        current_hidden, next_hidden, bias=not disable_last_layer_bias
                    )
                ]
            else:
                self.layers += [nn.Linear(current_hidden, next_hidden)]
            if not is_last:
                self.layers += [
                    nn.LayerNorm(next_hidden),
                    self.act_fn,
                    nn.Dropout(self.dropout_rate),
                ]

    def initialize_weights(self) -> None:
        """initialize model parameters"""

        def _initialize_weights(module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform(module.weight)
                module.bias.data.fill_(0.0)
            elif isinstance(module, torch.nn.LayerNorm):
                module.reset_parameters()

        self.layers.apply(_initialize_weights)

    def forward(
        self,
        input_data: torch.Tensor,
        output_data: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """forward path

        Args:
            input_data (torch.Tensor): input tensor (B, H)
            output_labels (Union[torch.Tensor, None]): output labels  (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights (B, O)

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:

        """
        can_apply_mixup = self.training and output_data is not None

        for idx, layer in enumerate(self.layers):
            input_data = layer(input_data)

            if can_apply_mixup and idx == self.mixup_layer_num:
                # apply mixup
                input_data, output_data, loss_weights = self.mixup_layer.forward(
                    input_data, output_data, loss_weights, True
                )

        return input_data, output_data, loss_weights


class MSDMlp(torch.nn.Module):
    """MLP + multi-sample dropout"""

    def __init__(
        self,
        hidden_units: List[int],
        mixup_layer_num: int,
        act_fn_name: str = "gelu",
        alpha: float = 0.2,
        dropout_rate: float = 0.1,
        is_last: bool = True,
        n_repeat: int = 5,
        disable_last_layer_bias: bool = False,
    ) -> None:
        """init"""
        super().__init__()

        self.hidden_units = hidden_units
        self.mixup_layer_num = mixup_layer_num
        self.mixup_layer = Mixup(alpha)
        if alpha < 0:
            self.mixup_layer_num = -1  # disable

        self.layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.last_layer = nn.Linear(
            self.hidden_units[-2],
            self.hidden_units[-1],
            bias=not disable_last_layer_bias,
        )

        if act_fn_name != "gelu":
            raise ValueError("Not supported")
        self.act_fn = torch.nn.GELU()
        self.dropout_rate = dropout_rate
        self.is_last = is_last
        self.n_repeat = n_repeat

        self.create_layers()

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int = 1,
        n_layers: int = 4,
        mixup_layer_num: int = 5,
        alpha: float = 0.2,
        dropout_rate: float = 0.1,
        is_last: bool = True,
        n_repeat: int = 5,
        disable_last_layer_bias: bool = False,
    ):
        """create a default MLP module"""
        return cls(
            [input_dim] * n_layers + [output_dim],
            mixup_layer_num,
            alpha=alpha,
            dropout_rate=dropout_rate,
            is_last=is_last,
            n_repeat=n_repeat,
            disable_last_layer_bias=disable_last_layer_bias,
        )

    def create_layers(self) -> None:
        """create MLP"""
        for idx in range(len(self.hidden_units) - 1):
            current_hidden = self.hidden_units[idx]
            next_hidden = self.hidden_units[idx + 1]
            is_last = (idx == len(self.hidden_units) - 2) and self.is_last

            if not is_last:
                self.layers += [
                    nn.Linear(current_hidden, next_hidden),
                    nn.LayerNorm(next_hidden),
                    self.act_fn,
                ]

    def forward(
        self,
        input_data: torch.Tensor,
        output_data: Union[torch.Tensor, None],
        loss_weights: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """forward path

        Args:
            input_data (torch.Tensor): input tensor (B, H)
            output_labels (Union[torch.Tensor, None]): output labels  (B, O)
            loss_weights (Union[torch.Tensor, None]): loss weights (B, O)

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:

        """
        can_apply_mixup = self.training and output_data is not None
        can_use_msd = self.training and output_data is not None

        for idx, layer in enumerate(self.layers):
            input_data = layer(input_data)
            if can_apply_mixup and idx == self.mixup_layer_num:
                # apply mixup
                input_data, output_data, loss_weights = self.mixup_layer.forward(
                    input_data, output_data, loss_weights, True
                )

        # last layer
        if can_use_msd:
            ret = []
            for _ in range(self.n_repeat):
                ret.append(self.last_layer(self.dropout(input_data)))
            input_data = torch.mean(torch.stack(ret, 0), 0)
        else:
            input_data = self.last_layer(input_data)
        return input_data, output_data, loss_weights
