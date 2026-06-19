#!/usr/bin/env python
# pylint: disable=no-member,line-too-long

from typing import Tuple, Union

import omegaconf
import torch

from elvef.utils.dataclasses.models.mixup import MixupParam


class Mixup:
    """mixup

    Attributes:
        mixup_param (MixupParam): parameters for mixup

    """

    def __init__(self, mixup_param: MixupParam) -> None:
        """init

        Args:
            mixup_param (MixupParam): parameters for mixup
        """
        self.mixup_param = mixup_param

    @staticmethod
    def create_arguments(config: omegaconf.dictconfig.DictConfig) -> MixupParam:
        """create arguments for Mixup

        Args:
            config (omegaconf.dictconfig.DictConfig): config

        Returns:
            MixupParam: created instance

        """

        can_apply = bool(config.model.decoder.mixup.can_apply)

        layer_index = int(config["model"]["decoder"]["mixup"]["layer_index"])

        alpha = float(config["model"]["decoder"]["mixup"]["alpha"])

        return MixupParam(
            can_apply=can_apply,
            layer_index=layer_index,
            alpha=alpha,
            can_concat_original=False,
        )

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
        if (not self.mixup_param.can_apply) or (output_labels is None):
            return input_data, output_labels, loss_weights

        # 1. interpolation coefficient
        batch_size: int = input_data.size(0)
        device = input_data.device
        coeff = self.mixup_param.get_interpolation_coefficient(batch_size).to(device)
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
            output_labels_a == self.mixup_param.ignore_id,
            output_labels_b == self.mixup_param.ignore_id,
        )
        mixed_output_labels = coeff * output_labels_a + (1.0 - coeff) * output_labels_b
        mixed_output_labels[empty_ids] = self.mixup_param.ignore_id

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
            mixed_loss_weights.data[empty_ids] = self.mixup_param.ignore_id  # type: ignore

        # 3. concat original samples
        if self.mixup_param.can_concat_original:
            mixed_input_data = torch.cat((mixed_input_data, input_data), 0)
            mixed_output_labels = torch.cat((mixed_output_labels, output_labels), 0)

            if loss_weights is None or mixed_loss_weights is None:
                mixed_loss_weights = None
            elif update_loss_weights:
                mixed_loss_weights = torch.vstack((mixed_loss_weights, loss_weights))
            else:
                mixed_loss_weights = torch.vstack((loss_weights, loss_weights))

        return mixed_input_data, mixed_output_labels, mixed_loss_weights


def create_mixup_module(config: omegaconf.dictconfig.DictConfig) -> Union[Mixup, None]:
    """create Mixup instance

    Args:
        config (omegaconf.dictconfig.DictConfig): config

    Returns:
        Union[Mixup, None]: created instance

    """
    if config["model"]["decoder"].get("mixup", None) is not None:
        return Mixup(Mixup.create_arguments(config))

    return None
