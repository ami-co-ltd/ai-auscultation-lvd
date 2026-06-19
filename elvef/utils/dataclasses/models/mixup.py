#!/usr/bin/env python
# pylint: disable=no-member

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class MixupParam:
    """parameters for Mixup

    Attributes:
        can_apply (bool): apply mixup if True
        layer_index (int): layer to apply mixup
        alpha (float): alpha parameter. interpolation coefficient: Beta(alpha, alpha)
        can_concat_original (bool): concat original samples if True
        ignore_id (int): id to represent unlabled data. Defaults to -1.

    """

    can_apply: bool
    layer_index: int
    alpha: float
    can_concat_original: bool
    ignore_id: int = -1

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()

    def check_parameters(self) -> None:
        """check parameters

        Raises:
            ValueError: got netagive layer_index

        """
        if self.layer_index < 0:
            raise ValueError("Got negative layer_index")

    def get_interpolation_coefficient(self, batch_size: int = -1) -> torch.Tensor:
        """generate interpolation coefficient

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
