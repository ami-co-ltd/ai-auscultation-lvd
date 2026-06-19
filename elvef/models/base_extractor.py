#!/usr/bin/env python
# pylint: disable=line-too-long,too-many-arguments,no-member,too-many-instance-attributes,no-else-raise,consider-using-f-string

import torch


class BaseExtractor(torch.nn.Module):
    """base feature extractor class

    Attributes:
        image_size (int): height and width size of the extracted time-frequency
            representation data.
        can_normalize (bool): apply min-max normalize if True.
        min_value (float): minimum value after min-max normalization.
        max_value (float): maximum value after min-max normalization.

    """

    def __init__(
        self,
        image_size: int = -1,
        can_normalize: bool = False,
        min_value: float = -1,
        max_value: float = -1,
    ) -> None:
        """init

        Args:
            image_size (int): height and width size of the extracted 2d stft
            can_normalize (bool): normalize data if True.
            min_value (float): minimum value after min-max normalization.
            max_value (float): maximum value after min-max normalization.
        """
        super().__init__()

        self.image_size = image_size

        self.can_normalize = can_normalize
        self.min_value = min_value
        self.max_value = max_value

    @torch.no_grad()
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """forward path

        Args:
            input_data (torch.Tensor): input data

        Returns:
            torch.Tensor: input data

        """
        return input_data
