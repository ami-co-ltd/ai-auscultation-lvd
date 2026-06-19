#!/usr/bin/env python
# pylint: disable=no-else-return

from typing import Union

import torch

ActivationType = Union[
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU,
    torch.nn.modules.activation.Sigmoid,
    torch.nn.modules.activation.Softmax,
    torch.nn.modules.activation.Mish,
]


def create_activation(fn_name: str) -> ActivationType:
    """create activation function

    Note:
        - MEMO: use dataclass if we need more parameters for activation module.

    Args:
        fn_name (str): activation function name

    Returns:
        ActivationType: activation function

    """
    if fn_name == "relu":
        return torch.nn.ReLU()

    if fn_name == "sigmoid":
        return torch.nn.Sigmoid()

    if fn_name == "softmax":
        return torch.nn.Softmax(dim=-1)

    if fn_name == "mish":
        return torch.nn.Mish()

    raise ValueError("Unknown activation function.")
