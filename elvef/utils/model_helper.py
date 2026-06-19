#!/usr/bin/env python
# pylint: disable=too-many-lines,line-too-long,logging-fstring-interpolation

import errno
import os
import random
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from torch.nn import Module as Predictor

logger = getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    """set random seed

    Args:
        seed (int): seed number
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = (
    # False  # <https://github.com/pytorch/pytorch/issues/6351>
    # )


def remove_inconsistent_shape_tensors(
    init_model: torch.nn.Module,
    fitted_state_dict: OrderedDict,
    can_throw_error: bool = False,
) -> None:
    """remove network module in `fitted_state_dict` when a size
    of fitted and init tensors are inconsistent.

    Args:
        init_model (torch.nn.Module): randomly initialized model
        fitted_state_dict (OrderedDict): fitted checkpoint
        can_throw_error (bool): terminate program if True. Defaults to False.

    """
    for init_name, init_tensor in init_model.state_dict().items():
        if init_name in fitted_state_dict:

            fitted_tensor = fitted_state_dict[init_name]
            if init_tensor.size() != fitted_tensor.size():

                logger.warning(
                    "inconsistent shape, param: {}, init shape: {}, fitted shape: {}".format(
                        init_name,
                        init_tensor.size(),
                        fitted_tensor.size(),
                    )
                )
                if can_throw_error:
                    raise ValueError
                del fitted_state_dict[init_name]


def is_batch_norm_param(name: str) -> bool:
    """return True if the param relates to batch normalization operation

    Note:
        - this result only affect to logging message and status value

    Args:
        name (str): parameter name

    Returns:
        bool: return True if the param relates to batch normalization
    """
    if "running_mean" in name:
        return True
    if "running_var" in name:
        return True
    if "num_batches_tracked" in name:
        return True
    return False


def load_saved_predictor_model(model_path: Union[str, Path], model: Predictor) -> int:
    """load saved Predictor model

    Args:
        model_path (Union[str, Path]): model file path
        model (Predictor): placeholder

    Raises:
        FileNotFoundError: model file does not exist.

    Returns:
        int: status
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

    model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    # check param names
    status: int = 0
    loaded_param_names = [
        name for name in model_state_dict.keys() if not is_batch_norm_param(name)
    ]
    names = [name_param[0] for name_param in model.named_parameters()]
    if len(set(loaded_param_names) - set(names)) > 0:
        logger.info(
            "The saved model file contains some extra network modules. "
            "These modules are not ported to the input model."
        )
        status += 1
    if len(set(names) - set(loaded_param_names)) > 0:
        logger.info(
            "The input model has additional network modules. "
            "These modules are not updated by this function."
        )
        status += 2

    remove_inconsistent_shape_tensors(model, model_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    del model_state_dict
    return status
