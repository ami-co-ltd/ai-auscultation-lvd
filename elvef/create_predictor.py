#!/usr/bin/env python
# pylint: disable=too-many-arguments

from logging import getLogger
from pathlib import Path

from omegaconf import DictConfig

from elvef.models.predictor import Predictor
from elvef.utils import model_helper

logger = getLogger(__name__)


def create_predictor(
    output_dim: int,
    config: DictConfig,
    pretrained_model_file: Path = Path(""),
    is_test: bool = False,
    can_use_cuda: bool = False,
) -> Predictor:
    """create Predictor module

    Args:
        output_dim (int): output dimension
        config (DictConfig): configuration
        pretrained_model_file (Union[str, Path]): pretrained model file path.
            Defaults to "" (train from scratch).
        module_names_to_transfer (Union[List[str], None]): list of module
            names to transfer. Defaults to None.
        is_test (bool): test mode if True. Defaults to False.
        can_use_cuda (bool): use cuda if True. Defaults to False.

    Returns:
        redictor: Predictor instance.
          - model training stage: Predictor
          - model evaluation stage: Union[Predictor, OnnxPredictor]

    """

    args = (output_dim, config, is_test, can_use_cuda)
    predictor = Predictor.build_model(*args)

    if pretrained_model_file.is_file():
        model_helper.load_saved_predictor_model(pretrained_model_file, predictor)

    if is_test:
        predictor.eval()
        if config["inference"]["test_time_augmentation"]["apply_dropout"]:
            for module in predictor.modules():
                if module.__class__.__name__.startswith("Dropout"):
                    module.train()

    return predictor
