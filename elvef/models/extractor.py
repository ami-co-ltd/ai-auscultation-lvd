#!/usr/bin/env python
# pylint: disable=bare-except,unused-argument

from omegaconf import DictConfig

from elvef.models.base_extractor import BaseExtractor


def create_feature_extractor(config: DictConfig) -> BaseExtractor:
    """create torch-based feature extraction module

    Args:
        config (DictConfig): config

    Returns:
        BaseExtractor: online feature extractor

    """
    return BaseExtractor(-1, False, 0, 0)
