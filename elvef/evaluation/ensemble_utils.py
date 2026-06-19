#!/usr/bin/env python
# pylint: disable=no-else-return,logging-format-interpolation,consider-using-f-string,too-many-locals,too-many-nested-blocks,too-many-arguments

from copy import deepcopy
from logging import getLogger
from typing import Iterator, List, Set

import numpy as np

from elvef.utils.dataclasses.prediction_result import PredictionResult

logger = getLogger(__name__)


def weights_generator_1() -> Iterator[List[float]]:
    """generate interpolation weights

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors
    """
    yield [1.0]


def weights_generator_2(n_steps: int = 101) -> Iterator[List[float]]:
    """generate interpolation weights for two predictors

    Args:
        n_steps (int): the number of steps. Defaults to 101.

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors
    """
    for i_weight in np.linspace(0.0, 1.0, n_steps):
        j_weight = 1.0 - i_weight
        yield [i_weight, j_weight]


def weights_generator_3(n_steps: int = 101) -> Iterator[List[float]]:
    """generate interpolation weights for three predictors

    Args:
        n_steps (int): the number of steps. Defaults to 101.

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors
    """
    for i_weight in np.linspace(0.0, 1.0, n_steps):
        for j_weight in np.linspace(0.0, 1.0, n_steps):
            k_weight = 1.0 - i_weight - j_weight
            if k_weight >= 0.0:
                yield [i_weight, j_weight, k_weight]


def weights_generator_4(n_steps: int) -> Iterator[List[float]]:
    """generate interpolation weights for three predictors

    Args:
        n_steps (int): the number of steps. Defaults to 101.

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors

    """
    for i_weight in np.linspace(0.0, 1.0, n_steps):
        for j_weight in np.linspace(0.0, 1.0, n_steps):
            cur_weight = i_weight + j_weight
            if cur_weight > 1.0:
                continue
            for k_weight in np.linspace(0.0, 1.0, n_steps):
                l_weight = 1.0 - cur_weight - k_weight
                if l_weight >= 0.0:
                    yield [i_weight, j_weight, k_weight, l_weight]


def weights_generator_5(n_steps: int) -> Iterator[List[float]]:
    """generate interpolation weights for 5 predictors

    Args:
        n_steps (int): the number of steps. Defaults to 101.

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors

    """
    for i_weight in np.linspace(0.0, 1.0, n_steps):
        for j_weight in np.linspace(0.0, 1.0, n_steps):
            cur_weight = i_weight + j_weight
            if cur_weight > 1.0:
                continue

            for k_weight in np.linspace(0.0, 1.0, n_steps):
                if cur_weight + k_weight > 1.0:
                    continue

                for l_weight in np.linspace(0.0, 1.0, n_steps):
                    m_weight = 1.0 - cur_weight - k_weight - l_weight

                    if m_weight >= 0.0:
                        yield [i_weight, j_weight, k_weight, l_weight, m_weight]


def weights_generator_8(n_steps: int) -> Iterator[List[float]]:
    """generate interpolation weights for three predictors

    Args:
        n_steps (int): the number of steps. Defaults to 101.

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors

    """
    for wt_1 in np.linspace(0.0, 1.0, n_steps):
        logger.info(f"{wt_1=}")
        for wt_2 in np.linspace(0.0, 1.0, n_steps):
            cum_wt_12 = wt_1 + wt_2
            if cum_wt_12 > 1.0:
                continue

            for wt_3 in np.linspace(0.0, 1.0, n_steps):
                if cum_wt_12 + wt_3 > 1.0:
                    continue

                for wt_4 in np.linspace(0.0, 1.0, n_steps):
                    cum_wt_14 = cum_wt_12 + wt_3 + wt_4
                    if cum_wt_14 > 1.0:
                        continue

                    for wt_5 in np.linspace(0.0, 1.0, n_steps):
                        if cum_wt_14 + wt_5 > 1.0:
                            continue

                        for wt_6 in np.linspace(0.0, 1.0, n_steps):
                            cum_wt_16 = cum_wt_14 + wt_5 + wt_6
                            if cum_wt_16 > 1.0:
                                continue

                            for wt_7 in np.linspace(0.0, 1.0, n_steps):
                                if cum_wt_16 + wt_7 > 1.0:
                                    continue

                                wt_8 = 1.0 - cum_wt_16 - wt_7
                                if wt_8 >= 0:
                                    weights = [wt_1, wt_2, wt_3, wt_4]
                                    weights.extend([wt_5, wt_6, wt_7, wt_8])
                                    yield weights


def get_predictor_weights_generator(
    n_predictors: int, n_steps: int = 101
) -> Iterator[List[float]]:
    """generate interpolation weights

    Args:
        n_predictors (int): the number of predictors for ensemble
        n_steps (int): the number of steps. Defaults to 101.

    Raises:
        ValueError: Got unexpected predictor number

    Yields:
        Iterator[List[float]]: interpolation weights for the predictors

    """
    if n_predictors == 1:
        return weights_generator_1()
    if n_predictors == 2:
        return weights_generator_2(n_steps)
    elif n_predictors == 3:
        return weights_generator_3(n_steps)
    elif n_predictors == 4:
        return weights_generator_4(n_steps)
    elif n_predictors == 5:
        return weights_generator_5(n_steps)
    elif n_predictors == 8:
        return weights_generator_8(n_steps)
    raise ValueError


def align_by_patient_name(
    pred_results: List[PredictionResult],
    accept_missing_position: bool = False,
    remove_position_name: bool = False,
) -> List[PredictionResult]:
    """remove patients who have missing position data

    Args:
        pred_results (List[PredictionResult]): prediction result
            seperated by auscultation position.

    Returns:
        List[PredictionResult]: position aligned.

    """
    if accept_missing_position:
        # return fill_unrecorded_positions_by_dup(pred_results)
        raise NotImplementedError

    # 1. pick patient names
    names_all: List[List[str]] = []
    for pred_result in pred_results:
        if remove_position_name:
            pred_result.accumulated_sample_names = [
                name.split("_")[0] for name in pred_result.accumulated_sample_names
            ]
        names_all.append(pred_result.accumulated_sample_names)

    # 2. intersection
    required_names: Set[str] = set(names_all[0])
    for names in names_all[1:]:
        required_names &= set(names)

    # 3. apply mask. remove patients who have missing position data.
    for pred_result in pred_results:

        sample_names = pred_result.accumulated_sample_names
        mask = np.array([name in required_names for name in sample_names])
        pred_result.accumulated_sample_names = np.array(sample_names)[mask].tolist()
        pred_result.accumulated_pred_prob = pred_result.accumulated_pred_prob[mask]  # type: ignore
        pred_result.accumulated_pred_bin = pred_result.accumulated_pred_bin[mask]  # type: ignore
        pred_result.accumulated_labels = pred_result.accumulated_labels[mask]  # type: ignore

    # 4. sort
    for pred_result in pred_results:

        src_pred_prob = deepcopy(pred_result.pred_prob)
        src_pred_bin = deepcopy(pred_result.pred_bin)
        src_labels = deepcopy(pred_result.labels)
        dst_pred_prob = deepcopy(pred_result.pred_prob)
        dst_pred_bin = deepcopy(pred_result.pred_bin)
        dst_labels = deepcopy(pred_result.labels)

        for dst_idx, name in enumerate(required_names):
            src_idx = pred_result.accumulated_sample_names.index(name)
            dst_pred_prob[dst_idx] = src_pred_prob[src_idx]
            dst_pred_bin[dst_idx] = src_pred_bin[src_idx]
            dst_labels[dst_idx] = src_labels[src_idx]

        pred_result.accumulated_sample_names = list(required_names)
        pred_result.accumulated_pred_prob = dst_pred_prob
        pred_result.accumulated_pred_bin = dst_pred_bin
        pred_result.accumulated_labels = dst_labels

    return pred_results
