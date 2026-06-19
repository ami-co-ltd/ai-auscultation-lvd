#!/usr/bin/env python
# pylint: disable=line-too-long,no-member,unnecessary-pass,logging-fstring-interpolation,too-many-locals,too-few-public-methods,using-constant-test,protected-access,too-many-arguments,too-many-instance-attributes,too-many-function-args,abstract-method

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import List, Optional

import numpy as np
import omegaconf
import torch
from sklearn import metrics

from elvef.evaluation.ensemble_utils import (
    align_by_patient_name,
    get_predictor_weights_generator,
)
from elvef.evaluation.k_prediction_result import (
    KPredictionResults,
    create_position_dependent_prediction_result,
    get_single_cutoff,
    select_pred_result_subset,
    stack_prediction_results,
)
from elvef.utils.dataclasses.prediction_result import PredictionResult

Predictor = torch.nn.Module
logger = getLogger(__name__)


def make_interpolated_pred_prob(
    pred_probs: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """ensemble pred_probs

    Args:
        pred_probs (torch.Tensor): predicted probability of shape
            (n_predictors, batch_size, output_dim)
        weights (torch.Tensor): ensemble weights (n_predictors, 1, 1).

    Returns:
        torch.Tensor: new predicted probability. shape: (batch_size, output_dim).

    """
    return torch.sum(pred_probs * weights, dim=0)


#
# Ensemble Modules
#
class Ensemble:
    """base ensemble class"""

    def __init__(
        self, predictors: List[Predictor], score_to_maximize: str = "macro_auroc"
    ) -> None:
        """init

        Args:
            predictors (List[Predictor]): list of predictors
            score_to_maximize (str): score name to maximize
                by changing interpolation coefficients.
                Defaults to"macro_auroc".
        """
        self.predictors = predictors
        self.score_to_maximize = score_to_maximize

    @property
    def n_predictors(self) -> int:
        """return the number of predictors"""
        return len(self.predictors)

    def get_predictions(self, input_data: torch.Tensor, n_expand: int) -> torch.Tensor:
        """compute posterior probabilities using multiple predictors

        Args:
            input_data (torch.Tensor): input data
            n_expand (int): expansion used in test-time augmentation.

        Returns:
            torch.Tensor: posterior probability. The shape is
                (n_predictors, batch, output_dim).
        """
        pred_probs: List[torch.Tensor] = [
            predictor.predict(input_data, n_expand)[1] for predictor in self.predictors  # type: ignore
        ]
        return torch.stack((pred_probs))

    def compute_score(self, pred_result: PredictionResult) -> float:
        """compute score

        Args:
            pred_result (PredictionResult): interpolated posterior probabilities

        Raises:
            KeyError: accessed to an unset dictionary key name

        Returns:
            float: score

        """
        label = pred_result.labels.reshape(-1)
        pred_prob = pred_result.pred_prob.reshape(-1)

        pred_prob = pred_prob[label >= 0]
        label = label[label >= 0]
        auroc = metrics.roc_auc_score(label, pred_prob)
        return float(auroc)

    def tune_weights(self, pred_results: List[PredictionResult]) -> None:
        """search the best interpolation coefficients"""
        raise NotImplementedError

    def predict(
        self,
        input_data: torch.Tensor,
        n_expand: int,
    ) -> torch.Tensor:
        """make prediction"""
        raise NotImplementedError


class WeightedAverageVoting(Ensemble):
    """class for weighted average voting

    Attributes:
        weights (torch.Tensor): interpolation coefficients of shape (#predictors,).
        n_steps (int): the number of steps.
    """

    def __init__(
        self,
        predictors: List[Predictor],
        np_weights: np.ndarray = np.array([]),
        n_steps: int = 101,
    ) -> None:
        """init

        Args:
            predictors (List[Predictor]): trained predictors
            np_weights (np.ndarray): interpolation coefficients of shape (#predictors,).
                Defaults to np.array([]).
            n_steps (int): the number of steps. Defaults to 101.
        """
        super().__init__(predictors)

        if np_weights.shape[0] != self.n_predictors:
            logger.info("initializing interpolation coefficients")
            np_weights = (
                np.ones(self.n_predictors, dtype=np.float32) / self.n_predictors
            )
        self.weights = torch.from_numpy(np_weights.reshape(-1, 1, 1))
        self.n_steps = n_steps

    @classmethod
    def build_from_omegaconf(cls, config: omegaconf.dictconfig.DictConfig):
        """create instance"""
        pass

    def tune_weights(self, pred_results: List[PredictionResult]) -> None:
        """search the best interpolation coefficients

        Note:
            - current code maximize auroc. the higher is the better.

        Args:
            pred_results (List[PredictionResult]): list of prediction probabilities and
                reference labels. `len(pred_results)` equals to `self.n_predictors`.

        """
        best_score = -1.0
        best_weights = torch.ones(self.n_predictors) / self.n_predictors

        if False:
            pass
            # for _weights in get_predictor_weights_generator(
            #     self.n_predictors, n_steps=self.n_steps
            # ):
            #     weights = np.array(_weights, dtype=np.float32)
            #     new_pred_result = make_interpolated_pred_result(
            #         pred_results,
            #         weights,
            #     )
            #     score = self.compute_score(new_pred_result)
            #     if best_score < score:
            #         best_score, best_weights = score, torch.from_numpy(weights)

        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            n_model_pred_probs = torch.from_numpy(
                np.stack([result.pred_prob for result in pred_results])
            ).to(device)

            new_pred_result = deepcopy(pred_results[0])

            for _weights in get_predictor_weights_generator(
                self.n_predictors, n_steps=self.n_steps
            ):
                weights = np.array(_weights, dtype=np.float32)
                new_pred_result.accumulated_pred_prob = (
                    make_interpolated_pred_prob(
                        n_model_pred_probs,
                        torch.from_numpy(weights)
                        .to(device)
                        .view(self.n_predictors, 1, 1),
                    )
                    .cpu()
                    .data.numpy()
                )

                score = self.compute_score(new_pred_result)
                if best_score < score:
                    best_score, best_weights = score, torch.from_numpy(weights)

        self.weights = best_weights.view(-1, 1, 1)


class EnsembleExpV2:
    """ensemble experiment. Use KpredictionResults

    Args:
        k_pred_results (List[KPredictionResults]): k-fold prediction results.
            (n_predictors,).
        pred_results (List[PredictionResult]): result wrt validation set.
            (n_predictors,).
        weights (np.ndarray): interpolation weights (n_predictors, odims).
        thresholds (np.ndarray): binalization thresholds (odims,).

    """

    def __init__(
        self,
        k_pred_results: List[KPredictionResults],
        valid_data_type: str = "valid",
        test_data_type: str = "test",
    ) -> None:
        """init

        Args:
            k_pred_results (List[KPredictionResults]): k-fold prediction results
                generated by multiple predictors.
            valid_data_type (str): data type used to select validation set.
                Defaults to "valid".

        """
        self.k_pred_results = k_pred_results
        # get validation data
        self.pred_results: List[PredictionResult] = self.parse_data(
            k_pred_results,
            valid_data_type,
        )

        self.valid_data_type = valid_data_type
        self.test_data_type = test_data_type

        self.weights: np.ndarray
        self.thresholds: np.ndarray

    @staticmethod
    def parse_data(
        k_pred_results: List[KPredictionResults],
        valid_data_type: str = "valid",
    ) -> List[PredictionResult]:
        """get validation set

        Args:
            k_pred_results: List[KPredictionResults]: k-fold prediction results
                generated by multiple predictors.
            valid_data_type (str): data type used to select validation set.
                Defaults to "valid".

        Returns:
            List[PredictionResult]: prediction result on validation set.

        """

        pred_results: List[PredictionResult] = []

        for k_pred_result in k_pred_results:
            n_cv = k_pred_result.n_cv
            pred_result = stack_prediction_results(
                [k_pred_result.summary[valid_data_type][i_cv] for i_cv in range(n_cv)]
            )
            pred_results.append(pred_result)
        return pred_results

    def optimize(
        self,
        target_positions: Optional[List[str]] = None,
        n_steps: int = 101,
    ) -> None:
        """search optimal weight and binalization thresholds

        Args:
            target_positions (Optional[List[str]]): position names used
                in ensemble. Use all pred_probs estimated by 4positions
                if it is None. Defaults to None.
            n_steps (int): interpolation step. Defaults to 101.

        """
        # 1. load prediction results
        if target_positions is not None:
            for idx, pred_result in enumerate(self.pred_results):
                pred_dict = create_position_dependent_prediction_result(
                    pred_result, True
                )
                pred_result = stack_prediction_results(
                    [pred_dict[pos] for pos in target_positions]
                )
                self.pred_results[idx] = pred_result

        n_predictors = len(self.pred_results)
        odims = self.pred_results[0].pred_prob.shape[1]
        final_pred_result = PredictionResult(0.3, None)
        final_thresholds: List[float] = []
        final_weights: List[np.ndarray] = []

        for odim in range(odims):

            # 2.1. get single-cutoff pred-probs
            single_pred_results = [
                get_single_cutoff(pred_result, odim)
                for pred_result in self.pred_results
            ]
            single_pred_results = align_by_patient_name(single_pred_results)

            ensemble = WeightedAverageVoting(
                [None] * n_predictors,  # type: ignore
                np.array([]),
                n_steps=n_steps,
            )
            ensemble.tune_weights(single_pred_results)
            weights = ensemble.weights.view(-1).numpy()  # (n_predictors,)

            # 2.2. evaluate ensembled prediction result
            labels = single_pred_results[0].labels
            pred_probs = np.zeros_like(labels).astype(np.float32)
            for idx in range(n_predictors):
                pred_probs += weights[idx] * single_pred_results[idx].pred_prob
            # if round_num > 0:
            # pred_probs = np.round(pred_probs, decimals=round_num)

            fpr, tpr, thresholds = metrics.roc_curve(
                labels.reshape(-1), pred_probs.reshape(-1)
            )
            best_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[best_idx]
            auc = metrics.auc(fpr, tpr)
            logger.info(f"{odim=}")
            logger.info(f"\t{weights=}")
            logger.info(f"\t{auc=}")
            logger.info(f"\t{best_threshold=}")

            if odim == 0:
                batch_size = labels.shape[0]
                final_pred_result.accumulated_sample_names = single_pred_results[
                    0
                ].accumulated_sample_names
                final_pred_result.accumulated_pred_prob = np.zeros(
                    (batch_size, odims), dtype=np.float32
                )
                final_pred_result.accumulated_labels = np.zeros(
                    (batch_size, odims), dtype=np.int32
                )

            final_pred_result.accumulated_pred_prob[:, odim] = pred_probs[:, 0]  # type: ignore
            final_pred_result.accumulated_labels[:, odim] = labels[:, 0]  # type: ignore
            final_thresholds.append(best_threshold)
            final_weights.append(weights)

        self.weights = np.stack(final_weights).astype(np.float32)
        self.thresholds = np.array(final_thresholds, dtype=np.float32)

        final_pred_result.set_thresholds(self.thresholds)

    def ensemble(self, pred_results: List[PredictionResult]) -> PredictionResult:
        """ensemble posterior probabilities

        Args:
            pred_results (List[PredictionResult]): prediction results
                estimated by multiple predictors.

        Returns:
            PredictionResult: ensembled result

        """
        n_predictors = len(pred_results)
        odims = pred_results[0].pred_prob.shape[1]
        final_pred_result = PredictionResult(0.3, None)

        for odim in range(odims):

            # 1. get single-cutoff pred-probs
            single_pred_results = [
                get_single_cutoff(pred_result, odim) for pred_result in pred_results
            ]
            single_pred_results = align_by_patient_name(single_pred_results)

            # 2.2. evaluate ensembled prediction result
            labels = single_pred_results[0].labels
            pred_probs = np.zeros_like(labels).astype(np.float32)
            for idx in range(n_predictors):
                pred_probs += (
                    self.weights[odim][idx] * single_pred_results[idx].pred_prob
                )

            if odim == 0:
                batch_size = labels.shape[0]
                final_pred_result.accumulated_sample_names = single_pred_results[
                    0
                ].accumulated_sample_names
                final_pred_result.accumulated_pred_prob = np.zeros(
                    (batch_size, odims), dtype=np.float32
                )
                final_pred_result.accumulated_labels = np.zeros(
                    (batch_size, odims), dtype=np.int32
                )

            final_pred_result.accumulated_pred_prob[:, odim] = pred_probs[:, 0]  # type: ignore
            final_pred_result.accumulated_labels[:, odim] = labels[:, 0]  # type: ignore

        final_pred_result.set_thresholds(self.thresholds)
        return final_pred_result

    def ensemble_all(
        self,
        k_pred_results: List[KPredictionResults],
        best_cvs: Optional[List[int]] = None,
        test_sample_ids: Optional[List[str]] = None,
    ) -> KPredictionResults:
        """ensemble k-fold valid and test set results

        Args:
            k_pred_results (List[KPredictionResults]): k-fold prediction results.
                len(): n_predictors.
            best_cvs (Optional[List[int]]): best fold ids.
                len(): n_predictors.
            test_sample_ids (Optional[List[str]]): sample names used in
                final evaluation. Defaults to None (use all samples).

        Returns:
            KPredictionResults: ensembled k-fold prediction results

        """
        res = deepcopy(k_pred_results[0])

        for dtype in [self.valid_data_type, self.test_data_type]:

            is_test = dtype == self.test_data_type

            if dtype == self.test_data_type and best_cvs is not None:
                batched = [
                    k_result.summary[dtype][best_cvs[idx]]
                    for idx, k_result in enumerate(k_pred_results)
                ]
                pred_result = self.ensemble(batched)
                if test_sample_ids is not None:
                    pred_result = select_pred_result_subset(
                        pred_result, test_sample_ids
                    )
                for i_cv in range(res.n_cv):
                    res._summary[dtype][i_cv] = pred_result  # type: ignore
                    res._summary_df[dtype][i_cv] = pred_result.to_dataframe()  # type: ignore

            else:
                for i_cv in range(res.n_cv):
                    pred_result = self.ensemble(
                        [k_result.summary[dtype][i_cv] for k_result in k_pred_results]
                    )
                    if is_test and test_sample_ids is not None:
                        pred_result = select_pred_result_subset(
                            pred_result, test_sample_ids
                        )
                    res._summary[dtype][i_cv] = pred_result  # type: ignore
                    res._summary_df[dtype][i_cv] = pred_result.to_dataframe()  # type: ignore

        return res

    def save(self, output_dir: Path) -> None:
        """save ensemble weights and binalization thresholds"""
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "weights.npy", self.weights)
        np.save(output_dir / "thresholds.npy", self.thresholds)
