#!/usr/bin/env python
# pylint: disable=protected-access,too-many-nested-blocks,too-many-arguments,unused-argument

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from elvef.utils.dataclasses.prediction_result import PredictionResult
from elvef.utils.terminology import AuscultationPosition

logger = getLogger(__name__)
KPredictionResultsT = TypeVar("KPredictionResultsT", bound="KPredictionResults")


def stack_prediction_results(
    pred_results: List[PredictionResult],
) -> PredictionResult:
    """stack prediction results"""
    summary = deepcopy(pred_results[0])
    for result in pred_results[1:]:
        summary.accumulate_results(
            result.pred_prob,
            result.labels,
            result.accumulated_sample_names,
            0,
        )

    summary.get_data()
    return summary


def get_single_cutoff(
    pred_result: PredictionResult, cutoff_idx: int
) -> PredictionResult:
    """get single-cutoff prediction result

    Args:
        pred_result (PredictionResult): multi-label prediction result
        cutoff_idx (int): target cutoff index

    Returns:
        PredictionResult: single-cutoff prediction result
            (shape of pred_prob: (batch, odim=1)).

    """
    new_pred_result = deepcopy(pred_result)
    pred_prob = pred_result.accumulated_pred_prob[:, [cutoff_idx]]  # type: ignore
    pred_bin = pred_result.accumulated_pred_bin[:, [cutoff_idx]]  # type: ignore
    labels = pred_result.accumulated_labels[:, [cutoff_idx]]  # type: ignore

    new_pred_result.accumulated_pred_prob = pred_prob
    new_pred_result.accumulated_pred_bin = pred_bin
    new_pred_result.accumulated_labels = labels
    return new_pred_result


def create_position_dependent_prediction_result(
    pred_result: PredictionResult,
    can_accept_ignore_id: bool = True,
) -> Dict[str, PredictionResult]:
    """compute position-dependent PredictionResult instances

    Args:
        pred_result (PredictionResult): position-independent prediction result

    Returns:
        Dict[str, PredictionResult]: position-dependent results

    """
    # 1. make position-dependent PredictionResult instances
    result_dict: Dict[str, PredictionResult] = {}

    for position_name in AuscultationPosition.available_names() + ["unknown"]:
        result_dict[position_name] = PredictionResult(
            pred_result.binalization_thr,
            pred_result.valid_symbol_ids,
            # can_accept_ignore_id=can_accept_ignore_id,
        )

    # 2. update `result_dict`
    for idx, sample_name in enumerate(pred_result.accumulated_sample_names):
        # 2.1. get dictionary key
        try:
            position_name = AuscultationPosition.get_position_name(sample_name)
        except ValueError:
            position_name = "unknown"

        # 2.2. set data
        i_pred_prob = (pred_result.accumulated_pred_prob[idx])[np.newaxis]  # type: ignore
        i_pred_bin = (pred_result.accumulated_pred_bin[idx])[np.newaxis]  # type: ignore
        i_labels = (pred_result.accumulated_labels[idx])[np.newaxis]  # type: ignore

        if result_dict[position_name].accumulated_pred_prob is None:
            result_dict[position_name].accumulated_pred_prob = i_pred_prob
            result_dict[position_name].accumulated_pred_bin = i_pred_bin
            result_dict[position_name].accumulated_labels = i_labels

        else:
            result_dict[position_name].accumulated_pred_prob = np.vstack(
                (result_dict[position_name].accumulated_pred_prob, i_pred_prob)  # type: ignore
            )
            result_dict[position_name].accumulated_pred_bin = np.vstack(
                (result_dict[position_name].accumulated_pred_bin, i_pred_bin)  # type: ignore
            )
            result_dict[position_name].accumulated_labels = np.vstack(
                (result_dict[position_name].accumulated_labels, i_labels)  # type: ignore
            )

        result_dict[position_name].accumulated_sample_names.append(
            pred_result.accumulated_sample_names[idx]
        )
        result_dict[position_name].process_times.append(-1)

    return result_dict


def get_single_cutoffs(pred_result: PredictionResult) -> List[PredictionResult]:
    """convert multi-cutoff result to a list of single-cutoff results

    Args:
        pred_result (PredictionResult): prediction result

    Returns:
        List[PredictionResult]: single-cutoff prediction results

    """
    odims = pred_result.pred_prob.shape[1]
    res: List[PredictionResult] = [deepcopy(pred_result) for _ in range(odims)]

    for odim in range(odims):
        pred_prob = pred_result.pred_prob[:, [odim]]
        pred_bin = pred_result.pred_bin[:, [odim]]
        labels = pred_result.labels[:, [odim]]
        res[odim].accumulated_pred_prob = pred_prob
        res[odim].accumulated_pred_bin = pred_bin
        res[odim].accumulated_labels = labels

    return res


def select_pred_result_subset(
    pred_result: PredictionResult,
    available_sample_names: List[str],
) -> PredictionResult:
    """select subset of prediction result

    Args:
        pred_result (PredictionResult): prediction result
        available_sample_names (List[str]): available names. Samples
            not in this list will be removed.
            The format is "{ID}" or "{ID}_{position}".

    Returns:
        PredictionResult: subset

    """
    subset = deepcopy(pred_result)

    names_all = pred_result.accumulated_sample_names
    if len(available_sample_names[0].split("_")) == 1:
        names_all = [name.split("_")[0] for name in names_all]
    elif len(available_sample_names[0].split("_")) > 2:
        raise NotImplementedError

    is_avail = np.array([name in available_sample_names for name in names_all])

    subset.accumulated_sample_names = [
        name for flg, name in zip(is_avail, pred_result.accumulated_sample_names) if flg
    ]
    subset.accumulated_pred_prob = pred_result.pred_prob[is_avail]
    subset.accumulated_pred_bin = pred_result.pred_bin[is_avail]
    subset.accumulated_labels = pred_result.labels[is_avail]

    return subset


class KPredictionResults:
    """manage k-fold prediction results

    Attributes:
        n_cv (int): #CV
        valid_pred_files (List[Path]): validation set prediction files
        test_pred_files (List[Path]): test set prediction files
        valid_data_type (str): unique data type (validation set). Defaults to "valid".
        test_data_type (str): unique data type (test set). Defaults to "test".
        _summary_df (Optional[Dict[str, Dict[int, pd.DataFrame]]]): storage
            key0: data type, key2: fold index, value=prediction result.
        _summary (Optional[Dict[str, Dict[int, PredictionResult]]]): storage
            key0: data type, key2: fold index, value=prediction result.
    """

    def __init__(
        self,
        valid_pred_files: List[Path],
        test_pred_files: List[Path],
        valid_data_type: str = "valid",
        test_data_type: str = "test",
    ) -> None:
        """init

        Args:
            valid_pred_files (List[Path]): validation set prediction files
            test_pred_files (List[Path]): test set prediction files
            valid_data_type (str): unique data type (validation set).
                Defaults to "valid".
            test_data_type (str): unique data type (test set). Defaults to "test".

        Raises:
            ValueError:

        """
        if len(valid_pred_files) != len(test_pred_files):
            raise ValueError
        self.n_cv = len(test_pred_files)

        self.valid_pred_files = valid_pred_files
        self.test_pred_files = test_pred_files
        self.valid_data_type = valid_data_type
        self.test_data_type = test_data_type

        # use_all_pos_on_valid: bool = False,
        self._summary_df: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None
        self._summary: Optional[Dict[str, Dict[int, PredictionResult]]] = None

    @classmethod
    def load_from_expdir(
        cls: Type[KPredictionResultsT],
        exp_dir: Path,
        pred_prob_columns: list[str],
        label_columns: list[str],
        n_cv: int = 5,
        valid_dir: str = "develop",
        test_dir: str = "external",
        exp_format: str = "w2v",
        fold_ids: Optional[List[int]] = None,
    ) -> KPredictionResultsT:
        """create instance

        Args:
            exp_dir (Path): experiment directory
            n_cv (int): #CV, Defaults to 5.
            valid_dir (str): subdirectory for validation set.
                Read prediction files in this directory. Defaults to "develop".
            test_dir (str): subdirectory for test set.
                Read prediction files in this directory. Defaults to "external".
            exp_format (str): experiment format

        Returns:
            KPredictionResultsT: k-fold prediction results

        """
        fold_ids = list(range(n_cv)) if fold_ids is None else fold_ids

        if exp_format == "w2v":
            valid_files = [
                exp_dir / valid_dir / f"prediction.{i_cv}.csv" for i_cv in fold_ids
            ]
        elif exp_format == "cnn":
            valid_files = [
                exp_dir / str(i_cv) / "prediction.valid.csv" for i_cv in fold_ids
            ]
        else:
            raise ValueError
        test_files = [
            exp_dir / test_dir / f"prediction.{i_cv}.csv" for i_cv in fold_ids
        ]
        k_results = cls(valid_files, test_files)
        k_results.load_dataframes()
        k_results.load_pred_results(pred_prob_columns, label_columns)
        return k_results

    @property
    def summary(self) -> Dict[str, Dict[int, PredictionResult]]:
        """return _summary"""
        if self._summary is None:
            raise ValueError
        return self._summary

    @property
    def summary_df(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """return _summary_df"""
        if self._summary_df is None:
            raise ValueError
        return self._summary_df

    @property
    def odims(self) -> int:
        """odims"""
        if self._summary is not None:
            return self.summary["valid"][0].labels.shape[0]
        raise ValueError

    @staticmethod
    def normalize_positions(positions: Optional[List[str]]) -> Optional[List[str]]:
        """normalize auscultation position names

        Args:
            positions (Optional[List[str]]): position names

        Returns:
            Optional[List[str]]: position names

        """
        if positions is None or len(positions) == 0:
            return None
        return [AuscultationPosition.normalize_name(pos) for pos in positions]

    def load_dataframes(
        self,
        available_valid_positions: Optional[List[str]] = None,
        available_test_positions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """load prediction result

        Args:
            available_valid_positions (Optional[List[str]]): available position names
                for validation set. position names not in this list will be removed.
                Use all positions if it is None. Defaults to None.
            available_test_positions (Optional[List[str]]): available position names
                for test set. position names not in this list will be removed.
                Use all positions if it is None. Defaults to None.

        Returns:
            Dict[str, Dict[int, pd.DataFrame]]: prediction results.
                key0: data type, key2: fold index, value=prediction result.

        """
        summary: Dict[str, Dict[int, pd.DataFrame]] = {}

        for data_type, pred_files in [
            ("valid", self.valid_pred_files),
            ("test", self.test_pred_files),
        ]:
            summary[data_type] = {}
            positions = (
                available_valid_positions
                if data_type == "valid"
                else available_test_positions
            )
            positions = self.normalize_positions(positions)

            for i_cv, pred_file in enumerate(pred_files):
                data = pd.read_csv(pred_file, index_col=0)

                if positions is not None:
                    index = data.index.to_series().map(
                        AuscultationPosition.get_position_name
                    )
                    mask = index.str.contains(positions[0])
                    for position in positions[1:]:
                        mask = mask | index.str.contains(position)
                    data = data[mask]

                summary[data_type][i_cv] = data

        self._summary_df = summary
        return summary

    def load_pred_results(
        self,
        pred_prob_columns: list[str],
        label_columns: list[str],
        available_valid_positions: Optional[List[str]] = None,
        available_test_positions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, PredictionResult]]:
        """load prediction result

        Args:
            available_valid_positions (Optional[List[str]]): available position names
                for validation set. position names not in this list will be removed.
                Use all positions if it is None. Defaults to None.
            available_test_positions (Optional[List[str]]): available position names
                for test set. position names not in this list will be removed.
                Use all positions if it is None. Defaults to None.

        Returns:
            Dict[str, Dict[int, PredictionResult]]: prediction results.
                key0: data type, key2: fold index, value=prediction result.

        """
        summary: Dict[str, Dict[int, PredictionResult]] = {}

        for data_type, pred_files in [
            ("valid", self.valid_pred_files),
            ("test", self.test_pred_files),
        ]:
            summary[data_type] = {}
            positions = (
                available_valid_positions
                if data_type == "valid"
                else available_test_positions
            )
            positions = self.normalize_positions(positions)

            for i_cv, pred_file in enumerate(pred_files):

                pred_result = PredictionResult.from_csv_file(
                    pred_file,
                    pred_prob_columns,
                    None,
                    label_columns,
                )

                if positions is not None:
                    subsets = create_position_dependent_prediction_result(pred_result)
                    pred_result = stack_prediction_results(
                        [subsets[position] for position in positions]
                    )
                summary[data_type][i_cv] = pred_result

        self._summary = summary
        return summary

    def save(
        self,
        output_dir: Path,
        symbol2id: Dict[str, int],
        can_save_scores: bool = True,
        concat_label_encoding: bool = False,
    ) -> DataFrame:
        """save k-fold prediction results

        Args:
            output_dir (Path): output directory
            symbols (int): symbols
            can_save_scores (bool): compute and save scores if True.
                Defaults to True.
            concat_label_encoding (bool): concatenate label encoding info
                if True. Defaults to False.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        symbols = list(symbol2id.keys())

        table: Optional[pd.DataFrame] = None
        if concat_label_encoding:
            table = None

        valid_all: List[DataFrame] = []
        valid_pred_results: List[PredictionResult] = []

        for dtype in [self.valid_data_type, self.test_data_type]:

            for i_cv in range(self.n_cv):
                output_file = output_dir / f"prediction.{dtype}.{i_cv}.csv"
                pred_result = self.summary[dtype][i_cv]
                valid = pred_result.save_file(output_file, symbols, table)
                assert isinstance(valid, pd.DataFrame)
                if dtype == self.valid_data_type:
                    valid["i_cv"] = i_cv
                    valid_all.append(valid)

                if dtype == self.valid_data_type:
                    valid_pred_results.append(pred_result)

        output_file = output_dir / f"prediction.{self.valid_data_type}.fold_all.csv"
        pd.concat(valid_all).to_csv(output_file)

        return pd.DataFrame()


def split_by_position(
    k_pred_result: KPredictionResults,
    can_remove_position_name: bool = False,
    data_types: Optional[List[str]] = None,
    dummy_position_name: Optional[str] = None,
) -> Dict[str, KPredictionResults]:
    """split k-fold prediction result by auscultation position

    Args:
        k_pred_result (KPredictionResults): k-fold prediction result
        can_remove_position_name (bool): remove position info
            if True. Defaults to False.
        dummy_position_name (Optional[str]): dummy position name added to
            sample_names (used when it is not None and can_remove_position_name=True).
            Merge [w2v@2lsb, w2v@4lsb, cnn@2lsb, w2v@4lsb].

    Returns:
        Dict[str, KPredictionResults]: subset of prediction results
            split by auscultation position ("2rsb", "2lsb", "4lsb", "5lmcl").

    """
    if data_types is None:
        data_types = ["valid", "test"]
    n_cv = k_pred_result.n_cv

    ret: Dict[str, KPredictionResults] = {}

    for position in AuscultationPosition.available_names():
        ret[position] = KPredictionResults(
            k_pred_result.valid_pred_files,
            k_pred_result.test_pred_files,
            k_pred_result.valid_data_type,
            k_pred_result.test_data_type,
        )
        ret[position].n_cv = n_cv
        ret[position]._summary = {dtype: {} for dtype in data_types}
        ret[position]._summary_df = {dtype: {} for dtype in data_types}

    if k_pred_result._summary is not None:

        for data_type in data_types:
            for i_cv in range(n_cv):
                subsets = create_position_dependent_prediction_result(
                    k_pred_result.summary[data_type][i_cv]
                )
                for position in AuscultationPosition.available_names():
                    subset = subsets[position]
                    if can_remove_position_name:
                        subset.accumulated_sample_names = [
                            name.split("_")[0]
                            for name in subset.accumulated_sample_names
                        ]
                        if dummy_position_name is not None:
                            subset.accumulated_sample_names = [
                                name + "_" + dummy_position_name.upper()
                                for name in subset.accumulated_sample_names
                            ]
                    ret[position]._summary[data_type][i_cv] = subset  # type: ignore[index]

    if k_pred_result._summary_df is not None:

        for data_type in data_types:
            for i_cv in range(n_cv):
                dfs = k_pred_result.summary_df[data_type][i_cv]
                for position in AuscultationPosition.available_names():
                    subset_df = dfs[
                        dfs.index.str.contains(position)
                        | dfs.index.str.contains(position.upper())
                    ]
                    if can_remove_position_name:
                        subset_df.index = (
                            subset_df.index.to_series().str.split("_").str.get(0)
                        )
                        if dummy_position_name is not None:
                            subset_df.index = (
                                dummy_position_name.upper() + "_" + subset_df.index
                            )
                    ret[position]._summary_df[data_type][i_cv] = subset_df  # type: ignore[index]

    return ret


def select_cutoff(
    k_pred_result: KPredictionResults,
    cutoff_idx: int,
    data_types: Optional[List[str]] = None,
) -> KPredictionResults:
    """select single cutoff

    Args:
        k_pred_result (KPredictionResults): k-fold prediction results
        cutoff_idx (int): cutoff index
        data_types (Optional[List[str]]): data types (valid and test)

    """
    if data_types is None:
        data_types = ["valid", "test"]
    n_cv = k_pred_result.n_cv
    ret = deepcopy(k_pred_result)

    for data_type in data_types:
        for i_cv in range(n_cv):
            ret._summary[data_type][i_cv] = get_single_cutoff(  # type: ignore[index]
                k_pred_result._summary[data_type][i_cv],  # type: ignore[index]
                cutoff_idx,
            )

    # symbols
    for data_type in data_types:
        for i_cv in range(n_cv):

            pred_df = k_pred_result._summary_df[data_type][i_cv]  # type: ignore[index]
            symbol = pred_df.columns[cutoff_idx].replace("_pred_prob", "")
            ret._summary_df[data_type][i_cv] = pred_df[  # type: ignore[index]
                [f"{symbol}_pred_prob", f"{symbol}_pred_bin", f"{symbol}_pred_bin"]
            ]

    return ret
