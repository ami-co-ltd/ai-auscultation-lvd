# pylint: disable=consider-using-f-string,too-many-instance-attributes,assignment-from-none,useless-return,invalid-name,unused-argument

import dataclasses
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

logger = getLogger(__name__)


@dataclasses.dataclass
class PredictionResult:
    """dataclass to store prediction result and reference label

    Note:
        - array shape
          - cutoff
            - pred_prob: (batch, n_cutoffs)
            - pred_bin: (batch, n_cutoffs)
            - labels: (batch, n_cutoffs)

    Attributes:
        binalization_thr (Union[float, np.ndarray]): thresholds to binarize predictions
        valid_symbol_ids (Union[List[int], None]): list of symbol ids to evaluate
        accumulated_sample_names (List[str]):
            list of unique identifiers to represent patients,
            e.g., ["kj0001_2rsb", "kj0001_2lsb", ...,  "tn0001_2rsb", ...]
        process_times (List[float]): list of processing times
        accumulated_pred_prob (Union[np.ndarray, None]): prediction results
        accumulated_bin (Union[np.ndarray, None]): binalized prediction results
        accumulated_labels (Union[np.ndarray, None]): reference labels
        can_round_probs (bool): round posteiror probabilities if True.
            This flag will be removed in future. Set can_round_probs to
            True for backward compatibility.

    """

    binalization_thr: Union[float, np.ndarray]
    valid_symbol_ids: Union[List[int], None]

    accumulated_sample_names: List[str] = dataclasses.field(default_factory=list)
    process_times: List[float] = dataclasses.field(default_factory=list)

    accumulated_pred_prob: Union[np.ndarray, None] = dataclasses.field(
        default=None, init=False
    )
    accumulated_pred_bin: Union[np.ndarray, None] = dataclasses.field(
        default=None, init=False
    )
    accumulated_labels: Union[np.ndarray, None] = dataclasses.field(
        default=None, init=False
    )
    # can_accept_ignore_id: bool = True
    # can_round_probs: bool = False
    is_multi_label: bool = True

    @classmethod
    def from_csv_file(
        cls,
        csv_file: Path,
        pred_prob_columns: list[str],
        pred_bin_columns: Optional[list[str]],
        label_columns: list[str],
    ):
        """read prediction result"""

        threshold = 0.5
        result = cls(threshold, None)

        df = pd.read_csv(csv_file, index_col=0)

        result.accumulated_sample_names = list(df.index)
        result.accumulated_pred_prob = df[pred_prob_columns].values
        if pred_bin_columns is None:
            # apply dummy threshold
            result.accumulated_pred_bin = (
                df[pred_prob_columns].values >= threshold
            ).astype(np.int32)
        else:
            result.accumulated_pred_bin = df[pred_bin_columns].values.astype(np.int32)
        result.accumulated_labels = df[label_columns].values

        return result

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()

    def check_parameters(self) -> None:
        """check parameters

        Raises:
            ValueError: Got unexpected binarization threshold
            ValueError: Got unexpected binarization thresholds
        """
        # - binalization thresholds
        if isinstance(self.binalization_thr, float):
            if not 0.0 <= self.binalization_thr <= 1.0:
                raise ValueError("Got unexpected binarization threshold")
        else:
            if (np.any(self.binalization_thr < 0)) or (
                np.any(self.binalization_thr > 1.0)
            ):
                raise ValueError("Got unexpected binarization thresholds")

    def accumulate_results(
        self,
        pred_prob: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        sample_names: List[str],
        process_time: float = -1.0,
    ) -> None:
        """accumulate results

        Args:
            pred_prob (Union[np.ndarray, torch.Tensor]): prediction of shape
                (batch, n_cutoffs) or (batch, n_diagnoses, n_severities)
            labels (Union[np.ndarray, torch.Tensor]): reference labels of shape
                (batch, n_cutoffs) or (batch, n_diagnoses).
            sample_names (List[str]): list of sample names (batch,)
            process_time (float): processing time [sec]

        """
        # 1. convert to np.ndarray
        np_pred_prob = (
            pred_prob.cpu().data.numpy().astype(np.float32)
            if isinstance(pred_prob, torch.Tensor)
            else pred_prob
        )
        if np.isnan(np_pred_prob).any():
            raise ValueError

        np_labels = (
            labels.cpu().data.numpy().astype(np.int32)
            if isinstance(labels, torch.Tensor)
            else labels
        )

        if (len(np_pred_prob.shape) != 3) and (np_pred_prob.shape != np_labels.shape):
            raise ValueError("Got inconsistent prediction and reference label shapes")
        if np_pred_prob.shape[0] != len(sample_names):
            raise ValueError("Got unexpected number of sample_names")

        # 2.1. accumulated_pred_prob
        if self.accumulated_pred_prob is None:
            self.accumulated_pred_prob = np_pred_prob
        else:
            self.accumulated_pred_prob = np.vstack(
                (self.accumulated_pred_prob, np_pred_prob)
            )
        # 2.2. accumulated_labels
        if self.accumulated_labels is None:
            self.accumulated_labels = np_labels
        else:
            self.accumulated_labels = np.vstack((self.accumulated_labels, np_labels))

        # 2.3. accumulated_sample_names
        self.accumulated_sample_names += sample_names
        self.process_times.append(process_time)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        """reset binalization thresholds

        Args:
            thresholds (np.ndarray): binalization thresholds (n_cutoffs,).

        """
        self.binalization_thr = thresholds
        self.get_data()

    def get_data(self) -> None:
        """return batched data

        Note:
            - shape of each ndarray: (B, O)
            - B: batch size
            - O: output dimension

        Raises:
            ValueError: No prediction result

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                prediction, binalized prediction, and ground-truth labels
        """
        # get_data w/o accumulate_results
        if (self.accumulated_pred_prob is None) or (self.accumulated_labels is None):
            raise ValueError("No prediction result")

        # binalization
        if len(self.accumulated_pred_prob.shape) == 2:
            self.accumulated_pred_bin = (
                self.accumulated_pred_prob >= self.binalization_thr
            ).astype(np.int32)

    #     else:
    #         self.accumulated_pred_bin = np.array(
    #             np.argmax(self.accumulated_pred_prob, axis=2), dtype=np.int32
    #         )

    #     if self.valid_symbol_ids is not None:
    #         # if self.accumulated_pred_prob is not None:
    #         self.accumulated_pred_prob = self.accumulated_pred_prob[
    #             :, self.valid_symbol_ids
    #         ]
    #         # if self.accumulated_pred_bin is not None:
    #         self.accumulated_pred_bin = self.accumulated_pred_bin[
    #             :, self.valid_symbol_ids
    #         ]
    #         # if self.accumulated_labels is not None:
    #         self.accumulated_labels = self.accumulated_labels[:, self.valid_symbol_ids]

    #         self.valid_symbol_ids = None

    #     # check NA symbol
    #     if not self.can_accept_ignore_id and np.any(self.accumulated_labels == -1):
    #         logger.warning("unknown labels will be converted to 0.")
    #         self.accumulated_labels[self.accumulated_labels == -1] = 0

    #     return (
    #         self.accumulated_pred_prob,
    #         self.accumulated_pred_bin,
    #         self.accumulated_labels,
    #     )

    def save_file(
        self,
        output_file: Path,
        symbols: Optional[List[str]] = None,
        table: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        """save model outputs and reference labels

        Args:
            output_file (Path): output file
            symbols (Optional[List[str]]): symbol names that are used for
                csv header. Defaults to None.
            table (Optional[DataFrame]): characteristics of patients.
                Defaults to None.

        Returns:
            Optional[DataFrame]: saved result

        """
        if output_file.suffix == ".txt":
            saved = self.save_txt_file(output_file)
        elif output_file.suffix == ".csv":
            saved = self.save_csv_file(output_file, symbols, table)
        return saved

    def to_dataframe(self, symbols: Optional[List[str]] = None) -> DataFrame:
        """convert to DataFrame"""
        if symbols is None:
            symbols = [f"dim-{idx}" for idx in range(self.pred_prob.shape[1])]

        columns_pred = [f"{symbol}_pred_prob" for symbol in symbols]
        columns_bin = [f"{symbol}_pred_bin" for symbol in symbols]
        columns_ref = [f"{symbol}_reference" for symbol in symbols]
        result = DataFrame(
            {},
            index=self.accumulated_sample_names,
            columns=columns_pred + columns_bin + columns_ref,
        )

        pred_prob = self.pred_prob
        pred_bin = (self.pred_prob >= self.thresholds).astype(np.int32)
        pred_bin[self.labels < 0] = -1
        labels = self.labels.astype(np.int32)
        sample_names = self.accumulated_sample_names

        # result.index = sample_names
        result.loc[sample_names, columns_pred] = pred_prob.tolist()
        result.loc[sample_names, columns_bin] = pred_bin.tolist()
        result.loc[sample_names, columns_ref] = labels.tolist()
        return result

    def save_csv_file(
        self,
        output_file: Path,
        symbols: Optional[List[str]],
        table: Optional[DataFrame],
    ) -> Optional[pd.DataFrame]:
        """save prediction results as csv file

        Note:
            - skip rounding operation

        Args:
            output_file (Path): output file
            symbols (Optional[List[str]]): symbol names that are used for
                csv header. Defaults to None.
            table (Optional[DataFrame]): characteristics of patients.

        Returns:
            Optional[pd.DataFrame]: saved data

        """
        result = self.to_dataframe(symbols)

        if table is not None:
            sample_name = result.index[0]
            if len(sample_name.split("_")) == 1:
                result["ID"] = result.index
            elif len(sample_name.split("_")) == 2:
                result[["ID", "position"]] = result.index.to_series().str.split(
                    "_", expand=True
                )
            else:
                raise ValueError
            table["ID"] = table.index
            table.index.name = "_ID"
            result = pd.merge(result, table, on="ID", how="left")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file)
        return result

    def save_txt_file(self, file_name: Union[str, Path]) -> Optional[DataFrame]:
        """save prediction results as txt file"""
        return None

    @property
    def pred_prob(self) -> np.ndarray:
        """return accumulated_pred_prob

        Raises:
            ValueError: No prediction result

        Returns:
            np.ndarray: accumulated_pred_prob
        """

        if self.accumulated_pred_prob is None:
            raise ValueError("No prediction result")
        return self.accumulated_pred_prob

    @property
    def pred_bin(self) -> np.ndarray:
        """return accumulated_pred_bin

        Note:
            - we need to call get_data first.

        Raises:
            ValueError: No prediction result

        Returns:
            np.ndarray: accumulated_pred_bin
        """
        if self.accumulated_pred_bin is None:
            raise ValueError("No prediction result")
        return self.accumulated_pred_bin

    @property
    def labels(self) -> np.ndarray:
        """return accumulated_labels

        Note:
            - we need to call get_data first.

        Raises:
            ValueError: No prediction result

        Returns:
            np.ndarray: accumulated_labels

        """
        if self.accumulated_labels is None:
            raise ValueError("No prediction result")
        return self.accumulated_labels

    @property
    def thresholds(self) -> Union[float, np.ndarray]:
        """return binalization thresholds"""

        return self.binalization_thr
