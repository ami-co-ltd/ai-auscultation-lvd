#!/usr/bin/env python
# pylint: disable=consider-using-f-string,too-many-locals,too-many-arguments,using-constant-test,too-many-statements,too-few-public-methods,protected-access

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class AurocFigure:
    """Plot AUROC"""

    def __init__(self, use_fp_tp: bool = True) -> None:
        """init"""
        self.fig, self.axes = plt.subplots(figsize=(6, 6))

        self.axes.set_xlim([-0.05, 1.05])
        self.axes.set_ylim([-0.05, 1.05])
        if use_fp_tp:
            self.axes.set_xlabel("False Positive Rate")
            self.axes.set_ylabel("True Positive Rate")
        else:
            self.axes.set_xlabel("1 - Specificity")
            self.axes.set_ylabel("Sensitivity")

    def _add_result(
        self,
        pred_prob: np.ndarray,
        labels: np.ndarray,
        color: Union[str, Tuple[float, ...]] = "tab:blue",
        label: str = "",
        threshold: float = -1.0,
        pred_bin: Optional[np.ndarray] = None,
    ) -> None:
        """append curves

        Note:
            - color example:
              - tab:blue, :orange, :green, :red, :purple, :brown, :pink, ...
            - https://bunsekikobako.com/matplotlib-default-colors/

        Args:
            pred_prob (np.ndarray): posterior probabilities (#batch,).
            labels (np.ndarray): labels (#batch,).
            color (Union[str, Tuple[float, ...]]): color of the curve. Defaults to "tab:green".
            label (str): label.
            threshold (float): binalization threshold. Compute an oracle
                value (from same dataset) if <=0.
            pred_bin (Optional[np.ndarray]): binalized prediction used to compute Sn&Sp.
                Use pred_prob and threshold if it is None. Defaults to None.

        """
        # 1. compute AUROC
        pred_prob = pred_prob[np.isin(labels, [0, 1])]
        labels = labels[np.isin(labels, [0, 1])]
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            labels, pred_prob, drop_intermediate=False
        )
        auc = sklearn.metrics.auc(fpr, tpr)

        # 2. compute tpr and fpr under the given binalization threshold
        if threshold <= 0:
            best_idx = np.argmax(tpr - fpr)
            threshold = thresholds[best_idx]

        if pred_bin is None:
            pred_bin = pred_prob >= threshold
        conf_matrix = sklearn.metrics.confusion_matrix(labels, pred_bin)
        true_neg, false_pos, false_neg, true_pos = conf_matrix.flatten()
        cutoff_fpr = false_pos / (false_pos + true_neg)
        cutoff_tpr = true_pos / (true_pos + false_neg)

        # 3. plot
        self.axes.plot(
            fpr,
            tpr,
            label="[%s] AUROC: %.3f, TPR: %.3f, FPR: %.3f"
            % (label, auc, cutoff_tpr, cutoff_fpr),
            color=color,
        )
        self.axes.vlines(
            x=cutoff_fpr, ymin=0, ymax=1, color=color, alpha=0.3, linewidth=1.0
        )
        self.axes.hlines(
            y=cutoff_tpr, xmin=0, xmax=1, color=color, alpha=0.3, linewidth=1.0
        )
        self.axes.scatter(cutoff_fpr, cutoff_tpr, color=color)

    def save(self, title: str, output_file: Union[str, Path]) -> None:
        """save figure

        Args:
            title (str): title
            output_file (str): output file

        """
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.savefig(output_file)
        plt.clf()
        plt.close()


def example() -> None:
    """plot ROC curve"""
    np.random.seed(0)

    batch_size = 100
    threshold = 0.2
    output_file = Path("auroc.png")

    develop = pd.DataFrame(
        {
            "pred_prob": np.random.uniform(batch_size),
            "label": np.random.randint(0, 2, size=batch_size),
        }
    )
    test = pd.DataFrame(
        {
            "pred_prob": np.random.uniform(batch_size),
            "label": np.random.randint(0, 2, size=batch_size),
        }
    )
    develop["pred_bin"] = (develop["pred_prob"] >= threshold).astype(int)
    test["pred_bin"] = (test["pred_prob"] >= threshold).astype(int)

    figure = AurocFigure(use_fp_tp=False)

    for label, color, data in [
        ("", "tab:blue", develop),
        ("", "tab:orange", test),
    ]:
        figure._add_result(
            data["pred_prob"].values,
            data["label"].values,
            color,
            label=label,
            pred_bin=data["pred_bin"].values,
        )

    figure.save("", output_file)


if __name__ == "__main__":
    example()
