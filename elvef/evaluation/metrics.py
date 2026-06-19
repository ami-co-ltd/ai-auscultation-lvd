# pylint: disable=bare-except,too-many-arguments

from logging import getLogger
from typing import Dict

import numpy as np
import sklearn
from sklearn.metrics import auc, precision_recall_curve

logger = getLogger(__name__)


def evaluate(
    pred_probs: np.ndarray,
    pred_bins: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False,
) -> Dict[str, float]:
    """evaluate

    Args:
        pred_probs (np.ndarray): posterior probabilities (batch,)
        pred_bins (np.ndarray): binalized predictions (batch,).
        labels (np.ndarray): reference labels (batch,)
        verbose (bool): print scores if True. Defaults to False.

    Returns:
        Dict[str, float]: scores

    """
    pred_probs = pred_probs[np.isin(labels, [0, 1])]
    pred_bins = pred_bins[np.isin(labels, [0, 1])]
    labels = labels[np.isin(labels, [0, 1])]

    true_neg, false_pos, false_neg, true_pos = 0, 0, 0, 0
    for label, pred_bin in zip(labels, pred_bins):
        if label == 0 and pred_bin == 0:
            true_neg += 1
        elif label == 0 and pred_bin == 1:
            false_pos += 1
        elif label == 1 and pred_bin == 0:
            false_neg += 1
        else:
            true_pos += 1

    try:
        auroc = sklearn.metrics.roc_auc_score(labels, pred_probs)
    except:
        auroc = float("nan")

    try:
        precision, recall, _ = precision_recall_curve(labels, pred_probs)
        auprc = auc(recall, precision)
    except:
        auprc = float("nan")

    sensitivity, specificity = float("nan"), float("nan")
    if true_pos + false_neg > 0:
        sensitivity = true_pos / (true_pos + false_neg)
    if true_neg + false_pos > 0:
        specificity = true_neg / (true_neg + false_pos)
    ppv, npv = float("nan"), float("nan")
    if true_pos + false_pos > 0:
        ppv = true_pos / (true_pos + false_pos)
    if true_neg + false_neg > 0:
        npv = true_neg / (true_neg + false_neg)

    result = {
        "n_positives": np.sum(labels == 1),
        "n_negatives": np.sum(labels == 0),
        "auroc": auroc,
        "auprc": auprc,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "true_pos": true_pos,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
    }
    if verbose:
        for key, val in result.items():
            logger.info(f"{key}={val}")
    return result
