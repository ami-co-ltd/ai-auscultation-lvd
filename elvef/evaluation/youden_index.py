#!/usr/bin/env python

import numpy as np
from sklearn.metrics import roc_curve


def youden_index_np(
    pred_probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """compute youden index

    Args:
        pred_probs (np.ndarray): predicted probabilities
            (batch,) or (batch, odims).
        labels (np.ndarray): reference labels
            (batch,) or (batch, odims).

    Returns:
        np.ndarray: binalization thresholds (odims,).

    """
    assert pred_probs.shape == labels.shape

    if len(pred_probs.shape) == 1:
        pred_probs = pred_probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)

    odims = labels.shape[1]
    thresholds = np.zeros(odims, dtype=np.float32)

    for odim in range(odims):
        pred_prob = pred_probs[:, odim]
        label = labels[:, odim]

        pred_prob = pred_prob[label >= 0]
        label = label[label >= 0]

        fpr, tpr, thr = roc_curve(label, pred_prob)
        best_idx = np.argmax(tpr - fpr)
        thresholds[odim] = float(thr[best_idx])

    return thresholds
