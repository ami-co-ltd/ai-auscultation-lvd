#!/usr/bin/env python

import pprint
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from elvef.evaluation import metrics
from elvef.evaluation.custom_ensemble import EnsembleExpV2
from elvef.evaluation.k_prediction_result import (
    KPredictionResults,
    select_cutoff,
    split_by_position,
)
from elvef.utils.terminology import AuscultationPosition
from script.s01_train_models import Params


def run_ensemble_npos_4(is_debug: bool = False) -> None:
    """ensemble"""

    params = Params()

    output_dir = params.exp_dir / "ensemble"

    w2v_k_results = KPredictionResults.load_from_expdir(
        params.exp_dir / "w2v",
        params.pred_prob_columns,
        params.label_columns,
        exp_format="w2v",
    )
    cnn_k_results = KPredictionResults.load_from_expdir(
        params.exp_dir / "cnn",
        params._pred_prob_columns,
        params._label_columns,
        exp_format="cnn",
    )

    batched: list[KPredictionResults] = []
    for k_results in [w2v_k_results, cnn_k_results]:
        position_dependent: dict[str, KPredictionResults] = split_by_position(
            deepcopy(k_results),
            can_remove_position_name=True,
        )
        for position in AuscultationPosition.available_names():
            batched.append(
                select_cutoff(position_dependent[position], params.cutoff_idx)
            )

    # 2. select the best model out of k-fold CV (2RSB/2LSB/4LSB/5LMCL).
    w2v_fold_ids = [3] * 4
    cnn_fold_ids = [1] * 4
    n_steps = 6 if is_debug else 21

    # 3. ensemble
    exp = EnsembleExpV2(batched)
    exp.optimize(n_steps=n_steps)
    res = exp.ensemble_all(batched, best_cvs=w2v_fold_ids + cnn_fold_ids)
    res.save(output_dir, params.symbol2id, concat_label_encoding=True)
    exp.save(output_dir)


def evaluate() -> None:
    """compute AUROC"""

    params = Params()

    develop = pd.concat(
        [
            pd.read_csv(
                params.exp_dir / f"ensemble/prediction.valid.{i_cv}.csv", index_col=0
            )
            for i_cv in range(params.n_cv)
        ],
        axis=0,
    )
    test = pd.read_csv(params.exp_dir / "ensemble/prediction.test.0.csv", index_col=0)

    for data in [develop, test]:
        scores = metrics.evaluate(
            data["dim-0_pred_prob"].values,
            data["dim-0_pred_bin"].values,
            data["dim-0_reference"].values,
        )
        pprint.pprint(scores)


def main() -> None:
    """main"""

    exp_dir = Path("exp.dummy")

    run_ensemble_npos_4(is_debug=True)
    evaluate()


if __name__ == "__main__":
    main()
