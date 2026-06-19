from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch
from omegaconf import DictConfig, OmegaConf

from elvef.create_predictor import create_predictor
from elvef.utils import data_helper, model_helper
from script.s01_train_models import Params, create_wav2vec_model


def preprocess() -> None:
    """save dummy reference label, PCG/ECG data, etc."""

    model_helper.set_seed()

    params = Params()
    output_dir = params.data_dir
    batch_size = params.batch_size

    # reference labels
    labels_dict = {
        idx: np.random.randint(0, 2, params.odims) for idx in range(batch_size)
    }

    # k-fold CV.
    for i_cv in range(params.n_cv):
        for data_type in params.data_types:

            index = [str(idx).zfill(6) + "_2RSB" for idx in range(batch_size)]
            index += [str(idx).zfill(6) + "_2LSB" for idx in range(batch_size)]
            index += [str(idx).zfill(6) + "_4LSB" for idx in range(batch_size)]
            index += [str(idx).zfill(6) + "_5LMCL" for idx in range(batch_size)]

            df = pd.DataFrame()
            df.index = index
            df["ID"] = df.index.to_series().str.split("_").str.get(0).astype(int)
            df["pcg_file"] = ""
            df["pcg_ecg_file"] = ""

            for index, row in df.iterrows():
                labels: np.ndarray = labels_dict[row["ID"]]
                df.loc[index, params.label_columns] = labels
                df.loc[index, params._label_columns] = labels

            output_file = output_dir / f"cv{i_cv}_{data_type}.csv"
            df.to_csv(output_file)


def load_npy_file(
    npy_file: Path,
    channels: int,
    sampling_frequency: int = 2000,
    trim_mode: int = 1,
) -> np.ndarray:
    """load time series data

    Args:
        npy_file (Path): time series data (channels, seqlen).
        channels (int): #chanenls
        sampling_frequency (int): sampling frequency [Hz]. Defaults to 2000.
        trim_mode (int): trim mode. Defaults to 1 (create 8[sec] data).
            0: returns the original data.

    Returns:
        np.ndarray: time series data

    """

    # create dummy data
    duration = 8.0
    seqlen = int(duration * sampling_frequency)
    data = np.random.randn(channels, seqlen)

    data = data_helper.trim_np_data(data, sampling_frequency, trim_mode=trim_mode)
    return data


def load_pcg_file(pcg_file: Path = Path("")) -> np.ndarray:
    """load PCG data

    Returns:
        np.ndarray: PCG data (channels=1, seqlen).

    """
    return load_npy_file(pcg_file, channels=1)


def load_pcg_ecg_file(pcg_ecg_file: Path = Path("")) -> np.ndarray:
    """load synchronized PCG/ECG data.

    Returns:
        np.ndarray: synchronized PCG/ECG data (channels=2, seqlen).

    """
    return load_npy_file(pcg_ecg_file, channels=2)


def preprocess_data(data: np.ndarray) -> np.ndarray:
    """preprocessing step"""
    return data


def predict_cnn(
    is_debug: bool,
    config_file: Path = Path("config/cnn.yaml"),
) -> None:
    """predict on k-fold develop/external-test sets using CNN.

    Args:
        is_debug (bool): debug mode. Output a variety of random prediction results if True.
        config_file (Path): CNN config file. Defaults to"config/cnn.yaml".

    """
    params = Params()

    config = DictConfig(OmegaConf.load(config_file))

    fs = 2000
    dummy_threshold = 0.5
    segment_duration = 5.0
    n_expand = 16

    data_dir = params.data_dir
    output_dir = params.exp_dir / "cnn"

    for i_cv in range(params.n_cv):

        # 1. load model
        model_file = params.cnn_model_dir / str(i_cv) / "model.pth"
        assert model_file.exists()
        model = create_predictor(params.odims, config, model_file, is_test=True)
        model = model.to(params.device)

        for data_type in params.data_types:

            dataset = pd.read_csv(data_dir / f"cv{i_cv}_{data_type}.csv", index_col=0)
            result = pd.DataFrame()

            for index, row in dataset.iterrows():
                # 2.1. prepare data
                pcg_data = load_pcg_file()
                pcg_data = preprocess_data(pcg_data)
                pcg_segments = data_helper.make_segments(
                    pcg_data,
                    segment_seqlen=int(fs * segment_duration),
                    n_expand=n_expand,
                )

                # 2.2. forward path
                with torch.no_grad():

                    if is_debug:
                        np_pred_probs = np.random.rand(params.odims)

                    else:
                        model_helper.set_seed()
                        # (n_expand, odims).
                        pred_probs = model.predict(
                            torch.from_numpy(pcg_segments).float().to(params.device)
                        )[1]
                        pred_probs = torch.mean(pred_probs, 0)
                        np_pred_probs = pred_probs.cpu().data.numpy()

                np_pred_bins = (np_pred_probs >= dummy_threshold).astype(np.int32)
                labels = [row[column] for column in params._label_columns]

                result.loc[index, params._pred_prob_columns] = np_pred_probs
                result.loc[index, params._pred_bin_columns] = np_pred_bins
                result.loc[index, params._label_columns] = labels

            # 3. save result
            if data_type == "develop":
                output_file = output_dir / str(i_cv) / "prediction.valid.csv"
            else:
                output_file = output_dir / "external" / f"prediction.{i_cv}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_file)


def predict_wav2vec(
    is_debug: bool,
    config_file: Path = Path("config/wav2vec.yaml"),
) -> None:
    """predict on k-fold develop/external-test sets using wav2vec

    Args:
        is_debug (bool): debug mode. Output a variety of random prediction results if True.
        config_file (Path): CNN config file. Defaults to"config/wav2vec.yaml".

    """

    params = Params()

    config = DictConfig(OmegaConf.load(config_file))

    fs = 2000
    dummy_threshold = 0.5
    segment_duration = 5.0
    n_expand = 4

    data_dir = params.data_dir
    output_dir = params.exp_dir / "w2v"

    for i_cv in range(params.n_cv):

        # 1. load model
        model_path = params.w2v_model_dir / str(i_cv) / "model.pth"
        assert model_path.exists()
        model = create_wav2vec_model(
            config,
            output_dim=params.odims,
            w2v_path=params.w2v_model_dir / "pretrained.ckpt",
            model_path=model_path,
        )
        model.eval()

        for data_type in params.data_types:

            dataset = pd.read_csv(data_dir / f"cv{i_cv}_{data_type}.csv", index_col=0)
            result = pd.DataFrame()

            for index, row in dataset.iterrows():
                # 2.1. prepare data
                pcg_ecg_data = load_pcg_ecg_file()
                pcg_ecg_data = preprocess_data(pcg_ecg_data)
                pcg_ecg_segments = data_helper.make_segments(
                    pcg_ecg_data,
                    segment_seqlen=int(fs * segment_duration),
                    n_expand=n_expand,
                )

                # 2.2. forward path
                with torch.inference_mode():

                    if is_debug:
                        np_pred_probs = np.random.rand(params.odims)

                    else:
                        model_helper.set_seed()
                        ret = model(torch.from_numpy(pcg_ecg_segments).float())

                        # (n_expand, odims).
                        pred_probs = torch.sigmoid(ret["encoder_out"])
                        pred_probs = torch.mean(pred_probs, 0)
                        np_pred_probs = pred_probs.cpu().data.numpy()

                np_pred_bins = (np_pred_probs >= dummy_threshold).astype(np.int32)
                labels = [row[column] for column in params.label_columns]

                result.loc[index, params.pred_prob_columns] = np_pred_probs
                result.loc[index, params.pred_bin_columns] = np_pred_bins
                result.loc[index, params.label_columns] = labels

            output_file = output_dir / data_type / f"prediction.{i_cv}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_file)


def main(is_dummy: bool = False) -> None:
    """predict"""

    preprocess()
    predict_cnn(is_debug=True)
    predict_wav2vec(is_debug=True)


if __name__ == "__main__":
    main()
