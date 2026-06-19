#!/usr/bin/env python

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from elvef.create_predictor import create_predictor
from elvef.models.nch_wav2vec2 import NchWav2Vec2Config, NchWav2Vec2Model
from elvef.models.nch_wav2vec2_maxpool import NchWav2Vec2Audio, NchWav2Vec2MaxPoolConfig
from elvef.utils import model_helper


class Params:
    """manage params and directories used in script/"""

    def __init__(
        self,
        data_dir: Path = Path("exp.dummy/data"),
        exp_dir: Path = Path("exp.dummy/exp"),
        device: str = "cuda",
    ) -> None:

        # data
        self.n_cv = 5  # 5-fold CV.
        self.batch_size = 100  # dummy samples
        self.data_types = ["develop", "external"]

        self.cutoff_idx = 0  # LVEF<40.
        self.symbol2id = {
            "dim-0": 0,
            "dim-1": 1,
            "dim-2": 2,
        }
        # predicted probabilities
        self.pred_prob_columns = [
            f"{symbol}_pred_prob" for symbol in self.symbol2id.keys()
        ]
        # binarized prediction
        self.pred_bin_columns = [
            f"{symbol}_pred_bin" for symbol in self.symbol2id.keys()
        ]
        # reference labels
        self.label_columns = [f"{symbol}_reference" for symbol in self.symbol2id.keys()]
        self._pred_prob_columns = self.pred_prob_columns
        self._pred_bin_columns = self.pred_bin_columns
        self._label_columns = self.label_columns

        # exp
        self.data_dir = data_dir
        self.exp_dir = exp_dir
        self.w2v_model_dir = exp_dir / "01_train_models" / "wav2vec"
        self.cnn_model_dir = exp_dir / "01_train_models" / "cnn"

        self.device = device if torch.cuda.is_available() else "cpu"

        self.initialize()

    def initialize(self) -> None:
        """mkdirs"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.w2v_model_dir.mkdir(parents=True, exist_ok=True)
        self.cnn_model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def odims(self) -> int:
        """output dimension"""
        return len(self.symbol2id.keys())


def create_wav2vec_model(
    config: DictConfig,
    output_dim: int,
    w2v_path: Path,
    model_path: Optional[Path] = None,
) -> torch.nn.Module:
    """create wav2vec model

    Args:
        config (DictConfig): wav2vec config file
        output_dim (int): output dimension
        w2v_path (Path): pretrained wav2vec encoder networks
        model_path (Optional[Path]): fitted model. Load weights if it is not None.

    Returns:
        torch.nn.Module: wav2vec model.

    """

    w2v_predictor_config = NchWav2Vec2MaxPoolConfig(
        w2v_path=str(w2v_path),
        activation_dropout=config.models_ft.activation_dropout,
        freeze_finetune_updates=config.models_ft.freeze_finetune_updates,
        feature_grad_mult=config.models_ft.feature_grad_mult,
        mlp_layers=config.models_ft.mlp_layers,
        use_msd=config.models_ft.use_msd,
        output_size=output_dim,
        tgt_layer=config.models_ft.tgt_layer,
    )
    w2v_predictor = NchWav2Vec2Audio(w2v_predictor_config)

    if model_path is not None:
        model_state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )
        w2v_predictor.load_state_dict(model_state_dict, strict=True)
    return w2v_predictor


def run_wav2vec(config_file: Path = Path("config/wav2vec.yaml")) -> None:
    """create dummy wav2vec models

    Args:
        config_file (Path): wav2vec config file. Defaults to"config/wav2vec.yaml".

    """

    params = Params()
    config = DictConfig(OmegaConf.load(config_file))

    # 1. self-supervised training
    w2v_encoder_config = NchWav2Vec2Config(
        encoder_layers=config.models_ssl.encoder_layers,
        dropout=config.models_ssl.dropout,
        attention_dropout=config.models_ssl.attention_dropout,
        activation_dropout=config.models_ssl.activation_dropout,
        dropout_features=config.models_ssl.dropout_features,
        final_dim=config.models_ssl.final_dim,
        conv_feature_layers=config.models_ssl.conv_feature_layers,
        quantize_targets=config.models_ssl.quantize_targets,
        mask_prob=config.models_ssl.mask_prob,
    )
    w2v_encoder = NchWav2Vec2Model(w2v_encoder_config)
    w2v_encoder_dict = asdict(w2v_encoder_config)
    w2v_encoder_dict["w2v_path"] = ""
    w2v_encoder_dict["_name"] = "nch_wav2vec2"

    ckpt = {
        "cfg": {
            "model": w2v_encoder_dict,
            "criterion": None,
            "lr_scheduler": None,
            "task": {
                "_name": "audio_finetuning",
                "data": {},
            },
        },
        "model": w2v_encoder.state_dict(),
        "data": {},
        "optimizer": {},
        "best_loss": -1,
        "extra_state": {},
    }
    ckpt_file = params.w2v_model_dir / "pretrained.ckpt"
    torch.save(ckpt, ckpt_file)

    for i_cv in range(params.n_cv):

        w2v_predictor = create_wav2vec_model(config, params.odims, ckpt_file)

        output_file = params.w2v_model_dir / str(i_cv) / "model.pth"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(w2v_predictor.state_dict(), output_file)


def run_cnn(config_file: Path = Path("config/cnn.yaml")) -> None:
    """create dummy CNN models

    Args:
        config_file (Path): CNN config file. Defaults to"config/cnn.yaml".

    """

    params = Params()

    for i_cv in range(params.n_cv):

        config = DictConfig(OmegaConf.load(config_file))

        cnn_predictor = create_predictor(params.odims, config)

        output_file = params.cnn_model_dir / str(i_cv) / "model.pth"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cnn_predictor.state_dict(), output_file)


def main() -> None:
    """train models"""

    model_helper.set_seed()
    run_wav2vec()
    run_cnn()


if __name__ == "__main__":
    main()
