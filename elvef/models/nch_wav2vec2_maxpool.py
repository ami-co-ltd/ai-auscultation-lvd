# pylint: disable=invalid-name,too-many-instance-attributes,too-many-function-args,missing-function-docstring,unused-argument,missing-class-docstring

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    LAYER_TYPE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
)
from fairseq.modules import GradMultiply
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, open_dict

from .attention_decoder import MultiHeadAttentionPooling
from .mixup_mlp import MixupMlp, MSDMlp
from .nch_wav2vec2 import NchWav2Vec2Model

logger = logging.getLogger(__name__)


@dataclass
class NchWav2Vec2MaxPoolConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    drop_path: float = 0
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    update_alibi: bool = True
    data: str = II("task.data")

    w2v_args: Any = None
    offload_activations: bool = field(
        default=False, metadata={"help": "offload_activations"}
    )
    min_params_to_wrap: int = field(
        default=int(1e8),
        metadata={
            "help": "minimum number of params for a layer to be wrapped with FSDP() when "
            "training with --ddp-backend=fully_sharded. Smaller values will "
            "improve memory efficiency, but may make torch.distributed "
            "communication less efficient due to smaller input sizes. This option "
            "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
            "--offload-activations are passed."
        },
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    ddp_backend: str = II("distributed_training.ddp_backend")

    zero_mask: bool = False
    load_ema: bool = False

    layer_decay: float = 1

    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )
    # Adapter num
    adp_num: int = field(default=-1)
    adp_dim: int = field(default=64)
    adp_act_fn: str = field(default="relu")
    adp_trf_idx: str = field(
        default="all",
    )

    freeze_regex: Optional[str] = field(
        default=None,
    )

    mixup_alpha: float = field(default=1.0)
    mixup_layer: int = field(default=-1)
    mlp_layers: int = field(default=4)
    mlp_head_dropout: float = field(default=0.0)
    mlp_head_pooling: str = field(default="maxpool")
    mlp_grad_scale: float = field(default=1.0)
    use_msd: bool = field(default=False)
    msd_n_repeat: int = field(default=1)
    msd_dropout: float = field(default=0.3)

    output_size: int = 9
    tgt_layer: Optional[int] = field(default=None)
    n_freezed_attn_layers: int = field(default=-1)

    can_input_position_ids: bool = field(default=False)
    aux_feat_file: Optional[str] = field(default=None)

    multi_task_heads: Any = None
    # Supervised SSL or Unsupervised SSL
    ssl_type: str = field(
        default="unsupervised",
        metadata={"help": "unsupervised or supervised"},
    )
    disable_last_layer_bias: bool = field(default=False)


@register_model("nch_wav2vec2_audio", dataclass=NchWav2Vec2MaxPoolConfig)
class NchWav2Vec2Audio(BaseFairseqModel):
    """

    Attributes:
        cfg (NchWav2Vec2MaxPoolConfig): config
        w2v_model (NchWav2Vec2Model): wav2vec encoder
        freeze_finetune_updates (int): warmup stage. freeze pretrained weights.
        num_updates (int): counts #iterations
        tgt_layer (int): use bottom transformer layer
        mlp_grad_scale (float): control gradients for MLP
        pool_fn (nn.Module): pooling layer
        proj (nn.Module): classification head
        n_freezed_attn_layers (int): used to freeze bottom attention layers
        can_input_position_ids (bool): input position id if True.

    """

    def __init__(self, cfg: NchWav2Vec2MaxPoolConfig) -> None:
        """init

        Args:
            cfg (NchWav2Vec2MaxPoolConfig): config

        """
        super().__init__()
        self.cfg = cfg

        self.w2v_model, d = self.create_unsupervised_ssl_w2v_encoder(cfg)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
        self.tgt_layer = cfg.tgt_layer

        self.mlp_grad_scale = cfg.mlp_grad_scale

        if cfg.mlp_head_pooling == "attention":
            self.pool_fn = MultiHeadAttentionPooling(1, d)  # type: ignore
        elif cfg.mlp_head_pooling == "maxpool":
            self.pool_fn = torch.nn.AdaptiveMaxPool1d(1)  # type: ignore
        else:
            self.pool_fn = torch.nn.AdaptiveAvgPool1d(1)  # type: ignore

        # MLP after max-pool
        if cfg.use_msd:
            self.proj = MSDMlp.build(
                input_dim=d,
                output_dim=cfg.output_size,
                n_layers=cfg.mlp_layers,
                mixup_layer_num=cfg.mixup_layer,
                alpha=cfg.mixup_alpha,
                dropout_rate=cfg.msd_dropout,
                n_repeat=cfg.msd_n_repeat,
                disable_last_layer_bias=cfg.disable_last_layer_bias,
            )
        else:
            self.proj = MixupMlp.build(
                input_dim=d,
                output_dim=cfg.output_size,
                n_layers=cfg.mlp_layers,
                mixup_layer_num=cfg.mixup_layer,
                alpha=cfg.mixup_alpha,
                dropout_rate=0.0,  # cfg.mlp_head_dropout,
                hidden_dim=d,
                disable_last_layer_bias=cfg.disable_last_layer_bias,
            )
        self.n_freezed_attn_layers = -1  # cfg.n_freezed_attn_layers  # -1
        self.can_input_position_ids = False  # cfg.can_input_position_ids  # False
        logger.info(f"{self.can_input_position_ids=}")

    @staticmethod
    def create_pooling_layer(layer_name: str, feat_dim: int = 1) -> torch.nn.Module:
        """create pooling layer"""
        if layer_name == "attention":
            return MultiHeadAttentionPooling(1, feat_dim)  # type: ignore
        if layer_name == "maxpool":
            return torch.nn.AdaptiveMaxPool1d(1)  # type: ignore
        if layer_name == "avgpool":
            return torch.nn.AdaptiveAvgPool1d(1)  # type: ignore
        raise NotImplementedError

    @staticmethod
    def create_unsupervised_ssl_w2v_encoder(
        cfg: NchWav2Vec2MaxPoolConfig,
    ) -> Tuple[NchWav2Vec2Model, int]:
        """create w2v encoder"""
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "require_same_masks": getattr(cfg, "require_same_masks", True),
            "pct_holes": getattr(cfg, "mask_dropout", 0),
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_before": cfg.mask_channel_before,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "checkpoint_activations": cfg.checkpoint_activations,
            "offload_activations": cfg.offload_activations,
            "min_params_to_wrap": cfg.min_params_to_wrap,
        }

        state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])

        print(f"{w2v_args=}")
        print(f"{w2v_args.model=}")
        print(f"{cfg.w2v_path=}")

        w2v_args.criterion = None
        w2v_args.lr_scheduler = None

        with open_dict(w2v_args):
            args_replacement = [
                "checkpoint_activations",
                "layer_type",
                "adp_num",
                "adp_dim",
                "adp_act_fn",
                "adp_trf_idx",
            ]
            for _args in args_replacement:
                if hasattr(cfg, _args) and getattr(cfg, _args, None) is not None:
                    w2v_args.model[_args] = getattr(cfg, _args, None)

            w2v_args.model["w2v_path"] = cfg.w2v_path

        # --
        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task, from_checkpoint=True)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()
        d = w2v_args.model.encoder_embed_dim

        if state is not None and not cfg.no_pretrained_weights:
            if cfg.load_ema:
                assert "_ema" in state["model"]
                for k in state["model"]["_ema"]:
                    mk = "encoder." + k
                    assert mk in state["model"], mk
                    state["model"][mk] = state["model"]["_ema"][k]
            NchWav2Vec2Audio.load_model_weights(state, model, cfg)

        return model, d

    @staticmethod
    def load_model_weights(state, model, cfg, strict=False):
        to_delete = {"_ema", "target_proj", "decoder"}
        for k in to_delete:
            if k in state["model"]:
                del state["model"][k]

        model.load_state_dict(state["model"], strict=strict)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: NchWav2Vec2MaxPoolConfig, task: FairseqTask):
        """Build a new model instance."""
        return cls(cfg)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = self.get_logits(net_output)
        return torch.nn.functional.sigmoid(logits)

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def extract_feats(self, source) -> torch.Tensor:
        x = self.w2v_model.extract_features(
            source=source,
            padding_mask=None,
            mask=None,
            layer=self.tgt_layer,
        )["x"]

        # B x T x C -> B x C
        x = self.pool_fn(x.transpose(1, 2))
        x = torch.squeeze(x, 2)
        return x

    def forward(self, source, **kwargs):
        """forward path

        Args:
            source (torch.Tensor): input data
            position_ids (Optional[torch.Tensor]): position ids (batch,).
            loss_weights (torch.Tensor):
            labels (torch.Tensor):

        """
        position_ids = kwargs["position_ids"] if self.can_input_position_ids else None

        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x = self.w2v_model.extract_features(
                source=source,
                padding_mask=None,
                mask=None,
                layer=self.tgt_layer,
                n_freezed_attn_layers=self.n_freezed_attn_layers,
                position_ids=position_ids,
            )["x"]

        # B x T x C -> B x C
        x = self.pool_fn(x.transpose(1, 2))
        x = torch.squeeze(x, 2)

        # MLP
        # sample_names = kwargs.get("sample_names", None)
        loss_weights = kwargs.get("loss_weights", None)
        labels = kwargs.get("labels", None)

        x = GradMultiply.apply(x, 1.0 / self.mlp_grad_scale)
        x, labels, loss_weights = self.proj(x, labels, loss_weights)
        x = GradMultiply.apply(x, self.mlp_grad_scale)

        return {
            "encoder_out": x,  # B x O
            "padding_mask": None,
            "layer_results": None,
            "labels": labels,
            "loss_weights": loss_weights,
        }
