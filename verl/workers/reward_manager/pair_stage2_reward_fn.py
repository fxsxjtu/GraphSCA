"""
Pair-based Stage2 reward model loader for MI-Peaks checkpoints.

Supports checkpoints produced by:
  training_reward_model/train_reward_model_in_memory.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class FlexiblePairEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        backbone: str = "mlp",
        mlp_hidden_dim: int = 1024,
        mlp_dropout: float = 0.1,
        rnn_hidden_dim: int = 512,
        rnn_layers: int = 2,
        rnn_bidirectional: bool = True,
        rnn_dropout: float = 0.1,
        tfm_d_model: int = 512,
        tfm_heads: int = 8,
        tfm_layers: int = 2,
        tfm_ffn_dim: int = 2048,
        tfm_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_dim = int(input_dim)
        self.proj_dim = int(proj_dim)

        if self.input_dim % 2 != 0:
            raise ValueError(f"Expected even input_dim for pair feature split, got {self.input_dim}")
        self.step_hidden_dim = self.input_dim // 2

        if backbone == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_dim // 2, self.proj_dim),
            )
        elif backbone == "rnn":
            self.rnn = nn.LSTM(
                input_size=self.step_hidden_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout if rnn_layers > 1 else 0.0,
            )
            rnn_out_dim = rnn_hidden_dim * (2 if rnn_bidirectional else 1)
            self.rnn_head = nn.Sequential(
                nn.LayerNorm(rnn_out_dim),
                nn.Linear(rnn_out_dim, self.proj_dim),
            )
        elif backbone == "transformer":
            self.tfm_in = nn.Linear(self.step_hidden_dim, tfm_d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, 2, tfm_d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=tfm_d_model,
                nhead=tfm_heads,
                dim_feedforward=tfm_ffn_dim,
                dropout=tfm_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.tfm = nn.TransformerEncoder(encoder_layer, num_layers=tfm_layers)
            self.tfm_head = nn.Sequential(
                nn.LayerNorm(tfm_d_model),
                nn.Linear(tfm_d_model, self.proj_dim),
            )
        else:
            raise ValueError(f"Unknown encoder backbone: {backbone}")

    def _to_pair_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.view(x.size(0), 2, self.step_hidden_dim)
        if x.dim() == 3:
            return x
        raise ValueError(f"Unsupported x shape for encoder: {tuple(x.shape)}")

    def forward(self, x, normalize: bool = True):
        if self.backbone == "mlp":
            z = self.net(x)
        elif self.backbone == "rnn":
            seq = self._to_pair_sequence(x)
            out, _ = self.rnn(seq)
            z = self.rnn_head(out[:, -1, :])
        else:
            seq = self._to_pair_sequence(x)
            h = self.tfm_in(seq) + self.pos_embed
            h = self.tfm(h)
            z = self.tfm_head(h.mean(dim=1))

        if normalize:
            return nn.functional.normalize(z, p=2, dim=1)
        return z


class HRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(512, 256), dropout: float = 0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PairStage2ActivationRewardFunction:
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        pair_mode: str = "prev_concat",
        stats_path: Optional[str] = None,
        auto_load_stats: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.pair_mode = pair_mode
        self.stats_path = stats_path
        self.auto_load_stats = auto_load_stats

        self.encoder = None
        self.regressor = None
        self._model_loaded = False

        self.input_dim = None
        self.proj_dim = None
        self.args = {}
        self.meta = {}

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._stats_loaded = False

    def _parse_hidden_dims(self, s: Any):
        if isinstance(s, (list, tuple)):
            return tuple(int(x) for x in s)
        if isinstance(s, str):
            out = [int(x.strip()) for x in s.split(",") if x.strip()]
            return tuple(out) if out else (512, 256)
        return (512, 256)

    def _maybe_load_stats(self):
        stats = None

        # Optional external stats file takes precedence.
        if self.stats_path:
            p = Path(self.stats_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    if "reward_mean" in cfg and "reward_std" in cfg:
                        stats = (float(cfg["reward_mean"]), float(cfg["reward_std"]))
                    elif "mean" in cfg and "std" in cfg:
                        stats = (float(cfg["mean"]), float(cfg["std"]))

        if stats is not None:
            self.reward_mean, self.reward_std = stats
            self._stats_loaded = True
            print(f"Loaded reward normalization stats: mean={self.reward_mean:.6f}, std={self.reward_std:.6f}")

    def _load_model(self):
        if self._model_loaded:
            return

        final_model = self.model_dir / "final_model.pth"
        if not final_model.exists():
            raise FileNotFoundError(f"Missing checkpoint: {final_model}")

        ckpt = torch.load(final_model, map_location=self.device, weights_only=False)
        required = {"encoder_state_dict", "regressor_state_dict", "input_dim", "proj_dim", "args"}
        missing = required - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint missing keys: {missing}")

        self.input_dim = int(ckpt["input_dim"])
        self.proj_dim = int(ckpt["proj_dim"])
        self.args = ckpt.get("args", {}) or {}
        self.meta = ckpt.get("test_results", {}) or {}

        backbone = self.args.get("encoder_backbone", "mlp")
        hidden_dim = int(self.args.get("encoder_hidden_dim", 1024))
        encoder_dropout = float(self.args.get("encoder_dropout", 0.1))

        self.encoder = FlexiblePairEncoder(
            input_dim=self.input_dim,
            proj_dim=self.proj_dim,
            backbone=backbone,
            mlp_hidden_dim=hidden_dim,
            mlp_dropout=encoder_dropout,
            rnn_hidden_dim=int(self.args.get("rnn_hidden_dim", 512)),
            rnn_layers=int(self.args.get("rnn_layers", 2)),
            rnn_bidirectional=bool(self.args.get("rnn_bidirectional", True)),
            rnn_dropout=float(self.args.get("rnn_dropout", 0.1)),
            tfm_d_model=int(self.args.get("tfm_d_model", 512)),
            tfm_heads=int(self.args.get("tfm_heads", 8)),
            tfm_layers=int(self.args.get("tfm_layers", 2)),
            tfm_ffn_dim=int(self.args.get("tfm_ffn_dim", 2048)),
            tfm_dropout=float(self.args.get("tfm_dropout", 0.1)),
        ).to(self.device)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.encoder.eval()

        regressor = HRegressor(
            input_dim=self.proj_dim,
            hidden_dims=self._parse_hidden_dims(self.args.get("regressor_hidden_dims", "512,256")),
            dropout=float(self.args.get("regressor_dropout", 0.2)),
        ).to(self.device)
        regressor.load_state_dict(ckpt["regressor_state_dict"])
        regressor.eval()
        self.regressor = regressor

        if self.normalize_reward and self.auto_load_stats:
            self._maybe_load_stats()
        if self.normalize_reward and not self._stats_loaded:
            print("⚠️  No reward stats found; normalization is skipped for pair-stage2 model.")

        self._model_loaded = True
        print(f"✓ Loaded pair-stage2 reward model from: {self.model_dir}")
        print(f"  backbone={backbone}, input_dim={self.input_dim}, proj_dim={self.proj_dim}, pair_mode={self.pair_mode}")

    def _to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor [N,H], got {tuple(x.shape)}")

        in_dim = x.shape[-1]
        if in_dim == self.input_dim:
            return x

        if self.input_dim == in_dim * 2:
            mode = (self.pair_mode or "prev_concat").lower()
            if mode in ("self", "self_concat"):
                left = x
            elif mode in ("prev", "prev_concat"):
                left = torch.cat([x[:1], x[:-1]], dim=0)
            elif mode in ("zero", "zero_prev"):
                left = torch.zeros_like(x)
            else:
                raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")
            return torch.cat([left, x], dim=-1)

        raise ValueError(
            f"Input dim mismatch: model expects {self.input_dim}, got {in_dim}. "
            f"Expected either exact match or 2x for pair conversion."
        )

    @torch.no_grad()
    def __call__(self, entity_embedding: torch.Tensor) -> torch.Tensor:
        if not self._model_loaded:
            self._load_model()

        if entity_embedding.device != torch.device(self.device):
            entity_embedding = entity_embedding.to(self.device)

        x = self._to_model_input(entity_embedding)
        enc_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != enc_dtype:
            x = x.to(enc_dtype)

        z = self.encoder(x, normalize=True)
        reward = self.regressor(z).squeeze()

        if self.normalize_reward and self._stats_loaded:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            reward = torch.sigmoid(reward)

        reward = reward * self.reward_scale
        return reward

    @torch.no_grad()
    def compute_batch(self, entity_embeddings: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        if not self._model_loaded:
            self._load_model()

        if entity_embeddings.dim() == 1:
            entity_embeddings = entity_embeddings.unsqueeze(0)
        if entity_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32, device=entity_embeddings.device)

        if entity_embeddings.device != torch.device(self.device):
            entity_embeddings = entity_embeddings.to(self.device)

        x = self._to_model_input(entity_embeddings)
        enc_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != enc_dtype:
            x = x.to(enc_dtype)

        z = self.encoder(x, normalize=True)
        rewards = self.regressor(z)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)

        if self.normalize_reward and self._stats_loaded:
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
            rewards = torch.sigmoid(rewards)

        rewards = rewards * self.reward_scale
        return rewards

    def get_input_dim(self) -> int:
        if not self._model_loaded:
            self._load_model()
        return int(self.input_dim)

    def get_model_info(self) -> Dict[str, Any]:
        if not self._model_loaded:
            self._load_model()
        return {
            "model_dir": str(self.model_dir),
            "model_type": "pair_stage2",
            "encoder_backbone": self.args.get("encoder_backbone", "unknown"),
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "pair_mode": self.pair_mode,
            "test_results": self.meta,
            "device": self.device,
            "normalize_reward": self.normalize_reward,
            "reward_scale": self.reward_scale,
        }


def create_pair_stage2_activation_reward_fn(
    model_dir: str,
    device: str = "cuda",
    normalize_reward: bool = True,
    reward_scale: float = 1.0,
    pair_mode: str = "prev_concat",
    stats_path: Optional[str] = None,
    auto_load_stats: bool = True,
):
    return PairStage2ActivationRewardFunction(
        model_dir=model_dir,
        device=device,
        normalize_reward=normalize_reward,
        reward_scale=reward_scale,
        pair_mode=pair_mode,
        stats_path=stats_path,
        auto_load_stats=auto_load_stats,
    )
