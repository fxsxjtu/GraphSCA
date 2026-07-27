"""
Hazard-rate prediction reward function for verl RL training.

Loads checkpoints produced by:
  training_reward_model/train_hazard_prediction.py

Supports three backbone types:
  - mlp:         Pair-wise. Input = [emb_{t-1}; emb_t] (2H,) -> scalar H.
  - rnn:         Sequence.  Input = (B, T, H) -> (B, T) H at each position.
  - transformer: Sequence.  Input = (B, T, H) -> (B, T) H at each position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Model definitions (mirrored from train_hazard_prediction.py)
# ---------------------------------------------------------------------------

class PairMLP(nn.Module):
    """MLP for pair-wise hazard prediction.
    Input:  (B, 2H) — concatenation of [emb_{t-1}, emb_t]
    Output: (B,)    — predicted H
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SequenceHazardModel(nn.Module):
    """Sequence-level hazard prediction (RNN or Transformer).
    Input:  x (B, T, H), lengths (B,), key_padding_mask (B, T)
    Output: (B, T) — predicted H at each position
    """

    def __init__(
        self,
        input_dim: int,
        backbone: str = "rnn",
        rnn_hidden_dim: int = 512,
        rnn_layers: int = 2,
        rnn_bidirectional: bool = True,
        rnn_dropout: float = 0.1,
        tfm_d_model: int = 256,
        tfm_heads: int = 4,
        tfm_layers: int = 2,
        tfm_ffn_dim: int = 1024,
        tfm_dropout: float = 0.1,
        reg_hidden_dims: Optional[List[int]] = None,
        reg_dropout: float = 0.2,
    ):
        super().__init__()
        if reg_hidden_dims is None:
            reg_hidden_dims = [256, 128]
        self.backbone_name = backbone

        if backbone == "rnn":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout if rnn_layers > 1 else 0.0,
            )
            enc_out_dim = rnn_hidden_dim * (2 if rnn_bidirectional else 1)
        elif backbone == "transformer":
            self.input_proj = nn.Linear(input_dim, tfm_d_model)
            self.pos_embed = nn.Embedding(8192, tfm_d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=tfm_d_model,
                nhead=tfm_heads,
                dim_feedforward=tfm_ffn_dim,
                dropout=tfm_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=tfm_layers)
            enc_out_dim = tfm_d_model
        else:
            raise ValueError(f"SequenceHazardModel does not support backbone={backbone!r}")

        layers: List[nn.Module] = []
        in_dim = enc_out_dim
        for h in reg_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(reg_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x, lengths, key_padding_mask=None):
        if self.backbone_name == "rnn":
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.encoder(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        elif self.backbone_name == "transformer":
            B, T, _ = x.shape
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            h = self.input_proj(x) + self.pos_embed(pos_ids)
            out = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.regressor(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Reward function wrapper
# ---------------------------------------------------------------------------

class HazardRewardFunction:
    """
    Wraps a trained hazard-rate prediction model for use as activation reward
    in verl RL training.

    Supports all three backbone types:
      - mlp: pair-wise, expects single entity embeddings (auto-constructs pairs)
      - rnn / transformer: sequence-level, can process full sequences or single tokens

    Args:
        model_path: Path to final_model.pth checkpoint
        device: Device to load model on
        normalize_reward: Whether to clamp rewards into [-1, 1] before scaling
        reward_scale: Multiplicative scale for final reward
        pair_mode: How to construct pairs for MLP backbone
                   'prev_concat' = [emb_{t-1}; emb_t]
                   'self_concat' = [emb_t; emb_t]
                   'zero_prev'   = [zeros; emb_t]
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        pair_mode: str = "prev_concat",
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.pair_mode = pair_mode

        self.model = None
        self.backbone = None
        self.input_dim = None
        self.args = {}
        self._loaded = False
        self._metadata_loaded = False

        # Legacy checkpoint stats kept for compatibility/debugging.
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._stats_loaded = False

    def _load_checkpoint_metadata(self):
        if self._metadata_loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.backbone = ckpt["backbone"]
        self.input_dim = int(ckpt["input_dim"])
        if not self.args:
            self.args = ckpt.get("args", {}) or {}
        self._metadata_loaded = True

    def _load_model(self):
        if self._loaded:
            return

        if not self._metadata_loaded:
            self._load_checkpoint_metadata()

        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if self.backbone == "mlp":
            hidden_dims_str = self.args.get("mlp_hidden_dims", "2048,1024,512")
            hidden_dims = [int(x) for x in hidden_dims_str.split(",")]
            dropout = float(self.args.get("mlp_dropout", 0.1))
            self.model = PairMLP(self.input_dim, hidden_dims, dropout)
        else:
            reg_hidden_str = self.args.get("reg_hidden_dims", "256,128")
            reg_hidden = [int(x) for x in reg_hidden_str.split(",")]
            self.model = SequenceHazardModel(
                input_dim=self.input_dim,
                backbone=self.backbone,
                rnn_hidden_dim=int(self.args.get("rnn_hidden_dim", 512)),
                rnn_layers=int(self.args.get("rnn_layers", 2)),
                rnn_bidirectional=bool(self.args.get("rnn_bidirectional", 1)),
                rnn_dropout=float(self.args.get("rnn_dropout", 0.1)),
                tfm_d_model=int(self.args.get("tfm_d_model", 256)),
                tfm_heads=int(self.args.get("tfm_heads", 4)),
                tfm_layers=int(self.args.get("tfm_layers", 2)),
                tfm_ffn_dim=int(self.args.get("tfm_ffn_dim", 1024)),
                tfm_dropout=float(self.args.get("tfm_dropout", 0.1)),
                reg_hidden_dims=reg_hidden,
                reg_dropout=float(self.args.get("reg_dropout", 0.2)),
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Try to extract normalization stats from test_results
        test_results = ckpt.get("test_results", {})
        if isinstance(test_results, dict):
            test_metrics = test_results.get("test_metrics", {})
            if "mae" in test_metrics:
                # Use MAE as a rough std estimate for normalization
                self.reward_std = max(float(test_metrics["mae"]) * 2.0, 0.01)
                self._stats_loaded = True

        self._loaded = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[HazardReward] Loaded {self.backbone} model from {self.model_path}")
        print(f"  input_dim={self.input_dim}, params={n_params:,}, device={self.device}")

    def _make_pair_input(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Construct pair input [emb_{t-1}; emb_t] for MLP backbone.
        embeddings: (N, H) single-token embeddings
        Returns: (N, 2H) pair features
        """
        mode = (self.pair_mode or "prev_concat").lower()
        if mode in ("prev", "prev_concat"):
            prev = torch.cat([embeddings[:1], embeddings[:-1]], dim=0)
        elif mode in ("self", "self_concat"):
            prev = embeddings
        elif mode in ("zero", "zero_prev"):
            prev = torch.zeros_like(embeddings)
        else:
            raise ValueError(f"Unknown pair_mode: {self.pair_mode}")
        return torch.cat([prev, embeddings], dim=-1)

    def _aggregate_span_rewards(
        self, rewards: torch.Tensor, span_indices: Optional[list[list[int]]] = None
    ) -> torch.Tensor:
        if span_indices is None:
            return rewards
        if len(span_indices) == 0:
            return torch.empty(0, dtype=rewards.dtype, device=rewards.device)

        aggregated = []
        reward_length = int(rewards.shape[0])
        for span in span_indices:
            if not span:
                aggregated.append(torch.zeros((), dtype=rewards.dtype, device=rewards.device))
                continue
            valid_span = [int(idx) for idx in span if 0 <= int(idx) < reward_length]
            if not valid_span:
                aggregated.append(torch.zeros((), dtype=rewards.dtype, device=rewards.device))
                continue
            idx_tensor = torch.as_tensor(valid_span, dtype=torch.long, device=rewards.device)
            aggregated.append(rewards.index_select(0, idx_tensor).mean())
        return torch.stack(aggregated)

    @torch.no_grad()
    def __call__(self, entity_embedding: torch.Tensor) -> torch.Tensor:
        """Compute reward for a single entity embedding (H,)."""
        if not self._loaded:
            self._load_model()

        if entity_embedding.device != torch.device(self.device):
            entity_embedding = entity_embedding.to(self.device)

        x = entity_embedding.unsqueeze(0) if entity_embedding.dim() == 1 else entity_embedding
        x = x.to(next(self.model.parameters()).dtype)

        if self.backbone == "mlp":
            x = self._make_pair_input(x)  # (1, 2H)
            reward = self.model(x)  # (1,)
        else:
            lengths = torch.tensor([x.shape[0]], device=self.device)
            x = x.unsqueeze(0)  # (1, 1, H)
            reward = self.model(x, lengths)  # (1, 1)
            reward = reward[0, -1]  # take last position

        reward = reward.squeeze()
        reward = self._normalize(reward)
        return reward

    @torch.no_grad()
    def compute_batch(
        self, entity_embeddings: torch.Tensor, span_indices: Optional[list[list[int]]] = None
    ) -> torch.Tensor:
        """Compute rewards for a batch of entity embeddings (N, H).
        Returns: (N,) reward values.
        """
        if not self._loaded:
            self._load_model()

        if entity_embeddings.dim() == 1:
            entity_embeddings = entity_embeddings.unsqueeze(0)
        if entity_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32, device=entity_embeddings.device)

        if entity_embeddings.device != torch.device(self.device):
            entity_embeddings = entity_embeddings.to(self.device)

        x = entity_embeddings.to(next(self.model.parameters()).dtype)

        if self.backbone == "mlp":
            x = self._make_pair_input(x)  # (N, 2H)
            rewards = self.model(x)  # (N,)
        else:
            # For sequence models, treat each embedding as a length-1 sequence
            N = x.shape[0]
            lengths = torch.ones(N, dtype=torch.long, device=self.device)
            x_seq = x.unsqueeze(1)  # (N, 1, H)
            rewards = self.model(x_seq, lengths)  # (N, 1)
            rewards = rewards.squeeze(-1)  # (N,)

        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)

        rewards = self._normalize(rewards)
        return self._aggregate_span_rewards(rewards, span_indices=span_indices)

    @torch.no_grad()
    def compute_sequence(
        self,
        sequence_embeddings: torch.Tensor,
        valid_length: int = None,
        span_indices: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        """Run the full response embedding sequence through the model and return per-position hazard rates.

        For RNN/Transformer backbones this preserves sequential context (unlike compute_batch
        which treats each entity as an independent length-1 sequence).
        For MLP backbone this falls back to compute_batch (pair-wise, no sequence context needed).

        Args:
            sequence_embeddings: (T, H) complete response embedding sequence for one sample.
            valid_length: number of valid (non-pad) positions. Defaults to T.

        Returns:
            (valid_length,) hazard rate predictions at each position.
        """
        if not self._loaded:
            self._load_model()

        T = sequence_embeddings.shape[0]
        if valid_length is None:
            valid_length = T

        if sequence_embeddings.device != torch.device(self.device):
            sequence_embeddings = sequence_embeddings.to(self.device)
        x = sequence_embeddings.to(next(self.model.parameters()).dtype)

        if self.backbone == "mlp":
            # MLP is pair-wise; just delegate to compute_batch
            rewards = self.compute_batch(x[:valid_length], span_indices=span_indices)
            return rewards  # (valid_length,)

        # RNN / Transformer: run the full sequence as a single batch element
        x = x.unsqueeze(0)  # (1, T, H)
        lengths = torch.tensor([valid_length], dtype=torch.long, device=self.device)
        rewards = self.model(x, lengths)  # (1, T)
        rewards = rewards[0, :valid_length]  # (valid_length,)
        rewards = self._normalize(rewards)
        return self._aggregate_span_rewards(rewards, span_indices=span_indices)

    def _normalize(self, reward: torch.Tensor) -> torch.Tensor:
        if self.normalize_reward:
            reward = reward.clamp(-1.0, 1.0)
        return reward * self.reward_scale

    def get_input_dim(self) -> int:
        if self.input_dim is None:
            self._load_checkpoint_metadata()
        if self.backbone == "mlp":
            return self.input_dim // 2  # MLP input_dim is 2*H
        return self.input_dim

    def get_backbone_name(self) -> str:
        if self.backbone is None:
            self._load_checkpoint_metadata()
        return self.backbone

    def get_model_info(self) -> Dict[str, Any]:
        if not self._loaded:
            self._load_model()
        return {
            "model_path": str(self.model_path),
            "backbone": self.backbone,
            "input_dim": self.input_dim,
            "pair_mode": self.pair_mode,
            "device": self.device,
            "normalize_reward": self.normalize_reward,
            "reward_scale": self.reward_scale,
        }


def create_hazard_reward_fn(
    model_path: str,
    device: str = "cuda",
    normalize_reward: bool = True,
    reward_scale: float = 1.0,
    pair_mode: str = "prev_concat",
) -> HazardRewardFunction:
    """Factory function to create a hazard reward function.

    Args:
        model_path: Path to final_model.pth (e.g. results/hazard_mlp/final_model.pth)
        device: Device to run on
        normalize_reward: Whether to apply sigmoid normalization
        reward_scale: Scale factor for rewards
        pair_mode: Pair construction mode for MLP backbone

    Returns:
        HazardRewardFunction instance
    """
    return HazardRewardFunction(
        model_path=model_path,
        device=device,
        normalize_reward=normalize_reward,
        reward_scale=reward_scale,
        pair_mode=pair_mode,
    )
