"""KL-based step reward.

For each entity span [t_start, t_end] in a rollout, accumulates the per-token
KL(actor || ref) over that span and emits a single scalar reward:

    KL_per_token[t] = log P_actor(y_t | y_<t)  -  log P_ref(y_t | y_<t)
    agg_k           = aggregate(KL_per_token[t_start : t_end])  (mean / sum / max)
    clip_k          = clip(agg_k, [-C, +C])
    final_k         = sign_factor × tanh(clip_k / scale)

``sign_factor`` depends on ``sign_mode``:
    with_outcome  : +1 if rollout correct, -1 otherwise (penalize wrong-rollout
                    divergence, reward correct-rollout divergence)
    raw           : no sign multiplier
    unsigned      : abs() — exploration bonus

No extra ref forward is needed: ``old_log_probs`` and ``ref_log_prob`` are
already produced by the standard PPO/GRPO pipeline before advantage is computed.

Output contract matches ProgressRewardComputer: returns four tensors GiGPO
consumes (pure_step_reward_tensor, step_rewards, anchor_obs, step_token_positions).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager._step_reward_utils import (
    extract_entity_positions,
    infer_sign_vector,
    sample_entities_uniform,
    scatter_entity_reward,
    select_shared_pool,
)


class KLStepRewardComputer:
    """Compute KL-anchored step rewards for a rollout batch."""

    def __init__(
        self,
        tokenizer,
        graph_reward_manager,
        *,
        mode: str = "shared",                      # all / n / shared
        n: int = 10,
        aggregation: str = "mean",                 # mean / sum / max
        sign_mode: str = "with_outcome",           # with_outcome / raw / unsigned
        scale: float = 1.0,                        # tanh wedge
        per_token_clip: float = 5.0,               # |kl_t| clip before aggregation
        agg_clip: float = 50.0,                    # |agg_k| clip before tanh
        entity_reward_mode: str = "span",          # span / first / last
        shared_pool_size: int = 30,
        min_rollout_freq: int = 2,
    ) -> None:
        if mode not in ("all", "n", "shared"):
            raise ValueError(f"mode must be 'all'/'n'/'shared', got {mode!r}")
        if aggregation not in ("mean", "sum", "max"):
            raise ValueError(f"aggregation must be 'mean'/'sum'/'max', got {aggregation!r}")
        if sign_mode not in ("with_outcome", "raw", "unsigned"):
            raise ValueError(f"sign_mode must be 'with_outcome'/'raw'/'unsigned', got {sign_mode!r}")
        if entity_reward_mode not in ("span", "first", "last"):
            raise ValueError(f"entity_reward_mode must be 'span'/'first'/'last', got {entity_reward_mode!r}")

        self.tokenizer = tokenizer
        self.graph_rm = graph_reward_manager
        self.mode = mode
        self.n = int(n)
        self.aggregation = aggregation
        self.sign_mode = sign_mode
        self.scale = float(scale) if float(scale) > 0 else 1e-6
        self.per_token_clip = float(per_token_clip)
        self.agg_clip = float(agg_clip)
        self.entity_reward_mode = entity_reward_mode
        self.shared_pool_size = max(1, int(shared_pool_size))
        self.min_rollout_freq = max(1, int(min_rollout_freq))

        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    # Public entry point.
    # ------------------------------------------------------------------ #

    def compute(
        self,
        data: DataProto,
        ref_policy_wg=None,                         # unused, kept for API parity
        outcome_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        batch = data.batch
        for required in ("responses", "prompts", "attention_mask",
                         "old_log_probs", "ref_log_prob"):
            if required not in batch:
                raise KeyError(f"KLStepRewardComputer requires '{required}' in data.batch")

        responses = batch["responses"]                  # (B, R)
        prompts = batch["prompts"]                      # (B, P)
        attn = batch["attention_mask"]                  # (B, P+R)
        old_lp = batch["old_log_probs"].to(torch.float32)   # (B, R)
        ref_lp = batch["ref_log_prob"].to(torch.float32)    # (B, R)

        B, R = responses.shape
        P = prompts.shape[-1]
        kl_per_token = (old_lp - ref_lp).detach().cpu()         # (B, R)
        kl_per_token = torch.clamp(kl_per_token, -self.per_token_clip, self.per_token_clip)

        sign_vec = infer_sign_vector(data, B, outcome_scores)

        # ------ Pass 1: extract entities per rollout ------
        raw_extractions: List[Optional[dict]] = []
        valid_count = 0
        no_entity = 0
        entity_total = 0
        entity_sampled_total = 0
        for i in range(B):
            valid_p_len = int(attn[i, :P].sum().item())
            valid_r_len = int(attn[i, P:].sum().item())
            if valid_p_len == 0 or valid_r_len == 0:
                raw_extractions.append(None)
                continue
            valid_response_ids = responses[i, :valid_r_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            flat = extract_entity_positions(self.graph_rm, response_str, valid_response_ids)
            if not flat:
                no_entity += 1
                raw_extractions.append(None)
                continue
            entity_total += len(flat)
            raw_extractions.append({
                "sample_idx": i,
                "valid_resp_len": valid_r_len,
                "flat_entities": flat,
            })

        # ------ Pass 2: select entities per mode ------
        if self.mode == "shared":
            uid_arr = None
            if data.non_tensor_batch is not None:
                uid_arr = data.non_tensor_batch.get("uid", None)
            shared_selected = select_shared_pool(
                uid_arr,
                raw_extractions,
                min_rollout_freq=self.min_rollout_freq,
                shared_pool_size=self.shared_pool_size,
            )
        else:
            shared_selected = None

        # ------ Pass 3: per-rollout aggregation ------
        pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
        per_sample_rewards: List[List[float]] = [[] for _ in range(B)]
        per_sample_anchors: List[List[object]] = [[] for _ in range(B)]
        per_sample_token_pos: List[List[tuple]] = [[] for _ in range(B)]

        agg_values_collected: List[float] = []
        reward_values_collected: List[float] = []
        pool_sizes: List[int] = []
        pool_coverage_rates: List[float] = []

        for i, extract in enumerate(raw_extractions):
            if extract is None:
                continue
            valid_r_len = extract["valid_resp_len"]
            sign_val = float(sign_vec[i])

            if self.mode == "shared":
                selected = shared_selected.get(i, [])
                pool_size = shared_selected.get(("__pool_size__", i), 0)
                if pool_size > 0:
                    pool_sizes.append(pool_size)
                    pool_coverage_rates.append(len(selected) / pool_size)
                if not selected:
                    continue
            elif self.mode == "n":
                selected = sample_entities_uniform(extract["flat_entities"], self.n)
            else:  # "all"
                selected = list(extract["flat_entities"])

            if not selected:
                continue
            entity_sampled_total += len(selected)
            valid_count += 1

            # KL slice for this rollout
            kl_row = kl_per_token[i]                             # (R,)

            for (etype, eid, token_span) in selected:
                t_s = max(0, min(int(token_span[0]), valid_r_len))
                t_e = max(0, min(int(token_span[-1]) + 1, valid_r_len))
                if t_e <= t_s:
                    continue
                seg = kl_row[t_s:t_e]
                if seg.numel() == 0:
                    continue
                if self.aggregation == "mean":
                    agg = float(seg.mean().item())
                elif self.aggregation == "sum":
                    agg = float(seg.sum().item())
                else:  # "max"
                    agg = float(seg.max().item())
                agg = max(-self.agg_clip, min(self.agg_clip, agg))
                agg_values_collected.append(agg)

                if self.sign_mode == "with_outcome":
                    s = sign_val
                elif self.sign_mode == "unsigned":
                    s = 0.0  # sign will be replaced by abs later
                else:  # "raw"
                    s = 1.0

                if self.sign_mode == "unsigned":
                    raw = abs(np.tanh(agg / self.scale))
                    final = float(raw)
                else:
                    raw = np.tanh(agg / self.scale)
                    final = float(s * raw)

                reward_values_collected.append(final)
                scatter_entity_reward(
                    pure_step_reward, i, token_span, final, R, self.entity_reward_mode
                )
                per_sample_rewards[i].append(final)
                per_sample_anchors[i].append((etype, eid))
                per_sample_token_pos[i].append(tuple(int(p) for p in token_span))

        # ------ Pack per-entity tensors (B, max_entities) ------
        max_entities = max((len(r) for r in per_sample_rewards), default=0)
        if max_entities == 0:
            max_entities = 1
        step_rewards_tensor = torch.full((B, max_entities), float("nan"), dtype=torch.float32)
        anchor_obs_arr = np.full((B, max_entities), None, dtype=object)
        step_token_positions_arr = np.full((B, max_entities), None, dtype=object)
        for i in range(B):
            r = per_sample_rewards[i]
            if not r:
                continue
            k = len(r)
            step_rewards_tensor[i, :k] = torch.tensor(r, dtype=torch.float32)
            ai = np.empty(k, dtype=object)
            ai[:] = per_sample_anchors[i]
            anchor_obs_arr[i, :k] = ai
            ti = np.empty(k, dtype=object)
            ti[:] = per_sample_token_pos[i]
            step_token_positions_arr[i, :k] = ti

        # ------ Stats ------
        def _safe(fn, xs, default=0.0):
            if not xs:
                return default
            return float(fn(xs))

        sat_count = sum(1 for v in reward_values_collected if abs(v) > 0.95)
        sat_rate = sat_count / max(1, len(reward_values_collected))
        clip_rate_per_token = float(
            (kl_per_token.abs() == self.per_token_clip).float().mean().item()
        )

        self.last_stats = {
            "kl_step/valid_samples": int(valid_count),
            "kl_step/no_entity_samples": int(no_entity),
            "kl_step/entities_sampled_sum": int(entity_sampled_total),
            "kl_step/agg_mean": _safe(np.mean, agg_values_collected),
            "kl_step/agg_std": _safe(np.std, agg_values_collected),
            "kl_step/agg_min": _safe(np.min, agg_values_collected),
            "kl_step/agg_max": _safe(np.max, agg_values_collected),
            "kl_step/reward_mean": _safe(np.mean, reward_values_collected),
            "kl_step/reward_std": _safe(np.std, reward_values_collected),
            "kl_step/reward_min": _safe(np.min, reward_values_collected),
            "kl_step/reward_max": _safe(np.max, reward_values_collected),
            "kl_step/saturation_rate": float(sat_rate),
            "kl_step/per_token_kl_clip_rate": clip_rate_per_token,
            "kl_step/per_token_kl_mean": float(kl_per_token.mean().item()),
            "kl_step/per_token_kl_std": float(kl_per_token.std().item()),
            "kl_step/shared_pool_size_mean": _safe(np.mean, pool_sizes),
            "kl_step/pool_coverage_rate_mean": _safe(np.mean, pool_coverage_rates),
            "kl_step/method": 1.0,  # marker: kl active
        }

        return {
            "pure_step_reward_tensor": pure_step_reward,
            "step_rewards": step_rewards_tensor,
            "anchor_obs": anchor_obs_arr,
            "step_token_positions": step_token_positions_arr,
        }
