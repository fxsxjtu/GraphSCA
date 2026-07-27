"""Self-consistency step reward.

Within each prompt's K rollouts, for every entity (type, id) in the shared
pool, look at the subset of rollouts that contain this entity and measure how
much their final boxed answers agree. High agreement = entity is a "convergent
reasoning step" worth rewarding (in correct rollouts) or penalising (in wrong
rollouts).

For each entity ``e`` in prompt ``p``:

    S_e            = {rollouts in p that contain entity e}
    answers_e      = [boxed(R_i) for R_i in S_e]
    raw_metric(e)  =
        - majority_share : max(count_a) / |S_e|                   (in [1/|S_e|, 1])
        - entropy_neg    : -H(answers_e) / log|S_e|               (in [-1, 0])
        - dominant_match : 1 if mode(answers_e) == boxed(R_i) else 0  (per-rollout)

Then per rollout R_i ∈ S_e:

    centred = raw_metric(e) - mean_per_prompt(raw_metric)   (if normalize=true)
    final   = sign_factor × centred                          (sign_mode='with_outcome')
                                                              -1 wrong, +1 correct
              centred                                        (sign_mode='raw')

Output contract identical to ProgressRewardComputer / KLStepRewardComputer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager._step_reward_utils import (
    extract_boxed_answer,
    extract_entity_positions,
    infer_sign_vector,
    sample_entities_uniform,
    scatter_entity_reward,
    select_shared_pool,
)


def _entropy_neg(answers: List[str]) -> float:
    """Negative normalized entropy of answer distribution. Range [-1, 0]."""
    n = len(answers)
    if n <= 1:
        return 0.0
    counts = Counter(answers)
    probs = np.array([c / n for c in counts.values()], dtype=np.float64)
    H = -float((probs * np.log(probs + 1e-12)).sum())
    H_max = float(np.log(n))
    if H_max <= 0:
        return 0.0
    return -H / H_max


class ConsistencyStepRewardComputer:
    """Self-consistency anchored on entities (no ref policy needed)."""

    def __init__(
        self,
        tokenizer,
        graph_reward_manager,
        *,
        mode: str = "shared",                     # all / n / shared (shared strongly recommended)
        n: int = 10,
        metric: str = "majority_share",           # majority_share / entropy_neg / dominant_match
        sign_mode: str = "with_outcome",          # with_outcome / raw
        normalize: bool = True,                   # subtract per-prompt mean
        min_group_size: int = 2,                  # min |S_e| to keep an entity's signal
        entity_reward_mode: str = "span",
        shared_pool_size: int = 30,
        min_rollout_freq: int = 2,
    ) -> None:
        if mode not in ("all", "n", "shared"):
            raise ValueError(f"mode must be 'all'/'n'/'shared', got {mode!r}")
        if metric not in ("majority_share", "entropy_neg", "dominant_match"):
            raise ValueError(f"metric must be 'majority_share'/'entropy_neg'/'dominant_match', got {metric!r}")
        if sign_mode not in ("with_outcome", "raw"):
            raise ValueError(f"sign_mode must be 'with_outcome'/'raw', got {sign_mode!r}")
        if entity_reward_mode not in ("span", "first", "last"):
            raise ValueError(f"entity_reward_mode must be 'span'/'first'/'last', got {entity_reward_mode!r}")

        self.tokenizer = tokenizer
        self.graph_rm = graph_reward_manager
        self.mode = mode
        self.n = int(n)
        self.metric = metric
        self.sign_mode = sign_mode
        self.normalize = bool(normalize)
        self.min_group_size = max(1, int(min_group_size))
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
        ref_policy_wg=None,                       # unused
        outcome_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        batch = data.batch
        for required in ("responses", "prompts", "attention_mask"):
            if required not in batch:
                raise KeyError(f"ConsistencyStepRewardComputer requires '{required}'")

        responses = batch["responses"]
        prompts = batch["prompts"]
        attn = batch["attention_mask"]
        B, R = responses.shape
        P = prompts.shape[-1]

        sign_vec = infer_sign_vector(data, B, outcome_scores)

        # ------ Pass 1: extract entities + boxed answer per rollout ------
        raw_extractions: List[Optional[dict]] = []
        boxed_answers: List[Optional[str]] = []
        no_answer = 0
        no_entity = 0
        entity_total = 0

        for i in range(B):
            valid_p_len = int(attn[i, :P].sum().item())
            valid_r_len = int(attn[i, P:].sum().item())
            if valid_p_len == 0 or valid_r_len == 0:
                raw_extractions.append(None)
                boxed_answers.append(None)
                continue
            valid_response_ids = responses[i, :valid_r_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ans = extract_boxed_answer(response_str)
            boxed_answers.append(ans)
            if ans is None:
                no_answer += 1

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

        # ------ Pass 2: select entity pool per mode ------
        uid_arr = None
        if data.non_tensor_batch is not None:
            uid_arr = data.non_tensor_batch.get("uid", None)

        if self.mode == "shared":
            shared_selected = select_shared_pool(
                uid_arr,
                raw_extractions,
                min_rollout_freq=self.min_rollout_freq,
                shared_pool_size=self.shared_pool_size,
            )
        else:
            shared_selected = None

        # Group sample indices by uid (we always need this for cross-rollout consistency).
        groups: dict = defaultdict(list)
        if uid_arr is not None:
            for i in range(B):
                if raw_extractions[i] is None:
                    continue
                try:
                    key = str(uid_arr[i])
                except Exception:
                    key = i
                groups[key].append(i)
        else:
            # No uid → cannot compute cross-rollout consistency; degrade to no-op.
            self.last_stats = {
                "consistency_step/valid_samples": 0,
                "consistency_step/no_uid": 1.0,
            }
            pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
            empty_step = torch.full((B, 1), float("nan"), dtype=torch.float32)
            empty_anchor = np.full((B, 1), None, dtype=object)
            empty_pos = np.full((B, 1), None, dtype=object)
            return {
                "pure_step_reward_tensor": pure_step_reward,
                "step_rewards": empty_step,
                "anchor_obs": empty_anchor,
                "step_token_positions": empty_pos,
            }

        # ------ Pass 3: build per-prompt {entity -> [rollouts that contain it]} ------
        # Per-rollout: entity_id -> first_span (used for scatter)
        per_sample_entities: dict = {}
        for i, extract in enumerate(raw_extractions):
            if extract is None:
                continue
            if self.mode == "shared":
                sel = shared_selected.get(i, [])
            elif self.mode == "n":
                sel = sample_entities_uniform(extract["flat_entities"], self.n)
            else:
                sel = list(extract["flat_entities"])
            seen: dict = {}
            for (etype, eid, span) in sel:
                k = (etype, eid)
                if k not in seen:
                    seen[k] = (span,)  # tuple to keep token_span
            per_sample_entities[i] = seen

        # ------ Pass 4: compute per-(prompt, entity) consistency, then per-rollout reward ------
        pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
        per_sample_rewards: List[List[float]] = [[] for _ in range(B)]
        per_sample_anchors: List[List[object]] = [[] for _ in range(B)]
        per_sample_token_pos: List[List[tuple]] = [[] for _ in range(B)]

        raw_metric_values: List[float] = []
        reward_values: List[float] = []
        per_prompt_unique_answers: List[int] = []
        per_prompt_majority_share: List[float] = []
        entities_per_pool: List[int] = []
        group_sizes: List[int] = []

        for uid_key, members in groups.items():
            # Build entity -> list of sample_idx that contain it.
            ent_to_samples: dict = defaultdict(list)
            for si in members:
                ent_dict = per_sample_entities.get(si, {})
                for ekey in ent_dict.keys():
                    ent_to_samples[ekey].append(si)

            entities_per_pool.append(len(ent_to_samples))

            # Per-prompt answer aggregate (for stats).
            prompt_answers = [boxed_answers[si] for si in members if boxed_answers[si] is not None]
            if prompt_answers:
                cnt = Counter(prompt_answers)
                per_prompt_unique_answers.append(len(cnt))
                per_prompt_majority_share.append(max(cnt.values()) / len(prompt_answers))

            # Per-entity raw metric.
            ent_raw: dict = {}
            for ekey, sample_list in ent_to_samples.items():
                if len(sample_list) < self.min_group_size:
                    continue
                group_sizes.append(len(sample_list))
                ans_list = [boxed_answers[si] for si in sample_list if boxed_answers[si] is not None]
                if not ans_list:
                    continue
                if self.metric == "majority_share":
                    cnt = Counter(ans_list)
                    raw_value_global = max(cnt.values()) / len(ans_list)
                    ent_raw[ekey] = ("global", float(raw_value_global))
                elif self.metric == "entropy_neg":
                    ent_raw[ekey] = ("global", float(_entropy_neg(ans_list)))
                else:  # dominant_match — per-rollout binary value
                    cnt = Counter(ans_list)
                    if not cnt:
                        continue
                    mode_ans = cnt.most_common(1)[0][0]
                    ent_raw[ekey] = ("per_rollout", mode_ans)

            if not ent_raw:
                continue

            # Per-prompt mean for normalization (only over global metrics).
            mean_for_centring = 0.0
            if self.normalize:
                global_vals = [v[1] for v in ent_raw.values() if v[0] == "global"]
                if global_vals:
                    mean_for_centring = float(np.mean(global_vals))

            # Assign per-rollout reward.
            for ekey, (kind, value) in ent_raw.items():
                for si in ent_to_samples[ekey]:
                    span_tuple = per_sample_entities[si].get(ekey)
                    if span_tuple is None:
                        continue
                    span = span_tuple[0]  # token span list
                    if kind == "global":
                        raw = float(value)
                        centred = raw - mean_for_centring if self.normalize else raw
                    else:  # per_rollout (dominant_match)
                        own = boxed_answers[si]
                        raw = 1.0 if own is not None and own == value else 0.0
                        centred = raw - 0.5 if self.normalize else raw  # naive 0.5 baseline

                    raw_metric_values.append(raw)

                    if self.sign_mode == "with_outcome":
                        s = float(sign_vec[si])
                    else:  # "raw"
                        s = 1.0
                    final = float(s * centred)
                    reward_values.append(final)

                    valid_r_len = raw_extractions[si]["valid_resp_len"]
                    scatter_entity_reward(
                        pure_step_reward, si, span, final, R, self.entity_reward_mode
                    )
                    per_sample_rewards[si].append(final)
                    per_sample_anchors[si].append(ekey)
                    per_sample_token_pos[si].append(tuple(int(p) for p in span))

        # ------ Pack per-entity tensors ------
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

        valid_samples = sum(1 for r in per_sample_rewards if r)
        self.last_stats = {
            "consistency_step/valid_samples": int(valid_samples),
            "consistency_step/no_answer_samples": int(no_answer),
            "consistency_step/no_entity_samples": int(no_entity),
            "consistency_step/entities_in_pool_mean": _safe(np.mean, entities_per_pool),
            "consistency_step/group_size_mean": _safe(np.mean, group_sizes),
            "consistency_step/unique_answers_per_prompt_mean": _safe(np.mean, per_prompt_unique_answers),
            "consistency_step/majority_share_per_prompt_mean": _safe(np.mean, per_prompt_majority_share),
            "consistency_step/raw_metric_mean": _safe(np.mean, raw_metric_values),
            "consistency_step/raw_metric_std": _safe(np.std, raw_metric_values),
            "consistency_step/reward_mean": _safe(np.mean, reward_values),
            "consistency_step/reward_std": _safe(np.std, reward_values),
            "consistency_step/reward_min": _safe(np.min, reward_values),
            "consistency_step/reward_max": _safe(np.max, reward_values),
            "consistency_step/method": 2.0,
        }

        return {
            "pure_step_reward_tensor": pure_step_reward,
            "step_rewards": step_rewards_tensor,
            "anchor_obs": anchor_obs_arr,
            "step_token_positions": step_token_positions_arr,
        }
