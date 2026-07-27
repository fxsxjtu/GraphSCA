"""Direction 3 — Invariant-Preservation Progress Reward.

Step-level reward derived from the frozen reference policy's log-prob of the
rollout's own final answer ``a_y`` evaluated at entity boundaries.

For each entity ``s_k`` with token span ``[t_start, t_end]``:

    r(s_k) = sign(r_acc) * Normalize[
        log P_ref(a_y | q, y_{<t_end}) - log P_ref(a_y | q, y_{<t_start})
    ]

Where ``Normalize`` is a FIPO-inspired 5-layer transform:

    L1: per-token log-prob clip (max(log_p, -logp_clip))
    L2: length normalization (divide summed log-prob by |a_y|)
    L3: Delta = avg_end - avg_start
    L4: tanh(Delta / scale)
    L5: asymmetric scale (pos_scale if raw > 0 else neg_scale)
    L6: sign(r_acc) multiplier

The result is written to a ``(B, response_len)`` ``pure_step_reward_tensor``
that plugs directly into the existing GiGPO pipeline.

Ref forward is implemented as a single synthetic DataProto call. Each real
rollout contributes ``n+1`` virtual samples (``n`` entity prefix positions +
``1`` baseline with prefix = prompt only). The virtual batch is left-padded
on the prompt portion and right-padded on the answer portion, matching the
ref worker's standard input contract.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from verl import DataProto


# --------------------------------------------------------------------------- #
# Helpers for locating the rollout's own \boxed{...} answer.
# --------------------------------------------------------------------------- #


def _last_boxed_only_string(string: str) -> Optional[str]:
    """Return the rightmost ``\\boxed{...}`` (including wrapper), or None.

    Kept as a standalone helper so this module does not import torch modules
    via ``verl.utils.reward_score`` at construction time.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left = 0
    while i < len(string):
        ch = string[i]
        if ch == "{":
            num_left += 1
        elif ch == "}":
            num_left -= 1
            if num_left == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    prefix_space = "\\boxed "
    if s.startswith(prefix_space):
        return s[len(prefix_space):]
    prefix_brace = "\\boxed{"
    if s.startswith(prefix_brace) and s.endswith("}"):
        return s[len(prefix_brace):-1]
    return None


def _extract_boxed_answer(response_str: str) -> Optional[str]:
    boxed = _last_boxed_only_string(response_str)
    inner = _remove_boxed(boxed)
    if inner is None:
        return None
    return inner.strip()


def _expand_answer(answer_str: str, task_class: Optional[str], mode: str) -> str:
    """Optional natural-language wrap for short answers.

    mode='raw' returns the extracted string as-is. mode='template' wraps
    per-task_class to improve variance of log P(a_y|q) estimates.
    """
    if mode == "raw" or not answer_str:
        return answer_str
    tc = (task_class or "").lower()
    if tc in ("integer", "int", "float", "number"):
        return f"The final answer is {answer_str}."
    # default for list / set / ordered_node_list / node_list / path / etc.
    return f"The answer is {answer_str}."


def _uniform_sample_indices(n_total: int, n_pick: int) -> List[int]:
    """Uniformly pick ``n_pick`` indices from ``range(n_total)``, always
    including 0 and n_total-1 when n_pick >= 2."""
    if n_total <= n_pick:
        return list(range(n_total))
    if n_pick == 1:
        return [n_total - 1]
    step = (n_total - 1) / (n_pick - 1)
    picked = sorted({int(round(i * step)) for i in range(n_pick)})
    # fill up to n_pick if rounding collapsed some
    i = 0
    while len(picked) < n_pick and i < n_total:
        if i not in picked:
            picked.append(i)
        i += 1
    return sorted(picked)[:n_pick]


# --------------------------------------------------------------------------- #
# Per-sample info assembled before the ref forward.
# --------------------------------------------------------------------------- #


@dataclass
class _SampleInfo:
    """Per-rollout bookkeeping needed to construct virtual samples and scatter
    rewards back to the (B, response_len) tensor."""

    sample_idx: int
    valid_prompt_ids: torch.Tensor    # 1D int tensor, actual prompt tokens (no padding)
    valid_response_ids: torch.Tensor  # 1D int tensor, actual response tokens (no padding)
    answer_ids: torch.Tensor          # 1D int tensor, expanded answer tokens (no padding)
    # sampled entities: list of (entity_type, entity_id, token_span_in_response)
    entities: List[Tuple[str, str, List[int]]]
    # prefix positions in the response we need log P at: [0, t_1, t_2, ..., t_n]
    # t_0 = 0 is the baseline (prefix = prompt only)
    prefix_positions: List[int]
    # for each entity, the (start_t, end_t) prefix positions used for Delta
    entity_to_prefixes: List[Tuple[int, int]]  # (prefix_pos_start, prefix_pos_end)


# --------------------------------------------------------------------------- #
# Main class.
# --------------------------------------------------------------------------- #


class ProgressRewardComputer:
    """Compute Direction 3 Progress step rewards for a batch.

    Usage (from ray_trainer after rollout + outcome reward, before advantage):
        computer = ProgressRewardComputer(tokenizer, graph_rm, mode='n', n=10, ...)
        pure_step_reward = computer.compute(batch, ref_policy_wg)
        batch.batch['pure_step_reward_tensor'] = pure_step_reward
    """

    def __init__(
        self,
        tokenizer,
        graph_reward_manager,
        *,
        mode: str = "shared",
        n: int = 10,
        expansion: str = "raw",
        logp_clip: float = 20.0,
        scale: float = 0.3,
        pos_scale: float = 1.0,
        neg_scale: float = 1.0,
        length_normalize: bool = True,
        entity_reward_mode: str = "span",
        anchor_mode: str = "span",
        max_combined_length: int = 10000,
        max_virtual_batch_per_call: int = 4096,
        shared_pool_size: int = 30,
        min_rollout_freq: int = 2,
    ) -> None:
        if mode not in ("all", "n", "shared"):
            raise ValueError(f"mode must be 'all' / 'n' / 'shared', got {mode!r}")
        if expansion not in ("raw", "template"):
            raise ValueError(f"expansion must be 'raw' or 'template', got {expansion!r}")
        if entity_reward_mode not in ("span", "first", "last"):
            raise ValueError(f"entity_reward_mode must be 'span'/'first'/'last', got {entity_reward_mode!r}")
        if anchor_mode not in ("span", "cumulative", "telescoping"):
            raise ValueError(
                f"anchor_mode must be 'span' / 'cumulative' / 'telescoping', got {anchor_mode!r}"
            )

        self.tokenizer = tokenizer
        self.graph_rm = graph_reward_manager
        self.mode = mode
        self.n = int(n)
        self.expansion = expansion
        self.logp_clip = float(logp_clip)
        self.scale = float(scale) if float(scale) > 0 else 1e-6
        self.pos_scale = float(pos_scale)
        self.neg_scale = float(neg_scale)
        self.length_normalize = bool(length_normalize)
        self.entity_reward_mode = entity_reward_mode
        self.anchor_mode = anchor_mode
        self.max_combined_length = int(max_combined_length)
        self.max_virtual_batch_per_call = max(1, int(max_virtual_batch_per_call))
        self.shared_pool_size = max(1, int(shared_pool_size))
        self.min_rollout_freq = max(1, int(min_rollout_freq))

        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

        # Tracked for training metric logging.
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    # Public entry point.
    # ------------------------------------------------------------------ #

    def compute(
        self,
        data: DataProto,
        ref_policy_wg,
        outcome_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        """Return a dict holding the four tensors GiGPO consumes.

        Keys:
            pure_step_reward_tensor: float (B, response_len); zero at non-entity positions.
            step_rewards: float (B, max_entities) with NaN padding; per-entity Progress.
            anchor_obs: numpy object array (B, max_entities); entity IDs.
            step_token_positions: numpy object array (B, max_entities); token positions.

        Args:
            data: rollout batch. Required keys: prompts, responses, attention_mask.
                Optional: reward_model.ground_truth / extra_info.task_class for template expansion.
            ref_policy_wg: ray worker group whose ``compute_ref_log_prob`` accepts
                a DataProto with input_ids / attention_mask / position_ids / responses.
            outcome_scores: optional ``(B,)`` tensor of outcome rewards. If None,
                we derive sign from ``token_level_scores`` / ``token_level_rewards`` in data.
        """
        batch = data.batch
        if "responses" not in batch:
            raise KeyError("ProgressRewardComputer expects 'responses' in data.batch")

        response_tensor = batch["responses"]            # (B, R)
        prompt_tensor = batch["prompts"]                # (B, P)
        attention_mask = batch["attention_mask"]        # (B, P+R)

        B, R = response_tensor.shape
        P = prompt_tensor.shape[-1]

        sign_vec = self._infer_sign_vector(data, B, outcome_scores)

        # -------- Pass 1a: per-sample raw extraction (CPU) --------
        # Collect per-sample entity occurrences (full span list, pre-selection) plus the
        # answer tokens. Selection logic (mode=all/n/shared) runs in Pass 1b.
        raw_extractions: List[Optional[dict]] = []
        valid_count = 0
        no_answer_count = 0
        no_entity_count = 0
        entity_total = 0
        entity_sampled_total = 0
        for i in range(B):
            valid_prompt_len = int(attention_mask[i, :P].sum().item())
            valid_resp_len = int(attention_mask[i, P:].sum().item())
            if valid_prompt_len == 0 or valid_resp_len == 0:
                raw_extractions.append(None)
                continue
            valid_prompt_ids = prompt_tensor[i, -valid_prompt_len:]
            valid_response_ids = response_tensor[i, :valid_resp_len]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_str = _extract_boxed_answer(response_str)
            if not answer_str:
                no_answer_count += 1
                raw_extractions.append(None)
                continue

            task_class = self._infer_task_class(data, i)
            expanded = _expand_answer(answer_str, task_class, self.expansion)
            answer_ids_list = self.tokenizer(expanded, add_special_tokens=False).get("input_ids", [])
            if len(answer_ids_list) == 0:
                no_answer_count += 1
                raw_extractions.append(None)
                continue
            answer_ids = torch.tensor(answer_ids_list, dtype=torch.long)

            flat_entities = self._extract_entity_positions(response_str, valid_response_ids)
            if len(flat_entities) == 0:
                no_entity_count += 1
                raw_extractions.append(None)
                continue
            entity_total += len(flat_entities)

            raw_extractions.append({
                "sample_idx": i,
                "valid_prompt_ids": valid_prompt_ids,
                "valid_response_ids": valid_response_ids,
                "valid_resp_len": valid_resp_len,
                "answer_ids": answer_ids,
                "flat_entities": flat_entities,  # List[(type, id, sorted_token_span)]
            })

        # -------- Pass 1b: per-mode entity selection --------
        # mode=all  : keep every span
        # mode=n    : uniform sample n spans per rollout
        # mode=shared: per-prompt (uid) shared pool filtered by min_rollout_freq, capped
        #              at shared_pool_size; each rollout keeps only entities in the pool,
        #              using its FIRST occurrence span.
        if self.mode == "shared":
            shared_selected = self._select_shared(data, raw_extractions)
        else:
            shared_selected = None  # per-sample uniform / all

        sample_infos: List[Optional[_SampleInfo]] = []
        group_size_hist: dict = defaultdict(int)  # for stats (via shared-pool only)
        pool_sizes: List[int] = []
        pool_coverage_rates: List[float] = []

        for i, extract in enumerate(raw_extractions):
            if extract is None:
                sample_infos.append(None)
                continue

            if self.mode == "shared":
                selected_entities = shared_selected.get(i, [])
                # Track coverage: how many of this prompt's shared_pool this rollout hit.
                pool_for_prompt = shared_selected.get(("__pool_size__", i), 0)
                if pool_for_prompt > 0:
                    pool_sizes.append(pool_for_prompt)
                    pool_coverage_rates.append(len(selected_entities) / pool_for_prompt)
                if not selected_entities:
                    sample_infos.append(None)
                    continue
            else:
                flat_entities = extract["flat_entities"]
                selected_entities = self._sample_entities(flat_entities)

            entity_sampled_total += len(selected_entities)
            valid_resp_len = extract["valid_resp_len"]

            # anchor_mode controls where (t_start, t_end) for each entity's Delta come from:
            #   span        : t_start = entity_span[0],          t_end = entity_span[-1]+1
            #   cumulative  : t_start = 0,                       t_end = entity_span[-1]+1
            #   telescoping : t_start = previous entity's t_end, t_end = entity_span[-1]+1
            #                 (first entity uses t_start = 0)
            #
            # cumulative/telescoping require entities sorted by token position; sort once
            # here so the order is consistent regardless of how upstream selection ordered them.
            if self.anchor_mode in ("cumulative", "telescoping"):
                selected_entities = sorted(
                    selected_entities, key=lambda e: int(e[2][-1])
                )

            prefix_set = {0}
            per_entity_prefixes: List[Tuple[int, int]] = []
            prev_end = 0
            for (_, _, token_span) in selected_entities:
                span_start = int(token_span[0])
                span_end = int(token_span[-1]) + 1
                span_start = max(0, min(span_start, valid_resp_len))
                span_end = max(0, min(span_end, valid_resp_len))

                if self.anchor_mode == "cumulative":
                    t_start, t_end = 0, span_end
                elif self.anchor_mode == "telescoping":
                    t_start, t_end = prev_end, span_end
                    if t_end < t_start:  # safety: out-of-order entity, fall back to span
                        t_start, t_end = span_start, span_end
                    prev_end = t_end
                else:  # "span" (legacy)
                    t_start, t_end = span_start, span_end

                prefix_set.add(t_start)
                prefix_set.add(t_end)
                per_entity_prefixes.append((t_start, t_end))
            prefix_positions = sorted(prefix_set)

            sample_infos.append(
                _SampleInfo(
                    sample_idx=extract["sample_idx"],
                    valid_prompt_ids=extract["valid_prompt_ids"],
                    valid_response_ids=extract["valid_response_ids"],
                    answer_ids=extract["answer_ids"],
                    entities=selected_entities,
                    prefix_positions=prefix_positions,
                    entity_to_prefixes=per_entity_prefixes,
                )
            )
            valid_count += 1

        # -------- Pass 2: build virtual batch --------
        virtual_samples: List[Tuple[int, int]] = []  # (sample_idx, prefix_pos_idx within sample)
        virtual_prefix_ids: List[torch.Tensor] = []  # each = prompt_ids + y[:t_k]
        virtual_answer_ids: List[torch.Tensor] = []

        for info in sample_infos:
            if info is None:
                continue
            for pp_idx, t_k in enumerate(info.prefix_positions):
                prefix = torch.cat(
                    [info.valid_prompt_ids, info.valid_response_ids[: int(t_k)]], dim=0
                )
                answer = info.answer_ids
                combined_len = prefix.shape[0] + answer.shape[0]
                if combined_len > self.max_combined_length:
                    # Truncate the prefix (keep prompt, drop leading response tokens).
                    # This is a safety shield — ideally never triggered.
                    drop = combined_len - self.max_combined_length
                    prompt_len = info.valid_prompt_ids.shape[0]
                    # Drop from the response-prefix portion, not the prompt.
                    new_prefix_resp_len = max(0, int(t_k) - drop)
                    prefix = torch.cat(
                        [info.valid_prompt_ids, info.valid_response_ids[:new_prefix_resp_len]],
                        dim=0,
                    )
                virtual_prefix_ids.append(prefix)
                virtual_answer_ids.append(answer)
                virtual_samples.append((info.sample_idx, pp_idx))

        # Sort virtual samples by (prefix_len + answer_len) to minimize per-chunk padding
        # waste. The final result lookup is keyed on (sample_idx, prefix_pos_idx), which is
        # preserved by any permutation of these three parallel lists.
        if len(virtual_samples) > 1:
            combined_lengths = [p.shape[0] + a.shape[0] for p, a in zip(virtual_prefix_ids, virtual_answer_ids)]
            order = sorted(range(len(virtual_samples)), key=lambda i: combined_lengths[i])
            virtual_samples = [virtual_samples[i] for i in order]
            virtual_prefix_ids = [virtual_prefix_ids[i] for i in order]
            virtual_answer_ids = [virtual_answer_ids[i] for i in order]

        pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
        progress_values: List[float] = []
        reward_values: List[float] = []

        if len(virtual_samples) == 0:
            self.last_stats = self._build_stats(
                valid_count=valid_count,
                no_answer_count=no_answer_count,
                no_entity_count=no_entity_count,
                entity_total=entity_total,
                entity_sampled_total=entity_sampled_total,
                virtual_count=0,
                progress_values=[],
                reward_values=[],
                pool_sizes=pool_sizes,
                pool_coverage_rates=pool_coverage_rates,
                mode=self.mode,
            )
            step_rewards_tensor = torch.full((B, 1), float("nan"), dtype=torch.float32)
            anchor_obs_arr = np.full((B, 1), None, dtype=object)
            step_token_positions_arr = np.full((B, 1), None, dtype=object)
            return {
                "pure_step_reward_tensor": pure_step_reward,
                "step_rewards": step_rewards_tensor,
                "anchor_obs": anchor_obs_arr,
                "step_token_positions": step_token_positions_arr,
            }

        # -------- Pass 3: chunked ref log_prob forward --------
        # Two constraints we respect per chunk:
        #   (a) len(virtual_dp) must be divisible by ref_policy_wg.world_size
        #       (DataProto.chunk raises otherwise).
        #   (b) total virtual samples can be huge (esp. mode=all), so we split into
        #       chunks of at most max_virtual_batch_per_call to keep activation memory
        #       bounded and allow progress during long forwards.
        try:
            dp_size = int(getattr(ref_policy_wg, "world_size", 1)) or 1
        except Exception:
            dp_size = 1

        per_prefix_values: List[float] = []
        total_v = len(virtual_samples)
        # Chunk size aligned down to dp_size so each chunk is safely divisible.
        chunk_size = max(dp_size, (self.max_virtual_batch_per_call // dp_size) * dp_size)

        for chunk_start in range(0, total_v, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_v)
            prefix_chunk = virtual_prefix_ids[chunk_start:chunk_end]
            answer_chunk = virtual_answer_ids[chunk_start:chunk_end]

            # Pad the chunk to a multiple of dp_size with "nothing" rows (attention_mask=0).
            pad_count = (-len(prefix_chunk)) % dp_size
            if pad_count > 0:
                pad_prefix = prefix_chunk[-1]
                pad_answer = answer_chunk[-1]
                prefix_chunk = prefix_chunk + [pad_prefix] * pad_count
                answer_chunk = answer_chunk + [pad_answer] * pad_count

            virtual_dp = self._assemble_virtual_dataproto(
                prefix_chunk, answer_chunk, data, pad_suffix=pad_count
            )

            ref_out = ref_policy_wg.compute_ref_log_prob(virtual_dp)
            ref_log_probs = ref_out.batch["ref_log_prob"].float()  # (V_chunk, A_pad)
            virt_resp_mask = virtual_dp.batch["attention_mask"][:, -ref_log_probs.shape[-1]:].float()
            virt_resp_mask = virt_resp_mask.to(ref_log_probs.device)

            # -------- Pass 4 (per chunk): clipped sum + length-normalize --------
            clipped = ref_log_probs.clamp(min=-self.logp_clip, max=0.0)
            sum_log_p = (clipped * virt_resp_mask).sum(dim=-1)
            answer_lengths = virt_resp_mask.sum(dim=-1).clamp(min=1.0)
            if self.length_normalize:
                chunk_vals = sum_log_p / answer_lengths
            else:
                chunk_vals = sum_log_p
            # Drop the padded suffix rows.
            real_count = chunk_end - chunk_start
            chunk_vals_np = chunk_vals.detach().cpu().numpy()[:real_count]
            per_prefix_values.extend(chunk_vals_np.tolist())

        per_prefix_value = np.asarray(per_prefix_values, dtype=np.float64)

        # Group virtual-sample outputs by (sample_idx, prefix_pos_idx).
        prefix_lookup: dict = {}
        for (samp_idx, pp_idx), val in zip(virtual_samples, per_prefix_value):
            prefix_lookup[(samp_idx, pp_idx)] = float(val)

        # -------- Pass 5: per-entity reward + scatter --------
        # Accumulate per-sample entity-level Progress (for GiGPO's (B, max_entities) tensors).
        per_sample_rewards: List[List[float]] = [[] for _ in range(B)]
        per_sample_anchor_ids: List[List[object]] = [[] for _ in range(B)]
        per_sample_token_positions: List[List[tuple]] = [[] for _ in range(B)]

        for info in sample_infos:
            if info is None:
                continue
            pp_index_of: dict = {t: k for k, t in enumerate(info.prefix_positions)}
            sign_val = float(sign_vec[info.sample_idx])
            for (entity_type, entity_id, token_span), (t_start, t_end) in zip(
                info.entities, info.entity_to_prefixes
            ):
                k_start = pp_index_of[t_start]
                k_end = pp_index_of[t_end]
                val_start = prefix_lookup.get((info.sample_idx, k_start), 0.0)
                val_end = prefix_lookup.get((info.sample_idx, k_end), 0.0)
                delta = val_end - val_start
                progress_values.append(delta)

                raw = np.tanh(delta / self.scale)
                if raw > 0:
                    raw_scaled = raw * self.pos_scale
                else:
                    raw_scaled = raw * self.neg_scale
                final = sign_val * float(raw_scaled)
                reward_values.append(final)

                self._scatter_entity_reward(
                    pure_step_reward,
                    info.sample_idx,
                    token_span,
                    final,
                    R,
                )
                per_sample_rewards[info.sample_idx].append(final)
                # anchor_obs must be hashable + group-comparable across rollouts of the same
                # prompt. Use the same (type, id) convention as the graph reward manager.
                per_sample_anchor_ids[info.sample_idx].append((entity_type, entity_id))
                per_sample_token_positions[info.sample_idx].append(tuple(int(p) for p in token_span))

        # -------- Pass 6: pack per-entity tensors to (B, max_entities) --------
        max_entities = max((len(r) for r in per_sample_rewards), default=0)
        if max_entities == 0:
            max_entities = 1
        step_rewards_tensor = torch.full((B, max_entities), float("nan"), dtype=torch.float32)
        anchor_obs_arr = np.full((B, max_entities), None, dtype=object)
        step_token_positions_arr = np.full((B, max_entities), None, dtype=object)
        for i in range(B):
            rewards_i = per_sample_rewards[i]
            if not rewards_i:
                continue
            k = len(rewards_i)
            step_rewards_tensor[i, :k] = torch.tensor(rewards_i, dtype=torch.float32)
            anchor_i = np.empty(k, dtype=object)
            anchor_i[:] = per_sample_anchor_ids[i]
            anchor_obs_arr[i, :k] = anchor_i
            tokpos_i = np.empty(k, dtype=object)
            tokpos_i[:] = per_sample_token_positions[i]
            step_token_positions_arr[i, :k] = tokpos_i

        # -------- stats --------
        self.last_stats = self._build_stats(
            valid_count=valid_count,
            no_answer_count=no_answer_count,
            no_entity_count=no_entity_count,
            entity_total=entity_total,
            entity_sampled_total=entity_sampled_total,
            virtual_count=len(virtual_samples),
            progress_values=progress_values,
            reward_values=reward_values,
            pool_sizes=pool_sizes,
            pool_coverage_rates=pool_coverage_rates,
            mode=self.mode,
        )

        return {
            "pure_step_reward_tensor": pure_step_reward,
            "step_rewards": step_rewards_tensor,
            "anchor_obs": anchor_obs_arr,
            "step_token_positions": step_token_positions_arr,
        }

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #

    def _infer_sign_vector(
        self,
        data: DataProto,
        B: int,
        outcome_scores: Optional[torch.Tensor],
    ) -> np.ndarray:
        """Derive per-sample sign = +1 if outcome positive else -1."""
        if outcome_scores is not None:
            vals = outcome_scores.detach().to("cpu").numpy().astype(float)
        else:
            for key in ("token_level_scores", "token_level_rewards"):
                if key in data.batch:
                    token_scores = data.batch[key]
                    vals = token_scores.sum(dim=-1).detach().to("cpu").numpy().astype(float)
                    break
            else:
                warnings.warn(
                    "ProgressRewardComputer: no outcome scores available; defaulting sign=+1 for all samples."
                )
                return np.ones(B, dtype=float)
        return np.where(vals > 0, 1.0, -1.0)

    def _infer_task_class(self, data: DataProto, sample_idx: int) -> Optional[str]:
        if self.expansion != "template":
            return None
        non_tensor = data.non_tensor_batch
        extra = non_tensor.get("extra_info", None) if non_tensor is not None else None
        if extra is None:
            return None
        try:
            item = extra[sample_idx]
        except (IndexError, TypeError, KeyError):
            return None
        if isinstance(item, dict):
            return item.get("task_class") or item.get("task") or item.get("answer_type")
        return None

    def _extract_entity_positions(
        self, response_str: str, valid_response_ids: torch.Tensor
    ):
        entities = self.graph_rm.extract_entities(response_str)
        entity_positions = self.graph_rm.find_entity_token_positions(
            response_str, valid_response_ids, entities
        )
        flat: List[Tuple[str, str, List[int]]] = []
        for entity_id, span_list in entity_positions.get("node_positions", {}).items():
            for token_span in span_list:
                token_span = [int(x) for x in token_span]
                if token_span:
                    flat.append(("node", entity_id, sorted(token_span)))
        for entity_id, span_list in entity_positions.get("edge_positions", {}).items():
            for token_span in span_list:
                token_span = [int(x) for x in token_span]
                if token_span:
                    flat.append(("edge", entity_id, sorted(token_span)))
        flat.sort(key=lambda e: e[2][0])
        return flat

    def _sample_entities(self, flat_entities):
        if self.mode == "all" or len(flat_entities) <= self.n:
            return flat_entities
        picks = _uniform_sample_indices(len(flat_entities), self.n)
        return [flat_entities[i] for i in picks]

    def _select_shared(self, data: DataProto, raw_extractions):
        """Build per-prompt shared entity pool and return a dict mapping
        sample_idx -> list of (entity_type, entity_id, first_span).

        Extra entries under key ``("__pool_size__", sample_idx)`` hold the pool size
        assigned to that sample's prompt, used downstream for coverage stats.
        """
        uid_arr = data.non_tensor_batch.get("uid", None) if data.non_tensor_batch is not None else None
        if uid_arr is None:
            # No uid → cannot group; fall back to mode=n on each sample
            return {
                extract["sample_idx"]: self._sample_entities(extract["flat_entities"])
                for extract in raw_extractions
                if extract is not None
            }

        # Group sample_idx by uid.
        groups: dict = defaultdict(list)
        for i, extract in enumerate(raw_extractions):
            if extract is None:
                continue
            key = uid_arr[i]
            try:
                key = str(key)
            except Exception:
                pass
            groups[key].append(i)

        selected: dict = {}
        for uid_key, member_idxs in groups.items():
            # Build per-sample unique entity sets (dedup by (type, id), first-span).
            per_sample_first_span: List[dict] = []
            for si in member_idxs:
                extract = raw_extractions[si]
                seen: dict = {}
                for (etype, eid, span) in extract["flat_entities"]:
                    ekey = (etype, eid)
                    if ekey not in seen:
                        seen[ekey] = span  # first occurrence (flat_entities is sorted by first-token)
                per_sample_first_span.append(seen)

            # Frequency across rollouts + earliest first-token across rollouts (tie-break).
            freq: dict = defaultdict(int)
            earliest: dict = {}
            for seen in per_sample_first_span:
                for ekey, span in seen.items():
                    freq[ekey] += 1
                    t0 = int(span[0]) if span else 0
                    if ekey not in earliest or t0 < earliest[ekey]:
                        earliest[ekey] = t0

            qualified = [(ekey, freq[ekey], earliest[ekey]) for ekey in freq if freq[ekey] >= self.min_rollout_freq]
            qualified.sort(key=lambda x: (-x[1], x[2]))  # higher freq first, earlier position second
            pool = [ekey for (ekey, _, _) in qualified[: self.shared_pool_size]]
            pool_set = set(pool)
            pool_size = len(pool)

            for si, seen in zip(member_idxs, per_sample_first_span):
                picked = []
                for ekey in pool:
                    if ekey in seen:
                        picked.append((ekey[0], ekey[1], seen[ekey]))
                selected[si] = picked
                selected[("__pool_size__", si)] = pool_size

        return selected

    def _scatter_entity_reward(
        self,
        tensor: torch.Tensor,
        sample_idx: int,
        token_span: Sequence[int],
        reward: float,
        response_len: int,
    ) -> None:
        positions = [int(p) for p in token_span if 0 <= int(p) < response_len]
        if not positions:
            return
        if self.entity_reward_mode == "first":
            targets = [positions[0]]
        elif self.entity_reward_mode == "last":
            targets = [positions[-1]]
        else:
            targets = positions
        for pos in targets:
            tensor[sample_idx, pos] = reward

    def _assemble_virtual_dataproto(
        self,
        prefixes: List[torch.Tensor],
        answers: List[torch.Tensor],
        ref_data: DataProto,
        pad_suffix: int = 0,
    ) -> DataProto:
        """Build a left/right-padded DataProto matching the ref worker's input contract.

        Prompt portion = prefixes[i]; response portion = answers[i]. Prompts are
        left-padded to max_prefix_len; responses are right-padded to max_answer_len.

        ``pad_suffix`` indicates the final N rows are dummy (attention_mask=0 everywhere,
        added solely so len(DataProto) is divisible by dp_size).
        """
        V = len(prefixes)
        assert V == len(answers), f"prefix/answer count mismatch: {V} vs {len(answers)}"

        max_prefix_len = max(x.shape[0] for x in prefixes)
        max_answer_len = max(x.shape[0] for x in answers)

        input_ids = torch.full(
            (V, max_prefix_len + max_answer_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        responses = torch.full(
            (V, max_answer_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )

        real_count = V - max(0, pad_suffix)
        for i, (p, a) in enumerate(zip(prefixes, answers)):
            pl = p.shape[0]
            al = a.shape[0]
            # left-pad the prompt portion: real tokens go to the RIGHT end of the prompt slot
            input_ids[i, max_prefix_len - pl : max_prefix_len] = p
            # right-pad the answer portion
            input_ids[i, max_prefix_len : max_prefix_len + al] = a
            if i < real_count:
                attention_mask[i, max_prefix_len - pl : max_prefix_len] = 1
                attention_mask[i, max_prefix_len : max_prefix_len + al] = 1
            responses[i, :al] = a

        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)

        virtual_tensor_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        }

        virtual_dp = DataProto.from_single_dict(virtual_tensor_dict)
        # Seed meta info the ref worker expects (micro_batch_size etc. are set by the
        # worker from its own config; temperature default mirrors the existing path).
        virtual_dp.meta_info.setdefault("temperature", 1.0)
        return virtual_dp

    def _build_stats(
        self,
        *,
        valid_count: int,
        no_answer_count: int,
        no_entity_count: int,
        entity_total: int,
        entity_sampled_total: int,
        virtual_count: int,
        progress_values: List[float],
        reward_values: List[float],
        pool_sizes: Optional[List[int]] = None,
        pool_coverage_rates: Optional[List[float]] = None,
        mode: str = "n",
    ) -> dict:
        def _safe(fn, xs, default=0.0):
            if not xs:
                return default
            return float(fn(xs))

        stats = {
            "direction3/valid_samples": int(valid_count),
            "direction3/no_answer_samples": int(no_answer_count),
            "direction3/no_entity_samples": int(no_entity_count),
            "direction3/entities_detected_sum": int(entity_total),
            "direction3/entities_sampled_sum": int(entity_sampled_total),
            "direction3/virtual_samples": int(virtual_count),
            "direction3/progress_mean": _safe(np.mean, progress_values),
            "direction3/progress_std": _safe(np.std, progress_values),
            "direction3/progress_min": _safe(np.min, progress_values),
            "direction3/progress_max": _safe(np.max, progress_values),
            "direction3/reward_mean": _safe(np.mean, reward_values),
            "direction3/reward_std": _safe(np.std, reward_values),
            "direction3/reward_min": _safe(np.min, reward_values),
            "direction3/reward_max": _safe(np.max, reward_values),
        }
        if mode == "shared":
            stats["direction3/shared_pool_size_mean"] = _safe(np.mean, pool_sizes or [])
            stats["direction3/shared_pool_size_min"] = _safe(np.min, pool_sizes or [])
            stats["direction3/shared_pool_size_max"] = _safe(np.max, pool_sizes or [])
            stats["direction3/pool_coverage_rate_mean"] = _safe(np.mean, pool_coverage_rates or [])
            stats["direction3/pool_coverage_rate_min"] = _safe(np.min, pool_coverage_rates or [])
        return stats


__all__ = ["ProgressRewardComputer"]
