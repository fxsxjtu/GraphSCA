"""Entity-info-only step reward computer.

Companion to the `entity_anchored_grouped_kl` policy loss. This "computer" does
NOT compute step rewards — the reward signal is computed inside the policy loss
during PPO update (because it depends on `log π_θ - log π_old` which is only
non-zero after the actor has been updated for at least one inner iteration).

Instead, this computer extracts entity anchor information and packages it as
batch tensors that the policy loss can consume:

    anchor_positions:        (B, max_K) int — token position of each anchor
    anchor_group_ids:        (B, max_K) int — globally unique group ID per anchor
    segment_id_per_token:    (B, R)     int — which anchor each token belongs to (-1 = past last anchor)
    num_anchors:             (B,)       int — actual anchor count per rollout

Group ID design: combine (prompt_uid, entity_type, entity_id) into a single int
through a per-batch dictionary. This lets the policy loss do GiGPO-style group
normalization via scatter operations (rollouts that share the same entity in
the same prompt land in the same group).

The computer also returns zero step rewards (so the GiGPO additive-step path is
neutralized — the entity-anchored signal flows through the loss function
instead).

Performance optimizations (applied 2026-05-05):
  - Batch GPU→CPU sync: one transfer for all valid_lens instead of B.
  - tokenizer.batch_decode: one Rust-native batched call instead of B.
  - ProcessPoolExecutor over CPU cores for entity extraction. (NOTE: switched from
    ThreadPoolExecutor on 2026-05-05 — the slow path is `_build_token_position_map`
    which uses pure-Python `difflib.SequenceMatcher` and holds the GIL, making
    threads useless. Process pool gives true parallelism.)
"""
from __future__ import annotations

# Set BEFORE any tokenizer / transformers / huggingface_hub imports run, to
# silence the "huggingface/tokenizers: process forked" warning AND avoid the
# Rust tokenizer's internal threadpool deadlocking when we fork() workers.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import multiprocessing as _mp
import time as _time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager._step_reward_utils import (
    extract_entity_positions,
    extract_entity_positions_fast,
    sample_entities_uniform,
    select_shared_pool,
)


# Tunable: workers for parallel entity extraction. Each worker does CPU-bound
# regex/tokenize work. Detection priority:
#   1. ENTITY_EXTRACT_WORKERS env var (explicit override)
#   2. len(os.sched_getaffinity(0)) — actual CPUs visible to the process
#      (preferred over os.cpu_count() because Ray actors often have a cgroup
#      that makes os.cpu_count() report 1 even when 9 cores are usable).
#   3. os.cpu_count() — fallback for non-Linux
#   4. 8 — last-resort default
def _detect_worker_count() -> int:
    env = os.environ.get("ENTITY_EXTRACT_WORKERS")
    if env:
        try:
            v = int(env)
            if v > 0:
                return min(32, v)
        except ValueError:
            pass
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity > 0:
            return min(32, affinity)
    except (AttributeError, OSError):
        pass
    return min(32, os.cpu_count() or 8)


_THREAD_POOL_WORKERS = _detect_worker_count()


# Module-global context shared with fork()'d worker processes.
# Populated each call before pool.map; workers read via _extract_worker().
# Using a module-level dict + fork mp_context lets workers access these
# without pickle (large objects like graph_rm + tokenizer are expensive to
# serialize per-task).
_WORKER_CTX: dict = {}


def _extract_worker(args):
    """Module-level worker for ProcessPool. MUST be top-level (not closure)
    so it can be discovered after fork or pickle.

    Reads input from _WORKER_CTX (which was populated in the parent before
    pool.map and is inherited via fork). Uses ``extract_entity_positions_fast``
    which expects pre-computed retokenized_ids + offset_mapping (batch-encoded
    in the parent for efficiency)."""
    local_idx, sample_idx = args
    try:
        graph_rm = _WORKER_CTX["graph_rm"]
        text = _WORKER_CTX["response_strs"][local_idx]
        response_ids = _WORKER_CTX["response_ids_list"][local_idx]
        extract_cap = _WORKER_CTX["extract_cap"]
        retok_ids = _WORKER_CTX["retok_ids_list"][local_idx]
        offset_map = _WORKER_CTX["offset_maps"][local_idx]
        flat = extract_entity_positions_fast(
            graph_rm, text, response_ids,
            retokenized_ids=retok_ids,
            offset_mapping=offset_map,
            max_entities=extract_cap,
            first_occurrence_only=True,
        )
    except Exception:
        flat = []
    return local_idx, sample_idx, flat


class EntityInfoOnlyComputer:
    """Extracts entity anchor info and packs it as tensors for downstream policy loss."""

    def __init__(
        self,
        tokenizer,
        graph_reward_manager,
        *,
        mode: str = "all",                        # all / n / shared
        n: int = 10,
        shared_pool_size: int = 30,
        min_rollout_freq: int = 2,
        max_anchors_per_rollout: int = 64,         # cap to control memory
    ) -> None:
        if mode not in ("all", "n", "shared"):
            raise ValueError(f"mode must be 'all'/'n'/'shared', got {mode!r}")

        self.tokenizer = tokenizer
        self.graph_rm = graph_reward_manager
        self.mode = mode
        self.n = int(n)
        self.shared_pool_size = max(1, int(shared_pool_size))
        self.min_rollout_freq = max(1, int(min_rollout_freq))
        self.max_anchors_per_rollout = max(1, int(max_anchors_per_rollout))

        self.last_stats: dict = {}

    def compute(
        self,
        data: DataProto,
        ref_policy_wg=None,                        # unused
        outcome_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        batch = data.batch
        for required in ("responses", "prompts", "attention_mask"):
            if required not in batch:
                raise KeyError(f"EntityInfoOnlyComputer requires '{required}'")

        responses = batch["responses"]
        prompts = batch["prompts"]
        attn = batch["attention_mask"]
        B, R = responses.shape
        P = prompts.shape[-1]
        device = responses.device

        # ---- Pass 1 (optimized): batched decode + parallel entity extraction ----
        # Optimization 2: ONE GPU→CPU sync for all valid lengths instead of B.
        valid_p_lens = attn[:, :P].sum(dim=1).cpu().tolist()         # (B,)
        valid_r_lens = attn[:, P:].sum(dim=1).cpu().tolist()         # (B,)

        # Build the list of valid (sample_idx, response_ids) for batch decode.
        # Pull responses to CPU once.
        responses_cpu = responses.cpu()
        valid_indices: List[int] = []
        valid_response_id_lists: List[torch.Tensor] = []
        for i in range(B):
            if valid_p_lens[i] == 0 or valid_r_lens[i] == 0:
                continue
            valid_indices.append(i)
            valid_response_id_lists.append(responses_cpu[i, : valid_r_lens[i]])

        # Optimization 1: batch_decode in a single Rust-native call (Hugging Face
        # tokenizers are multi-threaded internally). Replaces B separate decode calls.
        t_decode_start = _time.time()
        if valid_indices:
            response_strs = self.tokenizer.batch_decode(
                valid_response_id_lists, skip_special_tokens=True
            )
        else:
            response_strs = []
        t_decode = _time.time() - t_decode_start

        # Optimization 4 (NEW): batch-encode response_strs ONCE in parent process
        # to get retokenized_ids + offset_mapping for every rollout. The Rust
        # tokenizer parallelizes internally across all sequences. Workers then
        # re-use these pre-computed encodings instead of calling self.tokenizer()
        # per rollout — saves ~5ms × B sequential Rust calls. Per-call -> single
        # batched call.
        t_encode_start = _time.time()
        retok_ids_list: List[List[int]] = []
        offset_maps: List[List[tuple]] = []
        if response_strs:
            batch_enc = self.tokenizer(
                response_strs,
                return_offsets_mapping=True,
                add_special_tokens=False,
                padding=False,
            )
            retok_ids_list = [list(map(int, ids)) for ids in batch_enc["input_ids"]]
            offset_maps = [list(om) for om in batch_enc["offset_mapping"]]
        t_encode = _time.time() - t_encode_start

        # Optimization 3 + 5: parallel entity extraction via PROCESS pool, with
        # a fast extractor (extract_entity_positions_fast) that uses bisect
        # (O(log N)) instead of the O(N) linear scan in graph_rm._find_entity_in_text.
        # Combined with batch-encoded offsets (above), per-rollout work drops
        # from ~127ms to ~5-10ms — single-thread already a 10-25× speedup;
        # ProcessPool stacks on top.
        graph_rm = self.graph_rm

        # Hard-cap unique entities BEFORE find_entity_token_positions (still useful
        # since fewer entities means less per-rollout work).
        extract_cap = max(self.max_anchors_per_rollout * 2, 16)

        # Use module-global state so fork()'d worker can see them without pickle.
        # (closures don't survive fork in pickle-spawn scenarios; with fork they do.)
        _WORKER_CTX["graph_rm"] = graph_rm
        _WORKER_CTX["response_strs"] = response_strs
        _WORKER_CTX["response_ids_list"] = valid_response_id_lists
        _WORKER_CTX["retok_ids_list"] = retok_ids_list
        _WORKER_CTX["offset_maps"] = offset_maps
        _WORKER_CTX["extract_cap"] = extract_cap

        per_sample_flat: dict = {}    # sample_idx -> flat_entities
        backend_used = "none"
        t_extract_start = _time.time()
        if valid_indices:
            tasks = list(enumerate(valid_indices))   # list of (local_idx, sample_idx)

            # Two strategies based on env var:
            #   ENTITY_EXTRACT_BACKEND=process (default): ProcessPool, true parallel
            #   ENTITY_EXTRACT_BACKEND=thread: ThreadPool, falls back if process fails
            #   ENTITY_EXTRACT_BACKEND=serial: single-threaded (debug)
            backend = os.environ.get("ENTITY_EXTRACT_BACKEND", "process").lower()

            if backend == "process":
                try:
                    fork_ctx = _mp.get_context("fork")
                    with ProcessPoolExecutor(
                        max_workers=_THREAD_POOL_WORKERS,
                        mp_context=fork_ctx,
                    ) as pool:
                        for local_idx, sample_idx, flat in pool.map(
                            _extract_worker, tasks, chunksize=max(1, len(tasks) // (_THREAD_POOL_WORKERS * 4))
                        ):
                            per_sample_flat[sample_idx] = flat
                    backend_used = f"process({_THREAD_POOL_WORKERS})"
                except Exception as e:
                    print(f"[EntityInfoOnly] ProcessPool failed ({type(e).__name__}: {e}), falling back to ThreadPool")
                    backend = "thread"

            if backend == "thread":
                with ThreadPoolExecutor(max_workers=_THREAD_POOL_WORKERS) as pool:
                    for local_idx, sample_idx, flat in pool.map(_extract_worker, tasks):
                        per_sample_flat[sample_idx] = flat
                backend_used = f"thread({_THREAD_POOL_WORKERS})"
            elif backend == "serial":
                for task in tasks:
                    local_idx, sample_idx, flat = _extract_worker(task)
                    per_sample_flat[sample_idx] = flat
                backend_used = "serial"

        t_extract = _time.time() - t_extract_start

        # Clear worker context to free memory (large strings/tensors).
        _WORKER_CTX.clear()

        # Print diagnostic EVERY step so we can see if backend stays parallel.
        # (Not just first-step like the previous one-shot print.)
        n_tasks = len(valid_indices) if valid_indices else 0
        if n_tasks:
            per_task_ms = t_extract * 1000 / n_tasks
            print(
                f"[EntityInfoOnly] decode={t_decode:.1f}s encode_batch={t_encode:.1f}s "
                f"extract={t_extract:.1f}s ({backend_used}, "
                f"tasks={n_tasks}, per_task={per_task_ms:.1f}ms, "
                f"workers={_THREAD_POOL_WORKERS})"
            )

        # Build raw_extractions in original order, preserving None for padding.
        raw_extractions: List[Optional[dict]] = [None] * B
        no_entity = 0
        entity_total = 0
        entity_sampled_total = 0
        for local_idx, i in enumerate(valid_indices):
            flat = per_sample_flat.get(i, [])
            if not flat:
                no_entity += 1
                continue
            entity_total += len(flat)
            raw_extractions[i] = {
                "sample_idx": i,
                "valid_resp_len": valid_r_lens[i],
                "flat_entities": flat,
            }

        # ---- Pass 2: select entities per mode ----
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

        # ---- Pass 3: build per-rollout selected entity list ----
        per_sample_entities: List[List[tuple]] = [[] for _ in range(B)]  # list of (etype, eid, span)
        for i, extract in enumerate(raw_extractions):
            if extract is None:
                continue
            if self.mode == "shared":
                sel = shared_selected.get(i, [])
            elif self.mode == "n":
                sel = sample_entities_uniform(extract["flat_entities"], self.n)
            else:  # all
                sel = list(extract["flat_entities"])
            # Keep only first occurrence of each (etype, eid).
            seen_keys = set()
            unique_sel = []
            for (etype, eid, span) in sel:
                key = (etype, eid)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_sel.append((etype, eid, span))
            # Sort by token position (anchor order).
            unique_sel.sort(key=lambda e: int(e[2][-1]))
            # Cap to max_anchors_per_rollout.
            unique_sel = unique_sel[: self.max_anchors_per_rollout]
            per_sample_entities[i] = unique_sel
            entity_sampled_total += len(unique_sel)

        # ---- Pass 4: assign global group IDs ----
        # Group key = (uid, etype, eid). All rollouts that share this key go in
        # the same group (so GiGPO group-norm works at anchor level).
        uid_arr = None
        if data.non_tensor_batch is not None:
            uid_arr = data.non_tensor_batch.get("uid", None)

        group_id_map: dict = {}
        next_group_id = 0
        per_sample_group_ids: List[List[int]] = [[] for _ in range(B)]
        # Prompt-level grouping: same uid → same prompt_id, regardless of entity.
        # Used by ``size_1_strategy=prompt_mean`` in the policy loss to centre
        # size-1 anchors against the mean of all anchors of the SAME prompt
        # (across that prompt's rollouts), instead of falling back to a single
        # global batch mean.
        prompt_id_map: dict = {}
        next_prompt_id = 0
        per_sample_prompt_id: List[int] = [-1] * B
        for i in range(B):
            uid_i = str(uid_arr[i]) if uid_arr is not None else f"_no_uid_{i}"
            if uid_i not in prompt_id_map:
                prompt_id_map[uid_i] = next_prompt_id
                next_prompt_id += 1
            per_sample_prompt_id[i] = prompt_id_map[uid_i]
            for (etype, eid, _span) in per_sample_entities[i]:
                key = (uid_i, etype, eid)
                if key not in group_id_map:
                    group_id_map[key] = next_group_id
                    next_group_id += 1
                per_sample_group_ids[i].append(group_id_map[key])

        # ---- Pass 5: build tensors ----
        # max_K is determined by the actual maximum across the batch, capped.
        max_K = max(
            (len(es) for es in per_sample_entities),
            default=0,
        )
        if max_K == 0:
            # No entities at all in this batch — produce empty tensors.
            max_K = 1

        # End-of-segment position = entity span end + 1 (exclusive in original
        # encoding). For our segment [t_{prev_end}, t_curr_end] the "anchor
        # position" is t_curr_end (the last token of this segment).
        anchor_positions = torch.full((B, max_K), -1, dtype=torch.long, device=device)
        anchor_group_ids = torch.full((B, max_K), -1, dtype=torch.long, device=device)
        # Same shape as anchor_group_ids but keyed by prompt uid only.
        anchor_prompt_ids = torch.full((B, max_K), -1, dtype=torch.long, device=device)
        num_anchors = torch.zeros(B, dtype=torch.long, device=device)
        segment_id_per_token = torch.full((B, R), -1, dtype=torch.long, device=device)

        per_sample_token_positions: List[List[tuple]] = [[] for _ in range(B)]
        per_sample_anchor_obs: List[List[object]] = [[] for _ in range(B)]

        for i in range(B):
            entities = per_sample_entities[i]
            if not entities:
                continue
            valid_resp_len = raw_extractions[i]["valid_resp_len"] if raw_extractions[i] else 0
            # Pack kept entities CONTIGUOUSLY into slots [0, kept_idx) so the loss
            # function's mask `anchor_idx < num_anchors` correctly marks valid slots.
            # Previously we used the loop counter ``k`` which left "holes" at skipped
            # slots — those holes had anchor_group_ids != -1 but were beyond
            # num_anchors, leading to out-of-bounds indexing into group_mean.
            kept_idx = 0
            prev_end = -1
            for k, (etype, eid, span) in enumerate(entities):
                # Anchor position = the last token in the entity span (within the response).
                anchor_pos = min(int(span[-1]), valid_resp_len - 1)
                if anchor_pos <= prev_end:
                    # Anchor positions must strictly increase. Skip duplicates / out-of-order.
                    continue
                anchor_positions[i, kept_idx] = anchor_pos
                anchor_group_ids[i, kept_idx] = per_sample_group_ids[i][k]
                anchor_prompt_ids[i, kept_idx] = per_sample_prompt_id[i]
                # Tokens from prev_end+1 to anchor_pos (inclusive) belong to segment kept_idx.
                start = prev_end + 1
                end = anchor_pos + 1  # exclusive
                start = max(0, start)
                end = min(valid_resp_len, end)
                if end > start:
                    segment_id_per_token[i, start:end] = kept_idx
                prev_end = anchor_pos
                per_sample_token_positions[i].append(tuple(int(p) for p in span))
                per_sample_anchor_obs[i].append((etype, eid))
                kept_idx += 1
            num_anchors[i] = kept_idx
            # Tokens past the last anchor (response_len-1) keep segment_id = -1.

        # ---- Build the four output tensors GiGPO consumes (zero step rewards) ----
        pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
        # step_rewards: NaN-padded; zero values for valid anchor slots so GiGPO
        # group-norm doesn't crash, but step_advantage will end up ~0.
        max_entities_for_gigpo = max(max_K, 1)
        step_rewards_tensor = torch.full(
            (B, max_entities_for_gigpo), float("nan"), dtype=torch.float32
        )
        anchor_obs_arr = np.full((B, max_entities_for_gigpo), None, dtype=object)
        step_token_positions_arr = np.full((B, max_entities_for_gigpo), None, dtype=object)
        for i in range(B):
            k = len(per_sample_anchor_obs[i])
            if k == 0:
                continue
            step_rewards_tensor[i, :k] = 0.0
            ai = np.empty(k, dtype=object)
            ai[:] = per_sample_anchor_obs[i]
            anchor_obs_arr[i, :k] = ai
            ti = np.empty(k, dtype=object)
            ti[:] = per_sample_token_positions[i]
            step_token_positions_arr[i, :k] = ti

        # ---- Stats ----
        self.last_stats = {
            "entity_info/no_entity_samples": int(no_entity),
            "entity_info/entities_detected_sum": int(entity_total),
            "entity_info/entities_selected_sum": int(entity_sampled_total),
            "entity_info/num_unique_groups": int(next_group_id),
            "entity_info/max_anchors_per_rollout": int(num_anchors.max().item()) if num_anchors.numel() else 0,
            "entity_info/avg_anchors_per_rollout": float(num_anchors.float().mean().item())
            if num_anchors.numel() else 0.0,
        }

        # Numpy mirrors of anchor tensors. Stored in non_tensor_batch by
        # ray_trainer because batch.batch tensors get stripped during the
        # Ray dispatch path that splits and sends DataProto to FSDP workers.
        # numpy arrays in non_tensor_batch survive that dispatch reliably.
        anchor_positions_np = anchor_positions.cpu().numpy()
        anchor_group_ids_np = anchor_group_ids.cpu().numpy()
        anchor_prompt_ids_np = anchor_prompt_ids.cpu().numpy()
        segment_id_per_token_np = segment_id_per_token.cpu().numpy()
        num_anchors_np = num_anchors.cpu().numpy()

        return {
            "pure_step_reward_tensor": pure_step_reward,
            "step_rewards": step_rewards_tensor,
            "anchor_obs": anchor_obs_arr,
            "step_token_positions": step_token_positions_arr,
            # Tensor versions (kept for completeness; not relied on through dispatch).
            "anchor_positions": anchor_positions,
            "anchor_group_ids": anchor_group_ids,
            "anchor_prompt_ids": anchor_prompt_ids,
            "segment_id_per_token": segment_id_per_token,
            "num_anchors": num_anchors,
            # Numpy versions for non_tensor_batch (the path that survives dispatch).
            "anchor_positions_np": anchor_positions_np,
            "anchor_group_ids_np": anchor_group_ids_np,
            "anchor_prompt_ids_np": anchor_prompt_ids_np,
            "segment_id_per_token_np": segment_id_per_token_np,
            "num_anchors_np": num_anchors_np,
        }
