"""Shared utilities for entity-anchored step reward computers.

Used by ``progress_reward_fn`` (Direction 3), ``kl_step_reward`` (token-level KL
accumulated over entity spans), and ``consistency_step_reward`` (cross-rollout
self-consistency at entity positions).
"""
from __future__ import annotations

import bisect
import re
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Regex compile cache (shared across rollouts; same pattern strings recur).
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=20000)
def _compiled_pattern(pattern_str: str):
    """Compile a regex pattern with IGNORECASE. Cached across rollouts because
    the same node/edge IDs (e.g. node 5, edge (3,9)) recur across the batch."""
    return re.compile(pattern_str, re.IGNORECASE)


def _bisect_token_range(starts: List[int], ends: List[int], sc: int, ec: int) -> Tuple[int, int]:
    """Find token index range [left, right) that overlaps char range [sc, ec).

    Replaces the O(N) linear scan in ``graph_rm._find_entity_in_text`` with
    O(log N) binary search. ``offset_mapping`` is monotonically increasing in
    token order, so we can bisect:
      - left: first token whose ``end > sc``  (overlaps from left)
      - right: first token whose ``start >= ec`` (past the end)
    """
    # First token where end > sc (i.e. earliest token that touches/overlaps sc)
    left = bisect.bisect_right(ends, sc)
    # First token where start >= ec (one past the last covering token)
    right = bisect.bisect_left(starts, ec)
    if left >= right:
        return -1, -1
    return left, right


# --------------------------------------------------------------------------- #
# Boxed-answer extraction.
# --------------------------------------------------------------------------- #


def last_boxed_only_string(s: str) -> Optional[str]:
    """Return the rightmost ``\\boxed{...}`` (including wrapper), or None."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    end = None
    depth = 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        return None
    return s[idx : end + 1]


def remove_boxed_wrapper(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if s.startswith("\\boxed ") :
        return s[len("\\boxed "):]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1]
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    inner = remove_boxed_wrapper(last_boxed_only_string(text))
    return inner.strip() if inner else None


# --------------------------------------------------------------------------- #
# Entity extraction & uniform sampling.
# --------------------------------------------------------------------------- #


def extract_entity_positions(
    graph_rm,
    response_str: str,
    valid_response_ids: torch.Tensor,
    max_entities: int = 0,
    first_occurrence_only: bool = True,
):
    """Return list of (entity_type, entity_id, sorted_token_span) using graph_rm.

    Args:
        max_entities: if > 0, cap the unique entity ids passed to the expensive
            ``find_entity_token_positions`` call. Defaults to 0 (no cap, legacy).
            Setting to e.g. 32 avoids 600+ entity regex scans per rollout.
        first_occurrence_only: if True, keep only the FIRST occurrence of each
            (entity_type, entity_id). Same entity at multiple positions used to
            create separate flat entries — wasted downstream work since
            ``select_shared_pool`` and similar dedupe later anyway.
    """
    entities = graph_rm.extract_entities(response_str)

    # Cap unique entity ids BEFORE the expensive find_entity_token_positions step.
    # This step does O(unique_entities × response_length) regex work; capping at 32
    # vs 50+ saves 30-40% per rollout when avg unique entity count is high.
    if max_entities > 0:
        nodes = list(entities.get("nodes", []))
        edges = list(entities.get("edges", []))
        if len(nodes) > max_entities:
            nodes = nodes[:max_entities]
        if len(edges) > max_entities:
            edges = edges[:max_entities]
        entities = {"nodes": nodes, "edges": edges}

    entity_positions = graph_rm.find_entity_token_positions(
        response_str, valid_response_ids, entities
    )
    flat: List[Tuple[str, str, List[int]]] = []
    seen_keys: set = set()
    for eid, span_list in entity_positions.get("node_positions", {}).items():
        for tok_span in span_list:
            tok_span = [int(x) for x in tok_span]
            if not tok_span:
                continue
            key = ("node", eid)
            if first_occurrence_only and key in seen_keys:
                continue
            seen_keys.add(key)
            flat.append(("node", eid, sorted(tok_span)))
    for eid, span_list in entity_positions.get("edge_positions", {}).items():
        for tok_span in span_list:
            tok_span = [int(x) for x in tok_span]
            if not tok_span:
                continue
            key = ("edge", eid)
            if first_occurrence_only and key in seen_keys:
                continue
            seen_keys.add(key)
            flat.append(("edge", eid, sorted(tok_span)))
    flat.sort(key=lambda e: e[2][0])
    return flat


def extract_entity_positions_fast(
    graph_rm,
    response_str: str,
    valid_response_ids,
    retokenized_ids: List[int],
    offset_mapping: List[Tuple[int, int]],
    max_entities: int = 0,
    first_occurrence_only: bool = True,
):
    """Fast variant of ``extract_entity_positions``.

    Differences from the original (which calls ``graph_rm.find_entity_token_positions``):
      1. ``retokenized_ids`` and ``offset_mapping`` are passed in pre-computed
         (we batch-encode in the parent process and reuse here).
      2. Per-entity regex matches use ``_bisect_token_range`` (O(log N))
         instead of the O(N) linear scan over ``offset_mapping``.
      3. Regex patterns are LRU-cached across rollouts via ``_compiled_pattern``.
      4. We early-exit per entity once we find the first occurrence
         (the only one used downstream when ``first_occurrence_only=True``).

    Returns the same shape as ``extract_entity_positions``:
        list of (entity_type, entity_id, sorted_token_span)
    """
    entities = graph_rm.extract_entities(response_str)

    if max_entities > 0:
        nodes = list(entities.get("nodes", []))[:max_entities]
        edges = list(entities.get("edges", []))[:max_entities]
        entities = {"nodes": nodes, "edges": edges}

    # Build token_position_map from retokenized -> original_ids (reuse graph_rm's
    # logic; it has a fast path when retok==original which is the common case).
    original_ids_int = [int(t) for t in valid_response_ids]
    token_position_map = graph_rm._build_token_position_map(
        retokenized_ids=retokenized_ids,
        original_ids=original_ids_int,
    )

    # Pre-extract starts/ends arrays once per rollout for bisect.
    if offset_mapping:
        starts = [int(s) for s, _ in offset_mapping]
        ends = [int(e) for _, e in offset_mapping]
    else:
        starts, ends = [], []

    flat: List[Tuple[str, str, List[int]]] = []
    seen_keys: set = set()
    n_tokens = len(token_position_map)

    def _scan_entity(etype: str, eid, patterns: List[str]):
        key = (etype, eid)
        if first_occurrence_only and key in seen_keys:
            return
        for pattern_str in patterns:
            try:
                pattern = _compiled_pattern(pattern_str)
            except re.error:
                continue
            for match in pattern.finditer(response_str):
                sc, ec = match.start(), match.end()
                left, right = _bisect_token_range(starts, ends, sc, ec)
                if left < 0:
                    continue
                left = max(0, left)
                right = min(n_tokens, right)
                if left >= right:
                    continue
                # Map retokenized indices to original token positions via map.
                original_positions: List[int] = []
                for ti in range(left, right):
                    original_positions.extend(token_position_map[ti])
                # Dedupe preserving order.
                seen_pos = set()
                normalized: List[int] = []
                for p in original_positions:
                    pi = int(p)
                    if pi not in seen_pos:
                        seen_pos.add(pi)
                        normalized.append(pi)
                if not normalized:
                    continue
                seen_keys.add(key)
                flat.append((etype, eid, sorted(normalized)))
                if first_occurrence_only:
                    return  # done with this entity

    for node_id in entities.get("nodes", []):
        patterns = graph_rm._build_node_search_patterns(node_id)
        _scan_entity("node", node_id, patterns)

    for edge in entities.get("edges", []):
        patterns = graph_rm._build_edge_search_patterns(edge)
        _scan_entity("edge", edge, patterns)

    flat.sort(key=lambda e: e[2][0])
    return flat


def sample_entities_uniform(flat_entities, n_pick: int):
    """Pick ``n_pick`` evenly-spaced entities (incl. first and last)."""
    if n_pick <= 0 or len(flat_entities) <= n_pick:
        return list(flat_entities)
    n_total = len(flat_entities)
    if n_pick == 1:
        return [flat_entities[-1]]
    step = (n_total - 1) / (n_pick - 1)
    idxs = sorted({int(round(step * k)) for k in range(n_pick)})
    # Top up if rounding collisions reduced count.
    i = 0
    while len(idxs) < n_pick and i < n_total:
        if i not in idxs:
            idxs.append(i)
        i += 1
    return [flat_entities[i] for i in sorted(idxs)[:n_pick]]


# --------------------------------------------------------------------------- #
# Per-prompt shared entity pool (mode=shared).
# --------------------------------------------------------------------------- #


def select_shared_pool(
    uid_arr,
    raw_extractions: List[Optional[dict]],
    min_rollout_freq: int = 2,
    shared_pool_size: int = 30,
):
    """Build a shared entity pool per prompt (rollouts grouped by uid).

    Returns a dict with two kinds of keys:
    - ``sample_idx`` (int) -> list of (entity_type, entity_id, first_span)
    - ``("__pool_size__", sample_idx)`` -> int (the pool size assigned to this
      sample's prompt; useful for coverage stats)

    Each ``raw_extractions[i]`` is either None or a dict with
    ``flat_entities``: List[(etype, eid, span)] sorted by first-token position.
    """
    if uid_arr is None:
        return {
            i: list(extract["flat_entities"])
            for i, extract in enumerate(raw_extractions)
            if extract is not None
        }

    groups: dict = defaultdict(list)
    for i, extract in enumerate(raw_extractions):
        if extract is None:
            continue
        try:
            key = str(uid_arr[i])
        except Exception:
            key = i
        groups[key].append(i)

    selected: dict = {}
    for uid_key, member_idxs in groups.items():
        # Per-rollout unique entity sets (first-occurrence span).
        per_sample_first: List[dict] = []
        for si in member_idxs:
            seen: dict = {}
            for (etype, eid, span) in raw_extractions[si]["flat_entities"]:
                ekey = (etype, eid)
                if ekey not in seen:
                    seen[ekey] = span
            per_sample_first.append(seen)

        freq: dict = defaultdict(int)
        earliest: dict = {}
        for seen in per_sample_first:
            for ekey, span in seen.items():
                freq[ekey] += 1
                t0 = int(span[0]) if span else 0
                if ekey not in earliest or t0 < earliest[ekey]:
                    earliest[ekey] = t0

        qualified = [
            (ekey, freq[ekey], earliest[ekey])
            for ekey in freq
            if freq[ekey] >= min_rollout_freq
        ]
        qualified.sort(key=lambda x: (-x[1], x[2]))
        pool = [ekey for (ekey, _, _) in qualified[:shared_pool_size]]
        pool_size = len(pool)

        for si, seen in zip(member_idxs, per_sample_first):
            picked = []
            for ekey in pool:
                if ekey in seen:
                    picked.append((ekey[0], ekey[1], seen[ekey]))
            selected[si] = picked
            selected[("__pool_size__", si)] = pool_size

    return selected


# --------------------------------------------------------------------------- #
# Outcome sign vector (used by sign_mode='with_outcome').
# --------------------------------------------------------------------------- #


def infer_sign_vector(data, B: int, outcome_scores=None) -> np.ndarray:
    """+1 if outcome > 0, -1 otherwise. Falls back to +1 with warning."""
    if outcome_scores is not None:
        vals = outcome_scores.detach().to("cpu").numpy().astype(float)
    else:
        for key in ("token_level_scores", "token_level_rewards"):
            if key in data.batch:
                vals = data.batch[key].sum(dim=-1).detach().to("cpu").numpy().astype(float)
                break
        else:
            warnings.warn("infer_sign_vector: no outcome scores available; defaulting +1")
            return np.ones(B, dtype=float)
    return np.where(vals > 0, 1.0, -1.0)


# --------------------------------------------------------------------------- #
# Reward scatter to (B, R) tensor.
# --------------------------------------------------------------------------- #


def scatter_entity_reward(
    tensor: torch.Tensor,
    sample_idx: int,
    token_span: Sequence[int],
    reward: float,
    response_len: int,
    entity_reward_mode: str = "span",
) -> None:
    positions = [int(p) for p in token_span if 0 <= int(p) < response_len]
    if not positions:
        return
    if entity_reward_mode == "first":
        targets = [positions[0]]
    elif entity_reward_mode == "last":
        targets = [positions[-1]]
    else:  # "span"
        targets = positions
    for pos in targets:
        tensor[sample_idx, pos] = reward


# --------------------------------------------------------------------------- #
# Bool/string config parser.
# --------------------------------------------------------------------------- #


def parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value) if value is not None else default
