"""Deterministic candidate action generator for graph-logic routing.

This module intentionally uses only the public-ish attributes of
``GraphLogicSpec`` and ``GraphLogicAlgorithmState`` to avoid making the router a
second graph executor. It proposes actions; the main observer still verifies
them through the existing task state machine.
"""

from __future__ import annotations

import heapq
import math
import re
from typing import Any

from .action_schema import ActionCandidate


def _norm_edge(u: int, v: int, directed: bool = False) -> tuple[int, int]:
    return (u, v) if directed or u <= v else (v, u)


def _numbers(text: str) -> list[int]:
    out: list[int] = []
    for raw in re.findall(r"\b\d+\b", text):
        if len(raw) > 12:
            continue
        try:
            out.append(int(raw))
        except ValueError:
            continue
    return out


def _numeric_values(text: str) -> list[float]:
    vals: list[float] = []
    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _find(parent: dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _adjacency(spec: Any) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {int(node): [] for node in getattr(spec, "nodes", set())}
    directed = bool(getattr(spec, "directed", False))
    weights = getattr(spec, "weights", {}) or {}
    for edge in getattr(spec, "edges", set()) or set():
        if len(edge) < 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        w = float(weights.get((u, v), weights.get(_norm_edge(u, v, directed), 1.0)))
        adj.setdefault(u, []).append((v, w))
        if not directed:
            adj.setdefault(v, []).append((u, w))
    return adj


def _shortest_distance(spec: Any, source: int, target: int) -> float:
    if source == target:
        return 0.0
    adj = _adjacency(spec)
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    best: dict[int, float] = {int(source): 0.0}
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > best.get(node, math.inf) + 1e-12:
            continue
        if node == target:
            return float(dist)
        for nbr, weight in adj.get(node, []):
            nd = dist + float(weight)
            if nd + 1e-12 < best.get(nbr, math.inf):
                best[nbr] = nd
                heapq.heappush(heap, (nd, nbr))
    return math.inf


def _pair_edges_from_text(text: str, spec: Any) -> set[tuple[int, int]]:
    directed = bool(getattr(spec, "directed", False))
    nodes = set(getattr(spec, "nodes", set()) or set())
    edges: set[tuple[int, int]] = set()
    for a, b in re.findall(r"\((\d+)\s*,\s*(\d+)\)", text):
        u, v = int(a), int(b)
        if u in nodes and v in nodes:
            edges.add(_norm_edge(u, v, directed))
    for m in re.finditer(r"\b(\d+)\s*(?:->|--|-|to|and)\s*(\d+)\b", text, flags=re.I):
        u, v = int(m.group(1)), int(m.group(2))
        if u in nodes and v in nodes:
            edges.add(_norm_edge(u, v, directed))
    return edges


def _path_from_text(text: str, spec: Any) -> list[int]:
    nodes = set(getattr(spec, "nodes", set()) or set())
    patterns = [
        r"\b\d+(?:(?:\s*(?:->|--|-|to|via|through)\s*)\d+){2,}\b",
        r"\[([0-9,\s]+)\]",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            raw = m.group(1) if m.groups() else m.group(0)
            seq = [int(x) for x in re.findall(r"\d+", raw) if int(x) in nodes]
            if len(seq) >= 2:
                return seq
    return []


def _node_sequence_from_text(text: str, spec: Any) -> list[int]:
    nodes = set(getattr(spec, "nodes", set()) or set())
    seq = _path_from_text(text, spec)
    if seq:
        return seq
    patterns = [
        r"(?:order|sequence|path|list|set|answer)\s*(?:is|=|:)?\s*[\[\(\{]?\s*([0-9,\s]+)",
        r"\[([0-9,\s]+)\]",
        r"\{([0-9,\s]+)\}",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            vals = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in nodes]
            if vals:
                return vals
    return []


def _candidate(
    cid: str,
    action_type: str,
    args: dict[str, Any],
    unit_names: tuple[str, ...],
    *,
    edges: tuple[tuple[int, int], ...] = (),
    nodes: tuple[int, ...] = (),
    weak: bool = False,
    terminal: bool = False,
    reason: str = "",
) -> ActionCandidate:
    key = (action_type, tuple(sorted(args.items(), key=lambda item: str(item[0]))))
    return ActionCandidate(
        candidate_id=cid,
        action_type=action_type,
        args=args,
        canonical_key=key,
        unit_names=unit_names,
        edges=edges,
        nodes=nodes,
        weak=weak,
        terminal=terminal,
        reason=reason,
    )


def _candidate_text_relevance(text: str, candidate: ActionCandidate) -> float:
    """Cheap ranking so over-generated candidates keep text-relevant actions first."""

    low = text.lower()
    text_nums = {str(num) for num in _numbers(text)}
    score = 0.0
    for node in candidate.nodes:
        if str(node) in text_nums:
            score += 2.0
    for value in candidate.args.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and (
                str(int(value)) in text_nums or str(round(float(value), 6)) in text_nums
            ):
                score += 1.2
        elif isinstance(value, (list, tuple)):
            score += 0.5 * sum(1 for item in value if str(item) in text_nums)
    for u, v in candidate.edges:
        if re.search(rf"\b{u}\b.{0,16}\b{v}\b|\b{v}\b.{0,16}\b{u}\b", text):
            score += 3.0
    action = candidate.action_type
    if action == "SETTLE_NODE" and any(
        cue in low for cue in ("settle", "settled", "finalize", "finalise", "permanent", "minimum tentative", "unsettled")
    ):
        score += 5.0
    elif action == "RELAX_EDGE" and any(
        cue in low for cue in ("relax", "update", "improve", "better distance", "tentative cost", "through", "via")
    ):
        score += 3.0
    elif action == "NO_RELAX_EDGE" and any(
        cue in low for cue in ("does not improve", "no improvement", "not improve", "keep", "unchanged")
    ):
        score += 4.0
    elif action == "ACCEPT_EDGE" and any(cue in low for cue in ("accept", "add", "include", "union", "safe edge", "connects")):
        score += 3.0
    elif action == "REJECT_CYCLE_EDGE" and any(cue in low for cue in ("reject", "skip", "cycle", "same component", "do not add")):
        score += 3.0
    elif action == "AUGMENT_PATH" and any(cue in low for cue in ("augment", "push", "send", "residual path", "bottleneck")):
        score += 3.0
    elif action == "PAIR_DISTANCE" and any(cue in low for cue in ("distance", "d(", "shortest path", "contribution")):
        score += 2.0
    elif action == "NORMALIZER" and any(cue in low for cue in ("denominator", "normalizer", "divide", "ordered")):
        score += 3.0
    elif action == "MAPPING_PAIR" and any(cue in low for cue in ("map", "maps", "correspond", "matched", "graph1", "graph2")):
        score += 3.0
    elif action == "VISIT_NODE" and any(cue in low for cue in ("visit", "discover", "enqueue", "dequeue", "process", "mark")):
        score += 3.0
    elif action == "TRAVERSE_EDGE" and any(cue in low for cue in ("tree", "parent", "explore", "neighbor", "discover")):
        score += 3.0
    elif action == "EMIT_NODE" and any(cue in low for cue in ("emit", "output", "append", "order", "topological", "next")):
        score += 3.0
    elif action == "FINAL_NODE_SEQUENCE" and any(cue in low for cue in ("final", "answer", "path", "order", "sequence")):
        score += 3.0
    elif action == "SELECT_SET_NODE" and any(cue in low for cue in ("select", "choose", "include", "cover", "dominat", "independent")):
        score += 3.0
    elif action == "SELECT_SET_EDGE" and any(cue in low for cue in ("select", "choose", "include", "matching", "cover", "edge")):
        score += 3.0
    elif action in {"FINAL_NODE_SET", "FINAL_EDGE_SET"} and any(cue in low for cue in ("final", "answer", "solution", "set")):
        score += 3.0
    return score


def _sort_candidates(text: str, candidates: list[ActionCandidate], max_candidates: int) -> list[ActionCandidate]:
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda item: (_candidate_text_relevance(text, item[1]), -item[0]), reverse=True)
    return [candidate for _idx, candidate in indexed[:max_candidates]]


def _wsp_candidates(spec: Any, state: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    text_nums = set(_numbers(text))
    directed = bool(getattr(spec, "directed", False))
    weights = getattr(spec, "weights", {}) or {}
    dist = getattr(state, "dist", {}) or {}
    explicit_edges = _pair_edges_from_text(text, spec)
    seen_keys: set[tuple[int, int, float]] = set()

    for edge in list(explicit_edges) + list(getattr(spec, "edge_order", []) or []) + list(getattr(spec, "edges", []) or []):
        u, v = int(edge[0]), int(edge[1])
        orientations = [(u, v)] if directed else [(u, v), (v, u)]
        for a, b in orientations:
            if a not in dist:
                continue
            old = float(dist.get(b, math.inf))
            base = float(dist.get(a, math.inf))
            weight = float(weights.get((a, b), weights.get(_norm_edge(a, b, directed), 1.0)))
            new_dist = base + weight
            if not math.isfinite(base):
                continue
            improves = bool(new_dist + 1e-12 < old)
            if improves:
                key = (int(a), int(b), round(float(new_dist), 6))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                edge_key = _norm_edge(a, b, directed)
                candidates.append(
                    _candidate(
                        f"relax_{a}_{b}_{round(new_dist, 6)}",
                        "RELAX_EDGE",
                        {"u": a, "v": b, "weight": weight, "new_dist": round(new_dist, 6), "old_dist": old},
                        ("state_dijkstra_relax", "state_dijkstra_weighted_relax_cert"),
                        edges=(edge_key,),
                        nodes=(a, b),
                        reason="weighted shortest-path relax candidate",
                    )
                )
            elif b in text_nums or a in text_nums:
                key = (int(a), int(b), round(float(new_dist), 6))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                edge_key = _norm_edge(a, b, directed)
                candidates.append(
                    _candidate(
                        f"no_relax_{a}_{b}_{round(new_dist, 6)}",
                        "NO_RELAX_EDGE",
                        {"u": a, "v": b, "weight": weight, "candidate_dist": round(new_dist, 6), "old_dist": old},
                        ("state_dijkstra_no_relax",),
                        edges=(edge_key,),
                        nodes=(a, b),
                        weak=True,
                        reason="weighted shortest-path no-improvement candidate",
                    )
                )
    low = text.lower()
    settle_cues = any(
        cue in low
        for cue in (
            "settle",
            "settled",
            "finalize",
            "finalise",
            "permanent",
            "minimum tentative",
            "unsettled",
            "next settled",
            "next node",
            "choose node",
        )
    )
    unsettled = [
        (float(value), int(node))
        for node, value in dist.items()
        if node not in getattr(state, "settled_nodes", set()) and math.isfinite(float(value))
    ]
    if unsettled and len(candidates) < max_candidates and (settle_cues or not candidates):
        value, node = min(unsettled)
        candidates.append(
            _candidate(
                f"settle_{node}_{round(value, 6)}",
                "SETTLE_NODE",
                {
                    "v": node,
                    "node": node,
                    "dist": round(value, 6),
                    "rule": "choose the unsettled node with the minimum tentative distance",
                },
                ("state_dijkstra_settle",),
                nodes=(node,),
                reason=(
                    f"settle/finalize node {node}: it is the next unsettled node "
                    f"with minimum tentative distance {round(value, 6)}"
                ),
            )
        )
    path = _path_from_text(text, spec)
    if len(path) >= 2 and len(candidates) < max_candidates:
        candidates.append(
            _candidate(
                "final_path_" + "_".join(map(str, path[:16])),
                "FINAL_PATH",
                {"path": path},
                ("state_path_cost_valid", "state_weighted_path_final_cert"),
                nodes=tuple(path),
                terminal=True,
                reason="final weighted path candidate",
            )
        )
    return _sort_candidates(text, candidates, max_candidates)


def _mst_candidates(spec: Any, state: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    directed = bool(getattr(spec, "directed", False))
    weights = getattr(spec, "weights", {}) or {}
    parent = dict(getattr(state, "dsu_parent", {}) or {})
    explicit_edges = _pair_edges_from_text(text, spec)
    edge_order = list(explicit_edges) + list(getattr(spec, "edge_order", []) or [])
    if not edge_order:
        edge_order = sorted(getattr(spec, "edges", set()) or set(), key=lambda e: (float(weights.get(e, 1.0)), e))
    seen_edges: set[tuple[int, int]] = set()
    for edge in edge_order:
        if len(candidates) >= max_candidates:
            break
        u, v = int(edge[0]), int(edge[1])
        norm = _norm_edge(u, v, directed)
        if norm in seen_edges:
            continue
        seen_edges.add(norm)
        same_component = _find(parent, u) == _find(parent, v)
        weight = float(weights.get(norm, weights.get((u, v), 1.0)))
        if same_component:
            candidates.append(
                _candidate(
                    f"reject_cycle_{u}_{v}",
                    "REJECT_CYCLE_EDGE",
                    {"u": u, "v": v, "weight": weight},
                    ("state_mst_cycle_reject_cert",),
                    edges=(norm,),
                    nodes=(u, v),
                    reason="edge currently closes a DSU cycle",
                )
            )
        else:
            candidates.append(
                _candidate(
                    f"accept_edge_{u}_{v}_{round(weight, 6)}",
                    "ACCEPT_EDGE",
                    {"u": u, "v": v, "weight": weight},
                    ("state_mst_safe_edge",),
                    edges=(norm,),
                    nodes=(u, v),
                    reason="edge crosses current DSU components",
                )
            )
    return _sort_candidates(text, candidates, max_candidates)


def _flow_path_candidates(spec: Any, state: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    residual = getattr(state, "residual", {}) or {}
    path_nodes = _path_from_text(text, spec)
    candidates: list[ActionCandidate] = []
    directed = bool(getattr(spec, "directed", True))

    def add_path(nodes: list[int], source: str) -> None:
        if len(nodes) < 2 or len(candidates) >= max_candidates:
            return
        edges: list[tuple[int, int]] = []
        for u, v in zip(nodes, nodes[1:]):
            edge = _norm_edge(int(u), int(v), directed)
            if float(residual.get(edge, 0.0)) <= 1e-12:
                return
            edges.append(edge)
        bottleneck = min(float(residual.get(edge, 0.0)) for edge in edges)
        candidates.append(
            _candidate(
                "augment_" + "_".join(map(str, nodes[:12])) + f"_b{round(bottleneck, 6)}",
                "AUGMENT_PATH",
                {"path": nodes, "bottleneck": round(bottleneck, 6), "source": source},
                (
                    "state_flow_valid_augment",
                    "state_flow_valid_augment_inferred",
                    "state_flow_bottleneck",
                    "state_flow_residual_bottleneck_cert",
                    "state_flow_residual_update_after_augment",
                ),
                edges=tuple(edges),
                nodes=tuple(nodes),
                reason="residual feasible augmenting path",
            )
        )

    add_path(path_nodes, "text_path")
    if len(candidates) < max_candidates:
        source = getattr(spec, "source", None)
        target = getattr(spec, "target", None)
        if source is not None and target is not None:
            queue = [int(source)]
            parent: dict[int, int | None] = {int(source): None}
            for node in queue:
                if node == int(target):
                    break
                for (u, v), cap in residual.items():
                    if int(u) == int(node) and float(cap) > 1e-12 and int(v) not in parent:
                        parent[int(v)] = int(u)
                        queue.append(int(v))
            if int(target) in parent:
                rev = [int(target)]
                while parent[rev[-1]] is not None:
                    rev.append(int(parent[rev[-1]]))
                add_path(list(reversed(rev)), "residual_bfs")
    return _sort_candidates(text, candidates, max_candidates)


def _metric_candidates(spec: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    if getattr(spec, "task", "") not in {"global_efficiency", "wiener_index", "betweenness_centrality", "closeness_centrality"}:
        return []
    low = text.lower()
    nums = [n for n in _numbers(text) if n in set(getattr(spec, "nodes", set()) or set())]
    candidates: list[ActionCandidate] = []
    seen: set[tuple[int, int]] = set()
    explicit_pairs = sorted(_pair_edges_from_text(text, spec))
    pair_sources: list[tuple[int, int]] = list(explicit_pairs)
    pair_sources.extend((u, v) for i, u in enumerate(nums) for v in nums[i + 1 :])
    for u, v in pair_sources:
            if u == v:
                continue
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            dist = _shortest_distance(spec, a, b)
            if not math.isfinite(dist):
                continue
            unit_names = ["global_pair_distance_scope"]
            if getattr(spec, "task", "") == "wiener_index":
                unit_names.append("wiener_pair_distance_scope")
            if getattr(spec, "task", "") == "global_efficiency":
                unit_names.extend(["global_efficiency_pair_scope", "global_efficiency_pair_contribution"])
            candidates.append(
                _candidate(
                    f"pair_distance_{a}_{b}_{round(dist, 6)}",
                    "PAIR_DISTANCE",
                    {"u": a, "v": b, "distance": round(dist, 6)},
                    tuple(unit_names),
                    nodes=(a, b),
                    reason="pair-distance table candidate",
                )
            )
            if len(candidates) >= max_candidates:
                return _sort_candidates(text, candidates, max_candidates)
    if any(cue in low for cue in ("denominator", "normalizer", "divide", "n(n-1)", "n *")) and len(candidates) < max_candidates:
        candidates.append(
            _candidate(
                f"normalizer_{getattr(spec, 'task', 'metric')}",
                "NORMALIZER",
                {"task": getattr(spec, "task", "")},
                ("global_efficiency_denominator_value", "global_efficiency_normalizer", "wiener_pair_scope"),
                weak=True,
                reason="metric normalizer candidate",
            )
        )
    return _sort_candidates(text, candidates, max_candidates)


def _isomorphism_candidates(spec: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    if getattr(spec, "task", "") != "isomophic_mapping":
        return []
    candidates: list[ActionCandidate] = []
    for m in re.finditer(
        r"(?:graph\s*1\s*#?\s*)?(\d+)\s*(?:->|:|=|maps?\s+to|corresponds?\s+to|matched\s+with)\s*(?:graph\s*2\s*#?\s*)?(\d+)",
        text,
        flags=re.I,
    ):
        u, v = int(m.group(1)), int(m.group(2))
        candidates.append(
            _candidate(
                f"map_{u}_{v}",
                "MAPPING_PAIR",
                {"u": u, "v": v},
                ("mapping_pair", "mapping_valid_dict", "complete_mapping_dict_answer"),
                nodes=(u, v),
                reason="isomorphism mapping pair candidate",
            )
        )
        if len(candidates) >= max_candidates:
            break
    if len(candidates) < max_candidates:
        for m in re.finditer(
            r"(?:vertex|node)\s+(\d+).{0,48}?(?:corresponds?\s+to|maps?\s+to|matched\s+with).{0,24}?(?:vertex|node)\s+(\d+)",
            text,
            flags=re.I,
        ):
            u, v = int(m.group(1)), int(m.group(2))
            candidates.append(
                _candidate(
                    f"map_{u}_{v}",
                    "MAPPING_PAIR",
                    {"u": u, "v": v},
                    ("mapping_pair", "mapping_valid_dict", "complete_mapping_dict_answer"),
                    nodes=(u, v),
                    reason="isomorphism mapping pair candidate",
                )
            )
            if len(candidates) >= max_candidates:
                break
    return _sort_candidates(text, candidates, max_candidates)


def _traversal_candidates(spec: Any, state: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    task = str(getattr(spec, "task", "") or "")
    if task not in {"bfs", "dfs"}:
        return []
    low = text.lower()
    if not any(cue in low for cue in ("visit", "visited", "discover", "enqueue", "dequeue", "push", "pop", "process", "mark", "tree", "parent", "explore")):
        return []
    candidates: list[ActionCandidate] = []
    for edge in sorted(_pair_edges_from_text(text, spec)):
        if edge in set(getattr(spec, "edges", set()) or set()):
            u, v = int(edge[0]), int(edge[1])
            candidates.append(
                _candidate(
                    f"traverse_edge_{u}_{v}",
                    "TRAVERSE_EDGE",
                    {"task": task, "u": u, "v": v},
                    (
                        "state_traversal_order_edge",
                        "state_traversal_tree_edge",
                        "traversal_tree_edge",
                        "traversal_edge",
                    ),
                    edges=(edge,),
                    nodes=(u, v),
                    reason="BFS/DFS traversal edge candidate",
                )
            )
    seen_node_ids: set[int] = set()
    text_nodes = []
    for n in _numbers(text):
        if n in set(getattr(spec, "nodes", set()) or set()) and n not in seen_node_ids:
            seen_node_ids.add(n)
            text_nodes.append(n)
    for node in text_nodes:
        candidates.append(
            _candidate(
                f"visit_node_{node}",
                "VISIT_NODE",
                {"task": task, "node": int(node)},
                (
                    "state_traversal_order_node",
                    "state_traversal_visit",
                    "traversal_node",
                ),
                nodes=(int(node),),
                reason="BFS/DFS visit/discover node candidate",
            )
        )
        if len(candidates) >= max_candidates:
            break
    return _sort_candidates(text, candidates, max_candidates)


def _ordering_candidates(spec: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    task = str(getattr(spec, "task", "") or "")
    if task not in {"topological_sort", "hamiltonian_path", "traveling_salesman_problem"}:
        return []
    low = text.lower()
    if not any(cue in low for cue in ("order", "topological", "path", "tour", "sequence", "visit", "append", "emit", "output", "final", "answer")):
        return []
    candidates: list[ActionCandidate] = []
    seq = _node_sequence_from_text(text, spec)
    if seq:
        unit_names = ["ordering_answer_valid", "parseable_node_list_answer"]
        if task == "topological_sort":
            unit_names.extend(["topological_node", "topological_prefix"])
        else:
            unit_names.extend(["path_edge", "path_prefix", "path_reaches_target"])
        candidates.append(
            _candidate(
                "final_node_sequence_" + "_".join(map(str, seq[:16])),
                "FINAL_NODE_SEQUENCE",
                {"task": task, "sequence": seq[:32]},
                tuple(unit_names),
                nodes=tuple(seq),
                terminal=True,
                reason="ordered node sequence/path/order candidate",
            )
        )
    text_nodes = [n for n in _numbers(text) if n in set(getattr(spec, "nodes", set()) or set())]
    for node in text_nodes:
        candidates.append(
            _candidate(
                f"emit_node_{node}",
                "EMIT_NODE",
                {"task": task, "node": int(node)},
                ("topological_node", "topological_prefix", "path_prefix"),
                nodes=(int(node),),
                weak=task != "topological_sort",
                reason="ordered sequence node emission candidate",
            )
        )
        if len(candidates) >= max_candidates:
            break
    return _sort_candidates(text, candidates, max_candidates)


def _set_task_candidates(spec: Any, text: str, max_candidates: int) -> list[ActionCandidate]:
    task = str(getattr(spec, "task", "") or "")
    if task not in {
        "bipartite_maximum_matching",
        "max_weight_matching",
        "min_edge_covering",
        "min_vertex_cover",
        "dominating_set",
        "maximal_independent_set",
    }:
        return []
    low = text.lower()
    if not any(cue in low for cue in ("matching", "cover", "dominating", "dominate", "independent", "select", "choose", "include", "add", "take", "set", "solution", "final")):
        return []
    candidates: list[ActionCandidate] = []
    explicit_edges = sorted(edge for edge in _pair_edges_from_text(text, spec) if edge in set(getattr(spec, "edges", set()) or set()))
    if task in {"bipartite_maximum_matching", "max_weight_matching", "min_edge_covering"}:
        for edge in explicit_edges:
            u, v = int(edge[0]), int(edge[1])
            if task in {"bipartite_maximum_matching", "max_weight_matching"}:
                unit_names = ("matching_edge", "matching_size", "max_weight_matching_weight", "max_weight_matching_optimal_weight")
                action_type = "SELECT_SET_EDGE"
            else:
                unit_names = ("edge_cover_node", "edge_cover_matching_edge", "edge_set_answer_feasible", "edge_set_answer_optimal_size")
                action_type = "SELECT_SET_EDGE"
            candidates.append(
                _candidate(
                    f"select_edge_{task}_{u}_{v}",
                    action_type,
                    {"task": task, "u": u, "v": v},
                    unit_names,
                    edges=(edge,),
                    nodes=(u, v),
                    reason=f"{task} selected edge candidate",
                )
            )
            if len(candidates) >= max_candidates:
                break
        if explicit_edges and any(cue in low for cue in ("final", "answer", "solution", "set")) and len(candidates) < max_candidates:
            candidates.append(
                _candidate(
                    f"final_edge_set_{task}_{len(explicit_edges)}",
                    "FINAL_EDGE_SET",
                    {"task": task, "edges": explicit_edges[:16]},
                    ("edge_set_answer_feasible", "edge_set_answer_optimal_size", "set_answer_feasible", "set_answer_optimal_size"),
                    edges=tuple(explicit_edges[:16]),
                    nodes=tuple(sorted({node for edge in explicit_edges for node in edge})),
                    terminal=True,
                    reason=f"{task} final edge set candidate",
                )
            )
    else:
        text_nodes = [n for n in _numbers(text) if n in set(getattr(spec, "nodes", set()) or set())]
        for node in text_nodes:
            if task == "min_vertex_cover":
                unit_names = ("vertex_cover_edge", "vertex_cover_all_edges_covered", "set_answer_feasible", "set_answer_optimal_size")
            elif task == "dominating_set":
                unit_names = ("dominated_node", "dominating_all_nodes_covered", "set_answer_feasible", "set_answer_optimal_size")
            else:
                unit_names = (
                    "independent_node",
                    "independent_nonedge_check",
                    "maximal_independent_covered_node",
                    "maximal_independent_complete",
                    "set_answer_feasible",
                    "set_answer_optimal_size",
                )
            candidates.append(
                _candidate(
                    f"select_node_{task}_{node}",
                    "SELECT_SET_NODE",
                    {"task": task, "node": int(node)},
                    unit_names,
                    nodes=(int(node),),
                    reason=f"{task} selected node candidate",
                )
            )
            if len(candidates) >= max_candidates:
                break
        if text_nodes and any(cue in low for cue in ("final", "answer", "solution", "set")) and len(candidates) < max_candidates:
            candidates.append(
                _candidate(
                    f"final_node_set_{task}_{len(text_nodes)}",
                    "FINAL_NODE_SET",
                    {"task": task, "nodes": text_nodes[:24]},
                    ("set_answer_feasible", "set_answer_optimal_size", "vertex_cover_all_edges_covered", "dominating_all_nodes_covered", "maximal_independent_complete"),
                    nodes=tuple(text_nodes[:24]),
                    terminal=True,
                    reason=f"{task} final node set candidate",
                )
            )
    return _sort_candidates(text, candidates, max_candidates)


def enumerate_action_candidates(
    task: str,
    spec: Any,
    state: Any | None,
    segment_text: str,
    *,
    max_candidates: int = 16,
) -> list[ActionCandidate]:
    """Return deterministic candidate actions for one segment."""

    if state is None or max_candidates <= 0:
        return []
    task = str(task or getattr(spec, "task", "") or "")
    if task == "weighted_shortest_path":
        candidates = _wsp_candidates(spec, state, segment_text, max_candidates)
    elif task in {"bfs", "dfs"}:
        candidates = _traversal_candidates(spec, state, segment_text, max_candidates)
    elif task in {"minimum_spanning_tree", "weighted_minimum_spanning_tree"}:
        candidates = _mst_candidates(spec, state, segment_text, max_candidates)
    elif task == "maximum_flow":
        candidates = _flow_path_candidates(spec, state, segment_text, max_candidates)
    elif task in {"topological_sort", "hamiltonian_path", "traveling_salesman_problem"}:
        candidates = _ordering_candidates(spec, segment_text, max_candidates)
    elif task in {"global_efficiency", "wiener_index", "betweenness_centrality", "closeness_centrality"}:
        candidates = _metric_candidates(spec, segment_text, max_candidates)
    elif task in {
        "bipartite_maximum_matching",
        "max_weight_matching",
        "min_edge_covering",
        "min_vertex_cover",
        "dominating_set",
        "maximal_independent_set",
    }:
        candidates = _set_task_candidates(spec, segment_text, max_candidates)
    elif task == "isomophic_mapping":
        candidates = _isomorphism_candidates(spec, segment_text, max_candidates)
    else:
        candidates = []
    if candidates:
        candidates = candidates[: max(1, max_candidates - 1)]
    candidates.append(
        ActionCandidate(
            candidate_id="abstain",
            action_type="ABSTAIN",
            args={},
            canonical_key=("ABSTAIN",),
            unit_names=(),
            reason="no executable graph action",
        )
    )
    return candidates[:max_candidates]
