"""Graph-state progress step reward for Erdos graph tasks.

The signal is derived from structured fields already present in the dataset:
``extra_info.task``, ``extra_info.edges``, ``extra_info.direction``,
``extra_info.sample``, and ``reward_model.ground_truth``. It scores node/edge
mentions in the rollout against the oracle graph state, then emits GiGPO-style
step rewards.

For the target token span, this module supports prefix coverage by returning a
compact token-position spec ``("prefix", end_token)``. ``core_algos`` expands
that spec during scatter, so an entity mention can affect all response tokens up
to that mention without materializing long index lists in numpy object arrays.
"""
from __future__ import annotations

import ast
import heapq
import math
import multiprocessing as _mp
import os
import time as _time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager._step_reward_utils import (
    _bisect_token_range,
    extract_entity_positions,
    extract_entity_positions_fast,
)
from verl.workers.reward_manager.graph_logic_observer import (
    GraphLogicConfig,
    STATE_PROGRESS_COMPONENTS,
    build_graph_logic_spec,
    observe_graph_logic_response,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_GRAPH_LOCAL_STAT_TASKS = {
    "node_number",
    "edge_number",
    "degree",
    "degree_centrality",
    "neighbor",
    "avg_neighbor_degree",
    "edge_existence",
    "density",
    "common_neighbor",
    "jaccard_coefficient",
    "adamic_adar_index",
    "resource_allocation_index",
    "triangles",
    "clustering_coefficient",
    "is_regular",
    "is_tournament",
    "is_eularian",
    "has_cycle",
    "is_bipartite",
}
_GRAPH_TRAVERSAL_TASKS = {
    "bfs",
    "dfs",
    "connected_component_number",
    "strongly_connected_number",
    "bridges",
    "topological_sort",
    "local_connectivity",
}
_GRAPH_PATH_DISTANCE_TASKS = {
    "shortest_path",
    "weighted_shortest_path",
    "diameter",
    "radius",
    "center",
    "periphery",
    "barycenter",
    "harmonic_centrality",
}
_GRAPH_GLOBAL_METRIC_TASKS = {
    "betweenness_centrality",
    "closeness_centrality",
    "global_efficiency",
    "wiener_index",
}
_GRAPH_OPTIMIZATION_SET_TASKS = {
    "min_vertex_cover",
    "min_edge_covering",
    "dominating_set",
    "maximal_independent_set",
    "bipartite_maximum_matching",
    "max_weight_matching",
    "hamiltonian_path",
    "traveling_salesman_problem",
}
_GRAPH_TREE_FLOW_TASKS = {
    "minimum_spanning_tree",
    "weighted_minimum_spanning_tree",
    "maximum_flow",
}
_GRAPH_MAPPING_TASKS = {"isomophic_mapping"}
_GRAPH_LOGIC_FAMILIES = (
    "local_stat",
    "traversal_connectivity",
    "path_distance",
    "global_metric",
    "optimization_set",
    "tree_flow",
    "mapping",
    "other",
)


def _graph_logic_task_family(task: Any) -> str:
    task = str(task or "").lower()
    if task in _GRAPH_LOCAL_STAT_TASKS:
        return "local_stat"
    if task in _GRAPH_TRAVERSAL_TASKS:
        return "traversal_connectivity"
    if task in _GRAPH_PATH_DISTANCE_TASKS:
        return "path_distance"
    if task in _GRAPH_GLOBAL_METRIC_TASKS:
        return "global_metric"
    if task in _GRAPH_OPTIMIZATION_SET_TASKS:
        return "optimization_set"
    if task in _GRAPH_TREE_FLOW_TASKS:
        return "tree_flow"
    if task in _GRAPH_MAPPING_TASKS:
        return "mapping"
    return "other"


def _parse_graph_logic_family_scales(scale_csv: Any, default_scale: float) -> dict[str, float]:
    scales = {family: float(default_scale) for family in _GRAPH_LOGIC_FAMILIES}
    if not scale_csv:
        return scales
    if isinstance(scale_csv, dict):
        items = scale_csv.items()
    else:
        text = str(scale_csv).strip()
        if not text:
            return scales
        chunks = [chunk.strip() for chunk in text.replace(";", ",").split(",") if chunk.strip()]
        items = []
        for chunk in chunks:
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            items.append((key.strip(), value.strip()))
    for key, value in items:
        family = str(key).strip().lower()
        if family not in scales:
            continue
        try:
            scales[family] = max(0.0, min(3.0, float(value)))
        except Exception:
            continue
    return scales


def _parse_graph_logic_task_set(task_csv: Any) -> set[str]:
    """Parse a comma/space separated task list for graph-logic process controls."""

    if not task_csv:
        return set()
    if isinstance(task_csv, (list, tuple, set)):
        raw_items = task_csv
    else:
        raw = str(task_csv or "").replace(";", ",").replace("|", ",")
        raw_items = []
        for chunk in raw.split(","):
            raw_items.extend(str(chunk).split())
    tasks = {str(item).strip().lower() for item in raw_items if str(item).strip()}
    # Keep the dataset's historical typo usable while accepting the correct spelling.
    if "isomorphic_mapping" in tasks:
        tasks.add("isomophic_mapping")
    if "isomophic_mapping" in tasks:
        tasks.add("isomorphic_mapping")
    return tasks


def _graph_logic_step_unit_names(step: Any) -> set[str]:
    """Best-effort unit names for online unit-trust filtering."""

    names = getattr(step, "unit_names", None) or ()
    out = {str(name).strip() for name in names if str(name).strip()}
    if out:
        return out
    anchor = getattr(step, "anchor", None)
    if isinstance(anchor, tuple):
        if len(anchor) >= 3 and str(anchor[0]) == "graph_logic":
            return {str(anchor[2])}
        if anchor:
            return {str(anchor[0])}
    if anchor is not None:
        return {str(anchor)}
    return set()


def _detect_worker_count() -> int:
    env = os.environ.get("GRAPH_PROGRESS_EXTRACT_WORKERS") or os.environ.get("ENTITY_EXTRACT_WORKERS")
    if env:
        try:
            v = int(env)
            if v > 0:
                return min(64, v)
        except ValueError:
            pass
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity > 0:
            return min(64, affinity)
    except (AttributeError, OSError):
        pass
    return min(64, os.cpu_count() or 8)


_WORKER_CTX: dict = {}


def _extract_graph_progress_worker(args):
    local_idx, sample_idx = args
    try:
        graph_rm = _WORKER_CTX["graph_rm"]
        text = _WORKER_CTX["response_strs"][local_idx]
        response_ids = _WORKER_CTX["response_ids_list"][local_idx]
        max_entities = _WORKER_CTX["max_entities"]
        first_occurrence_only = _WORKER_CTX["first_occurrence_only"]
        retok_ids_list = _WORKER_CTX.get("retok_ids_list") or []
        offset_maps = _WORKER_CTX.get("offset_maps") or []
        if local_idx < len(retok_ids_list) and local_idx < len(offset_maps):
            flat = extract_entity_positions_fast(
                graph_rm,
                text,
                response_ids,
                retokenized_ids=retok_ids_list[local_idx],
                offset_mapping=offset_maps[local_idx],
                max_entities=max_entities,
                first_occurrence_only=first_occurrence_only,
            )
        else:
            flat = extract_entity_positions(
                graph_rm,
                text,
                response_ids,
                max_entities=max_entities,
                first_occurrence_only=first_occurrence_only,
            )
    except Exception:
        flat = []
    return local_idx, sample_idx, flat


def _observe_graph_logic_worker(args):
    local_idx, sample_idx = args
    try:
        text = _WORKER_CTX["response_strs"][local_idx]
        spec = _WORKER_CTX["logic_specs"][sample_idx]
        config = _WORKER_CTX["graph_logic_config"]
        entropy_offsets_list = _WORKER_CTX.get("model_entropy_offset_maps") or []
        entropy_values_list = _WORKER_CTX.get("model_entropy_values_list") or []
        entropy_offsets = entropy_offsets_list[local_idx] if local_idx < len(entropy_offsets_list) else None
        entropy_values = entropy_values_list[local_idx] if local_idx < len(entropy_values_list) else None
        steps, stats = observe_graph_logic_response(
            text,
            spec,
            config,
            token_entropy_offsets=entropy_offsets,
            token_entropies=entropy_values,
        )
    except Exception:
        steps, stats = [], {}
    return local_idx, sample_idx, steps, stats


def _literal(value: Any, default=None):
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in ("", "none", "null"):
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        return default


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _norm_edge(edge: Any, directed: bool) -> Optional[Tuple[int, int]]:
    if not isinstance(edge, (tuple, list)) or len(edge) < 2:
        return None
    u, v = _as_int(edge[0]), _as_int(edge[1])
    if u is None or v is None:
        return None
    if directed:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def _edge_nodes(edge: Tuple[int, int]) -> set[int]:
    return {int(edge[0]), int(edge[1])}


def _iter_ints(obj: Any) -> Iterable[int]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            ki = _as_int(k)
            vi = _as_int(v)
            if ki is not None:
                yield ki
            if vi is not None:
                yield vi
        return
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            xi = _as_int(x)
            if xi is not None:
                yield xi


def _parse_graph(extra_info: dict) -> tuple[set[int], set[Tuple[int, int]], bool, dict[int, set[int]]]:
    directed = str(extra_info.get("direction", "")).lower() == "directed"
    edges_raw = _literal(extra_info.get("edges"), default=[]) or []
    edges: set[Tuple[int, int]] = set()
    nodes: set[int] = set()
    for e in edges_raw:
        ne = _norm_edge(e, directed=directed)
        if ne is None:
            continue
        edges.add(ne)
        nodes.update(_edge_nodes(ne))

    nodes_spec = _literal(extra_info.get("nodes"), default=None)
    if isinstance(nodes_spec, (tuple, list)) and len(nodes_spec) >= 2:
        a, b = _as_int(nodes_spec[0]), _as_int(nodes_spec[1])
        if a is not None and b is not None and a <= b:
            nodes.update(range(a, b + 1))

    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        if not directed:
            adj[v].add(u)
    return nodes, edges, directed, adj


def _parse_sample_nodes(sample: Any) -> list[int]:
    if sample is None:
        return []
    if isinstance(sample, str) and sample.strip().lower() in ("", "none", "null"):
        return []
    if isinstance(sample, (list, tuple, set, dict)):
        vals = []
        seen = set()
        for x in _iter_ints(sample):
            if x not in seen:
                seen.add(x)
                vals.append(x)
        return vals
    x = _as_int(sample)
    return [x] if x is not None else []


def _weighted_graph(
    extra_info: dict,
    directed: bool,
) -> tuple[dict[int, list[tuple[int, float]]], dict[Tuple[int, int], float]]:
    edges_raw = _literal(extra_info.get("edges"), default=[]) or []
    wadj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    weights: dict[Tuple[int, int], float] = {}
    for e in edges_raw:
        if not isinstance(e, (tuple, list)) or len(e) < 2:
            continue
        u, v = _as_int(e[0]), _as_int(e[1])
        if u is None or v is None:
            continue
        try:
            w = float(e[2]) if len(e) >= 3 else 1.0
        except Exception:
            w = 1.0
        if w <= 0:
            w = 1.0
        key = _norm_edge((u, v), directed=directed)
        if key is None:
            continue
        old_w = weights.get(key)
        if old_w is None or w < old_w:
            weights[key] = w
        wadj[u].append((v, w))
        if not directed:
            wadj[v].append((u, w))
    return wadj, weights


def _dijkstra(wadj: dict[int, list[tuple[int, float]]], source: int) -> dict[int, float]:
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist.get(u):
            continue
        for v, w in wadj.get(u, []):
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _path_edges_from_nodes(order: list[int], graph_edges: set[Tuple[int, int]], directed: bool) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for u, v in zip(order, order[1:]):
        edge = _norm_edge((u, v), directed=directed)
        if edge is not None and edge in graph_edges:
            out.add(edge)
    return out


def _ordered_nodes_from_edge_list(answer: Any, start: Optional[int], directed: bool) -> list[int]:
    if not isinstance(answer, (list, tuple)) or not answer:
        return []
    edges = []
    for e in answer:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            return []
        u, v = _as_int(e[0]), _as_int(e[1])
        if u is None or v is None:
            return []
        edges.append((u, v))
    order: list[int] = []
    seen: set[int] = set()
    if start is not None:
        order.append(start)
        seen.add(start)
    for u, v in edges:
        candidates = [v] if directed else [v, u]
        if u not in seen and (not order or v in seen):
            candidates = [u]
        elif v not in seen and (not order or u in seen):
            candidates = [v]
        elif u not in seen:
            candidates = [u]
        next_node = None
        for cand in candidates:
            if cand not in seen:
                next_node = cand
                break
        if next_node is None:
            continue
        order.append(next_node)
        seen.add(next_node)
    return order


def _incident_edges(node: int, graph_edges: set[Tuple[int, int]], directed: bool, include_incoming: bool = True):
    for u, v in graph_edges:
        if u == node or (include_incoming and v == node) or (not directed and v == node):
            yield (u, v)


def _mark_edges(
    target_edges: set[Tuple[int, int]],
    target_nodes: set[int],
    edges_to_add: Iterable[Tuple[int, int]],
    graph_edges: set[Tuple[int, int]],
    directed: bool,
):
    for e in edges_to_add:
        edge = _norm_edge(e, directed=directed)
        if edge is None or edge not in graph_edges:
            continue
        target_edges.add(edge)
        target_nodes.update(_edge_nodes(edge))


def _unweighted_dist_and_tree(
    source: int,
    adj: dict[int, set[int]],
    graph_edges: set[Tuple[int, int]],
    directed: bool,
) -> tuple[set[int], set[Tuple[int, int]]]:
    seen = {source}
    tree_edges: set[Tuple[int, int]] = set()
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in sorted(adj.get(u, set())):
            edge = _norm_edge((u, v), directed=directed)
            if edge is None or edge not in graph_edges:
                continue
            if v not in seen:
                seen.add(v)
                queue.append(v)
                tree_edges.add(edge)
    return seen, tree_edges


def _component_edges(seed: int, adj: dict[int, set[int]], graph_edges: set[Tuple[int, int]], directed: bool):
    comp, _ = _unweighted_dist_and_tree(seed, adj, graph_edges, directed)
    out = set()
    for e in graph_edges:
        u, v = e
        if u in comp and v in comp:
            out.add(e)
    return comp, out


def _all_graph_soft(
    soft_nodes: set[int],
    soft_edges: set[Tuple[int, int]],
    nodes: set[int],
    graph_edges: set[Tuple[int, int]],
):
    soft_nodes.update(nodes)
    soft_edges.update(graph_edges)


def _answer_entities(
    task: str,
    answer: Any,
    sample: Any,
    nodes: set[int],
    edges: set[Tuple[int, int]],
    directed: bool,
    adj: dict[int, set[int]],
) -> tuple[set[int], set[Tuple[int, int]], set[int], list[int]]:
    """Return positive_nodes, positive_edges, focus_nodes, oracle_order."""
    del nodes
    positive_nodes: set[int] = set()
    positive_edges: set[Tuple[int, int]] = set()
    focus_nodes: set[int] = set()
    oracle_order: list[int] = []

    for x in _iter_ints(sample):
        focus_nodes.add(x)

    if isinstance(answer, (list, tuple)):
        if answer and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in answer):
            for e in answer:
                ne = _norm_edge(e, directed=directed)
                if ne is not None:
                    positive_edges.add(ne)
                    positive_nodes.update(_edge_nodes(ne))
        else:
            vals = [_as_int(x) for x in answer]
            oracle_order = [x for x in vals if x is not None]
            positive_nodes.update(oracle_order)
            for u, v in zip(oracle_order, oracle_order[1:]):
                ne = _norm_edge((u, v), directed=directed)
                if ne is not None and ne in edges:
                    positive_edges.add(ne)
    elif isinstance(answer, dict):
        for x in _iter_ints(answer):
            positive_nodes.add(x)

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        u, v = _as_int(sample[0]), _as_int(sample[1])
        if u is not None and v is not None:
            focus_nodes.update([u, v])
            if task in {"common_neighbor", "jaccard_coefficient", "adamic_adar_index", "resource_allocation_index"}:
                positive_nodes.update(adj.get(u, set()) & adj.get(v, set()))

    if task in {"shortest_path", "weighted_shortest_path", "bfs", "dfs"} and oracle_order:
        positive_nodes.update(oracle_order)
        for u, v in zip(oracle_order, oracle_order[1:]):
            ne = _norm_edge((u, v), directed=directed)
            if ne is not None and ne in edges:
                positive_edges.add(ne)

    return positive_nodes, positive_edges, focus_nodes, oracle_order


def _algorithmic_entities(
    task: str,
    answer: Any,
    sample: Any,
    nodes: set[int],
    graph_edges: set[Tuple[int, int]],
    directed: bool,
    adj: dict[int, set[int]],
    extra_info: dict,
) -> tuple[set[int], set[Tuple[int, int]], set[int], set[int], set[Tuple[int, int]], list[int], bool]:
    """Return strong/soft graph entities induced by a task-level algorithm.

    Shape:
      positive_nodes, positive_edges, focus_nodes, soft_nodes, soft_edges,
      oracle_order, algorithm_supported
    """
    base_nodes, base_edges, focus_nodes, base_order = _answer_entities(
        task, answer, sample, nodes, graph_edges, directed, adj
    )
    positive_nodes: set[int] = set()
    positive_edges: set[Tuple[int, int]] = set()
    soft_nodes: set[int] = set()
    soft_edges: set[Tuple[int, int]] = set()
    oracle_order: list[int] = []
    supported = False

    sample_nodes = _parse_sample_nodes(sample)
    focus_nodes.update(sample_nodes)

    answer_edge_tasks = {
        "bipartite_maximum_matching",
        "bridges",
        "max_weight_matching",
        "min_edge_covering",
        "minimum_spanning_tree",
        "weighted_minimum_spanning_tree",
    }
    answer_node_tasks = {
        "barycenter",
        "center",
        "dominating_set",
        "hamiltonian_path",
        "maximal_independent_set",
        "min_vertex_cover",
        "periphery",
        "topological_sort",
    }
    query_node_tasks = {
        "avg_neighbor_degree",
        "betweenness_centrality",
        "closeness_centrality",
        "clustering_coefficient",
        "degree_centrality",
        "harmonic_centrality",
        "triangles",
    }
    query_pair_tasks = {"local_connectivity", "maximum_flow"}
    global_scalar_tasks = {
        "connected_component_number",
        "density",
        "diameter",
        "edge_number",
        "global_efficiency",
        "node_number",
        "radius",
        "strongly_connected_number",
        "traveling_salesman_problem",
        "wiener_index",
    }
    global_bool_tasks = {"has_cycle", "is_bipartite", "is_eularian", "is_regular", "is_tournament"}

    def add_incident(node: int, *, strong: bool = False, include_incoming: bool = True):
        if directed and include_incoming:
            raw_neighbors = []
            for u, v in graph_edges:
                if u == node:
                    raw_neighbors.append((u, v))
                elif v == node:
                    raw_neighbors.append((u, v))
        else:
            raw_neighbors = [(node, nb) for nb in adj.get(node, set())]
        for raw_edge in raw_neighbors:
            edge = _norm_edge(raw_edge, directed=directed)
            if edge is None or edge not in graph_edges:
                continue
            if strong:
                positive_edges.add(edge)
            else:
                soft_edges.add(edge)
            soft_nodes.update(_edge_nodes(edge))

    def add_reachability_from(node: int):
        reach_nodes, tree_edges = _unweighted_dist_and_tree(node, adj, graph_edges, directed)
        soft_nodes.update(reach_nodes)
        soft_edges.update(tree_edges)

    if task in {"shortest_path", "weighted_shortest_path"} and len(sample_nodes) >= 2:
        supported = True
        src, dst = sample_nodes[0], sample_nodes[1]
        focus_nodes.update([src, dst])
        if base_order:
            oracle_order = base_order
        positive_nodes.update(oracle_order)
        positive_edges.update(_path_edges_from_nodes(oracle_order, graph_edges, directed))

        wadj, weights = _weighted_graph(extra_info, directed)
        dist_src = _dijkstra(wadj, src)
        if dst in dist_src:
            best = dist_src[dst]
            if directed:
                rev_adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
                for (u, v), w in weights.items():
                    rev_adj[v].append((u, w))
                dist_dst = _dijkstra(rev_adj, dst)
            else:
                dist_dst = _dijkstra(wadj, dst)
            eps = 1e-8
            for (u, v), w in weights.items():
                on_forward = (
                    u in dist_src
                    and v in dist_dst
                    and abs(dist_src[u] + w + dist_dst[v] - best) <= eps
                )
                on_reverse = (
                    not directed
                    and v in dist_src
                    and u in dist_dst
                    and abs(dist_src[v] + w + dist_dst[u] - best) <= eps
                )
                if on_forward or on_reverse:
                    soft_edges.add((u, v) if directed or u <= v else (v, u))
                    soft_nodes.update([u, v])

    elif task in {"bfs", "dfs"}:
        supported = True
        start = sample_nodes[0] if sample_nodes else None
        if start is not None:
            focus_nodes.add(start)
        if isinstance(answer, (list, tuple)) and answer and all(
            isinstance(x, (list, tuple)) and len(x) >= 2 for x in answer
        ):
            for e in answer:
                edge = _norm_edge(e, directed=directed)
                if edge is not None:
                    positive_edges.add(edge)
                    positive_nodes.update(_edge_nodes(edge))
            oracle_order = _ordered_nodes_from_edge_list(answer, start, directed)
            positive_nodes.update(oracle_order)
        else:
            oracle_order = base_order
            positive_nodes.update(base_nodes)
            positive_edges.update(base_edges)
        if start is not None:
            add_incident(start, strong=False)

    elif task in {"common_neighbor", "jaccard_coefficient", "adamic_adar_index", "resource_allocation_index"}:
        if len(sample_nodes) >= 2:
            supported = True
            u, v = sample_nodes[0], sample_nodes[1]
            focus_nodes.update([u, v])
            common = set(adj.get(u, set())) & set(adj.get(v, set()))
            positive_nodes.update(common)
            if base_order:
                oracle_order = base_order
            else:
                oracle_order = sorted(common)
            for c in common:
                for a in (u, v):
                    edge = _norm_edge((a, c), directed=directed)
                    if edge is not None and edge in graph_edges:
                        positive_edges.add(edge)
            for a in (u, v):
                add_incident(a, strong=False, include_incoming=False)

    elif task == "edge_existence" and len(sample_nodes) >= 2:
        supported = True
        u, v = sample_nodes[0], sample_nodes[1]
        focus_nodes.update([u, v])
        edge = _norm_edge((u, v), directed=directed)
        if edge is not None and edge in graph_edges:
            positive_edges.add(edge)
        add_incident(u, strong=False)
        add_incident(v, strong=False)

    elif task in {"neighbor", "degree"} and sample_nodes:
        supported = True
        u = sample_nodes[0]
        focus_nodes.add(u)
        nbrs = set(adj.get(u, set()))
        if task == "neighbor":
            positive_nodes.update(nbrs)
            oracle_order = base_order or sorted(nbrs)
            add_incident(u, strong=True, include_incoming=False)
        else:
            soft_nodes.update(nbrs)
            add_incident(u, strong=True, include_incoming=True)

    elif task in query_node_tasks and sample_nodes:
        supported = True
        u = sample_nodes[0]
        focus_nodes.add(u)
        nbrs = set(adj.get(u, set()))
        if task in {"degree_centrality", "avg_neighbor_degree", "clustering_coefficient", "triangles"}:
            positive_nodes.update(nbrs)
            add_incident(u, strong=True, include_incoming=directed)
            if task in {"clustering_coefficient", "triangles"}:
                for a in nbrs:
                    for b in nbrs:
                        if a == b:
                            continue
                        edge = _norm_edge((a, b), directed=directed)
                        if edge is not None and edge in graph_edges:
                            positive_edges.add(edge)
                            positive_nodes.update([a, b])
        else:
            add_reachability_from(u)
            add_incident(u, strong=False, include_incoming=True)

    elif task in query_pair_tasks and len(sample_nodes) >= 2:
        supported = True
        src, dst = sample_nodes[0], sample_nodes[1]
        focus_nodes.update([src, dst])
        add_reachability_from(src)
        add_incident(dst, strong=False, include_incoming=True)
        if directed:
            rev_adj: dict[int, set[int]] = defaultdict(set)
            for u, v in graph_edges:
                rev_adj[v].add(u)
            rev_nodes, rev_tree = _unweighted_dist_and_tree(dst, rev_adj, graph_edges, True)
            soft_nodes.update(rev_nodes)
            soft_edges.update(rev_tree)
        edge = _norm_edge((src, dst), directed=directed)
        if edge is not None and edge in graph_edges:
            positive_edges.add(edge)

    elif task in answer_edge_tasks:
        supported = True
        positive_edges.update(base_edges)
        positive_nodes.update(base_nodes)
        if task in {"minimum_spanning_tree", "weighted_minimum_spanning_tree"}:
            _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)
        elif task in {"max_weight_matching", "bipartite_maximum_matching", "min_edge_covering"}:
            soft_edges.update(graph_edges)
            soft_nodes.update(nodes)
        elif task == "bridges":
            soft_edges.update(graph_edges)
            soft_nodes.update(nodes)

    elif task in answer_node_tasks:
        supported = True
        oracle_order = base_order
        positive_nodes.update(base_nodes)
        positive_edges.update(base_edges)
        if task in {"hamiltonian_path", "topological_sort"} and oracle_order:
            positive_edges.update(_path_edges_from_nodes(oracle_order, graph_edges, directed))
        if task == "maximal_independent_set":
            for u in positive_nodes:
                add_incident(u, strong=False, include_incoming=False)
        elif task in {"min_vertex_cover", "dominating_set"}:
            for u in positive_nodes:
                add_incident(u, strong=True, include_incoming=True)
            soft_edges.update(graph_edges)
        elif task in {"center", "periphery", "barycenter"}:
            for u in positive_nodes:
                add_reachability_from(u)
        else:
            _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)

    elif task in global_scalar_tasks:
        supported = True
        positive_nodes.update(base_nodes)
        positive_edges.update(base_edges)
        if task in {"diameter", "radius"} and base_nodes:
            positive_nodes.update(base_nodes)
            for u in base_nodes:
                add_reachability_from(u)
        elif task in {"connected_component_number", "strongly_connected_number"}:
            remaining = set(nodes)
            while remaining:
                seed = min(remaining)
                comp_nodes, comp_edges = _component_edges(seed, adj, graph_edges, directed)
                soft_nodes.update(comp_nodes)
                soft_edges.update(comp_edges)
                remaining.difference_update(comp_nodes)
        else:
            _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)

    elif task in global_bool_tasks:
        supported = True
        if task in {"is_regular", "is_eularian"}:
            degs = {u: 0 for u in nodes}
            for u, v in graph_edges:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + (0 if directed else 1)
            if degs:
                vals = list(degs.values())
                witness_nodes = {u for u, d in degs.items() if d in {min(vals), max(vals)} or d % 2 == 1}
                positive_nodes.update(witness_nodes)
                for u in witness_nodes:
                    add_incident(u, strong=False, include_incoming=True)
        elif task == "has_cycle":
            _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)
        else:
            _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)

    elif task == "isomophic_mapping":
        supported = True
        positive_nodes.update(base_nodes)
        _all_graph_soft(soft_nodes, soft_edges, nodes, graph_edges)

    if not supported:
        return base_nodes, base_edges, focus_nodes, set(), set(), base_order, False

    if not positive_nodes and base_nodes:
        positive_nodes.update(base_nodes)
    if not positive_edges and base_edges:
        positive_edges.update(base_edges)
    if not oracle_order and base_order:
        oracle_order = base_order

    soft_nodes.difference_update(positive_nodes)
    soft_nodes.difference_update(focus_nodes)
    soft_edges.difference_update(positive_edges)
    return positive_nodes, positive_edges, focus_nodes, soft_nodes, soft_edges, oracle_order, True


class GraphProgressRewardComputer:
    """Structured graph progress reward anchored at node/edge mentions."""

    def __init__(
        self,
        tokenizer,
        graph_reward_manager,
        *,
        max_entities: int = 64,
        reward_clip: float = 1.0,
        positive_reward: float = 1.0,
        focus_reward: float = 0.2,
        irrelevant_node_reward: float = 0.0,
        irrelevant_edge_reward: float = -0.1,
        invalid_edge_reward: float = -1.0,
        order_hit_reward: float = 1.0,
        order_seen_reward: float = -0.2,
        order_skip_reward: float = 0.2,
        coverage_mode: str = "prefix",
        prefix_decay: float = 0.995,
        scorer: str = "answer",
        relative_alpha: float = 2.0,
        relative_min_position_weight: float = 0.2,
        relative_final_gate_eta: float = 0.2,
        graph_logic_ordered_reward: float = 1.0,
        graph_logic_progress_reward: float = 0.25,
        graph_logic_answer_reward: float = 0.5,
        graph_logic_legal_reward: float = 0.0,
        graph_logic_invalid_penalty: float = 0.2,
        graph_logic_fact_dump_penalty: float = 0.0,
        graph_logic_final_gate_eta: float = 0.2,
        graph_logic_correct_process_weight: Optional[float] = None,
        graph_logic_wrong_process_weight: Optional[float] = None,
        graph_logic_correct_only_positive: bool = False,
        graph_logic_keep_wrong_negative: bool = True,
        graph_logic_unit_trust_filter: bool = False,
        graph_logic_unit_trust_margin: float = 0.05,
        graph_logic_unit_trust_min_task_samples: int = 8,
        graph_logic_unit_trust_min_correct_hits: int = 1,
        graph_logic_unit_trust_low_sample_keep: bool = True,
        graph_logic_unit_trust_keep_unknown_positive: bool = False,
        graph_logic_process_task_blocklist: str = "",
        graph_logic_good_prefix_preserve: bool = False,
        graph_logic_good_prefix_wrong_weight: float = 0.25,
        graph_logic_good_prefix_wrong_cap: float = 0.0,
        graph_logic_token_mapping: str = "exact",
        graph_logic_segment_merge_strategy: str = "none",
        graph_logic_segment_merge_target_count: int = 0,
        graph_logic_segment_merge_max_chars: int = 1200,
        graph_logic_segment_merge_min_info_score: float = 0.0,
        graph_logic_reward_mode: str = "event",
        graph_logic_potential_scale: float = 1.0,
        graph_logic_state_shaping: bool = False,
        graph_logic_state_reward_scale: float = 1.0,
        graph_logic_state_tasks: str = "",
        graph_logic_state_calibrate_potential: bool = True,
        graph_logic_state_invalid_penalty: float = 0.03,
        graph_logic_state_mention_reward: float = 0.005,
        graph_logic_state_consistent_reward: float = 0.02,
        graph_logic_duplicate_no_progress_penalty: float = 0.002,
        graph_logic_state_context_max_age: int = 2,
        graph_logic_positive_reward_cap: float = 0.25,
        graph_logic_positive_reward_budget_mode: str = "quality",
        graph_logic_final_answer_checks: bool = False,
        graph_logic_step_span_mode: str = "segment",
        graph_logic_action_span_max_per_segment: int = 6,
        graph_logic_action_span_context_chars: int = 0,
        graph_logic_state_candidate_grounding: bool = True,
        graph_logic_state_candidate_max_edges: int = 3,
        graph_logic_state_commitment_weight: bool = True,
        graph_logic_state_commitment_min_multiplier: float = 0.75,
        graph_logic_state_commitment_max_multiplier: float = 1.20,
        graph_logic_executable_commitment_weight: bool = False,
        graph_logic_ambiguity_gate_positive: bool = False,
        graph_logic_ambiguity_gate_threshold: float = 0.50,
        graph_logic_ambiguity_gate_multiplier: float = 0.0,
        graph_logic_certificate_only_hard_gate: bool = False,
        graph_logic_certificate_gate_task_allowlist: str = "",
        graph_logic_certificate_gate_min_exec_score: float = 0.60,
        graph_logic_certificate_gate_max_ambiguity: float = 0.35,
        graph_logic_graph_sensitivity_gate_positive: bool = False,
        graph_logic_graph_sensitivity_gate_task_allowlist: str = "",
        graph_logic_graph_sensitivity_gate_multiplier: float = 0.25,
        graph_logic_component_precision_gate: bool = False,
        graph_logic_component_precision_filter_units: bool = False,
        graph_logic_component_precision_task_allowlist: str = "",
        graph_logic_component_vague_potential_multiplier: float = 0.20,
        graph_logic_component_noisy_positive_multiplier: float = 0.0,
        graph_logic_component_schema_parseable_multiplier: float = 0.05,
        graph_logic_oracle_milestones: bool = False,
        graph_logic_oracle_milestone_max_units: int = 96,
        graph_logic_oracle_milestone_positive_only: bool = False,
        graph_logic_oracle_milestone_count_actions: bool = True,
        graph_logic_oracle_milestone_task_allowlist: str = "",
        graph_logic_action_router_enable: bool = False,
        graph_logic_action_router_mode: str = "shadow",
        graph_logic_action_router_task_allowlist: str = "",
        graph_logic_action_router_conf_threshold: float = 0.75,
        graph_logic_action_router_high_conf_threshold: float = 0.90,
        graph_logic_action_router_positive_cap: float = 0.05,
        graph_logic_action_router_negative: bool = False,
        graph_logic_action_router_max_candidates: int = 16,
        graph_logic_action_router_backend: str = "heuristic",
        graph_logic_action_router_base_url: str = "",
        graph_logic_action_router_model: str = "qwen3-0.6b-action-router",
        graph_logic_action_router_timeout: float = 2.0,
        graph_logic_action_router_max_tokens: int = 96,
        graph_logic_action_router_heuristic_fallback: bool = False,
        graph_logic_action_router_heuristic_fallback_threshold: float = 0.75,
        graph_logic_action_router_trace: bool = True,
        graph_logic_action_router_payload_trace: bool = False,
        graph_logic_schema_conflict_gate_positive: bool = True,
        graph_logic_schema_conflict_penalty_scale: float = 0.50,
        graph_logic_legacy_core4_compat: bool = False,
        graph_logic_prefix_credit: bool = False,
        graph_logic_prefix_credit_window: int = 256,
        graph_logic_prefix_credit_decay: float = 0.99,
        graph_logic_prefix_credit_local_alpha: float = 0.6,
        graph_logic_prefix_credit_negative: bool = False,
        graph_logic_family_balance: bool = False,
        graph_logic_family_scale_csv: str = "",
        graph_logic_family_scale_default: float = 1.0,
        first_occurrence_only: bool = True,
        extract_backend: Optional[str] = None,
        extract_workers: Optional[int] = None,
        print_timing: bool = True,
    ) -> None:
        if coverage_mode not in ("prefix", "prefix_decay", "span", "segment"):
            raise ValueError(
                f"coverage_mode must be 'prefix'/'prefix_decay'/'span'/'segment', got {coverage_mode!r}"
            )
        scorer = str(scorer or "answer").lower()
        if scorer not in ("answer", "algorithmic", "algorithmic_relative", "graph_logic"):
            raise ValueError(
                f"scorer must be 'answer'/'algorithmic'/'algorithmic_relative'/'graph_logic', got {scorer!r}"
            )
        if scorer == "graph_logic":
            coverage_mode = "segment"
        graph_logic_positive_reward_budget_mode = str(graph_logic_positive_reward_budget_mode or "quality").lower()
        if graph_logic_positive_reward_budget_mode not in ("quality", "net_progress"):
            raise ValueError(
                "graph_logic_positive_reward_budget_mode must be 'quality' or 'net_progress', "
                f"got {graph_logic_positive_reward_budget_mode!r}"
            )
        self.tokenizer = tokenizer
        self.graph_rm = graph_reward_manager
        self.max_entities = max(1, int(max_entities))
        self.reward_clip = float(reward_clip)
        self.positive_reward = float(positive_reward)
        self.focus_reward = float(focus_reward)
        self.irrelevant_node_reward = float(irrelevant_node_reward)
        self.irrelevant_edge_reward = float(irrelevant_edge_reward)
        self.invalid_edge_reward = float(invalid_edge_reward)
        self.order_hit_reward = float(order_hit_reward)
        self.order_seen_reward = float(order_seen_reward)
        self.order_skip_reward = float(order_skip_reward)
        self.coverage_mode = coverage_mode
        self.prefix_decay = max(1e-6, min(1.0, float(prefix_decay)))
        self.scorer = scorer
        self.relative_alpha = max(0.0, float(relative_alpha))
        self.relative_min_position_weight = max(0.0, min(1.0, float(relative_min_position_weight)))
        self.relative_final_gate_eta = max(0.0, min(1.0, float(relative_final_gate_eta)))
        self.graph_logic_config = GraphLogicConfig(
            ordered_reward=float(graph_logic_ordered_reward),
            progress_reward=float(graph_logic_progress_reward),
            answer_reward=float(graph_logic_answer_reward),
            legal_reward=float(graph_logic_legal_reward),
            invalid_penalty=max(0.0, float(graph_logic_invalid_penalty)),
            fact_dump_penalty=max(0.0, float(graph_logic_fact_dump_penalty)),
            max_steps=self.max_entities,
            segment_merge_strategy=str(graph_logic_segment_merge_strategy or "none").lower(),
            segment_merge_target_count=max(0, int(graph_logic_segment_merge_target_count)),
            segment_merge_max_chars=max(0, int(graph_logic_segment_merge_max_chars)),
            segment_merge_min_info_score=max(0.0, float(graph_logic_segment_merge_min_info_score)),
            reward_mode=str(graph_logic_reward_mode or "event").lower(),
            potential_scale=float(graph_logic_potential_scale),
            state_shaping=bool(graph_logic_state_shaping),
            state_reward_scale=float(graph_logic_state_reward_scale),
            state_tasks=str(graph_logic_state_tasks or ""),
            state_calibrate_potential=bool(graph_logic_state_calibrate_potential),
            state_invalid_penalty=max(0.0, float(graph_logic_state_invalid_penalty)),
            state_mention_reward=max(0.0, float(graph_logic_state_mention_reward)),
            state_consistent_reward=max(0.0, float(graph_logic_state_consistent_reward)),
            duplicate_no_progress_penalty=max(0.0, float(graph_logic_duplicate_no_progress_penalty)),
            state_context_max_age=max(-1, int(graph_logic_state_context_max_age)),
            positive_reward_cap=max(0.0, float(graph_logic_positive_reward_cap)),
            positive_reward_budget_mode=graph_logic_positive_reward_budget_mode,
            final_answer_checks=bool(graph_logic_final_answer_checks),
            step_span_mode=str(graph_logic_step_span_mode or "segment").lower(),
            action_span_max_per_segment=max(1, int(graph_logic_action_span_max_per_segment)),
            action_span_context_chars=max(0, int(graph_logic_action_span_context_chars)),
            state_candidate_grounding=bool(graph_logic_state_candidate_grounding),
            state_candidate_max_edges=max(1, int(graph_logic_state_candidate_max_edges)),
            state_commitment_weight=bool(graph_logic_state_commitment_weight),
            state_commitment_min_multiplier=max(0.0, float(graph_logic_state_commitment_min_multiplier)),
            state_commitment_max_multiplier=max(0.0, float(graph_logic_state_commitment_max_multiplier)),
            executable_commitment_weight=bool(graph_logic_executable_commitment_weight),
            ambiguity_gate_positive=bool(graph_logic_ambiguity_gate_positive),
            ambiguity_gate_threshold=max(0.0, min(1.0, float(graph_logic_ambiguity_gate_threshold))),
            ambiguity_gate_multiplier=max(0.0, min(1.0, float(graph_logic_ambiguity_gate_multiplier))),
            certificate_only_hard_gate=bool(graph_logic_certificate_only_hard_gate),
            certificate_gate_task_allowlist=str(graph_logic_certificate_gate_task_allowlist or ""),
            certificate_gate_min_exec_score=max(0.0, min(1.0, float(graph_logic_certificate_gate_min_exec_score))),
            certificate_gate_max_ambiguity=max(0.0, min(1.0, float(graph_logic_certificate_gate_max_ambiguity))),
            graph_sensitivity_gate_positive=bool(graph_logic_graph_sensitivity_gate_positive),
            graph_sensitivity_gate_task_allowlist=str(graph_logic_graph_sensitivity_gate_task_allowlist or ""),
            graph_sensitivity_gate_multiplier=max(
                0.0, min(1.0, float(graph_logic_graph_sensitivity_gate_multiplier))
            ),
            component_precision_gate=bool(graph_logic_component_precision_gate),
            component_precision_filter_units=bool(graph_logic_component_precision_filter_units),
            component_precision_task_allowlist=str(graph_logic_component_precision_task_allowlist or ""),
            component_vague_potential_multiplier=max(
                0.0, min(1.0, float(graph_logic_component_vague_potential_multiplier))
            ),
            component_noisy_positive_multiplier=max(
                0.0, min(1.0, float(graph_logic_component_noisy_positive_multiplier))
            ),
            component_schema_parseable_multiplier=max(
                0.0, min(1.0, float(graph_logic_component_schema_parseable_multiplier))
            ),
            oracle_milestones=bool(graph_logic_oracle_milestones),
            oracle_milestone_max_units=max(0, int(graph_logic_oracle_milestone_max_units)),
            oracle_milestone_positive_only=bool(graph_logic_oracle_milestone_positive_only),
            oracle_milestone_count_actions=bool(graph_logic_oracle_milestone_count_actions),
            oracle_milestone_task_allowlist=str(graph_logic_oracle_milestone_task_allowlist or ""),
            action_router_enable=bool(graph_logic_action_router_enable),
            action_router_mode=str(graph_logic_action_router_mode or "shadow"),
            action_router_task_allowlist=str(graph_logic_action_router_task_allowlist or ""),
            action_router_conf_threshold=max(0.0, min(1.0, float(graph_logic_action_router_conf_threshold))),
            action_router_high_conf_threshold=max(
                0.0, min(1.0, float(graph_logic_action_router_high_conf_threshold))
            ),
            action_router_positive_cap=max(0.0, float(graph_logic_action_router_positive_cap)),
            action_router_negative=bool(graph_logic_action_router_negative),
            action_router_max_candidates=max(2, int(graph_logic_action_router_max_candidates)),
            action_router_backend=str(graph_logic_action_router_backend or "heuristic"),
            action_router_base_url=str(graph_logic_action_router_base_url or ""),
            action_router_model=str(graph_logic_action_router_model or "qwen3-0.6b-action-router"),
            action_router_timeout=max(0.1, float(graph_logic_action_router_timeout)),
            action_router_max_tokens=max(8, int(graph_logic_action_router_max_tokens)),
            action_router_heuristic_fallback=bool(graph_logic_action_router_heuristic_fallback),
            action_router_heuristic_fallback_threshold=max(
                0.0, min(1.0, float(graph_logic_action_router_heuristic_fallback_threshold))
            ),
            action_router_trace=bool(graph_logic_action_router_trace),
            action_router_payload_trace=bool(graph_logic_action_router_payload_trace),
            schema_conflict_gate_positive=bool(graph_logic_schema_conflict_gate_positive),
            schema_conflict_penalty_scale=max(0.0, float(graph_logic_schema_conflict_penalty_scale)),
            legacy_core4_compat=bool(graph_logic_legacy_core4_compat),
        )
        self.graph_logic_prefix_credit = bool(graph_logic_prefix_credit)
        self.graph_logic_prefix_credit_window = max(0, int(graph_logic_prefix_credit_window))
        self.graph_logic_prefix_credit_decay = max(1e-6, min(1.0, float(graph_logic_prefix_credit_decay)))
        self.graph_logic_prefix_credit_local_alpha = max(
            0.0, min(1.0, float(graph_logic_prefix_credit_local_alpha))
        )
        self.graph_logic_prefix_credit_negative = bool(graph_logic_prefix_credit_negative)
        self.graph_logic_family_balance = bool(graph_logic_family_balance)
        self.graph_logic_family_scale_default = max(0.0, min(3.0, float(graph_logic_family_scale_default)))
        self.graph_logic_family_scales = _parse_graph_logic_family_scales(
            graph_logic_family_scale_csv, self.graph_logic_family_scale_default
        )
        self.graph_logic_final_gate_eta = max(0.0, min(1.0, float(graph_logic_final_gate_eta)))
        self.graph_logic_correct_process_weight = (
            1.0
            if graph_logic_correct_process_weight is None
            else max(0.0, float(graph_logic_correct_process_weight))
        )
        self.graph_logic_wrong_process_weight = (
            self.graph_logic_final_gate_eta
            if graph_logic_wrong_process_weight is None
            else max(0.0, float(graph_logic_wrong_process_weight))
        )
        self.graph_logic_correct_only_positive = bool(graph_logic_correct_only_positive)
        self.graph_logic_legacy_core4_compat = bool(graph_logic_legacy_core4_compat)
        self.graph_logic_keep_wrong_negative = bool(graph_logic_keep_wrong_negative)
        self.graph_logic_unit_trust_filter = bool(graph_logic_unit_trust_filter)
        self.graph_logic_unit_trust_margin = float(graph_logic_unit_trust_margin)
        self.graph_logic_unit_trust_min_task_samples = max(1, int(graph_logic_unit_trust_min_task_samples))
        self.graph_logic_unit_trust_min_correct_hits = max(1, int(graph_logic_unit_trust_min_correct_hits))
        self.graph_logic_unit_trust_low_sample_keep = bool(graph_logic_unit_trust_low_sample_keep)
        self.graph_logic_unit_trust_keep_unknown_positive = bool(graph_logic_unit_trust_keep_unknown_positive)
        self.graph_logic_process_task_blocklist = _parse_graph_logic_task_set(
            graph_logic_process_task_blocklist
        )
        self.graph_logic_good_prefix_preserve = bool(graph_logic_good_prefix_preserve)
        self.graph_logic_good_prefix_wrong_weight = max(
            0.0, min(1.0, float(graph_logic_good_prefix_wrong_weight))
        )
        self.graph_logic_good_prefix_wrong_cap = max(0.0, float(graph_logic_good_prefix_wrong_cap))
        graph_logic_token_mapping = str(graph_logic_token_mapping or "exact").lower()
        if graph_logic_token_mapping not in ("exact", "proportional"):
            raise ValueError(
                f"graph_logic_token_mapping must be 'exact'/'proportional', got {graph_logic_token_mapping!r}"
            )
        self.graph_logic_token_mapping = graph_logic_token_mapping
        self.first_occurrence_only = bool(first_occurrence_only)
        self.extract_backend = (extract_backend or os.environ.get("GRAPH_PROGRESS_EXTRACT_BACKEND", "process")).lower()
        self.extract_workers = int(extract_workers) if extract_workers is not None else _detect_worker_count()
        self.extract_workers = max(1, min(64, self.extract_workers))
        self.print_timing = bool(print_timing)
        self.last_stats: dict = {}

    def _target_positions(self, token_span: list[int]):
        if self.coverage_mode == "prefix":
            return ("prefix", int(token_span[-1]))
        if self.coverage_mode == "prefix_decay":
            return ("decay_prefix", int(token_span[-1]), float(self.prefix_decay))
        return tuple(int(x) for x in token_span)

    def _graph_logic_target_position(self, token_range, reward: float):
        if (
            not self.graph_logic_prefix_credit
            or self.graph_logic_prefix_credit_window <= 0
            or (reward < 0.0 and not self.graph_logic_prefix_credit_negative)
            or not isinstance(token_range, (list, tuple))
            or len(token_range) < 3
            or str(token_range[0]).lower() != "range"
        ):
            return token_range
        action_start = int(token_range[1])
        action_end = int(token_range[2])
        if action_end < action_start:
            return token_range
        prefix_start = max(0, action_start - self.graph_logic_prefix_credit_window)
        if prefix_start >= action_start:
            return token_range
        return (
            "backward_decay_range",
            int(prefix_start),
            int(action_start),
            int(action_end),
            float(self.graph_logic_prefix_credit_decay),
            float(self.graph_logic_prefix_credit_local_alpha),
        )

    @staticmethod
    def _char_span_to_token_range(
        char_span: tuple[int, int],
        starts: list[int],
        ends: list[int],
        token_position_map: list[list[int]],
        valid_r_len: int,
    ):
        if not starts or not ends or not token_position_map or valid_r_len <= 0:
            return None
        sc, ec = int(char_span[0]), int(char_span[1])
        if ec <= sc:
            return None
        left, right = _bisect_token_range(starts, ends, sc, ec)
        if left < 0:
            return None
        left = max(0, left)
        right = min(len(token_position_map), right)
        original_positions: list[int] = []
        for ti in range(left, right):
            original_positions.extend(int(p) for p in token_position_map[ti] if 0 <= int(p) < valid_r_len)
        if not original_positions:
            return None
        return ("range", min(original_positions), max(original_positions))

    @staticmethod
    def _char_span_to_proportional_token_range(
        char_span: tuple[int, int],
        text_len: int,
        valid_r_len: int,
    ):
        if text_len <= 0 or valid_r_len <= 0:
            return None
        sc, ec = int(char_span[0]), int(char_span[1])
        if ec <= sc:
            return None
        start = int((max(0, min(sc, text_len)) / float(text_len)) * valid_r_len)
        end = int(np.ceil((max(0, min(ec, text_len)) / float(text_len)) * valid_r_len)) - 1
        start = max(0, min(start, valid_r_len - 1))
        end = max(start, min(end, valid_r_len - 1))
        return ("range", start, end)

    def _clip(self, value: float) -> float:
        c = max(1e-8, self.reward_clip)
        return max(-c, min(c, float(value)))

    def _relative_position_weight(self, token_span: list[int], valid_r_len: int) -> float:
        if valid_r_len <= 0 or not token_span:
            return 1.0
        rel = max(0.0, min(1.0, (float(token_span[-1]) + 1.0) / float(valid_r_len)))
        shaped = rel**self.relative_alpha if self.relative_alpha > 0 else 1.0
        return self.relative_min_position_weight + (1.0 - self.relative_min_position_weight) * shaped

    @staticmethod
    def _relative_state() -> dict:
        return {
            "positive_nodes": set(),
            "positive_edges": set(),
            "soft_nodes": set(),
            "soft_edges": set(),
            "focus_nodes": set(),
            "invalid_nodes": set(),
            "invalid_edges": set(),
            "off_nodes": set(),
            "off_edges": set(),
            "order_pos": 0,
            "seen_order_nodes": set(),
        }

    @staticmethod
    def _coverage(hit: set, target: set) -> float:
        return float(len(hit) / max(1, len(target))) if target else 0.0

    def _relative_potential(
        self,
        state: dict,
        graph_nodes: set[int],
        positive_nodes: set[int],
        positive_edges: set[Tuple[int, int]],
        focus_nodes: set[int],
        soft_nodes: set[int],
        soft_edges: set[Tuple[int, int]],
        oracle_order: list[int],
    ) -> float:
        del graph_nodes
        terms: list[tuple[float, float]] = []
        if oracle_order:
            terms.append((0.45, float(state["order_pos"]) / max(1, len(oracle_order))))
        if positive_nodes:
            terms.append((0.25 if oracle_order else 0.40, self._coverage(state["positive_nodes"], positive_nodes)))
        if positive_edges:
            terms.append((0.20 if oracle_order else 0.35, self._coverage(state["positive_edges"], positive_edges)))
        if soft_nodes:
            terms.append((0.07, self._coverage(state["soft_nodes"], soft_nodes)))
        if soft_edges:
            terms.append((0.08, self._coverage(state["soft_edges"], soft_edges)))
        if focus_nodes:
            terms.append((0.05, self._coverage(state["focus_nodes"], focus_nodes)))

        if not terms:
            base = 0.0
        else:
            weight_sum = max(1e-8, sum(w for w, _ in terms))
            base = sum(w * v for w, v in terms) / weight_sum

        wrong = (
            0.30 * len(state["invalid_edges"])
            + 0.20 * len(state["invalid_nodes"])
            + 0.06 * len(state["off_edges"])
            + 0.03 * len(state["off_nodes"])
        )
        penalty = min(1.0, wrong)
        return max(0.0, min(1.0, base - penalty))

    def _relative_update(
        self,
        state: dict,
        etype: str,
        eid: object,
        graph_nodes: set[int],
        graph_edges: set[Tuple[int, int]],
        directed: bool,
        positive_nodes: set[int],
        positive_edges: set[Tuple[int, int]],
        focus_nodes: set[int],
        soft_nodes: set[int],
        soft_edges: set[Tuple[int, int]],
        oracle_order: list[int],
    ) -> object | None:
        if etype == "node":
            node = _as_int(eid)
            if node is None:
                return None
            if oracle_order and state["order_pos"] < len(oracle_order) and node == oracle_order[state["order_pos"]]:
                state["order_pos"] += 1
                state["seen_order_nodes"].add(node)
            if node in positive_nodes:
                state["positive_nodes"].add(node)
            elif node in soft_nodes:
                state["soft_nodes"].add(node)
            elif node in focus_nodes:
                state["focus_nodes"].add(node)
            elif node not in graph_nodes:
                state["invalid_nodes"].add(node)
            else:
                state["off_nodes"].add(node)
            return ("graph_progress_relative", "node", node)

        edge = _norm_edge(eid, directed=directed)
        if edge is None:
            return None
        if edge in positive_edges:
            state["positive_edges"].add(edge)
        elif edge in soft_edges:
            state["soft_edges"].add(edge)
        elif edge not in graph_edges:
            state["invalid_edges"].add(edge)
        else:
            state["off_edges"].add(edge)
        return ("graph_progress_relative", "edge", edge)

    def compute(self, data: DataProto, ref_policy_wg=None, outcome_scores: Optional[torch.Tensor] = None) -> dict:
        del ref_policy_wg
        batch = data.batch
        for required in ("responses", "prompts", "attention_mask"):
            if required not in batch:
                raise KeyError(f"GraphProgressRewardComputer requires '{required}'")

        responses = batch["responses"]
        prompts = batch["prompts"]
        attn = batch["attention_mask"]
        B, R = responses.shape
        P = prompts.shape[-1]

        ntb = data.non_tensor_batch if data.non_tensor_batch is not None else {}
        extra_infos = ntb.get("extra_info", [None] * B)
        reward_models = ntb.get("reward_model", [None] * B)
        segment_merge_strategy = str(getattr(self.graph_logic_config, "segment_merge_strategy", "none") or "none").lower()
        needs_model_entropy_merge = (
            self.scorer == "graph_logic"
            and segment_merge_strategy in {"model_entropy", "token_entropy", "logit_entropy", "logits_entropy"}
        )
        response_token_entropys = batch.get("response_token_entropys", None)
        if needs_model_entropy_merge and response_token_entropys is None:
            raise RuntimeError(
                "graph_logic_segment_merge_strategy=model_entropy requires "
                "batch['response_token_entropys']; enable the pre-reward actor entropy path."
            )

        pure_step_reward = torch.zeros((B, R), dtype=torch.float32)
        per_sample_rewards: List[List[float]] = [[] for _ in range(B)]
        per_sample_anchors: List[List[object]] = [[] for _ in range(B)]
        per_sample_token_pos: List[List[object]] = [[] for _ in range(B)]
        per_sample_reward_sum = np.zeros(B, dtype=np.float64)
        per_sample_reward_count = np.zeros(B, dtype=np.int64)

        valid_samples = 0
        unsupported = 0
        algorithmic_supported = 0
        algorithmic_soft_only_supported = 0
        graph_logic_supported = 0
        answer_fallback = 0
        entity_total = 0
        positive_count = 0
        negative_count = 0
        zero_count = 0
        reward_values: list[float] = []
        relative_phi_values: list[float] = []
        relative_delta_values: list[float] = []
        relative_gate_values: list[float] = []
        graph_logic_segments = 0.0
        graph_logic_segment_initial_segments = 0.0
        graph_logic_segment_merge_enabled = 0.0
        graph_logic_segment_merge_count = 0.0
        graph_logic_segment_merge_target_count = 0.0
        graph_logic_segment_merge_model_entropy = 0.0
        graph_logic_segment_merge_initial_info_sum = 0.0
        graph_logic_segment_merge_final_info_sum = 0.0
        graph_logic_segment_merge_selected_info_sum = 0.0
        graph_logic_grounded = 0.0
        graph_logic_candidate = 0.0
        graph_logic_executor_valid = 0.0
        graph_logic_progressive = 0.0
        graph_logic_ordered = 0.0
        graph_logic_answer_hit = 0.0
        graph_logic_fact_dump = 0.0
        graph_logic_invalid_edges = 0.0
        graph_logic_suppressed_invalid_edges = 0.0
        graph_logic_node_only_progress = 0.0
        graph_logic_potential_delta_raw = 0.0
        graph_logic_potential_delta = 0.0
        graph_logic_potential_positive_steps = 0.0
        graph_logic_potential_evidence_units = 0.0
        graph_logic_potential_evidence_steps = 0.0
        graph_logic_potential_task_evidence_units = 0.0
        graph_logic_potential_task_evidence_steps = 0.0
        graph_logic_state_supported = 0.0
        graph_logic_process_blocked = 0.0
        graph_logic_state_supported_segments = 0.0
        graph_logic_state_actions = 0.0
        graph_logic_state_valid_transitions = 0.0
        graph_logic_state_invalid_transitions = 0.0
        graph_logic_state_progressive = 0.0
        graph_logic_state_raw_delta = 0.0
        graph_logic_state_delta = 0.0
        graph_logic_state_delta_quality = 0.0
        graph_logic_state_quality_segments = 0.0
        graph_logic_state_new_units = 0.0
        graph_logic_state_weighted_new_units = 0.0
        graph_logic_state_rewardable_new_units = 0.0
        graph_logic_state_rewardable_weighted_new_units = 0.0
        graph_logic_state_weak_new_units = 0.0
        graph_logic_state_rewardable_transitions = 0.0
        graph_logic_state_weak_only_transition_segments = 0.0
        graph_logic_state_terminal_units = 0.0
        graph_logic_state_completion_invalid = 0.0
        graph_logic_state_raw_segments = 0.0
        graph_logic_state_candidate_segments = 0.0
        graph_logic_state_not_candidate_segments = 0.0
        graph_logic_state_fact_dump_segments = 0.0
        graph_logic_state_not_grounded_segments = 0.0
        graph_logic_state_mentions = 0.0
        graph_logic_state_mention_grounded_segments = 0.0
        graph_logic_state_mention_only_segments = 0.0
        graph_logic_state_consistent_segments = 0.0
        graph_logic_state_reward_consistent_segments = 0.0
        graph_logic_state_context_grounded_segments = 0.0
        graph_logic_state_formula_grounded_segments = 0.0
        graph_logic_state_context_edges = 0.0
        graph_logic_state_context_nodes = 0.0
        graph_logic_state_candidate_grounded_segments = 0.0
        graph_logic_state_commitment_weighted_segments = 0.0
        graph_logic_state_commitment_score = 0.0
        graph_logic_state_commitment_multiplier = 0.0
        graph_logic_state_ambiguity_scored_segments = 0.0
        graph_logic_state_ambiguity_score = 0.0
        graph_logic_state_executable_commitment_score = 0.0
        graph_logic_state_hard_cert_segments = 0.0
        graph_logic_state_ambiguity_gate_segments = 0.0
        graph_logic_state_ambiguity_gate_multiplier = 0.0
        graph_logic_state_certificate_gate_segments = 0.0
        graph_logic_state_certificate_gate_pass_segments = 0.0
        graph_logic_state_certificate_gate_blocked_segments = 0.0
        graph_logic_component_precision_gate_segments = 0.0
        graph_logic_component_precision_gate_verified_segments = 0.0
        graph_logic_component_precision_gate_vague_segments = 0.0
        graph_logic_component_precision_gate_noisy_segments = 0.0
        graph_logic_component_precision_schema_scalar_scaled = 0.0
        graph_logic_component_precision_schema_object_scaled = 0.0
        graph_logic_component_precision_schema_scalar_blocked = 0.0
        graph_logic_component_precision_schema_object_blocked = 0.0
        graph_logic_component_precision_schema_multiplier_sum = 0.0
        graph_logic_component_precision_gate_vague_multiplier = 0.0
        graph_logic_component_precision_gate_noisy_multiplier = 0.0
        graph_logic_component_precision_unit_filter_segments = 0.0
        graph_logic_component_precision_unit_filter_drop_segments = 0.0
        graph_logic_component_precision_unit_filter_input_units = 0.0
        graph_logic_component_precision_unit_filter_kept_units = 0.0
        graph_logic_component_precision_unit_filter_dropped_units = 0.0
        graph_logic_correct_only_blocked_positive = 0.0
        graph_logic_correct_only_blocked_positive_reward = 0.0
        graph_logic_wrong_negative_kept = 0.0
        graph_logic_wrong_negative_blocked = 0.0
        graph_logic_unit_trust_filter_positive = 0.0
        graph_logic_unit_trust_kept_positive = 0.0
        graph_logic_unit_trust_blocked_positive = 0.0
        graph_logic_unit_trust_unknown_positive = 0.0
        graph_logic_unit_trust_low_sample_keep = 0.0
        graph_logic_unit_trust_score_sum = 0.0
        graph_logic_state_action_segments = 0.0
        graph_logic_state_no_action_segments = 0.0
        graph_logic_state_invalid_action_segments = 0.0
        graph_logic_state_duplicate_no_progress_segments = 0.0
        graph_logic_state_valid_zero_delta_segments = 0.0
        graph_logic_state_action_no_progress_segments = 0.0
        graph_logic_state_no_valid_units_segments = 0.0
        graph_logic_state_units = 0.0
        graph_logic_state_duplicate_units = 0.0
        graph_logic_action_span_rewarded_segments = 0.0
        graph_logic_action_span_emitted = 0.0
        graph_logic_action_span_split_segments = 0.0
        graph_logic_action_span_fallback_segments = 0.0
        graph_logic_action_event_rewarded_segments = 0.0
        graph_logic_action_event_bound_segments = 0.0
        graph_logic_action_event_bound_spans = 0.0
        graph_logic_action_router_segments = 0.0
        graph_logic_action_router_candidates = 0.0
        graph_logic_action_router_no_candidate_segments = 0.0
        graph_logic_action_router_selected_segments = 0.0
        graph_logic_action_router_abstain_segments = 0.0
        graph_logic_action_router_ambiguous_segments = 0.0
        graph_logic_action_router_confident_segments = 0.0
        graph_logic_action_router_grounded_segments = 0.0
        graph_logic_action_router_accepted_segments = 0.0
        graph_logic_action_router_invalid_selection_segments = 0.0
        graph_logic_action_router_http_error_segments = 0.0
        graph_logic_action_router_http_missing_endpoint_segments = 0.0
        graph_logic_action_router_confidence = 0.0
        graph_logic_action_router_positive_bonus = 0.0
        graph_logic_action_router_positive_bonus_segments = 0.0
        graph_logic_action_router_source_http = 0.0
        graph_logic_action_router_source_http_error = 0.0
        graph_logic_action_router_source_heuristic = 0.0
        graph_logic_action_router_source_http_heuristic_fallback = 0.0
        graph_logic_state_component_scores = {component: 0.0 for component in STATE_PROGRESS_COMPONENTS}
        graph_logic_state_component_deltas = {component: 0.0 for component in STATE_PROGRESS_COMPONENTS}
        graph_logic_potential_final_values: list[float] = []
        graph_logic_ordered_recall_values: list[float] = []
        graph_logic_gate_values: list[float] = []
        graph_logic_label_counts = defaultdict(float)
        graph_logic_good_prefix_preserved_positive = 0.0
        graph_logic_good_prefix_blocked_positive = 0.0
        graph_logic_good_prefix_first_error_steps = 0.0
        graph_logic_good_prefix_cap_exhausted = 0.0
        graph_logic_family_sample_count = defaultdict(float)
        graph_logic_family_event_count = defaultdict(float)
        graph_logic_family_scale_sum = defaultdict(float)
        graph_logic_family_reward_abs_before = defaultdict(float)
        graph_logic_family_reward_abs_after = defaultdict(float)
        graph_logic_family_reward_signed_before = defaultdict(float)
        graph_logic_family_reward_signed_after = defaultdict(float)

        t_prepare_start = _time.time()
        valid_p_lens = attn[:, :P].sum(dim=1).cpu().tolist()
        valid_r_lens = attn[:, P:].sum(dim=1).cpu().tolist()
        responses_cpu = responses.cpu()

        sample_infos: dict[int, tuple] = {}
        logic_specs: dict[int, Any] = {}
        valid_indices: List[int] = []
        valid_response_id_lists: List[torch.Tensor] = []
        for i in range(B):
            extra = extra_infos[i] if i < len(extra_infos) else None
            rm = reward_models[i] if i < len(reward_models) else None
            if not isinstance(extra, dict) or not isinstance(rm, dict):
                unsupported += 1
                continue

            valid_p_len = int(valid_p_lens[i])
            valid_r_len = int(valid_r_lens[i])
            if valid_p_len == 0 or valid_r_len == 0:
                continue

            task = str(extra.get("task", "")).lower()
            nodes, graph_edges, directed, adj = _parse_graph(extra)
            answer = _literal(rm.get("ground_truth"), default=rm.get("ground_truth"))
            sample = _literal(extra.get("sample"), default=None)
            if self.scorer == "graph_logic":
                if (
                    task in self.graph_logic_process_task_blocklist
                    or "all" in self.graph_logic_process_task_blocklist
                    or "*" in self.graph_logic_process_task_blocklist
                ):
                    graph_logic_process_blocked += 1.0
                    continue
                spec = build_graph_logic_spec(task, extra, answer, sample)
                if spec is None or not (
                    spec.target_edges
                    or spec.target_nodes
                    or spec.answer_numbers
                    or spec.answer_bool is not None
                ):
                    unsupported += 1
                    continue
                logic_specs[i] = spec
                graph_logic_supported += 1
                valid_indices.append(i)
                valid_response_id_lists.append(responses_cpu[i, :valid_r_len])
                continue

            soft_nodes: set[int] = set()
            soft_edges: set[Tuple[int, int]] = set()
            used_algorithmic = False
            if self.scorer in ("algorithmic", "algorithmic_relative"):
                (
                    positive_nodes,
                    positive_edges,
                    focus_nodes,
                    soft_nodes,
                    soft_edges,
                    oracle_order,
                    used_algorithmic,
                ) = _algorithmic_entities(task, answer, sample, nodes, graph_edges, directed, adj, extra)
            else:
                positive_nodes, positive_edges, focus_nodes, oracle_order = _answer_entities(
                    task, answer, sample, nodes, graph_edges, directed, adj
                )
            if self.scorer == "algorithmic_relative" and not used_algorithmic:
                unsupported += 1
                continue
            has_hard_signal = bool(positive_nodes or positive_edges or focus_nodes or oracle_order)
            has_soft_signal = bool(soft_nodes or soft_edges)
            if not (has_hard_signal or has_soft_signal):
                unsupported += 1
                continue
            if self.scorer in ("algorithmic", "algorithmic_relative"):
                if used_algorithmic:
                    algorithmic_supported += 1
                    if has_soft_signal and not has_hard_signal:
                        algorithmic_soft_only_supported += 1
                else:
                    answer_fallback += 1

            sample_infos[i] = (
                task,
                nodes,
                graph_edges,
                directed,
                positive_nodes,
                positive_edges,
                focus_nodes,
                soft_nodes,
                soft_edges,
                oracle_order,
            )
            valid_indices.append(i)
            valid_response_id_lists.append(responses_cpu[i, :valid_r_len])

        t_prepare = _time.time() - t_prepare_start

        t_decode_start = _time.time()
        response_strs = (
            self.tokenizer.batch_decode(valid_response_id_lists, skip_special_tokens=True) if valid_indices else []
        )
        t_decode = _time.time() - t_decode_start

        t_encode_start = _time.time()
        retok_ids_list: List[List[int]] = []
        offset_maps: List[List[tuple]] = []
        need_offset_mapping = (
            self.scorer != "graph_logic"
            or self.graph_logic_token_mapping == "exact"
            or needs_model_entropy_merge
        )
        if response_strs and need_offset_mapping:
            try:
                batch_enc = self.tokenizer(
                    response_strs,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    padding=False,
                )
                retok_ids_list = [list(map(int, ids)) for ids in batch_enc["input_ids"]]
                offset_maps = [list(om) for om in batch_enc["offset_mapping"]]
            except Exception as e:
                print(f"[GraphProgress] batch offset encode failed ({type(e).__name__}: {e}); using slow span fallback")
                retok_ids_list = []
                offset_maps = []
        t_encode = _time.time() - t_encode_start

        model_entropy_values_list: List[List[float]] = []
        if needs_model_entropy_merge:
            try:
                entropy_cpu = response_token_entropys.detach().float().cpu()
            except Exception as e:
                raise RuntimeError(f"response_token_entropys must be a tensor for model_entropy merge: {e}") from e
            for sample_idx in valid_indices:
                valid_r_len = int(valid_r_lens[sample_idx])
                model_entropy_values_list.append(entropy_cpu[sample_idx, :valid_r_len].tolist())

        _WORKER_CTX["graph_rm"] = self.graph_rm
        _WORKER_CTX["response_strs"] = response_strs
        _WORKER_CTX["response_ids_list"] = valid_response_id_lists
        _WORKER_CTX["retok_ids_list"] = retok_ids_list
        _WORKER_CTX["offset_maps"] = offset_maps
        _WORKER_CTX["model_entropy_offset_maps"] = offset_maps if needs_model_entropy_merge else []
        _WORKER_CTX["model_entropy_values_list"] = model_entropy_values_list if needs_model_entropy_merge else []
        _WORKER_CTX["max_entities"] = self.max_entities
        _WORKER_CTX["first_occurrence_only"] = self.first_occurrence_only
        _WORKER_CTX["logic_specs"] = logic_specs
        _WORKER_CTX["graph_logic_config"] = self.graph_logic_config

        per_sample_flat: dict[int, list] = {}
        logic_observations: dict[int, tuple[list, dict]] = {}
        backend_used = "none"
        t_extract_start = _time.time()
        if valid_indices and self.scorer != "graph_logic":
            tasks = list(enumerate(valid_indices))
            backend = self.extract_backend
            if backend == "process":
                try:
                    fork_ctx = _mp.get_context("fork")
                    chunksize = max(1, len(tasks) // (self.extract_workers * 4))
                    with ProcessPoolExecutor(max_workers=self.extract_workers, mp_context=fork_ctx) as pool:
                        for _, sample_idx, flat in pool.map(
                            _extract_graph_progress_worker,
                            tasks,
                            chunksize=chunksize,
                        ):
                            per_sample_flat[sample_idx] = flat
                    backend_used = f"process({self.extract_workers})"
                except Exception as e:
                    print(f"[GraphProgress] ProcessPool failed ({type(e).__name__}: {e}); falling back to ThreadPool")
                    backend = "thread"

            if backend == "thread":
                with ThreadPoolExecutor(max_workers=self.extract_workers) as pool:
                    for _, sample_idx, flat in pool.map(_extract_graph_progress_worker, tasks):
                        per_sample_flat[sample_idx] = flat
                backend_used = f"thread({self.extract_workers})"
            elif backend == "serial":
                for task in tasks:
                    _, sample_idx, flat = _extract_graph_progress_worker(task)
                    per_sample_flat[sample_idx] = flat
                backend_used = "serial"
            elif backend_used == "none":
                for task in tasks:
                    _, sample_idx, flat = _extract_graph_progress_worker(task)
                    per_sample_flat[sample_idx] = flat
                backend_used = f"serial(fallback:{backend})"
        elif valid_indices:
            tasks = list(enumerate(valid_indices))
            backend = self.extract_backend
            if backend == "process":
                try:
                    fork_ctx = _mp.get_context("fork")
                    chunksize = max(1, len(tasks) // (self.extract_workers * 4))
                    with ProcessPoolExecutor(max_workers=self.extract_workers, mp_context=fork_ctx) as pool:
                        for _, sample_idx, steps, stats in pool.map(
                            _observe_graph_logic_worker,
                            tasks,
                            chunksize=chunksize,
                        ):
                            logic_observations[sample_idx] = (steps, stats)
                    backend_used = f"graph_logic_process({self.extract_workers})"
                except Exception as e:
                    print(
                        f"[GraphProgress] GraphLogic ProcessPool failed "
                        f"({type(e).__name__}: {e}); falling back to ThreadPool"
                    )
                    backend = "thread"

            if backend == "thread":
                with ThreadPoolExecutor(max_workers=self.extract_workers) as pool:
                    for _, sample_idx, steps, stats in pool.map(_observe_graph_logic_worker, tasks):
                        logic_observations[sample_idx] = (steps, stats)
                backend_used = f"graph_logic_thread({self.extract_workers})"
            elif backend == "serial":
                for task in tasks:
                    _, sample_idx, steps, stats = _observe_graph_logic_worker(task)
                    logic_observations[sample_idx] = (steps, stats)
                backend_used = "graph_logic_serial"
            elif backend_used == "none":
                for task in tasks:
                    _, sample_idx, steps, stats = _observe_graph_logic_worker(task)
                    logic_observations[sample_idx] = (steps, stats)
                backend_used = f"graph_logic_serial(fallback:{backend})"
        t_extract = _time.time() - t_extract_start
        _WORKER_CTX.clear()
        sample_to_local = {sample_idx: local_idx for local_idx, sample_idx in enumerate(valid_indices)}

        outcome_values = np.ones(B, dtype=np.float64)
        if outcome_scores is not None:
            try:
                oscores = outcome_scores.detach().float().cpu().numpy()
                if oscores.ndim > 1:
                    oscores = oscores.reshape(oscores.shape[0], -1).sum(axis=-1)
                n_os = min(B, len(oscores))
                outcome_values[:n_os] = np.clip(oscores[:n_os], 0.0, 1.0)
            except Exception:
                outcome_values.fill(1.0)

        unit_trust_scores: dict[tuple[str, str], float] = {}
        unit_trust_hits: dict[tuple[str, str], tuple[int, int]] = {}
        unit_trust_task_counts: dict[str, tuple[int, int]] = {}
        unit_trust_enabled = bool(
            self.scorer == "graph_logic"
            and self.graph_logic_unit_trust_filter
            and outcome_scores is not None
        )
        if unit_trust_enabled:
            task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            unit_hits: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
            for sample_idx in valid_indices:
                spec = logic_specs.get(sample_idx)
                if spec is None:
                    continue
                task_name = str(getattr(spec, "task", "") or "").lower()
                correct_idx = 0 if float(outcome_values[sample_idx]) > 0.5 else 1
                task_counts[task_name][correct_idx] += 1
                steps_for_sample, _logic_stats_for_sample = logic_observations.get(sample_idx, ([], {}))
                sample_units: set[str] = set()
                for step in steps_for_sample:
                    if float(getattr(step, "reward", 0.0) or 0.0) <= 0.0:
                        continue
                    sample_units.update(_graph_logic_step_unit_names(step))
                for unit_name in sample_units:
                    unit_hits[(task_name, unit_name)][correct_idx] += 1
            unit_trust_task_counts = {
                task_name: (counts[0], counts[1]) for task_name, counts in task_counts.items()
            }
            for key, hits in unit_hits.items():
                task_name, _unit_name = key
                correct_samples, wrong_samples = unit_trust_task_counts.get(task_name, (0, 0))
                correct_rate = float(hits[0]) / max(1.0, float(correct_samples))
                wrong_rate = float(hits[1]) / max(1.0, float(wrong_samples))
                unit_trust_scores[key] = correct_rate - wrong_rate
                unit_trust_hits[key] = (int(hits[0]), int(hits[1]))

        t_score_start = _time.time()
        for i in valid_indices:
            if self.scorer == "graph_logic":
                local_idx = sample_to_local.get(i)
                spec = logic_specs.get(i)
                if local_idx is None or spec is None:
                    continue
                text = response_strs[local_idx] if local_idx < len(response_strs) else ""
                steps, logic_stats = logic_observations.get(i, ([], {}))
                graph_logic_segments += float(logic_stats.get("segments", 0.0))
                graph_logic_segment_initial_segments += float(logic_stats.get("segment_initial_segments", 0.0))
                graph_logic_segment_merge_enabled += float(logic_stats.get("segment_merge_enabled", 0.0))
                graph_logic_segment_merge_count += float(logic_stats.get("segment_merge_count", 0.0))
                graph_logic_segment_merge_target_count += float(logic_stats.get("segment_merge_target_count", 0.0))
                graph_logic_segment_merge_model_entropy += float(
                    logic_stats.get("segment_merge_model_entropy", 0.0)
                )
                graph_logic_segment_merge_initial_info_sum += float(
                    logic_stats.get("segment_merge_initial_info_sum", 0.0)
                )
                graph_logic_segment_merge_final_info_sum += float(
                    logic_stats.get("segment_merge_final_info_sum", 0.0)
                )
                graph_logic_segment_merge_selected_info_sum += float(
                    logic_stats.get("segment_merge_selected_info_sum", 0.0)
                )
                graph_logic_grounded += float(logic_stats.get("grounded", 0.0))
                graph_logic_candidate += float(logic_stats.get("candidate", 0.0))
                graph_logic_executor_valid += float(logic_stats.get("executor_valid", 0.0))
                graph_logic_progressive += float(logic_stats.get("progressive", 0.0))
                graph_logic_ordered += float(logic_stats.get("ordered", 0.0))
                graph_logic_answer_hit += float(logic_stats.get("answer_hit", 0.0))
                graph_logic_fact_dump += float(logic_stats.get("fact_dump", 0.0))
                graph_logic_invalid_edges += float(logic_stats.get("invalid_edges", 0.0))
                graph_logic_suppressed_invalid_edges += float(logic_stats.get("suppressed_invalid_edges", 0.0))
                graph_logic_node_only_progress += float(logic_stats.get("node_only_progress", 0.0))
                graph_logic_potential_delta_raw += float(logic_stats.get("potential_delta_raw", 0.0))
                graph_logic_potential_delta += float(logic_stats.get("potential_delta", 0.0))
                graph_logic_potential_positive_steps += float(logic_stats.get("potential_positive_steps", 0.0))
                graph_logic_potential_evidence_units += float(logic_stats.get("potential_evidence_units", 0.0))
                graph_logic_potential_evidence_steps += float(logic_stats.get("potential_evidence_steps", 0.0))
                graph_logic_potential_task_evidence_units += float(
                    logic_stats.get("potential_task_evidence_units", 0.0)
                )
                graph_logic_potential_task_evidence_steps += float(
                    logic_stats.get("potential_task_evidence_steps", 0.0)
                )
                graph_logic_state_supported += float(logic_stats.get("state_supported", 0.0))
                graph_logic_state_supported_segments += float(logic_stats.get("state_supported_segments", 0.0))
                graph_logic_state_actions += float(logic_stats.get("state_actions", 0.0))
                graph_logic_state_valid_transitions += float(logic_stats.get("state_valid_transitions", 0.0))
                graph_logic_state_invalid_transitions += float(logic_stats.get("state_invalid_transitions", 0.0))
                graph_logic_state_progressive += float(logic_stats.get("state_progressive", 0.0))
                graph_logic_state_raw_delta += float(logic_stats.get("state_raw_delta", 0.0))
                graph_logic_state_delta += float(logic_stats.get("state_delta", 0.0))
                graph_logic_state_delta_quality += float(logic_stats.get("state_delta_quality", 0.0))
                graph_logic_state_quality_segments += float(logic_stats.get("state_action_segments", 0.0))
                graph_logic_state_new_units += float(logic_stats.get("state_new_units", 0.0))
                graph_logic_state_weighted_new_units += float(logic_stats.get("state_weighted_new_units", 0.0))
                graph_logic_state_rewardable_new_units += float(
                    logic_stats.get("state_rewardable_new_units", 0.0)
                )
                graph_logic_state_rewardable_weighted_new_units += float(
                    logic_stats.get("state_rewardable_weighted_new_units", 0.0)
                )
                graph_logic_state_weak_new_units += float(logic_stats.get("state_weak_new_units", 0.0))
                graph_logic_state_rewardable_transitions += float(
                    logic_stats.get("state_rewardable_transitions", 0.0)
                )
                graph_logic_state_weak_only_transition_segments += float(
                    logic_stats.get("state_weak_only_transition", 0.0)
                )
                graph_logic_state_terminal_units += float(logic_stats.get("state_terminal_units", 0.0))
                graph_logic_state_completion_invalid += float(logic_stats.get("state_completion_invalid", 0.0))
                graph_logic_state_raw_segments += float(logic_stats.get("state_raw_segments", 0.0))
                graph_logic_state_candidate_segments += float(logic_stats.get("state_candidate_segments", 0.0))
                graph_logic_state_not_candidate_segments += float(
                    logic_stats.get("state_not_candidate_segments", 0.0)
                )
                graph_logic_state_fact_dump_segments += float(logic_stats.get("state_fact_dump_segments", 0.0))
                graph_logic_state_not_grounded_segments += float(logic_stats.get("state_not_grounded_segments", 0.0))
                graph_logic_state_mentions += float(logic_stats.get("state_mentions", 0.0))
                graph_logic_state_mention_grounded_segments += float(
                    logic_stats.get("state_mention_grounded_segments", 0.0)
                )
                graph_logic_state_mention_only_segments += float(logic_stats.get("state_mention_only_segments", 0.0))
                graph_logic_state_consistent_segments += float(logic_stats.get("state_consistent_segments", 0.0))
                graph_logic_state_reward_consistent_segments += float(
                    logic_stats.get("state_reward_consistent_segments", 0.0)
                )
                graph_logic_state_context_grounded_segments += float(
                    logic_stats.get("state_context_grounded_segments", 0.0)
                )
                graph_logic_state_formula_grounded_segments += float(
                    logic_stats.get("state_formula_grounded_segments", 0.0)
                )
                graph_logic_state_context_edges += float(logic_stats.get("state_context_edges", 0.0))
                graph_logic_state_context_nodes += float(logic_stats.get("state_context_nodes", 0.0))
                graph_logic_state_candidate_grounded_segments += float(
                    logic_stats.get("state_candidate_grounded_segments", 0.0)
                )
                graph_logic_state_commitment_weighted_segments += float(
                    logic_stats.get("state_commitment_weighted_segments", 0.0)
                )
                graph_logic_state_commitment_score += float(logic_stats.get("state_commitment_score", 0.0))
                graph_logic_state_commitment_multiplier += float(
                    logic_stats.get("state_commitment_multiplier", 0.0)
                )
                graph_logic_state_ambiguity_scored_segments += float(
                    logic_stats.get("state_ambiguity_scored_segments", 0.0)
                )
                graph_logic_state_ambiguity_score += float(logic_stats.get("state_ambiguity_score", 0.0))
                graph_logic_state_executable_commitment_score += float(
                    logic_stats.get("state_executable_commitment_score", 0.0)
                )
                graph_logic_state_hard_cert_segments += float(logic_stats.get("state_hard_cert_segments", 0.0))
                graph_logic_state_ambiguity_gate_segments += float(
                    logic_stats.get("state_ambiguity_gate_segments", 0.0)
                )
                graph_logic_state_ambiguity_gate_multiplier += float(
                    logic_stats.get("state_ambiguity_gate_multiplier", 0.0)
                )
                graph_logic_state_certificate_gate_segments += float(
                    logic_stats.get("state_certificate_gate_segments", 0.0)
                )
                graph_logic_state_certificate_gate_pass_segments += float(
                    logic_stats.get("state_certificate_gate_pass_segments", 0.0)
                )
                graph_logic_state_certificate_gate_blocked_segments += float(
                    logic_stats.get("state_certificate_gate_blocked_segments", 0.0)
                )
                graph_logic_component_precision_gate_segments += float(
                    logic_stats.get("component_precision_gate_segments", 0.0)
                )
                graph_logic_component_precision_gate_verified_segments += float(
                    logic_stats.get("component_precision_gate_verified_segments", 0.0)
                )
                graph_logic_component_precision_gate_vague_segments += float(
                    logic_stats.get("component_precision_gate_vague_segments", 0.0)
                )
                graph_logic_component_precision_gate_noisy_segments += float(
                    logic_stats.get("component_precision_gate_noisy_segments", 0.0)
                )
                graph_logic_component_precision_schema_scalar_scaled += float(
                    logic_stats.get("component_precision_schema_scalar_scaled", 0.0)
                )
                graph_logic_component_precision_schema_object_scaled += float(
                    logic_stats.get("component_precision_schema_object_scaled", 0.0)
                )
                graph_logic_component_precision_schema_scalar_blocked += float(
                    logic_stats.get("component_precision_schema_scalar_blocked", 0.0)
                )
                graph_logic_component_precision_schema_object_blocked += float(
                    logic_stats.get("component_precision_schema_object_blocked", 0.0)
                )
                graph_logic_component_precision_schema_multiplier_sum += float(
                    logic_stats.get("component_precision_schema_multiplier_sum", 0.0)
                )
                graph_logic_component_precision_gate_vague_multiplier += float(
                    logic_stats.get("component_precision_gate_vague_multiplier", 0.0)
                )
                graph_logic_component_precision_gate_noisy_multiplier += float(
                    logic_stats.get("component_precision_gate_noisy_multiplier", 0.0)
                )
                graph_logic_component_precision_unit_filter_segments += float(
                    logic_stats.get("component_precision_unit_filter_segments", 0.0)
                )
                graph_logic_component_precision_unit_filter_drop_segments += float(
                    logic_stats.get("component_precision_unit_filter_drop_segments", 0.0)
                )
                graph_logic_component_precision_unit_filter_input_units += float(
                    logic_stats.get("component_precision_unit_filter_input_units", 0.0)
                )
                graph_logic_component_precision_unit_filter_kept_units += float(
                    logic_stats.get("component_precision_unit_filter_kept_units", 0.0)
                )
                graph_logic_component_precision_unit_filter_dropped_units += float(
                    logic_stats.get("component_precision_unit_filter_dropped_units", 0.0)
                )
                graph_logic_state_action_segments += float(logic_stats.get("state_action_segments", 0.0))
                graph_logic_state_no_action_segments += float(logic_stats.get("state_no_action_segments", 0.0))
                graph_logic_state_invalid_action_segments += float(
                    logic_stats.get("state_invalid_action_segments", 0.0)
                )
                graph_logic_state_duplicate_no_progress_segments += float(
                    logic_stats.get("state_duplicate_no_progress_segments", 0.0)
                )
                graph_logic_state_valid_zero_delta_segments += float(
                    logic_stats.get("state_valid_zero_delta_segments", 0.0)
                )
                graph_logic_state_action_no_progress_segments += float(
                    logic_stats.get("state_action_no_progress_segments", 0.0)
                )
                graph_logic_state_no_valid_units_segments += float(
                    logic_stats.get("state_no_valid_units_segments", 0.0)
                )
                graph_logic_state_units += float(logic_stats.get("state_units", 0.0))
                graph_logic_state_duplicate_units += float(logic_stats.get("state_duplicate_units", 0.0))
                graph_logic_action_span_rewarded_segments += float(
                    logic_stats.get("action_span_rewarded_segments", 0.0)
                )
                graph_logic_action_span_emitted += float(logic_stats.get("action_span_emitted", 0.0))
                graph_logic_action_span_split_segments += float(
                    logic_stats.get("action_span_split_segments", 0.0)
                )
                graph_logic_action_span_fallback_segments += float(
                    logic_stats.get("action_span_fallback_segments", 0.0)
                )
                graph_logic_action_event_rewarded_segments += float(
                    logic_stats.get("action_event_rewarded_segments", 0.0)
                )
                graph_logic_action_event_bound_segments += float(
                    logic_stats.get("action_event_bound_segments", 0.0)
                )
                graph_logic_action_router_segments += float(logic_stats.get("action_router_segments", 0.0))
                graph_logic_action_router_candidates += float(logic_stats.get("action_router_candidates", 0.0))
                graph_logic_action_router_no_candidate_segments += float(
                    logic_stats.get("action_router_no_candidate_segments", 0.0)
                )
                graph_logic_action_router_selected_segments += float(
                    logic_stats.get("action_router_selected_segments", 0.0)
                )
                graph_logic_action_router_abstain_segments += float(
                    logic_stats.get("action_router_abstain_segments", 0.0)
                )
                graph_logic_action_router_ambiguous_segments += float(
                    logic_stats.get("action_router_ambiguous_segments", 0.0)
                )
                graph_logic_action_router_confident_segments += float(
                    logic_stats.get("action_router_confident_segments", 0.0)
                )
                graph_logic_action_router_grounded_segments += float(
                    logic_stats.get("action_router_grounded_segments", 0.0)
                )
                graph_logic_action_router_accepted_segments += float(
                    logic_stats.get("action_router_accepted_segments", 0.0)
                )
                graph_logic_action_router_invalid_selection_segments += float(
                    logic_stats.get("action_router_invalid_selection_segments", 0.0)
                )
                graph_logic_action_router_http_error_segments += float(
                    logic_stats.get("action_router_http_error_segments", 0.0)
                )
                graph_logic_action_router_http_missing_endpoint_segments += float(
                    logic_stats.get("action_router_http_missing_endpoint_segments", 0.0)
                )
                graph_logic_action_router_confidence += float(logic_stats.get("action_router_confidence", 0.0))
                graph_logic_action_router_positive_bonus += float(
                    logic_stats.get("action_router_positive_bonus", 0.0)
                )
                graph_logic_action_router_positive_bonus_segments += float(
                    logic_stats.get("action_router_positive_bonus_segments", 0.0)
                )
                graph_logic_action_router_source_http += float(logic_stats.get("action_router_source_http", 0.0))
                graph_logic_action_router_source_http_error += float(
                    logic_stats.get("action_router_source_http_error", 0.0)
                )
                graph_logic_action_router_source_heuristic += float(
                    logic_stats.get("action_router_source_heuristic", 0.0)
                )
                graph_logic_action_router_source_http_heuristic_fallback += float(
                    logic_stats.get("action_router_source_http_heuristic_fallback", 0.0)
                )
                graph_logic_action_event_bound_spans += float(
                    logic_stats.get("action_event_bound_spans", 0.0)
                )
                for component in STATE_PROGRESS_COMPONENTS:
                    graph_logic_state_component_scores[component] += float(
                        logic_stats.get(f"state_component_{component}", 0.0)
                    )
                    graph_logic_state_component_deltas[component] += float(
                        logic_stats.get(f"state_component_delta_{component}", 0.0)
                    )
                if "potential_final" in logic_stats:
                    graph_logic_potential_final_values.append(float(logic_stats.get("potential_final", 0.0)))
                graph_logic_ordered_recall_values.append(float(logic_stats.get("ordered_recall", 0.0)))
                if not steps:
                    continue
                logic_task = str(getattr(spec, "task", "") or "").lower()
                logic_family = _graph_logic_task_family(logic_task)
                logic_family_scale = float(
                    self.graph_logic_family_scales.get(logic_family, self.graph_logic_family_scale_default)
                )
                if not self.graph_logic_family_balance:
                    logic_family_scale = 1.0
                graph_logic_family_sample_count[logic_family] += 1.0
                graph_logic_family_scale_sum[logic_family] += logic_family_scale
                starts: list[int] = []
                ends: list[int] = []
                token_position_map: list[list[int]] = []
                if self.graph_logic_token_mapping == "exact":
                    if local_idx >= len(retok_ids_list) or local_idx >= len(offset_maps):
                        continue
                    try:
                        original_ids = [int(t) for t in valid_response_id_lists[local_idx]]
                        token_position_map = self.graph_rm._build_token_position_map(
                            retokenized_ids=retok_ids_list[local_idx],
                            original_ids=original_ids,
                        )
                    except Exception:
                        continue
                    starts = [int(s) for s, _ in offset_maps[local_idx]]
                    ends = [int(e) for _, e in offset_maps[local_idx]]
                wrong_process_weight = self.graph_logic_wrong_process_weight
                correct_process_weight = self.graph_logic_correct_process_weight
                final_gate = wrong_process_weight + (
                    correct_process_weight - wrong_process_weight
                ) * float(outcome_values[i])
                gate_lo = min(wrong_process_weight, correct_process_weight)
                gate_hi = max(wrong_process_weight, correct_process_weight)
                final_gate = max(gate_lo, min(gate_hi, final_gate))
                graph_logic_gate_values.append(final_gate)
                emitted = 0
                wrong_outcome = float(outcome_values[i]) <= 0.5
                good_prefix_blocked = False
                good_prefix_preserved_sum = 0.0
                for step in steps[: self.max_entities]:
                    if self.graph_logic_token_mapping == "exact":
                        token_range = self._char_span_to_token_range(
                            step.char_span,
                            starts,
                            ends,
                            token_position_map,
                            int(valid_r_lens[i]),
                        )
                    else:
                        token_range = self._char_span_to_proportional_token_range(
                            step.char_span,
                            len(text),
                            int(valid_r_lens[i]),
                        )
                    if token_range is None:
                        continue
                    reward = float(step.reward)
                    step_label = str(getattr(step, "label", "NEUTRAL") or "NEUTRAL").upper()
                    step_first_error = bool(getattr(step, "first_error", False))
                    graph_logic_label_counts[step_label] += 1.0
                    if (
                        self.graph_logic_good_prefix_preserve
                        and wrong_outcome
                        and (step_first_error or step_label == "NEG_IRREC")
                    ):
                        if not good_prefix_blocked:
                            graph_logic_good_prefix_first_error_steps += 1.0
                        good_prefix_blocked = True
                    if reward > 0:
                        if self.graph_logic_good_prefix_preserve and wrong_outcome:
                            if not good_prefix_blocked and step_label == "POS_STRONG":
                                reward *= self.graph_logic_good_prefix_wrong_weight
                                if self.graph_logic_good_prefix_wrong_cap > 0.0:
                                    remaining = max(0.0, self.graph_logic_good_prefix_wrong_cap - good_prefix_preserved_sum)
                                    if remaining <= 0.0:
                                        reward = 0.0
                                        graph_logic_good_prefix_cap_exhausted += 1.0
                                    else:
                                        reward = min(reward, remaining)
                                if reward > 0.0:
                                    good_prefix_preserved_sum += reward
                                    graph_logic_good_prefix_preserved_positive += 1.0
                                else:
                                    graph_logic_good_prefix_blocked_positive += 1.0
                            else:
                                reward *= self.graph_logic_wrong_process_weight
                                graph_logic_good_prefix_blocked_positive += 1.0
                        else:
                            reward *= final_gate
                    if self.graph_logic_correct_only_positive and wrong_outcome and reward > 0.0:
                        graph_logic_correct_only_blocked_positive += 1.0
                        graph_logic_correct_only_blocked_positive_reward += float(reward)
                        reward = 0.0
                    if wrong_outcome and reward < 0.0:
                        if self.graph_logic_keep_wrong_negative:
                            graph_logic_wrong_negative_kept += 1.0
                        else:
                            graph_logic_wrong_negative_blocked += 1.0
                            reward = 0.0
                    if unit_trust_enabled and reward > 0.0:
                        graph_logic_unit_trust_filter_positive += 1.0
                        unit_names = _graph_logic_step_unit_names(step)
                        correct_samples, wrong_samples = unit_trust_task_counts.get(logic_task, (0, 0))
                        total_task_samples = int(correct_samples) + int(wrong_samples)
                        if (
                            total_task_samples < self.graph_logic_unit_trust_min_task_samples
                            and self.graph_logic_unit_trust_low_sample_keep
                        ):
                            graph_logic_unit_trust_low_sample_keep += 1.0
                            graph_logic_unit_trust_kept_positive += 1.0
                        elif not unit_names:
                            graph_logic_unit_trust_unknown_positive += 1.0
                            if self.graph_logic_unit_trust_keep_unknown_positive:
                                graph_logic_unit_trust_kept_positive += 1.0
                            else:
                                graph_logic_unit_trust_blocked_positive += 1.0
                                reward = 0.0
                        else:
                            best_trust = max(
                                unit_trust_scores.get((logic_task, unit_name), float("-inf"))
                                for unit_name in unit_names
                            )
                            if math.isfinite(best_trust):
                                graph_logic_unit_trust_score_sum += float(best_trust)
                            trusted = bool(
                                best_trust >= self.graph_logic_unit_trust_margin
                                and any(
                                    unit_trust_scores.get((logic_task, unit_name), float("-inf"))
                                    >= self.graph_logic_unit_trust_margin
                                    for unit_name in unit_names
                                )
                            )
                            if not trusted:
                                graph_logic_unit_trust_blocked_positive += 1.0
                                reward = 0.0
                            else:
                                has_correct_hits = False
                                for unit_name in unit_names:
                                    if unit_trust_scores.get((logic_task, unit_name), float("-inf")) < self.graph_logic_unit_trust_margin:
                                        continue
                                    unit_correct_hits, _unit_wrong_hits = unit_trust_hits.get(
                                        (logic_task, unit_name), (0, 0)
                                    )
                                    has_correct_hits = (
                                        unit_correct_hits >= self.graph_logic_unit_trust_min_correct_hits
                                    )
                                    if has_correct_hits:
                                        break
                                if has_correct_hits:
                                    graph_logic_unit_trust_kept_positive += 1.0
                                else:
                                    graph_logic_unit_trust_blocked_positive += 1.0
                                    reward = 0.0
                    reward_before_family = reward
                    if self.graph_logic_family_balance:
                        reward *= logic_family_scale
                    graph_logic_family_event_count[logic_family] += 1.0
                    graph_logic_family_reward_abs_before[logic_family] += abs(float(reward_before_family))
                    graph_logic_family_reward_abs_after[logic_family] += abs(float(reward))
                    graph_logic_family_reward_signed_before[logic_family] += float(reward_before_family)
                    graph_logic_family_reward_signed_after[logic_family] += float(reward)
                    reward = self._clip(reward)
                    if reward > 0:
                        positive_count += 1
                    elif reward < 0:
                        negative_count += 1
                    else:
                        zero_count += 1
                    reward_values.append(reward)
                    per_sample_rewards[i].append(reward)
                    per_sample_anchors[i].append(step.anchor)
                    per_sample_token_pos[i].append(self._graph_logic_target_position(token_range, reward))
                    per_sample_reward_sum[i] += reward
                    per_sample_reward_count[i] += 1
                    emitted += 1
                if emitted:
                    valid_samples += 1
                    entity_total += emitted
                continue

            flat = per_sample_flat.get(i, [])
            if not flat:
                continue

            valid_samples += 1
            entity_total += len(flat)
            order_pos = 0
            seen_order_nodes: set[int] = set()
            (
                task,
                graph_nodes,
                graph_edges,
                directed,
                positive_nodes,
                positive_edges,
                focus_nodes,
                soft_nodes,
                soft_edges,
                oracle_order,
            ) = sample_infos[i]
            relative_state = self._relative_state()
            phi_prev = self._relative_potential(
                relative_state,
                graph_nodes,
                positive_nodes,
                positive_edges,
                focus_nodes,
                soft_nodes,
                soft_edges,
                oracle_order,
            )
            final_gate = self.relative_final_gate_eta + (1.0 - self.relative_final_gate_eta) * float(outcome_values[i])
            final_gate = max(self.relative_final_gate_eta, min(1.0, final_gate))

            for etype, eid, token_span in flat[: self.max_entities]:
                reward = 0.0
                if self.scorer == "algorithmic_relative":
                    bad_before = (
                        len(relative_state["invalid_edges"])
                        + len(relative_state["invalid_nodes"])
                        + 0.25 * len(relative_state["off_edges"])
                        + 0.10 * len(relative_state["off_nodes"])
                    )
                    anchor_id = self._relative_update(
                        relative_state,
                        etype,
                        eid,
                        graph_nodes,
                        graph_edges,
                        directed,
                        positive_nodes,
                        positive_edges,
                        focus_nodes,
                        soft_nodes,
                        soft_edges,
                        oracle_order,
                    )
                    if anchor_id is None:
                        continue
                    phi_after = self._relative_potential(
                        relative_state,
                        graph_nodes,
                        positive_nodes,
                        positive_edges,
                        focus_nodes,
                        soft_nodes,
                        soft_edges,
                        oracle_order,
                    )
                    bad_after = (
                        len(relative_state["invalid_edges"])
                        + len(relative_state["invalid_nodes"])
                        + 0.25 * len(relative_state["off_edges"])
                        + 0.10 * len(relative_state["off_nodes"])
                    )
                    delta = phi_after - phi_prev
                    if delta >= 0.0:
                        denom = max(1e-8, 1.0 - phi_prev)
                        relative_h = delta / denom
                        reward = final_gate * relative_h * self.positive_reward
                        if delta == 0.0 and bad_after > bad_before:
                            reward = -abs(self.invalid_edge_reward) * 0.25 * (bad_after - bad_before)
                    else:
                        denom = max(1e-8, phi_prev)
                        relative_h = delta / denom
                        reward = relative_h * max(1e-8, abs(self.invalid_edge_reward))
                    reward *= self._relative_position_weight(token_span, int(valid_r_lens[i]))
                    phi_prev = phi_after
                    relative_phi_values.append(phi_after)
                    relative_delta_values.append(delta)
                    relative_gate_values.append(final_gate)
                elif etype == "node":
                    node = _as_int(eid)
                    if node is None:
                        continue
                    anchor_id: object = ("graph_progress", task, "node", node)
                    if oracle_order:
                        if order_pos < len(oracle_order) and node == oracle_order[order_pos]:
                            reward = self.order_hit_reward
                            seen_order_nodes.add(node)
                            order_pos += 1
                        elif node in seen_order_nodes:
                            reward = self.order_seen_reward
                        elif node in oracle_order[order_pos:]:
                            reward = self.order_skip_reward
                        elif node in focus_nodes:
                            reward = self.focus_reward
                        elif node in positive_nodes:
                            reward = self.positive_reward
                        elif node in soft_nodes:
                            reward = max(self.focus_reward, 0.5 * self.positive_reward)
                        else:
                            reward = self.irrelevant_node_reward
                    elif node in positive_nodes:
                        reward = self.positive_reward
                    elif node in focus_nodes:
                        reward = self.focus_reward
                    elif node in soft_nodes:
                        reward = max(self.focus_reward, 0.5 * self.positive_reward)
                    else:
                        reward = self.irrelevant_node_reward
                else:
                    edge = _norm_edge(eid, directed=directed)
                    if edge is None:
                        continue
                    anchor_id = ("graph_progress", task, "edge", edge)
                    if edge in positive_edges:
                        reward = self.positive_reward
                    elif edge not in graph_edges:
                        reward = self.invalid_edge_reward
                    elif edge in soft_edges:
                        reward = max(self.focus_reward, 0.5 * self.positive_reward)
                    else:
                        reward = self.irrelevant_edge_reward

                reward = self._clip(reward)
                if reward > 0:
                    positive_count += 1
                elif reward < 0:
                    negative_count += 1
                else:
                    zero_count += 1

                reward_values.append(reward)
                per_sample_rewards[i].append(reward)
                per_sample_anchors[i].append(anchor_id)
                per_sample_token_pos[i].append(self._target_positions(token_span))
                per_sample_reward_sum[i] += reward
                per_sample_reward_count[i] += 1
        t_score = _time.time() - t_score_start

        max_steps = max((len(x) for x in per_sample_rewards), default=0)
        if max_steps == 0:
            max_steps = 1
        step_rewards = torch.full((B, max_steps), float("nan"), dtype=torch.float32)
        anchor_obs = np.full((B, max_steps), None, dtype=object)
        step_token_positions = np.full((B, max_steps), None, dtype=object)
        for i in range(B):
            rewards = per_sample_rewards[i]
            if not rewards:
                continue
            k = len(rewards)
            step_rewards[i, :k] = torch.tensor(rewards, dtype=torch.float32)
            ai = np.empty(k, dtype=object)
            ai[:] = per_sample_anchors[i]
            anchor_obs[i, :k] = ai
            ti = np.empty(k, dtype=object)
            ti[:] = per_sample_token_pos[i]
            step_token_positions[i, :k] = ti

        def _safe_stat(fn, values, default=0.0):
            return float(fn(values)) if values else default

        total_events = positive_count + negative_count + zero_count
        audit_stats = {}
        if outcome_scores is not None:
            try:
                oscores = outcome_scores.detach().float().cpu().numpy()
                if oscores.ndim > 1:
                    oscores = oscores.reshape(oscores.shape[0], -1).sum(axis=-1)
                oscores = oscores[:B]
                has_signal = per_sample_reward_count[: len(oscores)] > 0
                correct = oscores > 0.5
                wrong = ~correct

                def _group_stats(prefix: str, mask):
                    mask = np.asarray(mask, dtype=bool) & has_signal
                    if not np.any(mask):
                        audit_stats[f"graph_progress/audit/{prefix}_n"] = 0.0
                        return
                    sums = per_sample_reward_sum[: len(oscores)][mask]
                    counts = per_sample_reward_count[: len(oscores)][mask]
                    means = sums / np.maximum(counts, 1)
                    audit_stats[f"graph_progress/audit/{prefix}_n"] = float(mask.sum())
                    audit_stats[f"graph_progress/audit/{prefix}_sum_mean"] = float(np.mean(sums))
                    audit_stats[f"graph_progress/audit/{prefix}_event_mean"] = float(np.mean(means))
                    audit_stats[f"graph_progress/audit/{prefix}_events_mean"] = float(np.mean(counts))

                _group_stats("correct", correct)
                _group_stats("wrong", wrong)
                correct_sum = audit_stats.get("graph_progress/audit/correct_sum_mean")
                wrong_sum = audit_stats.get("graph_progress/audit/wrong_sum_mean")
                correct_event = audit_stats.get("graph_progress/audit/correct_event_mean")
                wrong_event = audit_stats.get("graph_progress/audit/wrong_event_mean")
                if correct_sum is not None and wrong_sum is not None:
                    audit_stats["graph_progress/audit/sum_gap"] = float(correct_sum - wrong_sum)
                if correct_event is not None and wrong_event is not None:
                    audit_stats["graph_progress/audit/event_gap"] = float(correct_event - wrong_event)
            except Exception as e:
                print(f"[GraphProgress] audit failed ({type(e).__name__}: {e})")
        n_tasks = len(valid_indices)
        if self.print_timing and n_tasks:
            per_task_ms = t_extract * 1000 / max(1, n_tasks)
            print(
                f"[GraphProgress] prepare={t_prepare:.1f}s decode={t_decode:.1f}s "
                f"encode_batch={t_encode:.1f}s extract={t_extract:.1f}s "
                f"score={t_score:.1f}s ({backend_used}, tasks={n_tasks}, "
                f"per_task={per_task_ms:.1f}ms, first_only={self.first_occurrence_only})"
            )
        self.last_stats = {
            "graph_progress/valid_samples": int(valid_samples),
            "graph_progress/unsupported_samples": int(unsupported),
            "graph_progress/entities_seen": int(entity_total),
            "graph_progress/events": int(total_events),
            "graph_progress/positive_frac": float(positive_count / max(1, total_events)),
            "graph_progress/negative_frac": float(negative_count / max(1, total_events)),
            "graph_progress/zero_frac": float(zero_count / max(1, total_events)),
            "graph_progress/reward_mean": _safe_stat(np.mean, reward_values),
            "graph_progress/reward_std": _safe_stat(np.std, reward_values),
            "graph_progress/reward_min": _safe_stat(np.min, reward_values),
            "graph_progress/reward_max": _safe_stat(np.max, reward_values),
            "graph_progress/scorer_algorithmic": 1.0 if self.scorer in ("algorithmic", "algorithmic_relative") else 0.0,
            "graph_progress/scorer_algorithmic_relative": 1.0 if self.scorer == "algorithmic_relative" else 0.0,
            "graph_progress/scorer_graph_logic": 1.0 if self.scorer == "graph_logic" else 0.0,
            "graph_progress/algorithmic_supported": int(algorithmic_supported),
            "graph_progress/algorithmic_soft_only_supported": int(algorithmic_soft_only_supported),
            "graph_progress/graph_logic_supported": int(graph_logic_supported),
            "graph_progress/answer_fallback": int(answer_fallback),
            "graph_progress/relative_phi_mean": _safe_stat(np.mean, relative_phi_values),
            "graph_progress/relative_delta_mean": _safe_stat(np.mean, relative_delta_values),
            "graph_progress/relative_final_gate_mean": _safe_stat(np.mean, relative_gate_values),
            "graph_progress/relative_progress_frac": float(
                sum(1 for x in relative_delta_values if x > 0) / max(1, len(relative_delta_values))
            ),
            "graph_progress/relative_regress_frac": float(
                sum(1 for x in relative_delta_values if x < 0) / max(1, len(relative_delta_values))
            ),
            "graph_progress/coverage_prefix": 1.0 if self.coverage_mode == "prefix" else 0.0,
            "graph_progress/coverage_prefix_decay": 1.0 if self.coverage_mode == "prefix_decay" else 0.0,
            "graph_progress/coverage_span": 1.0 if self.coverage_mode == "span" else 0.0,
            "graph_progress/coverage_segment": 1.0 if self.coverage_mode == "segment" else 0.0,
            "graph_progress/logic_segments": float(graph_logic_segments),
            "graph_progress/logic_segment_initial_segments": float(graph_logic_segment_initial_segments),
            "graph_progress/logic_segment_merge_enabled": float(
                graph_logic_segment_merge_enabled / max(1.0, graph_logic_supported)
            ),
            "graph_progress/logic_segment_merge_count": float(graph_logic_segment_merge_count),
            "graph_progress/logic_segment_merge_target_count_mean": float(
                graph_logic_segment_merge_target_count / max(1.0, graph_logic_segment_merge_enabled)
            ),
            "graph_progress/logic_segment_merge_model_entropy": float(
                graph_logic_segment_merge_model_entropy / max(1.0, graph_logic_segment_merge_enabled)
            ),
            "graph_progress/logic_segment_merge_reduction_frac": float(
                max(0.0, graph_logic_segment_initial_segments - graph_logic_segments)
                / max(1.0, graph_logic_segment_initial_segments)
            ),
            "graph_progress/logic_segment_merge_initial_info_mean": float(
                graph_logic_segment_merge_initial_info_sum / max(1.0, graph_logic_segment_initial_segments)
            ),
            "graph_progress/logic_segment_merge_final_info_mean": float(
                graph_logic_segment_merge_final_info_sum / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_segment_merge_selected_info_mean": float(
                graph_logic_segment_merge_selected_info_sum / max(1.0, graph_logic_segment_merge_count)
            ),
            "graph_progress/logic_grounded_frac": float(graph_logic_grounded / max(1.0, graph_logic_segments)),
            "graph_progress/logic_candidate_frac": float(graph_logic_candidate / max(1.0, graph_logic_segments)),
            "graph_progress/logic_executor_valid_frac": float(
                graph_logic_executor_valid / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_progressive_frac": float(graph_logic_progressive / max(1.0, graph_logic_segments)),
            "graph_progress/logic_ordered_frac": float(graph_logic_ordered / max(1.0, graph_logic_segments)),
            "graph_progress/logic_answer_hit_frac": float(graph_logic_answer_hit / max(1.0, graph_logic_segments)),
            "graph_progress/logic_fact_dump_frac": float(graph_logic_fact_dump / max(1.0, graph_logic_segments)),
            "graph_progress/logic_invalid_edges": float(graph_logic_invalid_edges),
            "graph_progress/logic_suppressed_invalid_edges": float(graph_logic_suppressed_invalid_edges),
            "graph_progress/logic_node_only_progress_frac": float(
                graph_logic_node_only_progress / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_mode": float(self.graph_logic_config.reward_mode == "potential"),
            "graph_progress/logic_potential_delta_per_segment": float(
                graph_logic_potential_delta / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_delta_raw_per_segment": float(
                graph_logic_potential_delta_raw / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_positive_frac": float(
                graph_logic_potential_positive_steps / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_evidence_units_per_segment": float(
                graph_logic_potential_evidence_units / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_evidence_step_frac": float(
                graph_logic_potential_evidence_steps / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_task_evidence_units_per_segment": float(
                graph_logic_potential_task_evidence_units / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_potential_task_evidence_step_frac": float(
                graph_logic_potential_task_evidence_steps / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_state_enabled": float(self.graph_logic_config.state_shaping),
            "graph_progress/logic_state_supported": float(graph_logic_state_supported),
            "graph_progress/logic_state_supported_segment_frac": float(
                graph_logic_state_supported_segments / max(1.0, graph_logic_segments)
            ),
            "graph_progress/logic_state_raw_segments": float(graph_logic_state_raw_segments),
            "graph_progress/logic_state_candidate_frac": float(
                graph_logic_state_candidate_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_not_candidate_frac": float(
                graph_logic_state_not_candidate_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_fact_dump_skip_frac": float(
                graph_logic_state_fact_dump_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_not_grounded_frac": float(
                graph_logic_state_not_grounded_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_mention_frac": float(
                graph_logic_state_mentions / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_mention_grounded_frac": float(
                graph_logic_state_mention_grounded_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_mention_only_frac": float(
                graph_logic_state_mention_only_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_consistent_frac": float(
                graph_logic_state_consistent_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_reward_consistent_frac": float(
                graph_logic_state_reward_consistent_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_context_grounded_frac": float(
                graph_logic_state_context_grounded_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_formula_grounded_frac": float(
                graph_logic_state_formula_grounded_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_context_edges_per_segment": float(
                graph_logic_state_context_edges / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_context_nodes_per_segment": float(
                graph_logic_state_context_nodes / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_state_actions_per_segment": float(
                graph_logic_state_actions / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_action_segment_frac": float(
                graph_logic_state_action_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_no_action_segment_frac": float(
                graph_logic_state_no_action_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_valid_transition_frac": float(
                graph_logic_state_valid_transitions / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_rewardable_transition_frac": float(
                graph_logic_state_rewardable_transitions / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_weak_only_transition_frac": float(
                graph_logic_state_weak_only_transition_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_invalid_transitions_per_segment": float(
                graph_logic_state_invalid_transitions / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_invalid_action_segment_frac": float(
                graph_logic_state_invalid_action_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_duplicate_no_progress_frac": float(
                graph_logic_state_duplicate_no_progress_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_valid_zero_delta_frac": float(
                graph_logic_state_valid_zero_delta_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_action_no_progress_frac": float(
                graph_logic_state_action_no_progress_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_no_valid_units_frac": float(
                graph_logic_state_no_valid_units_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_units_per_segment": float(
                graph_logic_state_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_duplicate_units_per_segment": float(
                graph_logic_state_duplicate_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_progressive_frac": float(
                graph_logic_state_progressive / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_delta_per_segment": float(
                graph_logic_state_delta / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_raw_delta_per_segment": float(
                graph_logic_state_raw_delta / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_delta_quality_mean": float(
                graph_logic_state_delta_quality / max(1.0, graph_logic_state_quality_segments)
            ),
            "graph_progress/logic_state_new_units_per_segment": float(
                graph_logic_state_new_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_weighted_new_units_per_segment": float(
                graph_logic_state_weighted_new_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_rewardable_new_units_per_segment": float(
                graph_logic_state_rewardable_new_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_rewardable_weighted_new_units_per_segment": float(
                graph_logic_state_rewardable_weighted_new_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_weak_new_units_per_segment": float(
                graph_logic_state_weak_new_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_terminal_units_per_segment": float(
                graph_logic_state_terminal_units / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_completion_invalid_frac": float(
                graph_logic_state_completion_invalid / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_potential_final_mean": _safe_stat(np.mean, graph_logic_potential_final_values),
            "graph_progress/logic_ordered_recall_mean": _safe_stat(np.mean, graph_logic_ordered_recall_values),
            "graph_progress/logic_final_gate_mean": _safe_stat(np.mean, graph_logic_gate_values),
            "graph_progress/logic_correct_process_weight": float(self.graph_logic_correct_process_weight),
            "graph_progress/logic_wrong_process_weight": float(self.graph_logic_wrong_process_weight),
            "graph_progress/logic_legacy_core4_compat_enabled": (
                1.0 if self.graph_logic_legacy_core4_compat else 0.0
            ),
            "graph_progress/logic_correct_only_positive_enabled": (
                1.0 if self.graph_logic_correct_only_positive else 0.0
            ),
            "graph_progress/logic_keep_wrong_negative_enabled": (
                1.0 if self.graph_logic_keep_wrong_negative else 0.0
            ),
            "graph_progress/logic_correct_only_blocked_positive": float(
                graph_logic_correct_only_blocked_positive
            ),
            "graph_progress/logic_correct_only_blocked_positive_reward": float(
                graph_logic_correct_only_blocked_positive_reward
            ),
            "graph_progress/logic_wrong_negative_kept": float(graph_logic_wrong_negative_kept),
            "graph_progress/logic_wrong_negative_blocked": float(graph_logic_wrong_negative_blocked),
            "graph_progress/logic_unit_trust_filter_enabled": (
                1.0 if self.graph_logic_unit_trust_filter else 0.0
            ),
            "graph_progress/logic_unit_trust_filter_active": 1.0 if unit_trust_enabled else 0.0,
            "graph_progress/logic_unit_trust_margin": float(self.graph_logic_unit_trust_margin),
            "graph_progress/logic_unit_trust_min_task_samples": float(
                self.graph_logic_unit_trust_min_task_samples
            ),
            "graph_progress/logic_unit_trust_min_correct_hits": float(
                self.graph_logic_unit_trust_min_correct_hits
            ),
            "graph_progress/logic_unit_trust_positive": float(
                graph_logic_unit_trust_filter_positive
            ),
            "graph_progress/logic_unit_trust_kept_positive": float(
                graph_logic_unit_trust_kept_positive
            ),
            "graph_progress/logic_unit_trust_blocked_positive": float(
                graph_logic_unit_trust_blocked_positive
            ),
            "graph_progress/logic_unit_trust_unknown_positive": float(
                graph_logic_unit_trust_unknown_positive
            ),
            "graph_progress/logic_unit_trust_low_sample_keep": float(
                graph_logic_unit_trust_low_sample_keep
            ),
            "graph_progress/logic_unit_trust_blocked_frac": float(
                graph_logic_unit_trust_blocked_positive
                / max(1.0, graph_logic_unit_trust_filter_positive)
            ),
            "graph_progress/logic_unit_trust_kept_frac": float(
                graph_logic_unit_trust_kept_positive
                / max(1.0, graph_logic_unit_trust_filter_positive)
            ),
            "graph_progress/logic_unit_trust_unknown_frac": float(
                graph_logic_unit_trust_unknown_positive
                / max(1.0, graph_logic_unit_trust_filter_positive)
            ),
            "graph_progress/logic_unit_trust_low_sample_keep_frac": float(
                graph_logic_unit_trust_low_sample_keep
                / max(1.0, graph_logic_unit_trust_filter_positive)
            ),
            "graph_progress/logic_unit_trust_score_mean": float(
                graph_logic_unit_trust_score_sum
                / max(
                    1.0,
                    graph_logic_unit_trust_filter_positive
                    - graph_logic_unit_trust_unknown_positive
                    - graph_logic_unit_trust_low_sample_keep,
                )
            ),
            "graph_progress/logic_process_task_blocklist_enabled": (
                1.0 if self.graph_logic_process_task_blocklist else 0.0
            ),
            "graph_progress/logic_process_task_blocked_samples": float(graph_logic_process_blocked),
            "graph_progress/logic_good_prefix_preserve_enabled": (
                1.0 if self.graph_logic_good_prefix_preserve else 0.0
            ),
            "graph_progress/logic_good_prefix_wrong_weight": float(self.graph_logic_good_prefix_wrong_weight),
            "graph_progress/logic_good_prefix_wrong_cap": float(self.graph_logic_good_prefix_wrong_cap),
            "graph_progress/logic_good_prefix_preserved_positive": float(
                graph_logic_good_prefix_preserved_positive
            ),
            "graph_progress/logic_good_prefix_blocked_positive": float(
                graph_logic_good_prefix_blocked_positive
            ),
            "graph_progress/logic_good_prefix_cap_exhausted": float(graph_logic_good_prefix_cap_exhausted),
            "graph_progress/logic_good_prefix_first_error_steps": float(
                graph_logic_good_prefix_first_error_steps
            ),
            "graph_progress/logic_step_label_pos_strong": float(graph_logic_label_counts.get("POS_STRONG", 0.0)),
            "graph_progress/logic_step_label_pos_weak": float(graph_logic_label_counts.get("POS_WEAK", 0.0)),
            "graph_progress/logic_step_label_neg_rec": float(graph_logic_label_counts.get("NEG_REC", 0.0)),
            "graph_progress/logic_step_label_neg_irrec": float(graph_logic_label_counts.get("NEG_IRREC", 0.0)),
            "graph_progress/logic_step_label_neutral": float(graph_logic_label_counts.get("NEUTRAL", 0.0)),
            "graph_progress/logic_positive_budget_mode_net_progress": (
                1.0
                if str(getattr(self.graph_logic_config, "positive_reward_budget_mode", "quality")).lower()
                == "net_progress"
                else 0.0
            ),
            "graph_progress/logic_token_mapping_exact": 1.0 if self.graph_logic_token_mapping == "exact" else 0.0,
            "graph_progress/logic_token_mapping_proportional": (
                1.0 if self.graph_logic_token_mapping == "proportional" else 0.0
            ),
            "graph_progress/logic_step_span_action": (
                1.0
                if str(self.graph_logic_config.step_span_mode).lower()
                in {"action", "actions", "action_based", "action-based"}
                else 0.0
            ),
            "graph_progress/logic_action_span_emitted_per_rewarded_segment": float(
                graph_logic_action_span_emitted / max(1.0, graph_logic_action_span_rewarded_segments)
            ),
            "graph_progress/logic_action_span_split_frac": float(
                graph_logic_action_span_split_segments / max(1.0, graph_logic_action_span_rewarded_segments)
            ),
            "graph_progress/logic_action_span_fallback_frac": float(
                graph_logic_action_span_fallback_segments / max(1.0, graph_logic_action_span_rewarded_segments)
            ),
            "graph_progress/logic_action_event_bound_frac": float(
                graph_logic_action_event_bound_segments / max(1.0, graph_logic_action_span_rewarded_segments)
            ),
            "graph_progress/logic_action_event_rewarded_frac": float(
                graph_logic_action_event_rewarded_segments / max(1.0, graph_logic_action_span_rewarded_segments)
            ),
            "graph_progress/logic_action_event_spans_per_bound_segment": float(
                graph_logic_action_event_bound_spans / max(1.0, graph_logic_action_event_bound_segments)
            ),
            "graph_progress/logic_action_router_enabled": (
                1.0 if self.graph_logic_config.action_router_enable else 0.0
            ),
            "graph_progress/logic_action_router_http_backend": (
                1.0 if str(self.graph_logic_config.action_router_backend).lower() in {"http", "openai", "vllm"} else 0.0
            ),
            "graph_progress/logic_action_router_segments": float(graph_logic_action_router_segments),
            "graph_progress/logic_action_router_segment_frac": float(
                graph_logic_action_router_segments / max(1.0, graph_logic_state_raw_segments)
            ),
            "graph_progress/logic_action_router_candidates_per_segment": float(
                graph_logic_action_router_candidates / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_no_candidate_frac": float(
                graph_logic_action_router_no_candidate_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_selected_frac": float(
                graph_logic_action_router_selected_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_abstain_frac": float(
                graph_logic_action_router_abstain_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_ambiguous_frac": float(
                graph_logic_action_router_ambiguous_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_confident_frac": float(
                graph_logic_action_router_confident_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_grounded_frac": float(
                graph_logic_action_router_grounded_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_accepted_frac": float(
                graph_logic_action_router_accepted_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_invalid_selection_frac": float(
                graph_logic_action_router_invalid_selection_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_http_error_frac": float(
                graph_logic_action_router_http_error_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_missing_endpoint_frac": float(
                graph_logic_action_router_http_missing_endpoint_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_http_source_frac": float(
                graph_logic_action_router_source_http / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_http_error_source_frac": float(
                graph_logic_action_router_source_http_error / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_heuristic_source_frac": float(
                graph_logic_action_router_source_heuristic / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_http_fallback_source_frac": float(
                graph_logic_action_router_source_http_heuristic_fallback
                / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_confidence_mean": float(
                graph_logic_action_router_confidence / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_positive_bonus_per_segment": float(
                graph_logic_action_router_positive_bonus / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_router_positive_bonus_segment_frac": float(
                graph_logic_action_router_positive_bonus_segments / max(1.0, graph_logic_action_router_segments)
            ),
            "graph_progress/logic_action_span_context_chars": float(
                self.graph_logic_config.action_span_context_chars
            ),
            "graph_progress/logic_state_candidate_grounding_enabled": (
                1.0 if self.graph_logic_config.state_candidate_grounding else 0.0
            ),
            "graph_progress/logic_state_candidate_grounded_frac": float(
                graph_logic_state_candidate_grounded_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_state_commitment_weight_enabled": (
                1.0 if self.graph_logic_config.state_commitment_weight else 0.0
            ),
            "graph_progress/logic_executable_commitment_enabled": (
                1.0 if self.graph_logic_config.executable_commitment_weight else 0.0
            ),
            "graph_progress/logic_state_commitment_score_mean": float(
                graph_logic_state_commitment_score / max(1.0, graph_logic_state_commitment_weighted_segments)
            ),
            "graph_progress/logic_state_commitment_multiplier_mean": float(
                graph_logic_state_commitment_multiplier / max(1.0, graph_logic_state_commitment_weighted_segments)
            ),
            "graph_progress/logic_state_ambiguity_score_mean": float(
                graph_logic_state_ambiguity_score / max(1.0, graph_logic_state_ambiguity_scored_segments)
            ),
            "graph_progress/logic_state_executable_commitment_score_mean": float(
                graph_logic_state_executable_commitment_score
                / max(1.0, graph_logic_state_ambiguity_scored_segments)
            ),
            "graph_progress/logic_state_hard_cert_segment_frac": float(
                graph_logic_state_hard_cert_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_ambiguity_gate_enabled": (
                1.0 if self.graph_logic_config.ambiguity_gate_positive else 0.0
            ),
            "graph_progress/logic_ambiguity_gate_frac": float(
                graph_logic_state_ambiguity_gate_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_ambiguity_gate_multiplier_mean": float(
                graph_logic_state_ambiguity_gate_multiplier / max(1.0, graph_logic_state_ambiguity_gate_segments)
            ),
            "graph_progress/logic_certificate_gate_enabled": (
                1.0 if self.graph_logic_config.certificate_only_hard_gate else 0.0
            ),
            "graph_progress/logic_certificate_gate_pass_frac": float(
                graph_logic_state_certificate_gate_pass_segments / max(1.0, graph_logic_state_certificate_gate_segments)
            ),
            "graph_progress/logic_certificate_gate_block_frac": float(
                graph_logic_state_certificate_gate_blocked_segments
                / max(1.0, graph_logic_state_certificate_gate_segments)
            ),
            "graph_progress/logic_component_precision_gate_enabled": (
                1.0 if self.graph_logic_config.component_precision_gate else 0.0
            ),
            "graph_progress/logic_component_precision_unit_filter_enabled": (
                1.0 if self.graph_logic_config.component_precision_filter_units else 0.0
            ),
            "graph_progress/logic_component_precision_unit_filter_frac": float(
                graph_logic_component_precision_unit_filter_segments
                / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_component_precision_unit_filter_drop_frac": float(
                graph_logic_component_precision_unit_filter_drop_segments
                / max(1.0, graph_logic_component_precision_unit_filter_segments)
            ),
            "graph_progress/logic_component_precision_unit_filter_kept_ratio": float(
                graph_logic_component_precision_unit_filter_kept_units
                / max(1.0, graph_logic_component_precision_unit_filter_input_units)
            ),
            "graph_progress/logic_component_precision_unit_filter_dropped_units": float(
                graph_logic_component_precision_unit_filter_dropped_units
            ),
            "graph_progress/logic_component_precision_gate_frac": float(
                graph_logic_component_precision_gate_segments / max(1.0, graph_logic_state_supported_segments)
            ),
            "graph_progress/logic_component_precision_verified_frac": float(
                graph_logic_component_precision_gate_verified_segments
                / max(1.0, graph_logic_component_precision_gate_segments)
            ),
            "graph_progress/logic_component_precision_vague_frac": float(
                graph_logic_component_precision_gate_vague_segments
                / max(1.0, graph_logic_component_precision_gate_segments)
            ),
            "graph_progress/logic_component_precision_noisy_frac": float(
                graph_logic_component_precision_gate_noisy_segments
                / max(1.0, graph_logic_component_precision_gate_segments)
            ),
            "graph_progress/logic_component_precision_schema_scaled": float(
                graph_logic_component_precision_schema_scalar_scaled
                + graph_logic_component_precision_schema_object_scaled
            ),
            "graph_progress/logic_component_precision_schema_blocked": float(
                graph_logic_component_precision_schema_scalar_blocked
                + graph_logic_component_precision_schema_object_blocked
            ),
            "graph_progress/logic_component_precision_schema_multiplier_mean": float(
                graph_logic_component_precision_schema_multiplier_sum
                / max(
                    1.0,
                    graph_logic_component_precision_schema_scalar_scaled
                    + graph_logic_component_precision_schema_object_scaled,
                )
            ),
            "graph_progress/logic_component_precision_vague_multiplier_mean": float(
                graph_logic_component_precision_gate_vague_multiplier
                / max(1.0, graph_logic_component_precision_gate_vague_segments)
            ),
            "graph_progress/logic_component_precision_noisy_multiplier_mean": float(
                graph_logic_component_precision_gate_noisy_multiplier
                / max(1.0, graph_logic_component_precision_gate_noisy_segments)
            ),
            "graph_progress/logic_prefix_credit_enabled": 1.0 if self.graph_logic_prefix_credit else 0.0,
            "graph_progress/logic_prefix_credit_window": float(self.graph_logic_prefix_credit_window),
            "graph_progress/logic_prefix_credit_decay": float(self.graph_logic_prefix_credit_decay),
            "graph_progress/logic_prefix_credit_local_alpha": float(self.graph_logic_prefix_credit_local_alpha),
            "graph_progress/logic_prefix_credit_negative": (
                1.0 if self.graph_logic_prefix_credit_negative else 0.0
            ),
            "graph_progress/logic_family_balance_enabled": (
                1.0 if self.graph_logic_family_balance else 0.0
            ),
            "graph_progress/logic_family_scale_default": float(self.graph_logic_family_scale_default),
            "graph_progress/prefix_decay": float(self.prefix_decay),
            "graph_progress/timing_prepare": float(t_prepare),
            "graph_progress/timing_decode": float(t_decode),
            "graph_progress/timing_encode": float(t_encode),
            "graph_progress/timing_extract": float(t_extract),
            "graph_progress/timing_score": float(t_score),
        }
        for family in _GRAPH_LOGIC_FAMILIES:
            sample_count = float(graph_logic_family_sample_count.get(family, 0.0))
            event_count = float(graph_logic_family_event_count.get(family, 0.0))
            scale = float(self.graph_logic_family_scales.get(family, self.graph_logic_family_scale_default))
            observed_scale = float(graph_logic_family_scale_sum.get(family, 0.0) / max(1.0, sample_count))
            self.last_stats[f"graph_progress/logic_family_scale/{family}"] = scale
            self.last_stats[f"graph_progress/logic_family_observed_scale/{family}"] = observed_scale
            self.last_stats[f"graph_progress/logic_family_samples/{family}"] = sample_count
            self.last_stats[f"graph_progress/logic_family_events/{family}"] = event_count
            self.last_stats[f"graph_progress/logic_family_abs_before/{family}"] = float(
                graph_logic_family_reward_abs_before.get(family, 0.0) / max(1.0, event_count)
            )
            self.last_stats[f"graph_progress/logic_family_abs_after/{family}"] = float(
                graph_logic_family_reward_abs_after.get(family, 0.0) / max(1.0, event_count)
            )
            self.last_stats[f"graph_progress/logic_family_signed_before/{family}"] = float(
                graph_logic_family_reward_signed_before.get(family, 0.0) / max(1.0, event_count)
            )
            self.last_stats[f"graph_progress/logic_family_signed_after/{family}"] = float(
                graph_logic_family_reward_signed_after.get(family, 0.0) / max(1.0, event_count)
            )
        for component in STATE_PROGRESS_COMPONENTS:
            self.last_stats[f"graph_progress/logic_state_component_{component}_mean"] = float(
                graph_logic_state_component_scores[component] / max(1.0, graph_logic_state_supported_segments)
            )
            self.last_stats[f"graph_progress/logic_state_component_{component}_delta_per_segment"] = float(
                graph_logic_state_component_deltas[component] / max(1.0, graph_logic_state_supported_segments)
            )
        self.last_stats.update(audit_stats)

        return {
            "pure_step_reward_tensor": pure_step_reward,
            "step_rewards": step_rewards,
            "anchor_obs": anchor_obs,
            "step_token_positions": step_token_positions,
        }
