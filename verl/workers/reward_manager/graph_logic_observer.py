"""Executable graph observer for graph-process step rewards.

The observer is intentionally conservative: text patterns only propose latent
graph actions, and Python graph executors decide whether a segment can receive
process credit. It is used by ``GraphProgressRewardComputer`` when
``graph_progress_scorer=graph_logic``.
"""

from __future__ import annotations

import ast
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from verl.workers.reward_manager.semantic_action_router import (
    ActionCandidate,
    RoutedAction,
    RouterPayload,
    build_router_payload,
    enumerate_action_candidates,
    route_action_from_payload,
)


KNOWN_NON_TASK_KEYS = {"meta_data"}

PATH_TASKS = {"shortest_path", "weighted_shortest_path"}
TRAVERSAL_TASKS = {"bfs", "dfs"}
MST_TASKS = {"minimum_spanning_tree", "weighted_minimum_spanning_tree"}
ANSWER_EDGE_TASKS = {
    "bipartite_maximum_matching",
    "max_weight_matching",
    "min_edge_covering",
    "bridges",
}
ANSWER_NODE_TASKS = {
    "barycenter",
    "center",
    "dominating_set",
    "maximal_independent_set",
    "min_vertex_cover",
    "periphery",
    "topological_sort",
}
GLOBAL_WITNESS_TASKS = {"global_efficiency", "wiener_index"}
STRICT_TEMPLATE_TASKS = {
    "betweenness_centrality",
    "closeness_centrality",
    "global_efficiency",
    "harmonic_centrality",
    "maximum_flow",
    "weighted_shortest_path",
    "wiener_index",
}
BOOL_WITNESS_TASKS = {"has_cycle", "is_bipartite", "is_eularian", "is_regular", "is_tournament"}
DISTANCE_WITNESS_TASKS = {
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "diameter",
    "radius",
    "center",
    "periphery",
    "barycenter",
    "global_efficiency",
    "wiener_index",
    "local_connectivity",
}
SET_FEASIBILITY_TASKS = {
    "bipartite_maximum_matching",
    "max_weight_matching",
    "min_edge_covering",
    "min_vertex_cover",
    "dominating_set",
    "maximal_independent_set",
}
COUNT_FORMULA_TASKS = {"density", "edge_number", "node_number", "triangles"}
DEGREE_WITNESS_TASKS = {
    "degree",
    "degree_centrality",
    "avg_neighbor_degree",
    "is_regular",
    "is_eularian",
}
SCALAR_FINAL_SCHEMA_TASKS = (
    DISTANCE_WITNESS_TASKS
    | GLOBAL_WITNESS_TASKS
    | COUNT_FORMULA_TASKS
    | {"maximum_flow", "connected_component_number", "strongly_connected_number", "local_connectivity"}
)
HARD_SCALAR_FINAL_SCHEMA_TASKS = {
    "betweenness_centrality",
    "closeness_centrality",
    "global_efficiency",
    "wiener_index",
    "maximum_flow",
}
NODE_LIST_FINAL_SCHEMA_TASKS = (
    ANSWER_NODE_TASKS
    | TRAVERSAL_TASKS
    | {"hamiltonian_path", "traveling_salesman_problem", "shortest_path", "weighted_shortest_path"}
)
# BFS/DFS answers in this benchmark are tree-edge objects. Keep traversal in
# NODE_LIST_FINAL_SCHEMA_TASKS for intermediate order parsing, but let the
# schema gate prefer edge-list final objects when target_edge_order is present.
EDGE_LIST_FINAL_SCHEMA_TASKS = ANSWER_EDGE_TASKS | MST_TASKS | TRAVERSAL_TASKS
MAPPING_FINAL_SCHEMA_TASKS = {"isomophic_mapping"}
OBJECT_FINAL_SCHEMA_TASKS = NODE_LIST_FINAL_SCHEMA_TASKS | EDGE_LIST_FINAL_SCHEMA_TASKS | MAPPING_FINAL_SCHEMA_TASKS
STRICT_OBJECT_FINAL_SCHEMA_TASKS = {
    "isomophic_mapping",
    "min_vertex_cover",
    "dominating_set",
    "maximal_independent_set",
    "hamiltonian_path",
}
LEGACY_CORE4_PROTECTED_TASKS = {
    "bfs",
    "dfs",
    "has_cycle",
    "bridges",
    "connected_component_number",
    "strongly_connected_number",
    "local_connectivity",
    "shortest_path",
    "minimum_spanning_tree",
    "topological_sort",
    "triangles",
    "is_bipartite",
}
LEGACY_CORE4_STATE_TASKS = {
    "weighted_shortest_path",
    "weighted_minimum_spanning_tree",
    "betweenness_centrality",
    "wiener_index",
}
# Tasks whose positive state units are allowed in legacy core4 compatibility
# mode. Other tasks may still run the executor to keep invalid/no-progress
# negative shaping, but their positive units are filtered out.
LEGACY_CORE4_COMPAT_STATE_TASKS = LEGACY_CORE4_STATE_TASKS | LEGACY_CORE4_PROTECTED_TASKS
LEGACY_CORE4_STATE_UNIT_NAMES_BY_TASK = {
    "weighted_shortest_path": {
        "source_distance",
        "dijkstra_candidate_path_cost",
        "dijkstra_predecessor",
        "dijkstra_relax_edge",
        "path_prefix",
        "path_reaches_target",
        "state_dijkstra_checked",
        "state_dijkstra_decreased_dist",
        "state_dijkstra_dist",
        "state_dijkstra_dist_observed",
        "state_dijkstra_init",
        "state_dijkstra_no_relax",
        "state_dijkstra_predecessor",
        "state_dijkstra_relax",
        "state_dijkstra_relax_inferred",
        "state_dijkstra_scan_edge",
        "state_dijkstra_settle",
        "state_dijkstra_target_cert",
        "state_dijkstra_weighted_relax_cert",
        "state_dijkstra_weighted_rule",
        "state_path_cost_valid",
        "state_path_edge",
        "state_path_feasible",
        "state_path_prefix",
        "state_path_reaches_target",
        "state_weighted_path_final_cert",
        "weighted_path_uses_weights",
    },
    "weighted_minimum_spanning_tree": {
        "mst_answer_edge_count",
        "mst_answer_spanning_tree",
        "mst_answer_valid",
        "mst_consider_edge",
        "mst_edge_weight_checked",
        "mst_weight_order",
        "parseable_edge_list_answer",
        "state_mst_accept",
        "state_mst_component_count",
        "state_mst_connected_acyclic",
        "state_mst_consider",
        "state_mst_cycle_reject_cert",
        "state_mst_edge_count_target",
        "state_mst_edge_weight_checked",
        "state_mst_final_weight_checked",
        "state_mst_kruskal_prefix_edge",
        "state_mst_n_minus_one_edges",
        "state_mst_optimal_weight_cert",
        "state_mst_reject_cycle",
        "state_mst_safe_edge",
        "state_mst_spanning_nodes_covered",
        "state_mst_target_edge",
        "state_mst_weight_order",
        "state_mst_weighted_rule",
    },
    "betweenness_centrality": {
        "betweenness_excludes_endpoints",
        "betweenness_formula",
        "betweenness_normalized_value",
        "betweenness_normalizer_value",
        "betweenness_pair_checked",
        "betweenness_pair_considered",
        "betweenness_pair_via",
        "betweenness_pair_via_cert",
        "betweenness_path_via",
        "betweenness_raw_sum",
        "betweenness_sigma_fraction",
        "betweenness_sigma_fraction_value",
        "metric_answer",
    },
    "wiener_index": {
        "all_pairs_pair",
        "all_pairs_scope",
        "distance_pair",
        "distance_sum",
        "distance_sum_node_considered",
        "distance_value",
        "global_metric_answer",
        "global_metric_pair_distance_value",
        "global_pair_distance_scope",
        "wiener_distance_sum_formula",
        "wiener_pair_distance_scope",
        "wiener_pair_distance_value",
        "wiener_pair_scope",
    },
}

ACTION_CUES = {
    "path": (
        "path",
        "route",
        "walk",
        "way",
        "move",
        "move to",
        "proceed",
        "proceed to",
        "travel",
        "arrive",
        "arrive at",
        "start",
        "end",
        "target",
        "source",
        "go to",
        "going to",
        "lead to",
        "leads to",
        "from",
        "then",
        "next",
        "connected",
        "neighbor",
        "dead end",
        "edge",
        "reach",
        "via",
        "through",
        "predecessor",
        "parent",
    ),
    "weighted_path": (
        "distance",
        "dist",
        "cost",
        "weight",
        "shorter",
        "better",
        "best",
        "cheaper",
        "lowest",
        "total",
        "new distance",
        "current distance",
        "update",
        "updated",
        "becomes",
        "set",
        "set to",
        "relax",
        "tentative",
        "priority queue",
        "heap",
        "pop",
        "settle",
        "settled",
        "extract",
        "minimum",
        "smallest",
        "least",
        "dijkstra",
        "candidate",
    ),
    "traversal": (
        "bfs",
        "dfs",
        "queue",
        "stack",
        "enqueue",
        "dequeue",
        "visit",
        "visited",
        "process",
        "mark",
        "unvisited",
        "frontier",
        "level",
        "layer",
        "child",
        "children",
        "traversal",
        "discover",
        "found",
        "expand",
        "scan",
        "explore",
        "neighbor",
        "backtrack",
    ),
    "mst": (
        "mst",
        "spanning tree",
        "kruskal",
        "prim",
        "add",
        "include",
        "select",
        "choose",
        "consider",
        "take",
        "pick",
        "accept",
        "reject",
        "skip",
        "discard",
        "safe edge",
        "safe",
        "connection",
        "connect",
        "join",
        "link",
        "lightest",
        "cheapest",
        "smallest",
        "lowest",
        "cycle",
        "loop",
        "component",
        "components",
        "different component",
        "same component",
        "union",
        "merge",
        "merged",
        "minimum",
        "weight",
        "sorted",
    ),
    "metric": (
        "degree",
        "centrality",
        "coefficient",
        "index",
        "distance",
        "shortest",
        "sum",
        "average",
        "count",
        "number",
        "reachable",
        "component",
        "scc",
        "strongly connected",
        "eccentricity",
        "density",
        "efficiency",
        "diameter",
        "radius",
        "center",
        "median",
        "barycenter",
        "neighbor",
        "common",
    ),
    "set": (
        "set",
        "cover",
        "matching",
        "independent",
        "dominating",
        "vertex",
        "edge cover",
        "include",
        "select",
        "choose",
        "candidate",
        "valid",
        "uncovered",
        "covered",
        "endpoint",
        "matched",
        "unmatched",
    ),
    "bool": (
        "yes",
        "no",
        "cycle",
        "bipartite",
        "regular",
        "tournament",
        "euler",
        "component",
        "connected",
        "odd",
        "even",
        "partition",
    ),
    "flow": (
        "flow",
        "capacity",
        "source",
        "sink",
        "augment",
        "augmenting",
        "send",
        "push",
        "increase",
        "cut",
        "residual",
        "remaining capacity",
        "reverse edge",
        "bottleneck",
        "min capacity",
        "minimum capacity",
    ),
    "ordering": (
        "topological",
        "hamiltonian",
        "salesman",
        "tsp",
        "tour",
        "route",
        "cycle",
        "order",
        "precedes",
        "before",
        "after",
        "path",
    ),
}


@dataclass
class GraphLogicConfig:
    ordered_reward: float = 1.0
    progress_reward: float = 0.25
    answer_reward: float = 0.5
    legal_reward: float = 0.0
    invalid_penalty: float = 0.2
    fact_dump_penalty: float = 0.0
    max_steps: int = 64
    segment_merge_strategy: str = "none"
    segment_merge_target_count: int = 0
    segment_merge_max_chars: int = 1200
    segment_merge_min_info_score: float = 0.0
    reward_mode: str = "event"
    potential_scale: float = 1.0
    state_shaping: bool = False
    state_reward_scale: float = 1.0
    state_tasks: str = ""
    state_calibrate_potential: bool = True
    state_invalid_penalty: float = 0.03
    state_mention_reward: float = 0.005
    state_consistent_reward: float = 0.02
    duplicate_no_progress_penalty: float = 0.002
    state_context_max_age: int = 2
    positive_reward_cap: float = 0.25
    positive_reward_budget_mode: str = "quality"
    final_answer_checks: bool = False
    step_span_mode: str = "segment"
    action_span_max_per_segment: int = 6
    action_span_context_chars: int = 0
    state_candidate_grounding: bool = True
    state_candidate_max_edges: int = 3
    state_commitment_weight: bool = True
    state_commitment_min_multiplier: float = 0.75
    state_commitment_max_multiplier: float = 1.20
    executable_commitment_weight: bool = False
    ambiguity_gate_positive: bool = False
    ambiguity_gate_threshold: float = 0.50
    ambiguity_gate_multiplier: float = 0.0
    certificate_only_hard_gate: bool = False
    certificate_gate_task_allowlist: str = ""
    certificate_gate_min_exec_score: float = 0.60
    certificate_gate_max_ambiguity: float = 0.35
    graph_sensitivity_gate_positive: bool = False
    graph_sensitivity_gate_task_allowlist: str = ""
    graph_sensitivity_gate_multiplier: float = 0.25
    component_precision_gate: bool = False
    component_precision_filter_units: bool = False
    component_precision_task_allowlist: str = ""
    component_vague_potential_multiplier: float = 0.20
    component_noisy_positive_multiplier: float = 0.0
    component_schema_parseable_multiplier: float = 0.05
    oracle_milestones: bool = False
    oracle_milestone_max_units: int = 96
    oracle_milestone_positive_only: bool = False
    oracle_milestone_count_actions: bool = True
    oracle_milestone_task_allowlist: str = ""
    action_router_enable: bool = False
    action_router_mode: str = "shadow"
    action_router_task_allowlist: str = ""
    action_router_conf_threshold: float = 0.75
    action_router_high_conf_threshold: float = 0.90
    action_router_positive_cap: float = 0.05
    action_router_negative: bool = False
    action_router_max_candidates: int = 16
    action_router_backend: str = "heuristic"
    action_router_base_url: str = ""
    action_router_model: str = "qwen3-0.6b-action-router"
    action_router_timeout: float = 2.0
    action_router_max_tokens: int = 96
    action_router_max_segments_per_response: int = 0
    action_router_heuristic_fallback: bool = False
    action_router_heuristic_fallback_threshold: float = 0.75
    action_router_trace: bool = True
    action_router_payload_trace: bool = False
    schema_conflict_gate_positive: bool = True
    schema_conflict_penalty_scale: float = 0.50
    schema_final_object_missing_penalty_scale: float = 0.12
    schema_final_object_missing_min_chars: int = 8000
    legacy_core4_compat: bool = False


@dataclass
class GraphLogicSpec:
    task: str
    nodes: set[int]
    edges: set[tuple[int, int]]
    weights: dict[tuple[int, int], float]
    edge_order: list[tuple[int, int]]
    directed: bool
    source: int | None = None
    target: int | None = None
    target_edges: set[tuple[int, int]] | None = None
    target_edge_order: list[tuple[int, int]] | None = None
    alt_target_edge_orders: list[list[tuple[int, int]]] | None = None
    target_nodes: set[int] | None = None
    target_node_order: list[int] | None = None
    answer_numbers: list[float] | None = None
    answer_bool: bool | None = None
    answer_mapping: dict[int, int] | None = None
    distance_cache: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    path_count_cache: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)


@dataclass
class GraphLogicStep:
    reward: float
    anchor: object
    char_span: tuple[int, int]
    label: str = "NEUTRAL"
    first_error: bool = False
    unit_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphLogicActionEvent:
    """Executable observer action localized to the response text."""

    kind: str
    unit: tuple
    char_span: tuple[int, int]
    confidence: float = 1.0
    rewardable: bool = False
    new: bool = False


@dataclass
class GraphLogicPotentialState:
    seen_edges: set[tuple[int, int]]
    seen_nodes: set[int]
    edge_positions: list[int]
    node_pos: int = 0
    answer_seen: bool = False
    seen_evidence: set[tuple] = field(default_factory=set)


@dataclass
class GraphLogicAlgorithmState:
    """Mutable task-state used by optional algorithm-state shaping."""

    task: str
    seen_units: set[tuple] = field(default_factory=set)
    order_pos: int = 0
    node_order_pos: int = 0
    visited_nodes: set[int] = field(default_factory=set)
    settled_nodes: set[int] = field(default_factory=set)
    dist: dict[int, float] = field(default_factory=dict)
    parent_node: dict[int, int] = field(default_factory=dict)
    dsu_parent: dict[int, int] = field(default_factory=dict)
    dsu_rank: dict[int, int] = field(default_factory=dict)
    accepted_edges: set[tuple[int, int]] = field(default_factory=set)
    rejected_edges: set[tuple[int, int]] = field(default_factory=set)
    total_weight: float = 0.0
    residual: dict[tuple[int, int], float] = field(default_factory=dict)
    flow_value: float = 0.0
    max_flow_value: float = 0.0
    partial_mapping: dict[int, int] = field(default_factory=dict)
    invalid_steps: set[tuple] = field(default_factory=set)
    invalid_weight: float = 0.0
    recent_nodes: list[int] = field(default_factory=list)
    recent_edges: list[tuple[int, int]] = field(default_factory=list)
    context_age: int = 10**9


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


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


def _norm_edge(u: int, v: int, directed: bool = False) -> tuple[int, int]:
    return (u, v) if directed or u <= v else (v, u)


def _parse_sample_nodes(sample: Any) -> list[int]:
    if sample is None:
        return []
    if isinstance(sample, str) and sample.strip().lower() in ("", "none", "null"):
        return []
    if isinstance(sample, dict):
        values = []
        for k, v in sample.items():
            values.extend([k, v])
    elif isinstance(sample, (list, tuple, set)):
        values = list(sample)
    else:
        values = [sample]
    out = []
    seen = set()
    for value in values:
        iv = _as_int(value)
        if iv is not None and iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out


def _answer_numbers(answer: Any) -> list[float]:
    if isinstance(answer, bool):
        return []
    if isinstance(answer, (int, float)):
        return [float(answer)]
    s = str(answer)
    if "[" in s or "(" in s or "{" in s:
        return []
    out = []
    for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", s, flags=re.I):
        if len(x) > 64:
            continue
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def _answer_bool(answer: Any) -> bool | None:
    s = str(answer).strip().lower()
    if s in {"yes", "true"}:
        return True
    if s in {"no", "false"}:
        return False
    return None


def _answer_mapping(answer: Any) -> dict[int, int] | None:
    parsed = _literal(answer, default=None)
    if isinstance(parsed, dict):
        out: dict[int, int] = {}
        for key, value in parsed.items():
            k = _as_int(key)
            v = _as_int(value)
            if k is not None and v is not None:
                out[k] = v
        return out or None
    return None


def _literal_edges_or_nodes(
    answer: Any,
    directed: bool,
) -> tuple[list[int], set[tuple[int, int]], list[tuple[int, int]]]:
    s = str(answer)
    if isinstance(answer, dict) or (s.strip().startswith("{") and ":" in s):
        return [int(x) for x in re.findall(r"\b\d+\b", s) if len(x) <= 12], set(), []
    triples = re.findall(r"\((\d+)\s*,\s*(\d+)\s*,\s*\d+\)", s)
    pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", s)
    edge_order = [_norm_edge(int(a), int(b), directed) for a, b in pairs + triples]
    edges = set(edge_order)
    nodes: list[int] = []
    if not edges and ("[" in s or "," in s):
        nodes = [int(x) for x in re.findall(r"\b\d+\b", s) if len(x) <= 12]
        edge_order = [_norm_edge(u, v, directed) for u, v in zip(nodes, nodes[1:])]
        edges = set(edge_order)
    return nodes, edges, edge_order


def build_graph_logic_spec(task: str, extra_info: dict, answer: Any, sample: Any) -> GraphLogicSpec | None:
    task = str(task or "").lower()
    directed = str(extra_info.get("direction", "")).lower() == "directed"
    if task == "isomophic_mapping":
        answer_mapping = _answer_mapping(answer)
        nodes_spec = _literal(extra_info.get("nodes"), default=None)
        g1_nodes: set[int] = set()
        g2_nodes: set[int] = set()
        if isinstance(nodes_spec, (list, tuple)) and len(nodes_spec) >= 2:
            for raw in nodes_spec[0]:
                node = _as_int(raw)
                if node is not None:
                    g1_nodes.add(node)
            for raw in nodes_spec[1]:
                node = _as_int(raw)
                if node is not None:
                    g2_nodes.add(node)
        if answer_mapping:
            g1_nodes.update(answer_mapping.keys())
            g2_nodes.update(answer_mapping.values())

        edges_raw = _literal(extra_info.get("edges"), default=[]) or []
        edge_order: list[tuple[int, int]] = []
        edges: set[tuple[int, int]] = set()
        weights: dict[tuple[int, int], float] = {}
        graph_edge_lists = []
        if (
            isinstance(edges_raw, (list, tuple))
            and len(edges_raw) == 2
            and all(isinstance(part, (list, tuple)) for part in edges_raw)
        ):
            graph_edge_lists = list(edges_raw)
        elif isinstance(edges_raw, (list, tuple)):
            graph_edge_lists = [edges_raw]
        for part in graph_edge_lists:
            for raw in part:
                if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                    continue
                u, v = _as_int(raw[0]), _as_int(raw[1])
                if u is None or v is None:
                    continue
                edge = _norm_edge(u, v, directed)
                if edge not in edges:
                    edges.add(edge)
                    edge_order.append(edge)
                weights[edge] = 1.0
                if g1_nodes and (u in g1_nodes or v in g1_nodes):
                    g1_nodes.update((u, v))
                elif g2_nodes and (u in g2_nodes or v in g2_nodes):
                    g2_nodes.update((u, v))
        nodes = set(g1_nodes) | set(g2_nodes)
        if not nodes or not answer_mapping:
            return None
        return GraphLogicSpec(
            task=task,
            nodes=nodes,
            edges=edges,
            weights=weights,
            edge_order=edge_order,
            directed=directed,
            target_nodes=set(answer_mapping.keys()) | set(answer_mapping.values()),
            answer_mapping=answer_mapping,
        )

    edges_raw = _literal(extra_info.get("edges"), default=[]) or []
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    weights: dict[tuple[int, int], float] = {}
    edge_order: list[tuple[int, int]] = []
    for raw in edges_raw:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        u, v = _as_int(raw[0]), _as_int(raw[1])
        if u is None or v is None:
            continue
        try:
            w = float(raw[2]) if len(raw) >= 3 else 1.0
        except Exception:
            w = 1.0
        edge = _norm_edge(u, v, directed)
        nodes.update((u, v))
        if edge not in edges:
            edge_order.append(edge)
            edges.add(edge)
        weights[edge] = min(float(weights.get(edge, w)), w)

    nodes_spec = _literal(extra_info.get("nodes"), default=None)
    if isinstance(nodes_spec, (list, tuple)) and len(nodes_spec) >= 2:
        a, b = _as_int(nodes_spec[0]), _as_int(nodes_spec[1])
        if a is not None and b is not None and a <= b:
            nodes.update(range(a, b + 1))
    if not nodes or not edges:
        return None

    sample_nodes = _parse_sample_nodes(sample)
    source = sample_nodes[0] if sample_nodes else None
    target = sample_nodes[1] if len(sample_nodes) >= 2 else None
    answer_nodes, answer_edges, answer_edge_order = _literal_edges_or_nodes(answer, directed)
    if answer_nodes and task not in {"shortest_path", "weighted_shortest_path", "hamiltonian_path"}:
        answer_edges = set()
        answer_edge_order = []
    spec = GraphLogicSpec(
        task=task,
        nodes=nodes,
        edges=edges,
        weights=weights,
        edge_order=edge_order,
        directed=directed,
        source=source,
        target=target,
        target_edges=answer_edges or None,
        target_edge_order=answer_edge_order or None,
        target_nodes=set(answer_nodes) if answer_nodes else None,
        target_node_order=answer_nodes or None,
        answer_numbers=_answer_numbers(answer) or None,
        answer_bool=_answer_bool(answer),
        answer_mapping=_answer_mapping(answer),
    )
    apply_executor_targets(spec)
    return spec


def build_adj(spec: GraphLogicSpec) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for u, v in spec.edge_order:
        if (u, v) not in seen:
            adj[u].append(v)
            seen.add((u, v))
        if not spec.directed and (v, u) not in seen:
            adj[v].append(u)
            seen.add((v, u))
    return adj


def _path_from_prev(prev: dict[int, int | None], target: int, directed: bool) -> list[tuple[int, int]]:
    if target not in prev:
        return []
    nodes = []
    cur: int | None = target
    while cur is not None:
        nodes.append(cur)
        cur = prev[cur]
    nodes.reverse()
    return [_norm_edge(u, v, directed) for u, v in zip(nodes, nodes[1:])]


def bfs_path_edges(spec: GraphLogicSpec) -> list[tuple[int, int]]:
    if spec.source is None or spec.target is None:
        return []
    adj = build_adj(spec)
    queue = [spec.source]
    prev: dict[int, int | None] = {spec.source: None}
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if u == spec.target:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                queue.append(v)
    return _path_from_prev(prev, spec.target, spec.directed)


def dijkstra_path_and_relax(spec: GraphLogicSpec) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if spec.source is None:
        return [], []
    import heapq

    adj = build_adj(spec)
    dist: dict[int, float] = {spec.source: 0.0}
    prev: dict[int, int | None] = {spec.source: None}
    heap = [(0.0, spec.source)]
    done: set[int] = set()
    relax: list[tuple[int, int]] = []
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in adj.get(u, []):
            edge = _norm_edge(u, v, spec.directed)
            nd = d + float(spec.weights.get(edge, 1.0))
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                relax.append(edge)
                heapq.heappush(heap, (nd, v))
    path = _path_from_prev(prev, spec.target, spec.directed) if spec.target is not None else []
    return path, relax


def bfs_tree_edges(spec: GraphLogicSpec) -> list[tuple[int, int]]:
    if spec.source is None:
        return []
    adj = build_adj(spec)
    queue = [spec.source]
    seen = {spec.source}
    out = []
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                queue.append(v)
                out.append(_norm_edge(u, v, spec.directed))
    return out


def dfs_tree_edges(spec: GraphLogicSpec) -> list[tuple[int, int]]:
    if spec.source is None:
        return []
    adj = build_adj(spec)
    seen: set[int] = set()
    out: list[tuple[int, int]] = []

    def visit(u: int) -> None:
        seen.add(u)
        for v in adj.get(u, []):
            if v not in seen:
                out.append(_norm_edge(u, v, spec.directed))
                visit(v)

    visit(spec.source)
    return out


def kruskal_edges(spec: GraphLogicSpec) -> list[tuple[int, int]]:
    parent = {n: n for n in spec.nodes}
    rank = {n: 0 for n in spec.nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    edge_index = {e: i for i, e in enumerate(spec.edge_order)}
    ordered = sorted(spec.edge_order, key=lambda e: (spec.weights.get(e, 1.0), edge_index[e]))
    out = []
    for u, v in ordered:
        if union(u, v):
            out.append(_norm_edge(u, v, spec.directed))
            if len(out) >= max(0, len(spec.nodes) - 1):
                break
    return out


def prim_edges(spec: GraphLogicSpec) -> list[tuple[int, int]]:
    if not spec.nodes:
        return []
    import heapq

    adj: dict[int, list[tuple[float, int, int, int, tuple[int, int]]]] = defaultdict(list)
    edge_index = {e: i for i, e in enumerate(spec.edge_order)}
    for edge in spec.edge_order:
        u, v = edge
        w = float(spec.weights.get(edge, 1.0))
        idx = edge_index[edge]
        adj[u].append((w, idx, u, v, edge))
        if not spec.directed:
            adj[v].append((w, idx, v, u, edge))

    out: list[tuple[int, int]] = []
    seen: set[int] = set()
    starts = [spec.source] if spec.source in spec.nodes else []
    starts.extend(n for n in sorted(spec.nodes) if n not in starts)
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        heap = list(adj.get(start, []))
        heapq.heapify(heap)
        while heap:
            _, _, _, to_node, edge = heapq.heappop(heap)
            if to_node in seen:
                continue
            seen.add(to_node)
            out.append(edge)
            for item in adj.get(to_node, []):
                if item[3] not in seen:
                    heapq.heappush(heap, item)
    return out


def tsp_cycle_edges(spec: GraphLogicSpec, max_nodes: int = 12) -> list[tuple[int, int]]:
    if not spec.nodes or len(spec.nodes) > max_nodes:
        return []
    nodes = sorted(spec.nodes)
    if len(nodes) <= 1:
        return []
    start = nodes[0]
    others = nodes[1:]

    def weight(u: int, v: int) -> float:
        return float(spec.weights.get(_norm_edge(u, v, spec.directed), math.inf))

    dp: dict[tuple[int, int], tuple[float, int | None]] = {}
    for j, node in enumerate(others):
        w = weight(start, node)
        if math.isfinite(w):
            dp[(1 << j, j)] = (w, None)
    full = (1 << len(others)) - 1
    for mask in range(1, full + 1):
        for j, node_j in enumerate(others):
            if not (mask & (1 << j)) or (mask, j) not in dp:
                continue
            cost, _ = dp[(mask, j)]
            for k, node_k in enumerate(others):
                if mask & (1 << k):
                    continue
                w = weight(node_j, node_k)
                if not math.isfinite(w):
                    continue
                state = (mask | (1 << k), k)
                cand = cost + w
                if state not in dp or cand < dp[state][0]:
                    dp[state] = (cand, j)
    best: tuple[float, int] | None = None
    for j, node_j in enumerate(others):
        state = (full, j)
        if state not in dp:
            continue
        back = weight(node_j, start)
        if not math.isfinite(back):
            continue
        cand = dp[state][0] + back
        if best is None or cand < best[0]:
            best = (cand, j)
    if best is None:
        return []
    order_rev = [others[best[1]]]
    mask, j = full, best[1]
    while True:
        _, parent = dp[(mask, j)]
        if parent is None:
            break
        mask ^= 1 << j
        j = parent
        order_rev.append(others[j])
    path = [start] + list(reversed(order_rev))
    return [_norm_edge(u, v, spec.directed) for u, v in zip(path, path[1:] + [start])]


def incident_edges(spec: GraphLogicSpec, node: int) -> set[tuple[int, int]]:
    return {edge for edge in spec.edges if node in edge}


def connected_component_from(spec: GraphLogicSpec, source: int) -> tuple[set[int], set[tuple[int, int]]]:
    adj = build_adj(spec)
    seen = {source}
    queue = [source]
    tree: set[tuple[int, int]] = set()
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj.get(u, []):
            edge = _norm_edge(u, v, spec.directed)
            if v not in seen:
                seen.add(v)
                queue.append(v)
                tree.add(edge)
    return seen, tree


def all_bfs_tree_edges(spec: GraphLogicSpec) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for source in sorted(spec.nodes):
        tmp = GraphLogicSpec(
            task=spec.task,
            nodes=spec.nodes,
            edges=spec.edges,
            weights=spec.weights,
            edge_order=spec.edge_order,
            directed=spec.directed,
            source=source,
        )
        out.update(bfs_tree_edges(tmp))
    return out


def bridge_edges(spec: GraphLogicSpec) -> set[tuple[int, int]]:
    if spec.directed:
        return set()
    adj = build_adj(spec)
    tin: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[tuple[int, int]] = set()
    timer = 0

    def dfs(u: int, parent: int | None) -> None:
        nonlocal timer
        timer += 1
        tin[u] = low[u] = timer
        for v in adj.get(u, []):
            if v == parent:
                continue
            if v in tin:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    bridges.add(_norm_edge(u, v, False))

    for node in sorted(spec.nodes):
        if node not in tin:
            dfs(node, None)
    return bridges


def cycle_witness_edges(spec: GraphLogicSpec) -> set[tuple[int, int]]:
    if spec.directed:
        return set()
    adj = build_adj(spec)
    seen: set[int] = set()
    parent: dict[int, int | None] = {}

    def dfs(u: int) -> set[tuple[int, int]] | None:
        seen.add(u)
        for v in adj.get(u, []):
            if v == parent.get(u):
                continue
            edge = _norm_edge(u, v, False)
            if v in seen:
                return {edge}
            parent[v] = u
            got = dfs(v)
            if got:
                got.add(edge)
                return got
        return None

    for node in sorted(spec.nodes):
        if node not in seen:
            parent[node] = None
            got = dfs(node)
            if got:
                return got
    return set()


def bipartite_witness(spec: GraphLogicSpec) -> tuple[set[int], set[tuple[int, int]]]:
    adj = build_adj(spec)
    color: dict[int, int] = {}
    tree: set[tuple[int, int]] = set()
    for start in sorted(spec.nodes):
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj.get(u, []):
                edge = _norm_edge(u, v, spec.directed)
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                    tree.add(edge)
                elif color[v] == color[u]:
                    return {u, v}, {edge}
    return set(spec.nodes), tree


def tournament_witness(spec: GraphLogicSpec) -> tuple[set[int], set[tuple[int, int]]]:
    if not spec.directed:
        return set(spec.nodes), set(spec.edges)
    for i, u in enumerate(sorted(spec.nodes)):
        for v in sorted(spec.nodes)[i + 1 :]:
            uv = (u, v) in spec.edges
            vu = (v, u) in spec.edges
            if uv == vu:
                edges = ({(u, v), (v, u)} & spec.edges) if uv else set()
                return {u, v}, edges
    return set(spec.nodes), set(spec.edges)


def apply_executor_targets(spec: GraphLogicSpec) -> None:
    task = spec.task
    if task == "shortest_path":
        order = bfs_path_edges(spec)
        spec.target_edge_order = order
        spec.target_edges = set(order)
        spec.target_nodes = {x for e in order for x in e}
    elif task == "weighted_shortest_path":
        path, relax = dijkstra_path_and_relax(spec)
        spec.target_edge_order = path
        spec.target_edges = set(path) | set(relax)
        spec.target_nodes = {x for e in spec.target_edges for x in e}
    elif task == "bfs":
        order = bfs_tree_edges(spec)
        spec.target_edge_order = order
        spec.target_edges = set(order)
        spec.target_nodes = {x for e in order for x in e}
    elif task == "dfs":
        order = dfs_tree_edges(spec)
        spec.target_edge_order = order
        spec.target_edges = set(order)
        spec.target_nodes = {x for e in order for x in e}
    elif task in MST_TASKS:
        kruskal_order = kruskal_edges(spec)
        prim_order = prim_edges(spec)
        spec.target_edge_order = kruskal_order
        if prim_order and prim_order != kruskal_order:
            spec.alt_target_edge_orders = [prim_order]
        target = set(kruskal_order) | set(prim_order)
        spec.target_edges = target
        spec.target_nodes = {x for e in target for x in e}
    elif task == "traveling_salesman_problem":
        order = tsp_cycle_edges(spec)
        spec.target_edge_order = order
        spec.target_edges = set(order)
        spec.target_nodes = {x for e in order for x in e}
    elif task == "topological_sort":
        if spec.target_node_order:
            spec.target_nodes = set(spec.target_node_order)
    elif task == "local_connectivity":
        order = bfs_path_edges(spec)
        spec.target_edge_order = order
        spec.target_edges = set(order)
        spec.target_nodes = {x for e in order for x in e} | ({spec.source, spec.target} - {None})
    elif task == "maximum_flow" and spec.source is not None:
        reach, tree = connected_component_from(spec, spec.source)
        spec.target_edges = tree | incident_edges(spec, spec.source)
        spec.target_nodes = reach | ({spec.target} if spec.target is not None else set())
    elif task in {"neighbor", "degree", "degree_centrality", "avg_neighbor_degree"} and spec.source is not None:
        adj = build_adj(spec)
        spec.target_nodes = set(adj.get(spec.source, [])) | {spec.source}
        spec.target_edges = incident_edges(spec, spec.source)
    elif task in {"common_neighbor", "jaccard_coefficient", "adamic_adar_index", "resource_allocation_index"}:
        if spec.source is not None and spec.target is not None:
            adj = build_adj(spec)
            common = set(adj.get(spec.source, [])) & set(adj.get(spec.target, []))
            spec.target_nodes = common | {spec.source, spec.target}
            edges = set()
            for c in common:
                for u in (spec.source, spec.target):
                    edge = _norm_edge(u, c, spec.directed)
                    if edge in spec.edges:
                        edges.add(edge)
            spec.target_edges = edges
    elif task == "bridges":
        target = bridge_edges(spec)
        spec.target_edges = target
        spec.target_edge_order = [e for e in spec.edge_order if e in target]
        spec.target_nodes = {x for e in target for x in e}
    elif task == "has_cycle":
        target = cycle_witness_edges(spec)
        spec.target_edges = target
        spec.target_edge_order = [e for e in spec.edge_order if e in target]
        spec.target_nodes = {x for e in target for x in e}
    elif task == "is_bipartite":
        nodes, edges = bipartite_witness(spec)
        spec.target_nodes = nodes
        spec.target_edges = edges
    elif task == "is_tournament":
        nodes, edges = tournament_witness(spec)
        spec.target_nodes = nodes
        spec.target_edges = edges
    elif task in {"is_regular", "is_eularian"}:
        degs = {u: 0 for u in spec.nodes}
        for u, v in spec.edges:
            degs[u] = degs.get(u, 0) + 1
            if not spec.directed:
                degs[v] = degs.get(v, 0) + 1
        if task == "is_regular":
            vals = list(degs.values())
            nodes = {u for u, d in degs.items() if vals and d in {min(vals), max(vals)}}
        else:
            odd = {u for u, d in degs.items() if d % 2 == 1}
            nodes = odd or set(spec.nodes)
        spec.target_nodes = nodes
        edges = set()
        for node in nodes:
            edges.update(incident_edges(spec, node))
        spec.target_edges = edges
    elif task in GLOBAL_WITNESS_TASKS:
        spec.target_nodes = set(spec.nodes)
        spec.target_edges = all_bfs_tree_edges(spec)
    elif task == "edge_number":
        spec.target_edges = set(spec.edges)
    elif task == "node_number":
        spec.target_nodes = set(spec.nodes)
    elif task in ANSWER_NODE_TASKS:
        if spec.target_nodes:
            edges = set()
            for node in spec.target_nodes:
                edges.update(incident_edges(spec, node))
            spec.target_edges = spec.target_edges or edges


def _segment_text_information_score(segment_text: str) -> float:
    """Cheap text-side information proxy used before the model/logit tensors exist.

    This is not model entropy. It approximates segment informativeness from lexical
    diversity, graph/action payload density, and action-cue evidence. The real
    model-token entropy can be plugged in later by replacing this scorer while
    keeping the same relative merge loop.
    """

    stripped = segment_text.strip()
    if not stripped:
        return 0.0
    words = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?", stripped.lower())
    if not words:
        words = list(stripped)
    counts: dict[str, int] = defaultdict(int)
    for word in words:
        counts[word] += 1
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = float(count) / max(1.0, total)
        entropy -= p * math.log(p + 1e-12)
    norm_entropy = entropy / max(1.0, math.log(max(2.0, float(len(counts)))))

    low = stripped.lower()
    action_cues = len(_ACTION_SPAN_CUE_RE.findall(stripped))
    edge_mentions = len(re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)", stripped))
    numeric_mentions = len(re.findall(r"\d+(?:\.\d+)?", stripped))
    formula_mentions = len(re.findall(r"(?:dist|distance|capacity|flow|weight|sum|sigma|d\s*\()", low))
    length_factor = min(1.0, len(stripped) / 400.0)

    density = (
        0.10 * min(1.0, action_cues / 3.0)
        + 0.08 * min(1.0, edge_mentions / 2.0)
        + 0.06 * min(1.0, numeric_mentions / 6.0)
        + 0.06 * min(1.0, formula_mentions / 3.0)
    )
    return float(0.70 * norm_entropy + 0.10 * length_factor + density)


def _relative_entropy_merge_segments(
    text: str,
    spans: list[tuple[int, int, str]],
    *,
    target_count: int,
    max_chars: int,
    min_info_score: float,
    score_fn: Callable[[int, int, str], float] | None = None,
    stats: dict[str, float] | None = None,
) -> list[tuple[int, int, str]]:
    if target_count <= 0 or len(spans) <= target_count:
        return spans
    if max_chars <= 0:
        max_chars = 10**9

    scorer = score_fn or (lambda _s, _e, seg: _segment_text_information_score(seg))
    merged = [(int(s), int(e), str(seg)) for s, e, seg in spans]
    raw_scores = [float(scorer(s, e, seg)) for s, e, seg in merged]
    if stats is not None:
        stats["segment_merge_initial_segments"] = float(len(merged))
        stats["segment_merge_initial_info_sum"] = float(sum(raw_scores))

    merge_count = 0
    score_sum = 0.0
    while len(merged) > target_count:
        best_idx = -1
        best_score = float("inf")
        best_text = ""
        for idx in range(len(merged) - 1):
            start = merged[idx][0]
            end = merged[idx + 1][1]
            if end <= start or end - start > max_chars:
                continue
            combined = text[start:end].strip()
            if not combined:
                continue
            info_score = float(scorer(start, end, combined))
            # Preserve clear high-information action boundaries unless the user
            # explicitly lowers the threshold to zero.
            if min_info_score > 0.0 and info_score >= min_info_score:
                continue
            if info_score < best_score:
                best_idx = idx
                best_score = info_score
                best_text = combined
        if best_idx < 0:
            break

        start = merged[best_idx][0]
        end = merged[best_idx + 1][1]
        merged[best_idx : best_idx + 2] = [(start, end, best_text)]
        merge_count += 1
        score_sum += float(best_score)

    if stats is not None:
        final_scores = [float(scorer(s, e, seg)) for s, e, seg in merged]
        stats["segment_merge_final_segments"] = float(len(merged))
        stats["segment_merge_count"] = float(merge_count)
        stats["segment_merge_selected_info_sum"] = float(score_sum)
        stats["segment_merge_final_info_sum"] = float(sum(final_scores))
    return merged


def _token_entropy_score_fn(
    token_entropy_offsets: Sequence[Sequence[int] | tuple[int, int]],
    token_entropies: Sequence[float],
) -> Callable[[int, int, str], float]:
    pairs: list[tuple[int, int, float]] = []
    for off, ent in zip(token_entropy_offsets, token_entropies):
        try:
            left = int(off[0])
            right = int(off[1])
            val = float(ent)
        except Exception:
            continue
        if right <= left or not math.isfinite(val):
            continue
        pairs.append((left, right, val))
    if not pairs:
        raise ValueError("model_entropy segment merge requires non-empty token entropy offsets")

    def score(start: int, end: int, segment_text: str) -> float:
        total = 0.0
        weight = 0.0
        for left, right, val in pairs:
            overlap = min(end, right) - max(start, left)
            if overlap <= 0:
                continue
            total += float(overlap) * val
            weight += float(overlap)
        if weight > 0.0:
            return total / weight
        # If tokenizer offset mapping cannot cover a whitespace-only or decoded
        # artifact span, fail loudly only for completely empty segments. For real
        # text, use an extremely high score so it will not be preferentially
        # merged as "low entropy".
        return 1e9 if segment_text.strip() else 0.0

    return score


def split_segments_with_spans(
    text: str,
    config: GraphLogicConfig | None = None,
    stats: dict[str, float] | None = None,
    token_entropy_offsets: Sequence[Sequence[int] | tuple[int, int]] | None = None,
    token_entropies: Sequence[float] | None = None,
) -> list[tuple[int, int, str]]:
    def trim_span(start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def line_looks_actionable(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r"(?:[-*]|\d+[\).:]|step\s*\d+[\).:]?)\s+", stripped, flags=re.I):
            return True
        if re.search(r"\(\d+\s*,\s*\d+", stripped):
            return True
        low = stripped.lower()
        return any(
            cue in low
            for cue in (
                "visit",
                "select",
                "choose",
                "consider",
                "update",
                "relax",
                "settle",
                "enqueue",
                "dequeue",
                "add",
                "add edge",
                "include",
                "accept",
                "skip edge",
                "reject",
                "discard",
                "union",
                "component",
                "distance",
                "augment",
                "bottleneck",
            )
        )

    def add_line_chunks(spans_out: list[tuple[int, int, str]], start: int, end: int) -> bool:
        lines = list(re.finditer(r"[^\n]+", text[start:end]))
        if len(lines) < 2:
            return False
        actionable = sum(1 for m in lines if line_looks_actionable(m.group(0)))
        if actionable < 2:
            return False
        for m in lines:
            line_start, line_end = trim_span(start + m.start(), start + m.end())
            if line_start >= line_end:
                continue
            if line_end - line_start <= 900:
                spans_out.append((line_start, line_end, text[line_start:line_end]))
            else:
                add_long_chunk(spans_out, line_start, line_end)
        return True

    def add_long_chunk(spans_out: list[tuple[int, int, str]], start: int, end: int) -> None:
        sent_start = start
        i = start
        while i < end:
            is_boundary = text[i] in ".!?" and (i + 1 >= end or text[i + 1].isspace())
            too_long = i - sent_start >= 900 and text[i].isspace()
            if is_boundary or too_long:
                seg_start, seg_end = trim_span(sent_start, i + 1)
                if seg_start < seg_end:
                    spans_out.append((seg_start, seg_end, text[seg_start:seg_end]))
                sent_start = i + 1
                while sent_start < end and text[sent_start].isspace():
                    sent_start += 1
                i = sent_start
                continue
            i += 1
        seg_start, seg_end = trim_span(sent_start, end)
        if seg_start < seg_end:
            spans_out.append((seg_start, seg_end, text[seg_start:seg_end]))

    spans: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"\n\s*\n", text):
        raw_start, raw_end = trim_span(start, m.start())
        start = m.end()
        if raw_start >= raw_end:
            continue
        if add_line_chunks(spans, raw_start, raw_end):
            continue
        if raw_end - raw_start <= 900:
            spans.append((raw_start, raw_end, text[raw_start:raw_end]))
            continue
        add_long_chunk(spans, raw_start, raw_end)
    raw_start, raw_end = trim_span(start, len(text))
    if raw_start < raw_end:
        if add_line_chunks(spans, raw_start, raw_end):
            pass
        elif raw_end - raw_start <= 900:
            spans.append((raw_start, raw_end, text[raw_start:raw_end]))
        else:
            add_long_chunk(spans, raw_start, raw_end)
    if stats is not None:
        stats["segment_initial_segments"] = float(len(spans))
    cfg = config
    strategy = str(getattr(cfg, "segment_merge_strategy", "none") or "none").lower() if cfg is not None else "none"
    if strategy in {"relative_entropy", "entropy", "relative", "model_entropy", "token_entropy", "logit_entropy", "logits_entropy"}:
        if stats is not None:
            stats["segment_merge_enabled"] = 1.0
            stats["segment_merge_target_count"] = float(getattr(cfg, "segment_merge_target_count", 0) or 0)
            stats["segment_merge_model_entropy"] = float(
                strategy in {"model_entropy", "token_entropy", "logit_entropy", "logits_entropy"}
            )
        score_fn = None
        if strategy in {"model_entropy", "token_entropy", "logit_entropy", "logits_entropy"}:
            if token_entropy_offsets is None or token_entropies is None:
                raise ValueError("graph_logic_segment_merge_strategy=model_entropy requires token entropy inputs")
            score_fn = _token_entropy_score_fn(token_entropy_offsets, token_entropies)
        spans = _relative_entropy_merge_segments(
            text,
            spans,
            target_count=max(0, int(getattr(cfg, "segment_merge_target_count", 0) or 0)),
            max_chars=max(0, int(getattr(cfg, "segment_merge_max_chars", 1200) or 0)),
            min_info_score=max(0.0, float(getattr(cfg, "segment_merge_min_info_score", 0.0) or 0.0)),
            score_fn=score_fn,
            stats=stats,
        )
    elif stats is not None:
        stats["segment_merge_enabled"] = 0.0
        stats["segment_merge_final_segments"] = float(len(spans))
    return spans


_ACTION_SPAN_CUE_RE = re.compile(
    r"\b("
    r"visit|visited|discover|mark|enqueue|dequeue|push|pop|process|"
    r"check|consider|scan|select|choose|extract|settle|compute|calculate|evaluate|"
    r"relax|update|tentative|distance|dist|cost|predecessor|parent|"
    r"add|include|accept|skip|reject|discard|union|cycle|component|"
    r"augment|bottleneck|residual|capacity|flow|cut|min-cut|mincut|level|"
    r"match|matching|cover|dominate|independent|color|map|mapping|bijection|one-to-one|"
    r"preserve|adjacency|isomorphism|dict|dictionary|domain|range|"
    r"sum|formula|normalize|normalise|normalization|divide|denominator|pair|"
    r"sigma|dependency|contribution|reciprocal|eccentricity|reachable|unreachable|"
    r"n-1|acyclic|connected|spanning|safe|lightest|cheapest|lower\s+bound|unmatched|"
    r"source|sink|mincut|minimum\s+cut|cut\s+capacity|conservation|reverse|"
    r"path|edge|node|neighbor|neighbour|queue|stack|heap"
    r")\b|"
    r"\(\s*\d+\s*,\s*\d+\s*\)|"
    r"\b\d+\s*(?:->|=>|:|maps?\s+to)\s*\d+\b",
    flags=re.I,
)


def _action_subspans_within_segment(
    segment_text: str,
    segment_start: int,
    segment_end: int,
    max_spans: int,
    context_chars: int = 0,
) -> list[tuple[int, int]]:
    """Split one observer segment into non-overlapping action-like clauses.

    This is intentionally a localization helper, not a second executor. The
    algorithm state transition is still computed once for the full segment; this
    only decides where to place the resulting process reward in token space.
    """
    if max_spans <= 0 or not segment_text:
        return []

    cuts = {0, len(segment_text)}
    for match in re.finditer(
        r"(?:[;\n]+|(?<=[.!?])\s+|\s+(?=(?:then|next|after that|finally|now|therefore|so)\b))",
        segment_text,
        flags=re.I,
    ):
        cuts.add(match.start())
        cuts.add(match.end())
    ordered_cuts = sorted(c for c in cuts if 0 <= c <= len(segment_text))
    raw_spans: list[tuple[int, int]] = []
    for left, right in zip(ordered_cuts, ordered_cuts[1:]):
        while left < right and segment_text[left].isspace():
            left += 1
        while right > left and segment_text[right - 1].isspace():
            right -= 1
        if left >= right:
            continue
        clause = segment_text[left:right]
        if _ACTION_SPAN_CUE_RE.search(clause):
            ctx = max(0, int(context_chars))
            raw_spans.append((segment_start + max(0, left - ctx), segment_start + min(len(segment_text), right + ctx)))

    if not raw_spans and _ACTION_SPAN_CUE_RE.search(segment_text):
        left, right = 0, len(segment_text)
        while left < right and segment_text[left].isspace():
            left += 1
        while right > left and segment_text[right - 1].isspace():
            right -= 1
        if left < right:
            ctx = max(0, int(context_chars))
            raw_spans.append((segment_start + max(0, left - ctx), segment_start + min(len(segment_text), right + ctx)))

    normalized: list[tuple[int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for left, right in sorted(raw_spans):
        left = max(segment_start, min(left, segment_end))
        right = max(left, min(right, segment_end))
        if left >= right:
            continue
        span = (left, right)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        normalized.append(span)

    # Merge accidental overlaps introduced by regex boundaries only in strict
    # action-span mode. When a context window is requested, overlap is expected:
    # preserving separate action spans keeps action-level credit distinguishable,
    # while downstream scatter pooling controls duplicate token credit.
    if max(0, int(context_chars)) > 0:
        return normalized[:max_spans]

    merged: list[tuple[int, int]] = []
    for left, right in normalized:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged[:max_spans]


def _step_reward_char_spans(
    spec: GraphLogicSpec,
    cfg: GraphLogicConfig,
    segment_text: str,
    segment_start: int,
    segment_end: int,
    state_action_count: float,
    remaining_steps: int,
    action_event_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    if remaining_steps <= 0:
        return []
    mode = str(getattr(cfg, "step_span_mode", "segment") or "segment").lower()
    if mode not in {"action", "actions", "action_based", "action-based"}:
        return [(segment_start, segment_end)]

    configured_cap = max(1, int(getattr(cfg, "action_span_max_per_segment", 6)))
    if state_action_count > 0:
        configured_cap = min(configured_cap, max(1, int(math.ceil(float(state_action_count)))))
    cap = min(configured_cap, remaining_steps)
    if action_event_spans:
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for left, right in action_event_spans:
            left = max(segment_start, min(int(left), segment_end))
            right = max(left, min(int(right), segment_end))
            if left >= right:
                continue
            span = (left, right)
            if span in seen:
                continue
            seen.add(span)
            normalized.append(span)
        if normalized:
            return normalized[:cap]

    spans = _action_subspans_within_segment(
        segment_text,
        segment_start,
        segment_end,
        cap,
        int(getattr(cfg, "action_span_context_chars", 0)),
    )
    return spans or [(segment_start, segment_end)]


def _local_edge_span(text: str, edge: tuple[int, int], directed: bool) -> tuple[int, int] | None:
    candidates = [edge]
    if not directed and edge[0] != edge[1]:
        candidates.append((edge[1], edge[0]))
    for u, v in candidates:
        u_s, v_s = re.escape(str(int(u))), re.escape(str(int(v)))
        arrow = re.escape(chr(0x2192))
        patterns = (
            rf"\(\s*{u_s}\s*,\s*{v_s}(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\s*\)",
            rf"\[\s*{u_s}\s*,\s*{v_s}(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\s*\]",
            rf"\b{u_s}\s*(?:->|=>|--|-|–|—|{arrow}|to)\s*{v_s}\b",
            rf"\b{u_s}\s*,\s*{v_s}\b",
        )
        for pat in patterns:
            m = re.search(pat, text, flags=re.I)
            if m:
                return int(m.start()), int(m.end())
    return None


def _local_mapping_pair_span(text: str, src: int, dst: int) -> tuple[int, int] | None:
    a, b = re.escape(str(int(src))), re.escape(str(int(dst)))
    patterns = (
        rf"(?:graph\s*1|graph1|g1)?\s*#?\s*{a}\s*(?:->|=>|:|maps?\s+to|corresponds?\s+to|is\s+mapped\s+to)\s*(?:graph\s*2|graph2|g2)?\s*#?\s*{b}",
        rf"(?:graph\s*1|graph1|g1)\s*#?\s*{a}.{{0,70}}?(?:graph\s*2|graph2|g2)\s*#?\s*{b}",
        rf"(?:node\s+)?{a}.{{0,50}}?(?:maps?\s+to|corresponds?\s+to|image\s+is).{{0,20}}?(?:node\s+)?{b}",
    )
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            return int(m.start()), int(m.end())
    return None


def _unit_edges(unit: tuple) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in unit[1:]:
        if isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
            out.append((int(item[0]), int(item[1])))
        elif isinstance(item, tuple):
            for sub in item:
                if isinstance(sub, tuple) and len(sub) == 2 and all(isinstance(x, int) for x in sub):
                    out.append((int(sub[0]), int(sub[1])))
    return out


def _unit_local_span(spec: GraphLogicSpec, text: str, unit: tuple) -> tuple[int, int] | None:
    if not unit:
        return None
    name = str(unit[0])
    for edge in _unit_edges(unit):
        span = _local_edge_span(text, edge, spec.directed)
        if span is not None:
            return span
    if spec.task == "isomophic_mapping":
        ints = [int(x) for x in unit[1:] if isinstance(x, int)]
        if len(ints) >= 4 and name in {"mapping_adjacency_consistent", "mapping_preserves_edge"}:
            span = _local_mapping_pair_span(text, ints[0], ints[2])
            if span is not None:
                return span
        if len(ints) >= 2:
            span = _local_mapping_pair_span(text, ints[0], ints[1])
            if span is not None:
                return span
    for value in unit[1:]:
        if isinstance(value, int):
            m = re.search(rf"\b{re.escape(str(int(value)))}\b", text)
            if m:
                return int(m.start()), int(m.end())
    cue = name.replace("state_", "").replace("_", " ")
    cue_parts = [part for part in cue.split() if len(part) >= 4]
    for part in cue_parts[:3]:
        m = re.search(rf"\b{re.escape(part)}", text, flags=re.I)
        if m:
            return int(m.start()), int(m.end())
    return None


def _expand_local_span(text: str, span: tuple[int, int], context_chars: int) -> tuple[int, int]:
    left, right = span
    ctx = max(0, int(context_chars))
    if ctx > 0:
        left = max(0, int(left) - ctx)
        right = min(len(text), int(right) + ctx)
    else:
        left = max(0, int(left))
        right = min(len(text), int(right))
    return left, max(left, right)


def _state_action_events_for_units(
    spec: GraphLogicSpec,
    text: str,
    units: Iterable[tuple],
    new_units: set[tuple],
    rewardable_units: set[tuple],
) -> list[GraphLogicActionEvent]:
    ordered_units = sorted(set(units), key=lambda unit: (not _state_rewardable_unit(spec, unit), -_state_unit_weight(spec, unit), repr(unit)))
    events: list[GraphLogicActionEvent] = []
    seen_spans: set[tuple[int, int]] = set()
    for unit in ordered_units:
        span = _unit_local_span(spec, text, unit)
        if span is None:
            continue
        if span in seen_spans:
            continue
        seen_spans.add(span)
        events.append(
            GraphLogicActionEvent(
                kind=str(unit[0]) if unit else "unit",
                unit=unit,
                char_span=span,
                confidence=1.0,
                rewardable=unit in rewardable_units or _state_rewardable_unit(spec, unit),
                new=unit in new_units,
            )
        )
    return events


def _trace_action_events(events: Iterable[GraphLogicActionEvent]) -> list[dict[str, Any]]:
    return [
        {
            "kind": event.kind,
            "unit": _trace_value(event.unit),
            "char_span": [int(event.char_span[0]), int(event.char_span[1])],
            "confidence": float(event.confidence),
            "rewardable": bool(event.rewardable),
            "new": bool(event.new),
        }
        for event in events
    ]


def _global_action_event_spans(
    state_step_trace: dict[str, Any],
    segment_start: int,
    segment_end: int,
    segment_text: str,
    cfg: GraphLogicConfig,
) -> list[tuple[int, int]]:
    events = state_step_trace.get("action_events") if isinstance(state_step_trace, dict) else None
    if not isinstance(events, list):
        return []
    out: list[tuple[int, int]] = []
    context_chars = int(getattr(cfg, "action_span_context_chars", 0))
    for event in events:
        if not isinstance(event, dict):
            continue
        span = event.get("char_span")
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        left, right = _expand_local_span(segment_text, (int(span[0]), int(span[1])), context_chars)
        out.append((segment_start + left, min(segment_end, segment_start + right)))
    return out


def _has_any(text: str, cues: Iterable[str]) -> bool:
    low = text.lower()
    return any(cue in low for cue in cues)


def _answer_like(text: str) -> bool:
    low = text.lower()
    return (
        "\\boxed" in text
        or "final answer" in low
        or "the answer is" in low
        or bool(re.search(r"\[[0-9,\s()]+\]", text))
    )


def _fact_dump(text: str) -> bool:
    low = text.lower()
    tuple_count = len(re.findall(r"\(\d+\s*,\s*\d+", text))
    return tuple_count >= 8 or ("edges are" in low and tuple_count >= 4) or ("list all" in low and tuple_count >= 4)


def _allow_solution_list_segment(task: str, text: str) -> bool:
    low = text.lower()
    if task in SET_FEASIBILITY_TASKS:
        return any(
            cue in low
            for cue in (
                "matching",
                "edge cover",
                "vertex cover",
                "dominating set",
                "independent set",
                "select",
                "choose",
                "include",
                "cover",
                "matched",
            )
        )
    if task in {"hamiltonian_path", "traveling_salesman_problem", "topological_sort"}:
        return any(cue in low for cue in ("path", "route", "tour", "order", "sequence", "visit"))
    return False


def _scalar_answer_hit(text: str, spec: GraphLogicSpec) -> bool:
    if spec.answer_bool is not None:
        low = text.lower()
        if spec.answer_bool and re.search(r"\b(yes|true)\b", low):
            return True
        if not spec.answer_bool and re.search(r"\b(no|false)\b", low):
            return True
        task_phrases = {
            "has_cycle": ("cycle", ("acyclic", "no cycle", "does not contain a cycle")),
            "is_bipartite": ("bipartite", ("not bipartite", "isn't bipartite")),
            "is_eularian": ("eulerian", ("not eulerian", "not eularian", "isn't eulerian")),
            "is_regular": ("regular", ("not regular", "isn't regular")),
            "is_tournament": ("tournament", ("not a tournament", "not tournament", "isn't a tournament")),
        }
        phrase_info = task_phrases.get(spec.task)
        if phrase_info is not None:
            positive_phrase, negative_phrases = phrase_info
            has_positive = positive_phrase in low
            has_negative = any(neg in low for neg in negative_phrases)
            if spec.answer_bool and has_positive and not has_negative:
                return True
            if not spec.answer_bool and has_negative:
                return True
    if not spec.answer_numbers:
        return False
    vals = []
    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.I):
        if len(raw) > 64:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            pass
    for target in spec.answer_numbers:
        tol = max(1e-6, abs(target) * 1e-4)
        if any(abs(v - target) <= tol for v in vals):
            return True
    return False


def _infer_family(task: str, text: str) -> bool:
    if task == "shortest_path":
        return _has_any(text, ACTION_CUES["path"])
    if task == "weighted_shortest_path":
        return _has_any(text, ACTION_CUES["path"] + ACTION_CUES["weighted_path"])
    if task in TRAVERSAL_TASKS:
        return _has_any(text, ACTION_CUES["traversal"])
    if task in MST_TASKS:
        return _has_any(text, ACTION_CUES["mst"])
    if task == "maximum_flow":
        return _has_any(text, ACTION_CUES["flow"] + ACTION_CUES["path"])
    if task in {"hamiltonian_path", "topological_sort", "traveling_salesman_problem"}:
        return _has_any(text, ACTION_CUES["ordering"] + ACTION_CUES["path"])
    if task in ANSWER_EDGE_TASKS or task in {"min_vertex_cover", "maximal_independent_set", "dominating_set"}:
        return _has_any(text, ACTION_CUES["set"] + ACTION_CUES["mst"])
    if task == "isomophic_mapping":
        return _has_any(
            text,
            (
                "map",
                "mapping",
                "mapped",
                "correspond",
                "isomorphic",
                "isomorphism",
                "preserve",
                "adjacency",
                "degree",
                "neighbor",
                "g1",
                "g2",
                "image",
            ),
        )
    if task in BOOL_WITNESS_TASKS or task == "local_connectivity":
        return _has_any(text, ACTION_CUES["bool"] + ACTION_CUES["path"])
    return _has_any(text, ACTION_CUES["metric"] + ACTION_CUES["bool"])


def _state_process_action_mentioned(spec: GraphLogicSpec, text: str) -> bool:
    """Detect algorithm-step language without requiring a full executable action."""

    task = spec.task
    if task in PATH_TASKS:
        cues = (
            "relax",
            "update",
            "tentative",
            "settle",
            "settled",
            "extract",
            "pop",
            "priority queue",
            "predecessor",
            "parent",
            "via",
            "through",
            "move",
            "reach",
            "distance",
            "dist",
            "cost",
        )
        return _has_any(text, cues)
    if task in TRAVERSAL_TASKS:
        return _has_any(
            text,
            (
                "visit",
                "visited",
                "discover",
                "enqueue",
                "dequeue",
                "push",
                "pop",
                "process",
                "mark",
                "frontier",
                "stack",
                "queue",
                "backtrack",
                "expand",
                "explore",
            ),
        )
    if task in MST_TASKS:
        return _has_any(
            text,
            (
                "add",
                "include",
                "select",
                "choose",
                "take",
                "pick",
                "accept",
                "reject",
                "skip",
                "discard",
                "consider",
                "safe edge",
                "union",
                "component",
                "cycle",
                "lightest",
                "minimum",
                "weight",
            ),
        )
    if task == "maximum_flow":
        return _has_any(
            text,
            (
                "augment",
                "augmenting",
                "send",
                "push",
                "increase",
                "residual",
                "capacity",
                "bottleneck",
                "cut",
                "flow value",
            ),
        )
    if task in {"hamiltonian_path", "topological_sort", "traveling_salesman_problem"}:
        return _has_any(text, ("extend", "append", "order", "precedes", "before", "after", "cycle", "path", "tour"))
    if task in SET_FEASIBILITY_TASKS or task in ANSWER_EDGE_TASKS:
        return _has_any(
            text,
            (
                "select",
                "choose",
                "include",
                "add",
                "cover",
                "covered",
                "uncovered",
                "matching",
                "matched",
                "unmatched",
                "independent",
                "dominating",
                "endpoint",
                "candidate",
            ),
        )
    if task == "isomophic_mapping":
        return _has_any(
            text,
            (
                "map",
                "mapping",
                "mapped",
                "correspond",
                "preserve",
                "adjacency",
                "image",
                "degree",
                "neighbor",
                "neighbour",
                "partition",
                "refine",
                "bijection",
                "one-to-one",
            ),
        )
    if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return _has_any(
            text,
            (
                "distance",
                "dist",
                "d(",
                "d[",
                "shortest",
                "pair",
                "sigma",
                "contribution",
                "matrix",
                "table",
                "sum",
                "normalize",
                "normalise",
                "denominator",
                "reachable",
                "unreachable",
                "eccentricity",
                "centrality",
                "efficiency",
                "wiener",
            ),
        )
    if task in BOOL_WITNESS_TASKS or task == "local_connectivity":
        return _has_any(text, ("check", "component", "cycle", "color", "partition", "odd", "even", "bridge", "degree"))
    return _infer_family(task, text)



def _state_formula_or_certificate_grounded(spec: GraphLogicSpec, text: str) -> bool:
    """Allow hard-task semantic steps that have no explicit graph entity.

    Many weak tasks contain useful algorithm state updates such as
    "divide by n(n-1)", "use residual graph", or "sort edges by weight".  These
    do not always mention a concrete node/edge, so the old grounded check skipped
    them before task-specific validators could decide whether they were real
    progress.  Keep this gate deliberately task-specific so generic explanations
    do not become rewardable by themselves.
    """

    low = text.lower()
    task = spec.task
    if task == "weighted_shortest_path":
        return _has_any(
            low,
            (
                "dijkstra",
                "priority queue",
                "heap",
                "tentative distance",
                "initialize",
                "initialise",
                "source distance",
                "infinity",
                "weighted path",
                "use weights",
            ),
        )
    if task in MST_TASKS:
        return _has_any(
            low,
            (
                "sort by weight",
                "sorted by weight",
                "ascending weight",
                "in increasing weight",
                "lightest edge",
                "cheapest edge",
                "safe edge",
                "n-1",
                "acyclic",
                "spanning tree",
                "connected and acyclic",
                "kruskal",
                "prim",
            ),
        )
    if task == "maximum_flow":
        return _has_any(
            low,
            (
                "residual network",
                "residual graph",
                "reverse edge",
                "back edge",
                "flow conservation",
                "capacity constraint",
                "source side",
                "sink side",
                "min cut",
                "minimum cut",
                "cut capacity",
                "no augmenting path",
                "no residual path",
            ),
        )
    if task == "isomophic_mapping":
        return _has_any(
            low,
            (
                "python dict",
                "dictionary",
                "bijection",
                "one-to-one",
                "no duplicate",
                "unique values",
                "preserve adjacency",
                "preserves edges",
                "degree sequence",
                "degree partition",
                "same number of edges",
                "same number of nodes",
            ),
        )
    if task == "betweenness_centrality":
        return _has_any(
            low,
            (
                "betweenness",
                "sigma",
                "σ",
                "dependency",
                "shortest path fraction",
                "fraction of shortest paths",
                "normalize",
                "normalise",
                "denominator",
                "exclude endpoints",
                "raw contribution",
                "total contribution",
            ),
        )
    if task == "closeness_centrality":
        return _has_any(
            low,
            (
                "closeness",
                "reachable count",
                "reachable nodes",
                "distance sum",
                "sum distance",
                "normalization",
                "normalisation",
                "wasserman",
                "faust",
                "unreachable",
            ),
        )
    if task == "global_efficiency":
        return _has_any(
            low,
            (
                "global efficiency",
                "efficiency",
                "reciprocal",
                "1/d",
                "inverse distance",
                "ordered pairs",
                "unordered pairs",
                "n(n-1)",
                "denominator",
                "unreachable",
                "zero contribution",
            ),
        )
    if task == "wiener_index":
        return _has_any(
            low,
            (
                "wiener",
                "all pairs",
                "pairwise distance",
                "unordered pairs",
                "distance sum",
                "sum of distances",
            ),
        )
    if task in {"center", "periphery", "barycenter", "radius", "diameter"}:
        return _has_any(
            low,
            (
                "eccentricity",
                "farthest",
                "maximum distance",
                "minimum eccentricity",
                "maximum eccentricity",
                "radius",
                "diameter",
                "all centers",
                "all peripheral",
                "all periphery",
                "all barycenter",
                "tie",
                "ties",
                "distance sum",
                "minimum distance sum",
                "smallest distance sum",
            ),
        )
    if task in {"min_vertex_cover", "dominating_set", "min_edge_covering", "bipartite_maximum_matching"}:
        return _has_any(
            low,
            (
                "lower bound",
                "upper bound",
                "optimal size",
                "minimum size",
                "maximum matching",
                "konig",
                "könig",
                "uncovered",
                "all edges are covered",
                "all nodes are dominated",
                "every vertex is covered",
            ),
        )
    return False


def _state_has_implicit_reference(text: str) -> bool:
    return _has_any(
        text,
        (
            "this edge",
            "that edge",
            "the edge",
            "this node",
            "that node",
            "the node",
            "this vertex",
            "that vertex",
            "the vertex",
            "this path",
            "that path",
            "the path",
            "it",
            "its",
            "them",
            "these",
            "those",
            "same",
            "previous",
            "above",
            "then",
            "next one",
            "current",
        ),
    )


def _advance_state_context_age(state: GraphLogicAlgorithmState | None) -> None:
    if state is not None and state.context_age < 10**8:
        state.context_age += 1


def _remember_state_context(
    state: GraphLogicAlgorithmState | None,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
) -> None:
    if state is None or (not valid_edges and not nodes):
        return
    if valid_edges:
        merged_edges = list(state.recent_edges)
        for edge in sorted(valid_edges):
            if edge in merged_edges:
                merged_edges.remove(edge)
            merged_edges.insert(0, edge)
        state.recent_edges = merged_edges[:6]
    if nodes:
        merged_nodes = list(state.recent_nodes)
        for node in sorted(nodes):
            if node in merged_nodes:
                merged_nodes.remove(node)
            merged_nodes.insert(0, int(node))
        state.recent_nodes = merged_nodes[:8]
    state.context_age = 0


def _state_contextual_grounding(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState | None,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    cfg: GraphLogicConfig,
) -> tuple[set[tuple[int, int]], set[int]]:
    if state is None or valid_edges or nodes:
        return set(), set()
    if int(getattr(cfg, "state_context_max_age", 0)) < 0:
        return set(), set()
    if state.context_age > int(getattr(cfg, "state_context_max_age", 2)):
        return set(), set()
    if not _state_process_action_mentioned(spec, text) or not _state_has_implicit_reference(text):
        return set(), set()

    inherited_edges: set[tuple[int, int]] = set()
    inherited_nodes: set[int] = set()
    edge_tasks = PATH_TASKS | MST_TASKS | ANSWER_EDGE_TASKS | {
        "maximum_flow",
        "hamiltonian_path",
        "traveling_salesman_problem",
        "min_edge_covering",
        "bipartite_maximum_matching",
        "max_weight_matching",
    }
    node_tasks = TRAVERSAL_TASKS | PATH_TASKS | SET_FEASIBILITY_TASKS | {
        "topological_sort",
        "hamiltonian_path",
        "traveling_salesman_problem",
        "center",
        "periphery",
        "barycenter",
        "radius",
        "diameter",
        "closeness_centrality",
        "betweenness_centrality",
    }

    if spec.task in edge_tasks and state.recent_edges:
        limit = 1 if ("this" in text.lower() or "that" in text.lower() or "it" in text.lower()) else 2
        inherited_edges.update(edge for edge in state.recent_edges[:limit] if edge in spec.edges)
    if spec.task in node_tasks and state.recent_nodes:
        limit = 1 if ("this" in text.lower() or "that" in text.lower() or "it" in text.lower()) else 3
        inherited_nodes.update(node for node in state.recent_nodes[:limit] if node in spec.nodes)
    for u, v in inherited_edges:
        inherited_nodes.update((u, v))
    return inherited_edges, inherited_nodes



def _state_action_commitment_score(spec: GraphLogicSpec, text: str) -> float:
    """How strongly the segment commits to an algorithm action.

    This is not a correctness signal. It only distinguishes explicit state
    updates from speculative prose so that executable progress receives credit
    mostly where the model appears to adopt that state.
    """

    low = text.lower()
    score = 0.50
    strong_cues = (
        "therefore",
        "thus",
        "so ",
        "we get",
        "we have",
        "this gives",
        "hence",
        "must",
        "equals",
        "=",
        "becomes",
        "updated to",
        "set to",
        "choose",
        "select",
        "add",
        "accept",
        "reject",
        "skip",
        "settle",
        "augment",
        "bottleneck",
        "final",
        "answer",
        "boxed",
    )
    action_cues = (
        "update",
        "relax",
        "distance",
        "dist",
        "parent",
        "predecessor",
        "union",
        "component",
        "cycle",
        "residual",
        "capacity",
        "flow",
        "cover",
        "dominate",
        "matching",
        "independent",
        "map",
        "preserve",
        "normalize",
        "normalise",
        "denominator",
        "sum",
    )
    hedge_cues = (
        "maybe",
        "perhaps",
        "might",
        "could be",
        "seems",
        "seem",
        "probably",
        "not sure",
        "guess",
        "try ",
        "let's try",
        "?",
    )
    score += 0.08 * min(3, sum(1 for cue in strong_cues if cue in low))
    score += 0.05 * min(3, sum(1 for cue in action_cues if cue in low))
    score -= 0.10 * min(4, sum(1 for cue in hedge_cues if cue in low))
    if spec.task in {"weighted_shortest_path", "weighted_minimum_spanning_tree", "maximum_flow"}:
        if any(cue in low for cue in ("weighted", "weight", "residual", "dijkstra", "kruskal", "prim")):
            score += 0.08
    if spec.task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index"}:
        if any(cue in low for cue in ("formula", "divide", "denominator", "normaliz", "reachable", "unreachable")):
            score += 0.08
    return _clamp_float(score, 0.0, 1.0)


def _oriented_edge_candidates(spec: GraphLogicSpec, edge: tuple[int, int]) -> list[tuple[int, int, tuple[int, int]]]:
    u, v = edge
    out = [(u, v, edge)]
    if not spec.directed and u != v:
        out.append((v, u, edge))
    return out


def _state_candidate_weighted_path_grounding(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
    nodes: set[int],
) -> tuple[set[tuple[int, int]], set[int], float, str]:
    low = text.lower()
    if spec.source is None or not _has_any(text, ACTION_CUES["weighted_path"]):
        return set(), set(), 0.0, ""
    finite_dist = {int(node): float(dist) for node, dist in state.dist.items() if math.isfinite(float(dist))}
    if not finite_dist:
        return set(), set(), 0.0, ""

    mentioned_nodes = set(nodes)
    update_values = {node: _distance_update_values_for_node(text, node) for node in spec.nodes}
    for node, values in update_values.items():
        if values:
            mentioned_nodes.add(int(node))
    relax_like = any(cue in low for cue in ("relax", "update", "improve", "shorter", "cheaper", "tentative", "distance", "dist", "cost"))
    if not relax_like:
        return set(), set(), 0.0, ""

    candidates: list[tuple[float, tuple[int, int], int, int, float]] = []
    for edge in spec.edges:
        for u, v, norm_edge in _oriented_edge_candidates(spec, edge):
            if u not in finite_dist or v == spec.source:
                continue
            cand = finite_dist[u] + float(spec.weights.get(norm_edge, 1.0))
            old = float(state.dist.get(v, math.inf))
            if cand + 1e-9 >= old:
                continue
            values = update_values.get(v, [])
            value_match = any(_close_number(value, cand) for value in values)
            via_match = u in mentioned_nodes or re.search(rf"(?:via|through|from)\s+(?:node\s+)?{u}\b", text, flags=re.I)
            target_match = v in mentioned_nodes or bool(values)
            if not target_match:
                continue
            # Counterfactual audits showed that inferred weighted relax steps
            # survive weight perturbations. Require the response to state the
            # new tentative distance before granting a latent relax action.
            if not value_match:
                continue
            confidence = 0.52
            if value_match:
                confidence += 0.24
            if via_match:
                confidence += 0.14
            if any(cue in low for cue in ("dijkstra", "priority queue", "heap", "settle")):
                confidence += 0.06
            candidates.append((confidence, norm_edge, u, v, cand))

    if not candidates and any(cue in low for cue in ("next", "minimum", "smallest", "settle", "extract", "pop")):
        unsettled = [node for node, dist in finite_dist.items() if node not in state.settled_nodes]
        if unsettled:
            min_dist = min(finite_dist[node] for node in unsettled)
            best = [node for node in unsettled if _close_number(finite_dist[node], min_dist)]
            if len(best) == 1 and (not mentioned_nodes or best[0] in mentioned_nodes):
                return set(), {int(best[0])}, 0.62, "wsp_next_settle"

    if not candidates:
        return set(), set(), 0.0, ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] > best[0] - 0.08:
        return set(), set(), 0.0, ""
    if best[0] < 0.58:
        return set(), set(), 0.0, ""
    _, edge, u, v, _ = best
    return {edge}, {int(u), int(v)}, float(best[0]), "wsp_candidate_relax"


def _state_candidate_mst_grounding(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
    nodes: set[int],
) -> tuple[set[tuple[int, int]], set[int], float, str]:
    low = text.lower()
    if not _has_any(text, ACTION_CUES["mst"]):
        return set(), set(), 0.0, ""
    add_like = any(cue in low for cue in ("add", "include", "select", "choose", "take", "pick", "accept", "safe", "lightest", "cheapest", "minimum"))
    reject_like = any(cue in low for cue in ("reject", "skip", "discard", "cycle", "same component", "already connected"))
    if not add_like and not reject_like:
        return set(), set(), 0.0, ""

    if add_like:
        candidates = [
            edge
            for edge in spec.edges
            if edge not in state.accepted_edges and _state_find(state, edge[0]) != _state_find(state, edge[1])
        ]
        if nodes:
            node_filtered = [edge for edge in candidates if edge[0] in nodes or edge[1] in nodes]
            if node_filtered:
                candidates = node_filtered
        if not candidates:
            return set(), set(), 0.0, ""
        min_weight = min(float(spec.weights.get(edge, 1.0)) for edge in candidates)
        best = [edge for edge in candidates if _close_number(float(spec.weights.get(edge, 1.0)), min_weight)]
        values = _numeric_values(text)
        if values:
            value_best = [edge for edge in best if any(_close_number(value, float(spec.weights.get(edge, 1.0))) for value in values)]
            if value_best:
                best = value_best
        if len(best) == 1:
            edge = best[0]
            confidence = 0.64
            if values and any(_close_number(value, float(spec.weights.get(edge, 1.0))) for value in values):
                confidence += 0.12
            if any(cue in low for cue in ("kruskal", "prim", "safe edge", "different component", "union")):
                confidence += 0.12
            return {edge}, {edge[0], edge[1]}, _clamp_float(confidence, 0.0, 1.0), "mst_candidate_safe_edge"

    if reject_like:
        candidates = [
            edge
            for edge in spec.edges
            if edge not in state.rejected_edges and _state_find(state, edge[0]) == _state_find(state, edge[1])
        ]
        if nodes:
            candidates = [edge for edge in candidates if edge[0] in nodes or edge[1] in nodes]
        if len(candidates) == 1:
            edge = candidates[0]
            return {edge}, {edge[0], edge[1]}, 0.68, "mst_candidate_cycle_reject"
    return set(), set(), 0.0, ""


def _residual_shortest_path(spec: GraphLogicSpec, state: GraphLogicAlgorithmState) -> list[tuple[int, int]]:
    if spec.source is None or spec.target is None:
        return []
    source, target = int(spec.source), int(spec.target)
    parent: dict[int, int] = {}
    queue = [source]
    seen = {source}
    while queue:
        u = queue.pop(0)
        if u == target:
            break
        for (a, b), cap in sorted(state.residual.items()):
            if a != u or cap <= 1e-12 or b in seen:
                continue
            seen.add(b)
            parent[b] = a
            queue.append(b)
    if target not in seen:
        return []
    path_nodes = [target]
    cur = target
    while cur != source:
        cur = parent[cur]
        path_nodes.append(cur)
    path_nodes.reverse()
    return [(u, v) for u, v in zip(path_nodes, path_nodes[1:])]


def _state_candidate_flow_grounding(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
    nodes: set[int],
) -> tuple[set[tuple[int, int]], set[int], float, str]:
    low = text.lower()
    if spec.source is None or spec.target is None:
        return set(), set(), 0.0, ""
    if not any(cue in low for cue in ("augment", "augmenting", "bottleneck", "send", "push", "residual")):
        return set(), set(), 0.0, ""
    path = _residual_shortest_path(spec, state)
    if not path or len(path) > 5:
        return set(), set(), 0.0, ""
    bottleneck = min(float(state.residual.get(edge, 0.0)) for edge in path)
    values = _flow_amount_values(text)
    value_match = any(_close_number(value, bottleneck) for value in values)
    endpoint_match = bool(nodes & {int(spec.source), int(spec.target)}) or not nodes
    if value_match or (endpoint_match and any(cue in low for cue in ("shortest augmenting", "next augmenting", "residual path"))):
        confidence = 0.58 + (0.22 if value_match else 0.0) + (0.10 if "residual" in low else 0.0)
        edge_set = {_norm_edge(u, v, spec.directed) for u, v in path}
        node_set = {node for edge in path for node in edge}
        return edge_set, node_set, _clamp_float(confidence, 0.0, 1.0), "flow_candidate_residual_path"
    return set(), set(), 0.0, ""


def _state_candidate_action_grounding(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState | None,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    cfg: GraphLogicConfig,
) -> tuple[set[tuple[int, int]], set[int], dict[str, float | str]]:
    if state is None or not bool(getattr(cfg, "state_candidate_grounding", True)):
        return set(), set(), {}
    if not _state_process_action_mentioned(spec, text):
        return set(), set(), {}
    max_edges = max(1, int(getattr(cfg, "state_candidate_max_edges", 3)))
    if len(valid_edges) >= max_edges:
        return set(), set(), {}

    candidates: tuple[set[tuple[int, int]], set[int], float, str]
    if spec.task == "weighted_shortest_path" and not valid_edges:
        candidates = _state_candidate_weighted_path_grounding(spec, text, state, nodes)
    elif spec.task in MST_TASKS and not valid_edges:
        candidates = _state_candidate_mst_grounding(spec, text, state, nodes)
    elif spec.task == "maximum_flow" and not valid_edges:
        candidates = _state_candidate_flow_grounding(spec, text, state, nodes)
    else:
        return set(), set(), {}

    edges, new_nodes, confidence, source = candidates
    if confidence <= 0.0:
        return set(), set(), {}
    edges = {edge for edge in edges if edge in spec.edges}
    if len(edges) > max_edges:
        return set(), set(), {}
    new_nodes = {int(node) for node in new_nodes if node in spec.nodes}
    if not edges and not new_nodes:
        return set(), set(), {}
    commitment = _state_action_commitment_score(spec, text)
    if confidence * max(0.50, commitment) < 0.35:
        return set(), set(), {}
    return edges, new_nodes, {
        "state_candidate_grounding_confidence": float(confidence),
        "state_candidate_grounding_commitment": float(commitment),
        "state_candidate_grounding_source": source,
    }


def _numbers_in_text(text: str, nodes: set[int]) -> set[int]:
    out = set()
    for raw in re.findall(r"\b\d+\b", text):
        if len(raw) > 12:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value in nodes:
            out.add(value)
    return out


def _explicit_edges(text: str, spec: GraphLogicSpec) -> set[tuple[int, int]]:
    triples = re.findall(r"\((\d+)\s*,\s*(\d+)\s*,\s*\d+\)", text)
    pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", text)
    out = {_norm_edge(int(a), int(b), spec.directed) for a, b in pairs + triples}

    def add_sequence(nums: list[int]) -> None:
        if len(nums) < 2 or (spec.nodes and len(nums) > len(spec.nodes) + 2):
            return
        edges = [_norm_edge(u, v, spec.directed) for u, v in zip(nums, nums[1:])]
        valid = sum(1 for edge in edges if edge in spec.edges)
        if valid and valid / max(1, len(edges)) >= 0.45:
            out.update(edges)

    arrow = re.escape(chr(0x2192))
    chain_pattern = rf"\b\d+(?:(?:\s*(?:-|->|=>)\s*)\d+|(?:\s*{arrow}\s*)\d+){{1,}}\b"
    for chain in re.findall(chain_pattern, text):
        add_sequence([int(x) for x in re.findall(r"\d+", chain) if len(x) <= 12])
    path_like_tasks = {
        "shortest_path",
        "weighted_shortest_path",
        "hamiltonian_path",
        "traveling_salesman_problem",
        "topological_sort",
    }
    weighted_or_capacity_context = spec.task in {
        "minimum_spanning_tree",
        "weighted_minimum_spanning_tree",
        "weighted_shortest_path",
        "maximum_flow",
    } or _has_any(text, ("weight", "weighted", "capacity", "cap", "flow", "mst", "kruskal", "prim"))
    if weighted_or_capacity_context:
        bare_weight_patterns = (
            r"\b(\d+)\s*[-–]\s*(\d+)\s*(?::|=)\s*[-+]?\d+(?:\.\d+)?(?:\s*\((?:weight|w|capacity|cap)[^)]*\))?",
            r"\b(\d+)\s*[-–]\s*(\d+)\b.{0,32}(?:weight|w|capacity|cap)\D{0,12}[-+]?\d+(?:\.\d+)?",
        )
        for pat in bare_weight_patterns:
            for m in re.finditer(pat, text, flags=re.I | re.S):
                u, v = int(m.group(1)), int(m.group(2))
                if u in spec.nodes and v in spec.nodes:
                    out.add(_norm_edge(u, v, spec.directed))
    if spec.task in path_like_tasks:
        for bracketed in re.findall(r"\[([0-9,\s]+)\]", text):
            add_sequence([int(x) for x in re.findall(r"\d+", bracketed) if len(x) <= 12])
        for m in re.finditer(
            r"(?:path|route|walk|way|cycle|order)\D{0,30}"
            r"((?:\d+\s*(?:,|then|to|via|through)\s*){1,}\d+)",
            text,
            flags=re.I,
        ):
            add_sequence([int(x) for x in re.findall(r"\d+", m.group(1)) if len(x) <= 12])

    for pat in (
        r"(?:node\s+)?(\d+)\s+(?:is\s+)?(?:connected|adjacent)\s+to\s+([0-9,\sandor]+)",
        r"from\s+(?:node\s+)?(\d+).*?(?:go|move|proceed|visit|reach)\s+to\s+([0-9,\sandor]+)",
        r"(?:neighbors?|neighbours?)\s+of\s+(?:node\s+)?(\d+)\s*(?:are|is|:)\s+([0-9,\sandor]+)",
    ):
        for m in re.finditer(pat, text, flags=re.I):
            u = int(m.group(1))
            for raw_v in re.findall(r"\d+", m.group(2)):
                out.add(_norm_edge(u, int(raw_v), spec.directed))

    pair_patterns = (
        r"(?:edge|arc)\s*(?:from\s+)?(?:node\s+)?(\d+)\s*(?:-|--|->|=>|to|and)\s*(?:node\s+)?(\d+)",
        r"(?:edge|arc)\s*(?:between|connecting)\s*(?:node\s+)?(\d+)\s*(?:and|to)\s*(?:node\s+)?(\d+)",
        r"connecting\s+(?:node\s+)?(\d+)\s+and\s+(?:node\s+)?(\d+)",
        r"(?:node\s+)?(\d+)\s+(?:and|with)\s+(?:node\s+)?(\d+)\s+(?:are\s+)?(?:connected|adjacent|matched)",
        r"(?:connect|join|match)\s+(?:node\s+)?(\d+)\s+(?:and|to|with)\s+(?:node\s+)?(\d+)",
        r"(?:take|pick|choose|select|add|include|accept|reject|skip|discard|consider|use)\D{0,35}"
        r"(?:node\s+)?(\d+)\s*(?:-|--|->|=>|to|and|with)\s*(?:node\s+)?(\d+)",
        r"(?:safe|lightest|cheapest|smallest|lowest|next)\s+(?:edge|connection|link)\D{0,35}"
        r"(?:node\s+)?(\d+)\s*(?:-|--|->|=>|to|and|with)\s*(?:node\s+)?(\d+)",
        r"(?:loop|cycle|same component|different component)\D{0,45}"
        r"(?:node\s+)?(\d+)\s*(?:-|--|->|=>|to|and|with)\s*(?:node\s+)?(\d+)",
        r"(?:from|via|through)\s+(?:node\s+)?(\d+)\s+(?:to|towards?)\s+(?:node\s+)?(\d+)",
    )
    for pat in pair_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            u, v = int(m.group(1)), int(m.group(2))
            if u in spec.nodes and v in spec.nodes:
                out.add(_norm_edge(u, v, spec.directed))
    return out


def _edge_contexts(text: str, edge: tuple[int, int], directed: bool, window: int = 80) -> list[str]:
    u, v = edge
    pairs = [(u, v)] if directed or u == v else [(u, v), (v, u)]
    patterns = []
    arrow = re.escape(chr(0x2192))
    edge_sep = rf"(?:-|--|->|=>|{arrow}|to|and|with)"
    for a, b in pairs:
        patterns.extend(
            [
                rf"\({a}\s*,\s*{b}(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\)",
                rf"(?:edge|arc)\s*(?:from\s+)?(?:node\s+)?{a}\s*{edge_sep}\s*(?:node\s+)?{b}",
                rf"(?:edge|arc)\s*(?:between|connecting)\s*(?:node\s+)?{a}\s*(?:and|to|{arrow})\s*(?:node\s+)?{b}",
                rf"\b{a}\s*[-–]\s*{b}\s*(?::|=)\s*[-+]?\d+(?:\.\d+)?(?:\s*\((?:weight|w|capacity|cap)[^)]*\))?",
                rf"\b{a}\s*[-–]\s*{b}\b.{{0,36}}(?:weight|w|capacity|cap)\D{{0,12}}[-+]?\d+(?:\.\d+)?",
                rf"connecting\s+(?:node\s+)?{a}\s+and\s+(?:node\s+)?{b}",
                rf"(?:node\s+)?{a}\s+(?:and|with)\s+(?:node\s+)?{b}\s+(?:are\s+)?(?:connected|adjacent|matched)",
                rf"(?:take|pick|choose|select|add|include|accept|reject|skip|discard|consider|use)\D{{0,35}}"
                rf"(?:node\s+)?{a}\s*{edge_sep}\s*(?:node\s+)?{b}",
                rf"(?:safe|lightest|cheapest|smallest|lowest|next)\s+(?:edge|connection|link)\D{{0,35}}"
                rf"(?:node\s+)?{a}\s*{edge_sep}\s*(?:node\s+)?{b}",
                rf"(?:loop|cycle|same component|different component)\D{{0,45}}"
                rf"(?:node\s+)?{a}\s*{edge_sep}\s*(?:node\s+)?{b}",
            ]
        )
    contexts = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            contexts.append(text[max(0, m.start() - window) : min(len(text), m.end() + window)].lower())
    return contexts


def _node_contexts(text: str, node: int, window: int = 80) -> list[str]:
    patterns = (
        rf"(?:node|vertex)\s+{node}\b",
        rf"\b{node}\b",
        rf"(?:dist|distance|cost)\s*(?:\[|\(|to|of|for)?\s*{node}\b",
    )
    contexts: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            contexts.append(text[max(0, m.start() - window) : min(len(text), m.end() + window)].lower())
    return contexts


def _node_context_has_any(text: str, node: int, cues: Iterable[str], window: int = 80) -> bool:
    contexts = _node_contexts(text, node, window=window)
    if not contexts:
        return False
    return any(any(cue in ctx for cue in cues) for ctx in contexts)


def _edge_has_hard_invalid_mention(text: str, edge: tuple[int, int], directed: bool) -> bool:
    u, v = edge
    pairs = [(u, v)] if directed or u == v else [(u, v), (v, u)]
    for a, b in pairs:
        patterns = (
            rf"\({a}\s*,\s*{b}(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\)",
            rf"(?:edge|arc)\s*(?:from\s+)?(?:node\s+)?{a}\s*(?:-|--|->|=>|to|and)\s*(?:node\s+)?{b}",
            rf"(?:edge|arc)\s*(?:between|connecting)\s*(?:node\s+)?{a}\s*(?:and|to)\s*(?:node\s+)?{b}",
        )
        if any(re.search(pat, text, flags=re.I) for pat in patterns):
            return True
    return False


def _edge_is_negated_or_rejected(text: str, edge: tuple[int, int], spec: GraphLogicSpec) -> bool:
    contexts = _edge_contexts(text, edge, spec.directed)
    if not contexts:
        return False
    negation_cues = (
        "no edge",
        "not an edge",
        "not connected",
        "not adjacent",
        "does not exist",
        "doesn't exist",
        "do not exist",
        "don't exist",
        "is absent",
        "invalid edge",
        "cannot use",
        "can't use",
        "skip",
        "reject",
        "discard",
        "ignore",
        "forms a cycle",
        "form a cycle",
        "creates a cycle",
        "would create a cycle",
        "same component",
        "already visited",
    )
    return any(any(cue in ctx for cue in negation_cues) for ctx in contexts)


def _edge_is_structural_pair_reference(text: str, edge: tuple[int, int], spec: GraphLogicSpec) -> bool:
    contexts = _edge_contexts(text, edge, spec.directed)
    if not contexts:
        return False
    direct_edge_cues = ("edge", "arc", "connected", "adjacent", "connect", "join", "matched")
    pair_cues = (
        "pair",
        "pairs",
        "node pair",
        "vertices",
        "distance",
        "dist",
        "shortest path",
        "path between",
        "from node",
        "to node",
        "between node",
        "all-pairs",
        "via",
    )
    if spec.task in DISTANCE_WITNESS_TASKS:
        return True
    if spec.task == "triangles":
        return True
    if spec.task == "is_tournament":
        return True
    if spec.task == "isomophic_mapping":
        return any(
            any(cue in ctx for cue in ("map", "mapping", "mapped", "correspond", "g1", "g2", "graph1", "graph2"))
            for ctx in contexts
        )
    if spec.task in {"is_regular", "is_eularian", "density", "edge_number", "node_number"}:
        return any(not any(cue in ctx for cue in direct_edge_cues) for ctx in contexts)
    if spec.task in {"min_vertex_cover", "dominating_set", "maximal_independent_set"}:
        return any(
            any(cue in ctx for cue in ("vertex cover", "vertices", "nodes", "set", "dominating", "independent"))
            and not any(cue in ctx for cue in ("edge", "connected", "adjacent", "matched"))
            for ctx in contexts
        )
    return False


def _invalid_penalty_scale(task: str) -> float:
    if task in DISTANCE_WITNESS_TASKS:
        return 0.25
    if task in {"triangles", "is_tournament"}:
        return 0.25
    if task == "min_edge_covering":
        return 0.10
    if task in SET_FEASIBILITY_TASKS:
        return 0.50
    if task in {"edge_existence", "maximum_flow", "hamiltonian_path"}:
        return 0.75
    return 1.0


def _task_weight(task: str) -> float:
    if task in PATH_TASKS | TRAVERSAL_TASKS | MST_TASKS | {"topological_sort", "traveling_salesman_problem"}:
        return 1.0
    if task in {"isomophic_mapping", "maximum_flow"}:
        return 0.55
    if task == "bridges":
        return 0.70
    if task in GLOBAL_WITNESS_TASKS:
        return 0.35
    if task in BOOL_WITNESS_TASKS:
        return 0.35
    if task in ANSWER_EDGE_TASKS:
        return 0.5
    return 0.4


def _allow_node_only_progress(task: str) -> bool:
    if task in PATH_TASKS:
        return False
    if task in TRAVERSAL_TASKS | {"hamiltonian_path", "topological_sort", "local_connectivity"}:
        return True
    if task in GLOBAL_WITNESS_TASKS | BOOL_WITNESS_TASKS | ANSWER_NODE_TASKS:
        return True
    if task in {
        "neighbor",
        "degree",
        "degree_centrality",
        "avg_neighbor_degree",
        "common_neighbor",
        "jaccard_coefficient",
        "adamic_adar_index",
        "resource_allocation_index",
        "clustering_coefficient",
        "triangles",
        "closeness_centrality",
        "harmonic_centrality",
        "betweenness_centrality",
        "connected_component_number",
        "strongly_connected_number",
        "node_number",
        "edge_number",
        "density",
        "diameter",
        "radius",
    }:
        return True
    return False


def _has_node_progress_cue(text: str) -> bool:
    low = text.lower()
    return any(
        cue in low
        for cue in (
            "visit",
            "visited",
            "discover",
            "enqueue",
            "dequeue",
            "update",
            "distance",
            "dist",
            "relax",
            "select",
            "choose",
            "pick",
            "process",
            "mark",
            "settle",
            "settled",
            "reach",
            "neighbor",
            "component",
            "length",
            "step",
            "steps",
            "eccentricity",
            "maximum distance",
            "total distance",
            "sum",
            "reciprocal",
            "current node",
            "next node",
        )
    )


def _has_selection_cue(text: str) -> bool:
    low = text.lower()
    return any(
        cue in low
        for cue in (
            "select",
            "choose",
            "chosen",
            "include",
            "add",
            "take",
            "pick",
            "candidate",
            "matching",
            "cover",
            "dominating",
            "independent",
            "matched",
            "unmatched",
            "covered",
            "uncovered",
        )
    )


def _selected_node_mentions(text: str, nodes: set[int]) -> set[int]:
    out = set()
    for m in re.finditer(r"(?:node|vertex|vertices)\s+([0-9,\sand]+)", text, flags=re.I):
        for raw in re.findall(r"\d+", m.group(1)):
            value = int(raw)
            if value in nodes:
                out.add(value)
    selection_list_patterns = (
        r"(?:select|choose|chosen|include|add|take|pick|candidate(?:s)?|cover(?:\s+set)?|solution)\s*"
        r"(?:nodes?|vertices?)?\s*(?:are|is|=|:|as|to be)?\s*[\{\[\(]?\s*([0-9,\sand]+)",
        r"(?:put|place)\s+(?:nodes?|vertices?)?\s*([0-9,\sand]+)\s+(?:in|into)\s+(?:the\s+)?(?:cover|set|solution)",
    )
    for pat in selection_list_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            for raw in re.findall(r"\d+", m.group(1)):
                value = int(raw)
                if value in nodes:
                    out.add(value)
    for bracketed in re.findall(r"\[([0-9,\s]+)\]", text):
        for raw in re.findall(r"\d+", bracketed):
            value = int(raw)
            if value in nodes:
                out.add(value)
    return out


def _shortest_distance(spec: GraphLogicSpec, source: int, target: int) -> float:
    if source not in spec.nodes or target not in spec.nodes:
        return math.inf
    if source == target:
        return 0.0
    key = (source, target) if spec.directed or source <= target else (target, source)
    if key in spec.distance_cache:
        return spec.distance_cache[key]

    def remember(value: float) -> float:
        spec.distance_cache[key] = value
        return value

    adj = build_adj(spec)
    weighted = any(abs(float(w) - 1.0) > 1e-9 for w in spec.weights.values())
    if weighted:
        import heapq

        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            if u == target:
                return remember(d)
            for v in adj.get(u, []):
                edge = _norm_edge(u, v, spec.directed)
                nd = d + float(spec.weights.get(edge, 1.0))
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return remember(math.inf)
    queue = [(source, 0)]
    seen = {source}
    head = 0
    while head < len(queue):
        u, d = queue[head]
        head += 1
        for v in adj.get(u, []):
            if v in seen:
                continue
            if v == target:
                return remember(float(d + 1))
            seen.add(v)
            queue.append((v, d + 1))
    return remember(math.inf)


def _shortest_path_count(spec: GraphLogicSpec, source: int, target: int) -> int:
    """Count shortest paths between two nodes under the task graph.

    This is intentionally lightweight and deterministic. It is used only by the
    observer to validate betweenness-style process statements, not by the final
    answer verifier.
    """
    if source not in spec.nodes or target not in spec.nodes:
        return 0
    if source == target:
        return 1
    key = (source, target) if spec.directed or source <= target else (target, source)
    if key in spec.path_count_cache:
        return int(spec.path_count_cache[key])

    def remember(value: int) -> int:
        spec.path_count_cache[key] = int(value)
        return int(value)

    adj = build_adj(spec)
    weighted = any(abs(float(w) - 1.0) > 1e-9 for w in spec.weights.values())
    if weighted:
        import heapq

        dist: dict[int, float] = {source: 0.0}
        count: dict[int, int] = {source: 1}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf) + 1e-9:
                continue
            for v in adj.get(u, []):
                edge = _norm_edge(u, v, spec.directed)
                nd = d + float(spec.weights.get(edge, 1.0))
                old = dist.get(v, math.inf)
                if nd + 1e-9 < old:
                    dist[v] = nd
                    count[v] = count.get(u, 0)
                    heapq.heappush(heap, (nd, v))
                elif _close_number(nd, old):
                    count[v] = count.get(v, 0) + count.get(u, 0)
        return remember(count.get(target, 0) if math.isfinite(dist.get(target, math.inf)) else 0)

    dist = {source: 0}
    count = {source: 1}
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj.get(u, []):
            nd = dist[u] + 1
            if v not in dist:
                dist[v] = nd
                count[v] = count[u]
                queue.append(v)
            elif dist[v] == nd:
                count[v] += count[u]
    return remember(count.get(target, 0))


def _node_sequences(text: str, spec: GraphLogicSpec) -> list[list[int]]:
    seqs: list[list[int]] = []

    def add_nums(nums: list[int]) -> None:
        if len(nums) >= 2 and len(nums) <= len(spec.nodes) + 4 and all(n in spec.nodes for n in nums):
            if nums not in seqs:
                seqs.append(nums)

    arrow = re.escape(chr(0x2192))
    chain_pattern = rf"\b\d+(?:(?:\s*(?:-|->|=>)\s*)\d+|(?:\s*{arrow}\s*)\d+){{1,}}\b"
    for chain in re.findall(chain_pattern, text):
        nums = [int(x) for x in re.findall(r"\d+", chain) if len(x) <= 12]
        add_nums(nums)
    for bracketed in re.findall(r"\[([0-9,\s]+)\]", text):
        nums = [int(x) for x in re.findall(r"\d+", bracketed) if len(x) <= 12]
        add_nums(nums)
    for m in re.finditer(
        r"(?:path|route|walk|way|tour|order)\D{0,30}((?:\d+\s*(?:,|then|to|via|through)\s*){1,}\d+)",
        text,
        flags=re.I,
    ):
        add_nums([int(x) for x in re.findall(r"\d+", m.group(1)) if len(x) <= 12])
    for m in re.finditer(
        r"(?:from|start(?:ing)?\s+at)\s+(?:node\s+)?(\d+)\D{0,30}"
        r"(?:to|towards?|through|via)\s+((?:node\s+)?\d+(?:\D{0,20}(?:to|then|via|through)\D{0,20}(?:node\s+)?\d+)*)",
        text,
        flags=re.I,
    ):
        add_nums([int(m.group(1))] + [int(x) for x in re.findall(r"\d+", m.group(2)) if len(x) <= 12])
    for m in re.finditer(
        r"\b(\d+)\s*(?:then|next|after that|followed by)\s*(\d+)"
        r"(?:(?:\s*(?:then|next|after that|followed by)\s*)\d+)*",
        text,
        flags=re.I,
    ):
        add_nums([int(x) for x in re.findall(r"\d+", m.group(0)) if len(x) <= 12])
    return seqs


def _degrees(spec: GraphLogicSpec) -> dict[int, int]:
    degs = {u: 0 for u in spec.nodes}
    for u, v in spec.edges:
        degs[u] = degs.get(u, 0) + 1
        if not spec.directed:
            degs[v] = degs.get(v, 0) + 1
    return degs


def _close_number(value: float, target: float) -> bool:
    tol = max(1e-6, abs(float(target)) * 1e-4)
    return abs(float(value) - float(target)) <= tol


def _parse_float_token(raw: str) -> float | None:
    token = str(raw).strip()
    if "/" in token:
        parts = token.split("/", 1)
        try:
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
        except ValueError:
            return None
        if abs(denominator) <= 1e-12:
            return None
        return numerator / denominator
    try:
        return float(token)
    except ValueError:
        return None


def _numeric_values(text: str) -> list[float]:
    out = []
    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?\s*/\s*[-+]?\d+(?:\.\d+)?", text, flags=re.I):
        if len(raw) > 64:
            continue
        value = _parse_float_token(raw)
        if value is not None:
            out.append(value)
    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.I):
        if len(raw) > 64:
            continue
        value = _parse_float_token(raw)
        if value is not None:
            out.append(value)
    return out


def _mentions_number_value(text: str, target: int | float) -> bool:
    return any(_close_number(value, float(target)) for value in _numeric_values(text))


def _last_boxed_content(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    rest = text[idx + len("\\boxed") :].lstrip()
    if not rest:
        return None
    if rest[0] != "{":
        return rest.split()[0] if rest.split() else None
    depth = 0
    start = None
    for offset, ch in enumerate(rest):
        if ch == "{":
            if depth == 0:
                start = offset + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return rest[start:offset]
    return None


def _final_answer_numeric_values(text: str) -> list[float]:
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return _numeric_values(boxed)
    low = text.lower()
    cut = -1
    for cue in ("final answer", "the answer is", "answer:", "therefore", "thus"):
        pos = low.rfind(cue)
        if pos > cut:
            cut = pos
    if cut >= 0:
        return _numeric_values(text[cut:])
    return []


def _numeric_final_invalid(spec: GraphLogicSpec, text: str) -> bool:
    if not spec.answer_numbers:
        return False
    scalar_tasks = (
        DISTANCE_WITNESS_TASKS
        | GLOBAL_WITNESS_TASKS
        | COUNT_FORMULA_TASKS
        | {"maximum_flow", "connected_component_number", "strongly_connected_number", "local_connectivity"}
    )
    if spec.task not in scalar_tasks:
        return False
    if not _has_answer_assertion_cue(spec, text):
        return False
    values = _final_answer_numeric_values(text)
    if not values:
        if spec.task in {"radius", "diameter"} and _has_answer_assertion_cue(spec, text):
            values = _numeric_values(text)
        else:
            return False
    if not values:
        return False
    targets = list(spec.answer_numbers[:3])
    return not any(_close_number(value, target) for value in values for target in targets)


def _numeric_final_missing_invalid(spec: GraphLogicSpec, text: str) -> bool:
    if not spec.answer_numbers or not _has_strong_final_answer_cue(text):
        return False
    if spec.task not in SCALAR_FINAL_SCHEMA_TASKS:
        return False
    return not _final_answer_numeric_values(text)


def _schema_final_numeric_missing(spec: GraphLogicSpec, text: str) -> bool:
    """Schema-only final numeric check for scalar graph tasks.

    This intentionally does not check correctness against the golden answer. It
    only catches long scalar-task generations that never land on a parseable
    final number, which was a major failure mode in the offline eval.
    """

    if not spec.answer_numbers or spec.task not in SCALAR_FINAL_SCHEMA_TASKS:
        return False
    return not _final_answer_numeric_values(text)


def _schema_final_scalar_parseable(spec: GraphLogicSpec, text: str) -> bool:
    """Schema-only positive counterpart for scalar tasks.

    This only checks that the response lands on a parseable final number. The
    value is still judged by the outcome verifier, not by the observer.
    """

    if not spec.answer_numbers or spec.task not in SCALAR_FINAL_SCHEMA_TASKS:
        return False
    return bool(_final_answer_numeric_values(text))


def _schema_final_object_units(spec: GraphLogicSpec, text: str) -> tuple[set[tuple], bool]:
    """Return schema-only final-object units and whether the final object is missing.

    The units intentionally avoid golden-answer correctness. They only reward
    landing on a task-appropriate parseable object: node list, edge list, or
    mapping dict. This gives the policy a process-level signal to stop with a
    verifier-compatible object while leaving correctness to outcome reward.
    """

    if spec.task not in OBJECT_FINAL_SCHEMA_TASKS:
        return set(), False

    units: set[tuple] = set()
    final_text = _final_answer_text(text)
    tail = text[-2048:] if text else ""
    finalish = bool(final_text) or _has_strong_final_answer_cue(text) or "\\boxed" in text
    if not final_text and finalish:
        final_text = tail

    if spec.task in MAPPING_FINAL_SCHEMA_TASKS:
        scan_text = final_text or tail
        mappings = _int_mapping_dicts(scan_text) if scan_text else []
        source_nodes = set(spec.answer_mapping.keys()) if spec.answer_mapping else set()
        target_nodes = set(spec.answer_mapping.values()) if spec.answer_mapping else set()
        for mapping in mappings:
            if not mapping:
                continue
            keys = set(mapping.keys())
            values = list(mapping.values())
            in_domain = True
            if source_nodes:
                in_domain = keys <= source_nodes and all(value in target_nodes for value in values)
            if not in_domain:
                continue
            units.add(("parseable_mapping_dict_answer", min(len(mapping), 64)))
            if len(set(values)) == len(values):
                units.add(("mapping_unique_values", min(len(values), 64)))
            if source_nodes and keys == source_nodes and len(set(values)) == len(values):
                units.add(("complete_mapping_dict_answer", len(source_nodes)))
        return units, not bool(units)

    if spec.task in EDGE_LIST_FINAL_SCHEMA_TASKS:
        edge_order = _final_answer_edge_order(text, spec)
        if not edge_order and final_text:
            edge_order = _edge_order_from_text(final_text, spec)
        if edge_order:
            signature = tuple(edge_order[:12])
            units.add(("parseable_edge_list_answer", spec.task, signature))
            if spec.task in TRAVERSAL_TASKS:
                target_edges = list(spec.target_edge_order or [])
                prefix_len = 0
                for got, expected in zip(edge_order, target_edges):
                    if got != expected:
                        break
                    prefix_len += 1
                if prefix_len > 0:
                    units.add(
                        (
                            "state_traversal_answer_edge_prefix",
                            spec.task,
                            int(prefix_len),
                            tuple(edge_order[: min(12, prefix_len)]),
                        )
                    )
                if target_edges and prefix_len == len(target_edges) and len(edge_order) == len(target_edges):
                    units.add(("ordering_answer_valid", spec.task, signature))
            if spec.task in MST_TASKS:
                target_edge_count = max(0, len(spec.nodes) - 1)
                edge_set = set(edge_order)
                if len(edge_order) == target_edge_count and len(edge_set) == target_edge_count:
                    units.add(("mst_answer_edge_count", spec.task, target_edge_count))
                    valid, total = _spanning_tree_valid(spec, edge_set)
                    if valid:
                        units.add(("mst_answer_spanning_tree", signature))
                        units.add(("state_mst_connected_acyclic", signature))
                        if spec.task == "weighted_minimum_spanning_tree":
                            optimal_weight = _mst_optimal_weight(spec)
                            if any(_close_number(value, float(total)) for value in _numeric_values(text)):
                                units.add(("state_mst_final_weight_checked", round(float(total), 6)))
                            if _close_number(float(total), float(optimal_weight)):
                                units.add(("state_mst_optimal_weight_cert", round(float(total), 6)))
                                units.add(("mst_answer_valid", signature, round(float(total), 6)))
        return units, not bool(units)

    if spec.task in NODE_LIST_FINAL_SCHEMA_TASKS:
        if spec.task in TRAVERSAL_TASKS and spec.target_edge_order:
            # In this benchmark BFS/DFS are evaluated as traversal tree-edge
            # lists. A final node sequence is parseable text, but it is the
            # wrong verifier schema and must not receive object-schema credit.
            return set(), True
        scan_text = final_text or ""
        seqs = _node_sequences(scan_text, spec) if scan_text else []
        nodes = seqs[-1] if seqs else []
        if nodes:
            signature = tuple(nodes[: min(12, len(nodes))])
            units.add(("parseable_node_list_answer", spec.task, signature))
            if spec.task in {"hamiltonian_path", "traveling_salesman_problem"} and len(set(nodes)) == len(nodes):
                units.add(("schema_simple_node_sequence", spec.task, min(len(nodes), 64)))
            selected = set(nodes) & spec.nodes
            if spec.task == "hamiltonian_path" and _hamiltonian_path_valid(spec, nodes):
                units.add(("hamiltonian_complete",))
                units.add(("ordering_answer_valid", spec.task, signature))
            elif spec.task == "weighted_shortest_path" and len(nodes) >= 2 and spec.source is not None:
                path_edges = _valid_path_edges(spec, nodes)
                if path_edges and nodes[0] == spec.source and (spec.target is None or nodes[-1] == spec.target):
                    path_cost = _path_cost(spec, path_edges)
                    target_cost = _shortest_distance(spec, int(spec.source), int(nodes[-1]))
                    if math.isfinite(target_cost) and _close_number(path_cost, target_cost):
                        units.add(("state_path_cost_valid", tuple(path_edges[: min(8, len(path_edges))]), round(float(path_cost), 6)))
                        if spec.target is not None and nodes[-1] == spec.target:
                            units.add(("state_weighted_path_final_cert", tuple(path_edges[: min(8, len(path_edges))]), round(float(target_cost), 6)))
            elif spec.task == "min_vertex_cover" and selected:
                if all(u in selected or v in selected for u, v in spec.edges):
                    units.add(("vertex_cover_all_edges_covered", len(selected)))
                    units.add(("set_answer_feasible", spec.task, tuple(sorted(selected))[:12]))
                    target_node_count = len(spec.target_nodes or set())
                    if target_node_count and len(selected) <= target_node_count:
                        units.add(("set_answer_optimal_size", spec.task, len(selected)))
            elif spec.task == "dominating_set" and selected:
                adj = build_adj(spec)
                dominated = set(selected)
                for node in selected:
                    dominated.update(adj.get(node, []))
                if set(spec.nodes) <= dominated:
                    units.add(("dominating_all_nodes_covered", len(selected)))
                    units.add(("set_answer_feasible", spec.task, tuple(sorted(selected))[:12]))
                    target_node_count = len(spec.target_nodes or set())
                    if target_node_count and len(selected) <= target_node_count:
                        units.add(("set_answer_optimal_size", spec.task, len(selected)))
            elif spec.task == "maximal_independent_set" and selected and len(selected) == len(nodes):
                conflict = any(
                    _norm_edge(u, v, spec.directed) in spec.edges
                    for u in selected
                    for v in selected
                    if u < v
                )
                if not conflict:
                    units.add(("set_answer_feasible", spec.task, tuple(sorted(selected))[:12]))
                    maximal = all(
                        any(_norm_edge(node, chosen, spec.directed) in spec.edges for chosen in selected)
                        for node in spec.nodes - selected
                    )
                    if maximal:
                        units.add(("maximal_independent_complete", len(selected)))
            elif spec.task in {"center", "periphery", "barycenter"} and selected and len(selected) == len(nodes):
                target_nodes = _target_answer_nodes(spec)
                if target_nodes and selected == target_nodes:
                    units.add(("metric_target_set_complete", spec.task, tuple(sorted(selected))))
                    units.add(("node_answer_valid", spec.task, tuple(sorted(selected))[:12]))
        return units, not bool(units)

    return set(), False


def _schema_final_answer_conflict_reason(spec: GraphLogicSpec, text: str) -> str:
    """Detect task/output-schema conflicts in explicit final answers.

    This is narrower than generic missing-object checks: it only fires on
    explicit final-answer cues. Intermediate process text can still receive
    process credit, but a final answer with the wrong object type cannot use
    process reward to cancel out the schema error.
    """

    if not _has_final_answer_cue(text):
        return ""
    final_text = _final_answer_text(text) or text[-2048:]
    if not final_text.strip():
        return ""

    final_edge_order = _final_answer_edge_order(text, spec)
    if not final_edge_order:
        final_edge_order = _edge_order_from_text(final_text, spec)
    final_nodes = _final_answer_nodes(text, spec)
    final_node_sequences = _node_sequences(final_text, spec)
    has_edge_object = bool(final_edge_order)
    has_node_object = bool(final_nodes or final_node_sequences)
    has_mapping_object = bool(_int_mapping_dicts(final_text))
    has_scalar_value = bool(_numeric_values(final_text))

    expects_edge_object = (
        spec.task in EDGE_LIST_FINAL_SCHEMA_TASKS
        and bool(spec.target_edges or spec.target_edge_order or spec.task in MST_TASKS)
    )
    if expects_edge_object:
        if has_edge_object:
            return ""
        if has_node_object:
            return "node_list_for_edge_task"
        if has_mapping_object:
            return "mapping_for_edge_task"
        if has_scalar_value:
            return "scalar_for_edge_task"
        return "missing_edge_list"

    expects_node_object = (
        spec.task in NODE_LIST_FINAL_SCHEMA_TASKS
        and not (spec.task in TRAVERSAL_TASKS and spec.target_edge_order)
        and bool(
            spec.target_nodes
            or spec.target_node_order
            or spec.task in {"topological_sort", "hamiltonian_path", "traveling_salesman_problem"}
        )
    )
    if expects_node_object:
        if has_edge_object:
            return "edge_list_for_node_task"
        if has_node_object:
            return ""
        if has_mapping_object:
            return "mapping_for_node_task"
        if has_scalar_value and spec.task not in PATH_TASKS:
            return "scalar_for_node_task"
        return "missing_node_list"

    if spec.task in MAPPING_FINAL_SCHEMA_TASKS:
        units, _ = _schema_final_object_units(spec, text)
        if units and any(
            unit and str(unit[0]) in {"complete_mapping_dict_answer", "mapping_valid_dict"}
            for unit in units
        ):
            return ""
        if has_edge_object:
            return "edge_list_for_mapping_task"
        if has_node_object:
            return "node_list_for_mapping_task"
        if has_scalar_value:
            return "scalar_for_mapping_task"
        return "missing_mapping"

    if spec.task in SCALAR_FINAL_SCHEMA_TASKS:
        if _schema_final_scalar_parseable(spec, text):
            return ""
        if has_edge_object or has_node_object or has_mapping_object:
            return "object_for_scalar_task"
        if has_scalar_value:
            return "wrong_scalar"
        return "missing_scalar"

    return ""


def _final_answer_text(text: str) -> str:
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return boxed
    low = text.lower()
    cut = -1
    for cue in ("final answer", "the answer is", "answer:", "therefore", "thus", "result is", "conclusion"):
        pos = low.rfind(cue)
        if pos > cut:
            cut = pos
    return text[cut:] if cut >= 0 else ""


def _path_final_invalid(spec: GraphLogicSpec, text: str) -> bool:
    if spec.task not in PATH_TASKS:
        return False
    if not _has_answer_assertion_cue(spec, text):
        return False
    final_text = _final_answer_text(text)
    if not final_text:
        final_text = text
    saw_candidate = False
    saw_shortest = False
    for seq in _node_sequences(final_text, spec):
        if spec.source is not None and seq[0] != spec.source:
            continue
        if spec.target is not None and seq[-1] != spec.target:
            continue
        saw_candidate = True
        path_edges = _valid_path_edges(spec, seq)
        if path_edges and _path_is_shortest_prefix(spec, seq, path_edges):
            saw_shortest = True
    return saw_candidate and not saw_shortest


def _has_final_answer_cue(text: str) -> bool:
    low = text.lower()
    return "\\boxed" in text or any(cue in low for cue in ("final answer", "the answer is", "answer:"))


def _has_strong_final_answer_cue(text: str) -> bool:
    low = text.lower()
    return _has_final_answer_cue(text) or any(
        cue in low for cue in ("therefore", "thus", "so the answer", "result is", "conclusion")
    )


def _has_answer_assertion_cue(spec: GraphLogicSpec, text: str) -> bool:
    if _has_final_answer_cue(text):
        return True
    low = text.lower()
    if _has_strong_final_answer_cue(text):
        return True
    task_terms = {
        "center": ("center", "centre"),
        "periphery": ("periphery", "peripheral nodes"),
        "barycenter": ("barycenter", "barycentre"),
        "radius": ("radius",),
        "diameter": ("diameter",),
        "dominating_set": ("dominating set",),
        "min_vertex_cover": ("vertex cover",),
        "maximal_independent_set": ("independent set",),
        "min_edge_covering": ("edge cover",),
        "bipartite_maximum_matching": ("matching",),
        "max_weight_matching": ("matching",),
        "bridges": ("bridges", "bridge edges"),
        "topological_sort": ("topological order", "topological sort", "order"),
        "hamiltonian_path": ("hamiltonian path",),
        "traveling_salesman_problem": ("tour", "tsp", "salesman"),
        "shortest_path": ("shortest path",),
        "weighted_shortest_path": ("shortest path", "weighted path"),
        "minimum_spanning_tree": ("minimum spanning tree", "spanning tree"),
        "weighted_minimum_spanning_tree": ("minimum spanning tree", "weighted mst", "spanning tree"),
        "maximum_flow": ("maximum flow", "max flow", "flow value"),
        "betweenness_centrality": ("betweenness", "betweenness centrality"),
        "closeness_centrality": ("closeness", "closeness centrality"),
        "global_efficiency": ("global efficiency", "efficiency"),
        "wiener_index": ("wiener index", "wiener"),
        "isomophic_mapping": ("mapping", "isomorphism"),
    }
    terms = task_terms.get(spec.task, ())
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b\D{{0,35}}(?:is|are|=|:)", low):
            return True
    return False


def _final_answer_nodes(text: str, spec: GraphLogicSpec) -> list[int]:
    final_text = _final_answer_text(text)
    if not final_text and _has_answer_assertion_cue(spec, text):
        final_text = text
    if not final_text:
        return []
    nodes = [int(raw) for raw in re.findall(r"\b\d+\b", final_text) if len(raw) <= 12 and int(raw) in spec.nodes]
    out: list[int] = []
    seen: set[int] = set()
    for node in nodes:
        if node not in seen:
            out.append(node)
            seen.add(node)
    return out


def _final_answer_edges(text: str, spec: GraphLogicSpec) -> set[tuple[int, int]]:
    final_text = _final_answer_text(text)
    if not final_text and _has_answer_assertion_cue(spec, text):
        final_text = text
    if not final_text:
        return set()
    return {edge for edge in _explicit_edges(final_text, spec) if edge in spec.edges}


def _ambiguous_final_answer_invalid(text: str) -> bool:
    if not _has_strong_final_answer_cue(text):
        return False
    final_text = _final_answer_text(text) or text
    return (
        "..." in final_text
        or "\\ldots" in final_text
        or "\\dots" in final_text
        or len(re.findall(r"\\boxed\s*\{", text)) > 1
    )


def _spanning_tree_valid(spec: GraphLogicSpec, edges: set[tuple[int, int]]) -> tuple[bool, float]:
    if len(edges) != max(0, len(spec.nodes) - 1):
        return False, 0.0
    parent = {node: node for node in spec.nodes}
    rank = {node: 0 for node in spec.nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    for u, v in edges:
        edge = _norm_edge(u, v, spec.directed)
        if edge not in spec.edges or u not in parent or v not in parent or not union(u, v):
            return False, 0.0
    roots = {find(node) for node in spec.nodes}
    if len(roots) != 1:
        return False, 0.0
    return True, sum(float(spec.weights.get(edge, 1.0)) for edge in edges)


def _mst_final_units(spec: GraphLogicSpec, text: str) -> tuple[set[tuple], int]:
    if spec.task not in MST_TASKS or not _has_answer_assertion_cue(spec, text):
        return set(), 0
    ambiguous = int(_ambiguous_final_answer_invalid(text))
    edges = _final_answer_edges(text, spec)
    if not edges:
        return set(), int(_has_strong_final_answer_cue(text)) + ambiguous
    valid, total = _spanning_tree_valid(spec, edges)
    if not valid:
        return set(), 1 + ambiguous
    optimal_edges = set(spec.target_edges or set())
    optimal_weight = _mst_optimal_weight(spec)
    units = {("mst_answer_spanning_tree", tuple(sorted(edges))[:12])}
    if optimal_edges and _close_number(total, optimal_weight):
        units.add(("mst_answer_valid", tuple(sorted(edges))[:12], round(float(total), 6)))
        return units, ambiguous
    return units, 1 + ambiguous


def _mst_parseable_object_units(
    spec: GraphLogicSpec,
    text: str,
    check_optimal: bool = False,
) -> tuple[set[tuple], int]:
    if spec.task not in MST_TASKS or not _has_answer_assertion_cue(spec, text):
        return set(), 0
    units: set[tuple] = set()
    invalid = int(_ambiguous_final_answer_invalid(text))
    edge_order = _final_answer_edge_order(text, spec)
    edges = set(edge_order)
    if not edge_order:
        return units, invalid + int(_has_strong_final_answer_cue(text))

    signature = tuple(edge_order[:12])
    units.add(("parseable_edge_list_answer", spec.task, signature))
    target_edge_count = max(0, len(spec.nodes) - 1)
    if len(edge_order) == target_edge_count and len(edges) == target_edge_count:
        units.add(("mst_answer_edge_count", spec.task, target_edge_count))
    else:
        invalid += 1
    valid, total = _spanning_tree_valid(spec, edges)
    if not valid:
        return units, invalid + 1

    units.add(("mst_answer_spanning_tree", signature))
    if check_optimal or spec.task == "weighted_minimum_spanning_tree":
        optimal_edges = set(spec.target_edges or set())
        optimal_weight = _mst_optimal_weight(spec)
        if optimal_edges and _close_number(total, optimal_weight):
            units.add(("state_mst_optimal_weight_cert", round(float(total), 6)))
            if any(_close_number(value, float(total)) for value in _numeric_values(text)):
                units.add(("state_mst_final_weight_checked", round(float(total), 6)))
            units.add(("mst_answer_valid", signature, round(float(total), 6)))
        elif optimal_edges:
            invalid += 1
    return units, invalid


def _topological_parseable_object_units(spec: GraphLogicSpec, text: str) -> tuple[set[tuple], int]:
    if spec.task != "topological_sort" or not _has_answer_assertion_cue(spec, text):
        return set(), 0
    units: set[tuple] = set()
    invalid = int(_ambiguous_final_answer_invalid(text))
    final_text = _final_answer_text(text) or text
    seqs = _node_sequences(final_text, spec)
    if not seqs:
        return units, invalid + int(_has_strong_final_answer_cue(text))

    saw_valid = False
    for seq in seqs:
        if not seq:
            continue
        signature = tuple(seq[: min(12, len(seq))])
        units.add(("parseable_node_list_answer", spec.task, signature))
        if _topological_order_valid(spec, seq):
            saw_valid = True
            units.add(("ordering_answer_valid", spec.task, signature))
    if not saw_valid and _has_strong_final_answer_cue(text):
        invalid += 1
    return units, invalid


def _edge_order_from_text(text: str, spec: GraphLogicSpec) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for m in re.finditer(r"\((\d+)\s*,\s*(\d+)(?:\s*,\s*[-+]?\d+(?:\.\d+)?)?\)", text):
        edge = _norm_edge(int(m.group(1)), int(m.group(2)), spec.directed)
        if edge in spec.edges:
            out.append(edge)
    return out


def _final_answer_edge_order(text: str, spec: GraphLogicSpec) -> list[tuple[int, int]]:
    final_text = _final_answer_text(text)
    if not final_text and _has_answer_assertion_cue(spec, text):
        final_text = text
    if not final_text:
        return []
    return _edge_order_from_text(final_text, spec)


def _traversal_parseable_and_order_units(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    if spec.task not in TRAVERSAL_TASKS:
        return set(), 0, 0
    low = text.lower()
    ordered_cue = any(
        cue in low
        for cue in (
            "order",
            "sequence",
            "visit",
            "visited",
            "discover",
            "traverse",
            "traversal",
            "queue",
            "stack",
            "enqueue",
            "dequeue",
            "push",
            "pop",
        )
    )
    finalish = _has_answer_assertion_cue(spec, text)
    if not ordered_cue and not finalish:
        return set(), 0, 0

    target = list(spec.target_node_order or [])
    target_edges = list(spec.target_edge_order or [])
    # The benchmark can provide both a traversal node order and the evaluated
    # traversal tree-edge order. If target_edge_order is present, final answers
    # must use the edge-list schema even when target_node_order is also known.
    expects_edge_answer = bool(target_edges)
    units: set[tuple] = set()
    actions = 0
    invalid = int(_ambiguous_final_answer_invalid(text)) if finalish else 0

    if finalish:
        saw_final_candidate = False
        saw_valid_final = False
        final_edge_order = _final_answer_edge_order(text, spec)
        if final_edge_order:
            saw_final_candidate = True
            actions += 1
            signature = tuple(final_edge_order[: min(12, len(final_edge_order))])
            units.add(("parseable_edge_list_answer", spec.task, signature))
            target_edges = list(spec.target_edge_order or [])
            prefix_len = 0
            for got, expected in zip(final_edge_order, target_edges):
                if got != expected:
                    break
                prefix_len += 1
            if prefix_len > 0:
                units.add(
                    (
                        "state_traversal_answer_edge_prefix",
                        spec.task,
                        int(prefix_len),
                        tuple(final_edge_order[: min(12, prefix_len)]),
                    )
                )
            if target_edges and prefix_len == len(target_edges) and len(final_edge_order) == len(target_edges):
                units.add(("ordering_answer_valid", spec.task, signature))
                saw_valid_final = True

        final_text = _final_answer_text(text)
        if not final_text and _has_answer_assertion_cue(spec, text):
            final_text = text[-2048:]
        final_node_sequences: list[list[int]] = []
        if final_text:
            final_node_sequences.extend(_node_sequences(final_text, spec))
            final_nodes = _final_answer_nodes(text, spec)
            if final_nodes:
                final_node_sequences.append(final_nodes)

        seen_node_sequences: set[tuple[int, ...]] = set()
        saw_wrong_node_schema = False
        for seq in final_node_sequences:
            if not seq:
                continue
            seq_key = tuple(seq)
            if seq_key in seen_node_sequences:
                continue
            seen_node_sequences.add(seq_key)
            saw_final_candidate = True
            if expects_edge_answer:
                if not final_edge_order:
                    saw_wrong_node_schema = True
                continue
            actions += 1
            signature = tuple(seq[: min(12, len(seq))])
            units.add(("parseable_node_list_answer", spec.task, signature))
            prefix_len = 0
            for got, expected in zip(seq, target):
                if got != expected:
                    break
                prefix_len += 1
            if prefix_len > 0:
                units.add(
                    (
                        "state_traversal_answer_prefix",
                        spec.task,
                        int(prefix_len),
                        tuple(seq[: min(12, prefix_len)]),
                    )
                )
            if target and prefix_len == len(target) and len(seq) == len(target):
                units.add(("ordering_answer_valid", spec.task, signature))
                saw_valid_final = True
        if saw_wrong_node_schema:
            actions += 1
            units.add(("schema_wrong_final_node_list", spec.task, len(seen_node_sequences)))
            invalid += 1

        if _has_strong_final_answer_cue(text):
            if not saw_final_candidate:
                actions += 1
                invalid += 1
            elif (target or spec.target_edge_order) and not saw_valid_final:
                invalid += 1

    seqs = [] if finalish or not target else _node_sequences(text, spec)

    for seq in seqs:
        if not seq:
            continue
        if ordered_cue or finalish:
            actions += 1

        start = state.node_order_pos
        advance = 0
        if seq and target and seq[0] == target[0]:
            full_prefix = 0
            for got, expected in zip(seq, target):
                if got != expected:
                    break
                full_prefix += 1
            if full_prefix > start:
                advance = full_prefix - start
        else:
            for got in seq:
                idx = start + advance
                if idx < len(target) and got == target[idx]:
                    advance += 1
                elif advance > 0:
                    break

        if advance > 0:
            for idx in range(start, min(start + advance, len(target))):
                units.add(("state_traversal_order_node", spec.task, int(target[idx]), int(idx + 1)))
            state.node_order_pos = min(len(target), start + advance)

    return units, actions, invalid


def _set_parseable_edge_answer_units(
    spec: GraphLogicSpec,
    text: str,
    check_optimal: bool = False,
) -> tuple[set[tuple], int]:
    if spec.task not in {"bipartite_maximum_matching", "max_weight_matching", "min_edge_covering"}:
        return set(), 0
    if not _has_answer_assertion_cue(spec, text):
        return set(), 0

    units: set[tuple] = set()
    invalid = int(_ambiguous_final_answer_invalid(text))
    edges = _final_answer_edges(text, spec)
    if not edges:
        return units, invalid + int(_has_strong_final_answer_cue(text))

    signature = tuple(sorted(edges))[:12]
    units.add(("parseable_edge_list_answer", spec.task, signature))
    feasible = _set_task_feasible(spec, set(), edges)
    if feasible:
        units.add(("edge_set_answer_feasible", spec.task, signature))
        if spec.task == "bipartite_maximum_matching":
            units.add(("matching_size", int(len(edges))))
        elif spec.task == "max_weight_matching":
            total_weight = _edge_set_weight(spec, edges)
            if any(_close_number(value, total_weight) for value in _numeric_values(text)):
                units.add(("max_weight_matching_weight", round(float(total_weight), 6)))
        elif spec.task == "min_edge_covering":
            units.add(("edge_cover_size", int(len(edges))))
        if check_optimal:
            target_edges = set(spec.target_edges or set())
            if spec.task == "max_weight_matching":
                target_weight = _max_weight_matching_target_weight(spec)
                total_weight = _edge_set_weight(spec, edges)
                total_weight_observed = any(_close_number(value, total_weight) for value in _numeric_values(text))
                if target_weight is not None and total_weight_observed and _close_number(total_weight, target_weight):
                    units.add(("max_weight_matching_optimal_weight", round(float(total_weight), 6)))
                elif target_weight is not None and any(cue in text.lower() for cue in ("maximum", "max weight", "optimal")):
                    invalid += 1
            elif spec.task == "min_edge_covering":
                if target_edges and len(edges) == len(target_edges):
                    units.add(("edge_set_answer_optimal_size", spec.task, len(edges)))
                    matching_size = max(0, len(spec.nodes) - len(edges))
                    units.add(("edge_cover_matching_size", int(matching_size)))
                else:
                    invalid += int(bool(target_edges))
            elif not target_edges or len(edges) <= len(target_edges):
                units.add(("edge_set_answer_optimal_size", spec.task, len(edges)))
            elif spec.task == "min_edge_covering":
                invalid += 1
    else:
        invalid += 1
    return units, invalid


def _weighted_path_cost_final_invalid(spec: GraphLogicSpec, text: str) -> bool:
    if spec.task != "weighted_shortest_path" or spec.source is None or spec.target is None:
        return False
    if not _has_answer_assertion_cue(spec, text):
        return False
    final_text = _final_answer_text(text) or text
    low = final_text.lower()
    if not any(cue in low for cue in ("cost", "distance", "weight", "total", "length")):
        return False
    values: list[float] = []
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    for pat in (
        rf"(?:cost|distance|weight|total|length)\D{{0,24}}(?:is|=|:)?\D{{0,12}}{number}",
        rf"{number}\D{{0,18}}(?:cost|distance|weight|total|length)",
    ):
        for m in re.finditer(pat, final_text, flags=re.I | re.S):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    if not values:
        return False
    target = _shortest_distance(spec, spec.source, spec.target)
    return math.isfinite(target) and not any(_close_number(value, target) for value in values)


def _true_center_nodes(spec: GraphLogicSpec) -> set[int]:
    ecc = {node: _eccentricity(spec, node) for node in spec.nodes}
    if not ecc:
        return set()
    best = min(ecc.values())
    return {node for node, value in ecc.items() if _close_number(value, best)}


def _true_periphery_nodes(spec: GraphLogicSpec) -> set[int]:
    ecc = {node: _eccentricity(spec, node) for node in spec.nodes}
    if not ecc:
        return set()
    best = max(ecc.values())
    return {node for node, value in ecc.items() if _close_number(value, best)}


def _true_barycenter_nodes(spec: GraphLogicSpec) -> set[int]:
    sums = {node: _distance_sum(spec, node) for node in spec.nodes}
    if not sums:
        return set()
    best = min(sums.values())
    return {node for node, value in sums.items() if _close_number(value, best)}


def _target_answer_nodes(spec: GraphLogicSpec) -> set[int]:
    if spec.task == "center":
        return _true_center_nodes(spec)
    if spec.task == "periphery":
        return _true_periphery_nodes(spec)
    if spec.task == "barycenter":
        return _true_barycenter_nodes(spec)
    return set(spec.target_nodes or set())


def _topological_order_valid(spec: GraphLogicSpec, seq: list[int]) -> bool:
    if set(seq) != set(spec.nodes) or len(seq) != len(spec.nodes):
        return False
    rank = {node: idx for idx, node in enumerate(seq)}
    return all(rank.get(u, -1) < rank.get(v, -1) for u, v in spec.edges)


def _hamiltonian_path_valid(spec: GraphLogicSpec, seq: list[int]) -> bool:
    return len(seq) == len(spec.nodes) and len(set(seq)) == len(spec.nodes) and bool(_valid_path_edges(spec, seq))


def _tsp_tour_valid(spec: GraphLogicSpec, seq: list[int]) -> bool:
    if len(seq) >= 2 and seq[0] == seq[-1]:
        seq = seq[:-1]
    if len(seq) != len(spec.nodes) or len(set(seq)) != len(spec.nodes):
        return False
    if not _valid_path_edges(spec, seq):
        return False
    return _norm_edge(seq[-1], seq[0], spec.directed) in spec.edges


def _tsp_tour_cost(spec: GraphLogicSpec, seq: list[int]) -> float | None:
    if len(seq) >= 2 and seq[0] == seq[-1]:
        seq = seq[:-1]
    if not _tsp_tour_valid(spec, seq):
        return None
    edges = _valid_path_edges(spec, seq)
    closing_edge = _norm_edge(seq[-1], seq[0], spec.directed)
    return _path_cost(spec, edges + [closing_edge])


def _set_task_feasible(spec: GraphLogicSpec, nodes: set[int], edges: set[tuple[int, int]]) -> bool:
    if spec.task == "min_vertex_cover":
        return bool(nodes) and all(u in nodes or v in nodes for u, v in spec.edges)
    if spec.task == "dominating_set":
        if not nodes:
            return False
        adj = build_adj(spec)
        dominated = set(nodes)
        for node in nodes:
            dominated.update(adj.get(node, []))
        return set(spec.nodes) <= dominated
    if spec.task == "maximal_independent_set":
        if not nodes:
            return False
        for u in nodes:
            for v in nodes:
                if u < v and _norm_edge(u, v, spec.directed) in spec.edges:
                    return False
        for node in spec.nodes - nodes:
            if all(_norm_edge(node, chosen, spec.directed) not in spec.edges for chosen in nodes):
                return False
        return True
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching"}:
        used: set[int] = set()
        for u, v in edges:
            if u in used or v in used:
                return False
            used.update((u, v))
        return bool(edges)
    if spec.task == "min_edge_covering":
        covered = {node for edge in edges for node in edge}
        return bool(edges) and set(spec.nodes) <= covered
    return False


def _final_answer_validation_units(
    spec: GraphLogicSpec,
    text: str,
) -> tuple[set[tuple], int]:
    if not _has_answer_assertion_cue(spec, text):
        return set(), 0
    units: set[tuple] = set()
    invalid = 0

    if spec.task in {"center", "periphery", "barycenter"}:
        invalid += int(_ambiguous_final_answer_invalid(text))
        nodes = set(_final_answer_nodes(text, spec))
        target = _target_answer_nodes(spec)
        if nodes:
            if nodes == target:
                units.add(("node_answer_valid", spec.task, tuple(sorted(target))))
            else:
                invalid += 1
        elif _has_strong_final_answer_cue(text):
            invalid += 1

    elif spec.task in {"topological_sort", "hamiltonian_path", "traveling_salesman_problem"}:
        invalid += int(_ambiguous_final_answer_invalid(text))
        final_text = _final_answer_text(text)
        saw_candidate = False
        saw_valid = False
        for seq in _node_sequences(final_text, spec):
            saw_candidate = True
            if spec.task == "topological_sort" and _topological_order_valid(spec, seq):
                saw_valid = True
                units.add(("ordering_answer_valid", spec.task, tuple(seq[: min(12, len(seq))])))
            elif spec.task == "hamiltonian_path" and _hamiltonian_path_valid(spec, seq):
                saw_valid = True
                units.add(("ordering_answer_valid", spec.task, tuple(seq[: min(12, len(seq))])))
            elif spec.task == "traveling_salesman_problem" and _tsp_tour_valid(spec, seq):
                cost = _tsp_tour_cost(spec, seq)
                target_order = list(spec.target_edge_order or [])
                target_cost = _path_cost(spec, target_order) if target_order else None
                if cost is not None and (target_cost is None or cost <= target_cost + 1e-9):
                    saw_valid = True
                    units.add(("ordering_answer_valid", spec.task, tuple(seq[: min(12, len(seq))])))
                else:
                    invalid += 1
        if saw_candidate and not saw_valid:
            invalid += 1
        elif not saw_candidate and _has_strong_final_answer_cue(text):
            invalid += 1

    elif spec.task in {"min_vertex_cover", "dominating_set", "maximal_independent_set"}:
        invalid += int(_ambiguous_final_answer_invalid(text))
        nodes = set(_final_answer_nodes(text, spec))
        target_size = len(spec.target_nodes or set())
        if nodes:
            feasible = _set_task_feasible(spec, nodes, set())
            if feasible:
                units.add(("set_answer_feasible", spec.task, tuple(sorted(nodes))[:12]))
                if not target_size or len(nodes) <= target_size:
                    units.add(("set_answer_optimal_size", spec.task, len(nodes)))
                elif spec.task in {"min_vertex_cover", "dominating_set"}:
                    invalid += 1
            else:
                invalid += 1
        elif _has_strong_final_answer_cue(text):
            invalid += 1

    elif spec.task in {"bipartite_maximum_matching", "max_weight_matching", "min_edge_covering", "bridges"}:
        invalid += int(_ambiguous_final_answer_invalid(text))
        edges = _final_answer_edges(text, spec)
        target_edges = set(spec.target_edges or set())
        if edges:
            if spec.task == "bridges":
                if edges == target_edges:
                    units.add(("edge_answer_valid", spec.task, tuple(sorted(edges))[:12]))
                else:
                    invalid += 1
            else:
                feasible = _set_task_feasible(spec, set(), edges)
                if feasible:
                    units.add(("edge_set_answer_feasible", spec.task, tuple(sorted(edges))[:12]))
                    if spec.task == "bipartite_maximum_matching":
                        units.add(("matching_size", int(len(edges))))
                    elif spec.task == "max_weight_matching":
                        total_weight = _edge_set_weight(spec, edges)
                        if any(_close_number(value, total_weight) for value in _numeric_values(text)):
                            units.add(("max_weight_matching_weight", round(float(total_weight), 6)))
                    elif spec.task == "min_edge_covering":
                        units.add(("edge_cover_size", int(len(edges))))
                    if spec.task == "max_weight_matching":
                        target_weight = _max_weight_matching_target_weight(spec)
                        total_weight = _edge_set_weight(spec, edges)
                        total_weight_observed = any(_close_number(value, total_weight) for value in _numeric_values(text))
                        if target_weight is not None and total_weight_observed and _close_number(total_weight, target_weight):
                            units.add(("max_weight_matching_optimal_weight", round(float(total_weight), 6)))
                        elif target_weight is not None and any(
                            cue in low for cue in ("maximum", "max weight", "optimal")
                        ):
                            invalid += 1
                    elif spec.task == "min_edge_covering":
                        if target_edges and len(edges) == len(target_edges):
                            units.add(("edge_set_answer_optimal_size", spec.task, len(edges)))
                            matching_size = max(0, len(spec.nodes) - len(edges))
                            units.add(("edge_cover_matching_size", int(matching_size)))
                        else:
                            invalid += int(bool(target_edges))
                    elif not target_edges or len(edges) <= len(target_edges):
                        units.add(("edge_set_answer_optimal_size", spec.task, len(edges)))
                    elif spec.task == "min_edge_covering":
                        invalid += 1
                else:
                    invalid += 1
        elif _has_strong_final_answer_cue(text):
            invalid += 1

    return units, invalid


def _finite_distances_from(spec: GraphLogicSpec, source: int) -> list[float]:
    values = []
    for node in sorted(spec.nodes):
        if node == source:
            continue
        dist = _shortest_distance(spec, source, node)
        if math.isfinite(dist):
            values.append(float(dist))
    return values


def _eccentricity(spec: GraphLogicSpec, source: int) -> float:
    values = _finite_distances_from(spec, source)
    return max(values) if values else 0.0


def _distance_sum(spec: GraphLogicSpec, source: int) -> float:
    return sum(_finite_distances_from(spec, source))


def _connected_components(spec: GraphLogicSpec) -> list[set[int]]:
    if not spec.directed:
        adj = build_adj(spec)
        seen: set[int] = set()
        comps: list[set[int]] = []
        for start in sorted(spec.nodes):
            if start in seen:
                continue
            comp = {start}
            seen.add(start)
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj.get(u, []):
                    if v not in seen:
                        seen.add(v)
                        comp.add(v)
                        queue.append(v)
            comps.append(comp)
        return comps

    adj = build_adj(spec)
    rev: dict[int, list[int]] = defaultdict(list)
    for u, vs in adj.items():
        for v in vs:
            rev[v].append(u)
    order: list[int] = []
    seen: set[int] = set()

    def dfs1(u: int) -> None:
        seen.add(u)
        for v in adj.get(u, []):
            if v not in seen:
                dfs1(v)
        order.append(u)

    def dfs2(u: int, comp: set[int]) -> None:
        comp.add(u)
        seen.add(u)
        for v in rev.get(u, []):
            if v not in seen:
                dfs2(v, comp)

    for node in sorted(spec.nodes):
        if node not in seen:
            dfs1(node)
    seen.clear()
    comps = []
    for node in reversed(order):
        if node in seen:
            continue
        comp: set[int] = set()
        dfs2(node, comp)
        comps.append(comp)
    for node in sorted(spec.nodes):
        if node not in seen:
            comps.append({node})
    return comps


def _reachable_from(spec: GraphLogicSpec, source: int) -> set[int]:
    if source not in spec.nodes:
        return set()
    adj = build_adj(spec)
    seen = {source}
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def _valid_path_edges(spec: GraphLogicSpec, nodes: list[int]) -> list[tuple[int, int]]:
    if len(nodes) < 2:
        return []
    edges = [_norm_edge(u, v, spec.directed) for u, v in zip(nodes, nodes[1:])]
    return edges if all(edge in spec.edges for edge in edges) else []


def _path_cost(spec: GraphLogicSpec, path_edges: Iterable[tuple[int, int]]) -> float:
    return sum(float(spec.weights.get(edge, 1.0)) for edge in path_edges)


def _edge_set_weight(spec: GraphLogicSpec, edges: Iterable[tuple[int, int]]) -> float:
    return sum(float(spec.weights.get(edge, 1.0)) for edge in edges)


def _max_weight_matching_target_weight(spec: GraphLogicSpec) -> float | None:
    if spec.task != "max_weight_matching" or not spec.target_edges:
        return None
    return _edge_set_weight(spec, spec.target_edges)


def _path_is_shortest_prefix(spec: GraphLogicSpec, nodes: list[int], path_edges: list[tuple[int, int]]) -> bool:
    if spec.task not in PATH_TASKS:
        return True
    if not path_edges:
        return False
    if spec.source is None:
        order = list(spec.target_edge_order or [])
        return bool(order) and path_edges == order[: len(path_edges)]
    if nodes[0] != spec.source:
        return False
    cost = _path_cost(spec, path_edges)
    dist_to_last = _shortest_distance(spec, spec.source, nodes[-1])
    if not math.isfinite(dist_to_last) or not _close_number(cost, dist_to_last):
        return False
    if spec.target is None:
        return True
    total = _shortest_distance(spec, spec.source, spec.target)
    suffix = _shortest_distance(spec, nodes[-1], spec.target)
    if not math.isfinite(total):
        return False
    if nodes[-1] == spec.target:
        return _close_number(cost, total)
    return math.isfinite(suffix) and _close_number(cost + suffix, total)


def _degree_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task not in DEGREE_WITNESS_TASKS:
        return set()
    low = text.lower()
    if "degree" not in low and "odd" not in low and "even" not in low:
        return set()
    degs = _degrees(spec)
    units: set[tuple] = set()
    patterns = (
        r"degree\s+of\s+(?:node\s+)?(\d+)\D{0,20}(\d+)",
        r"(?:node\s+)?(\d+)\D{0,12}(?:has|with|degree)\D{0,12}degree\D{0,12}(\d+)",
        r"(?:node\s+)?(\d+)\D{0,12}(?:has|with)\D{0,12}(\d+)\D{0,12}(?:neighbors?|neighbours?)",
        r"(?:deg|d)\s*\(\s*(\d+)\s*\)\s*=\s*(\d+)",
        r"(?:deg|degree|d)\s*\[\s*(\d+)\s*\]\s*=\s*(\d+)",
        r"(?:deg|degree|d)\s*_\s*(\d+)\s*=\s*(\d+)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            node, degree = int(m.group(1)), int(m.group(2))
            if degs.get(node) == degree:
                units.add(("degree", node, degree))
    if "degree" in low:
        for m in re.finditer(r"(?:node\s+)?(\d+)\s*[:=]\s*(\d+)", text, flags=re.I):
            node, degree = int(m.group(1)), int(m.group(2))
            if degs.get(node) == degree:
                units.add(("degree", node, degree))
    if spec.task == "is_regular":
        vals = set(degs.values())
        if len(vals) == 1 and re.search(r"\b(all|every|each)\b.{0,40}\bdegree\b", low):
            units.add(("regular_all_degrees", next(iter(vals))))
        if len(vals) > 1 and any(cue in low for cue in ("different degree", "not regular", "degrees differ")):
            units.add(("regular_counterexample", min(vals), max(vals)))
    if spec.task == "is_eularian":
        odd = {node for node, degree in degs.items() if degree % 2 == 1}
        if not odd and "even" in low:
            units.add(("euler_all_even",))
        for node in odd:
            if re.search(rf"(?:node\s+)?{node}\D{{0,30}}\bodd\b|\bodd\b\D{{0,30}}(?:node\s+)?{node}", low):
                units.add(("euler_odd_node", node))
    return units


def _count_formula_units(spec: GraphLogicSpec, text: str, answer_hit: bool) -> set[tuple]:
    task = spec.task
    if task not in COUNT_FORMULA_TASKS and task not in {"is_regular", "is_eularian"}:
        return set()
    low = text.lower()
    units: set[tuple] = set()
    n = len(spec.nodes)
    m = len(spec.edges)
    if re.search(rf"(?:number\s+of\s+)?(?:nodes|vertices|n)\D{{0,20}}{n}\b", low) or re.search(
        rf"\b{n}\s+(?:nodes|vertices)\b", low
    ) or re.search(
        rf"(?:there\s+(?:are|is)|has|with)\D{{0,20}}{n}\D{{0,10}}(?:nodes|vertices)\b", low
    ) or re.search(
        rf"(?:\|v\||#v|\bv\b|\bn\b)\s*=\s*{n}\b", text, flags=re.I
    ):
        units.add(("node_count", n))
    if re.search(rf"(?:number\s+of\s+)?(?:edges|edge\s+count|m)\D{{0,20}}{m}\b", low) or re.search(
        rf"\b{m}\s+edges?\b", low
    ) or re.search(
        rf"(?:there\s+(?:are|is)|has|with)\D{{0,20}}{m}\D{{0,10}}edges?\b", low
    ) or re.search(
        rf"(?:\|e\||#e|\be\b|\bm\b)\s*=\s*{m}\b", text, flags=re.I
    ):
        units.add(("edge_count", m))
    if task == "density" and (
        "density" in low
        or "2m" in low
        or "n(n-1)" in low
        or "|v|(|v|-1)" in low
        or "choose" in low
        or "possible edges" in low
    ):
        if answer_hit or ("2m" in low and "n" in low) or ("edge" in low and "node" in low):
            units.add(("density_formula", n, m))
    if task == "density" and any(cue in low for cue in ("possible edges", "total possible", "complete graph")):
        if "edge" in low and ("node" in low or "vertex" in low):
            units.add(("density_formula", n, m))
    if task == "triangles" and answer_hit:
        units.add(("triangle_count_answer",))
    return units


def _triangle_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "triangles":
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("triangle", "triad", "3-cycle", "three nodes", "connected", "adjacent")):
        return set()
    units: set[tuple] = set()
    for m in re.finditer(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", text):
        tri = tuple(sorted(int(x) for x in m.groups()))
        if len(set(tri)) != 3:
            continue
        a, b, c = tri
        edges = {
            _norm_edge(a, b, spec.directed),
            _norm_edge(a, c, spec.directed),
            _norm_edge(b, c, spec.directed),
        }
        if edges <= spec.edges:
            units.add(("triangle", tri))
    if spec.source is not None and spec.source in spec.nodes:
        adj = set(build_adj(spec).get(spec.source, []))
        negated = any(
            cue in low
            for cue in (
                "not connected",
                "not directly connected",
                "not adjacent",
                "no edge",
                "does not have an edge",
                "doesn't have an edge",
            )
        )
        for u, v in _pair_mentions(text, spec.nodes):
            if spec.source in {u, v} or u not in adj or v not in adj:
                continue
            edge = _norm_edge(u, v, spec.directed)
            tri = tuple(sorted((int(spec.source), int(u), int(v))))
            if edge in spec.edges:
                units.add(("triangle", tri))
            elif negated:
                units.add(("triangle_nonedge_checked", tri))
    return units


def _tournament_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "is_tournament":
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("pair", "tournament", "direction", "exactly one", "edge")):
        return set()
    units: set[tuple] = set()
    nums = sorted(n for n in _numbers_in_text(text, spec.nodes))
    if len(nums) >= 2:
        for i, u in enumerate(nums):
            for v in nums[i + 1 :]:
                uv = (u, v) in spec.edges
                vu = (v, u) in spec.edges
                if uv != vu:
                    units.add(("tournament_pair", u, v))
                elif "not" in low or "both" in low or "missing" in low or "no edge" in low:
                    units.add(("tournament_violation", u, v))
    if spec.answer_bool is True and "tournament" in low and not any(cue in low for cue in ("not", "missing", "both")):
        units.add(("tournament_answer",))
    if spec.answer_bool is False and any(cue in low for cue in ("not a tournament", "not tournament", "violate")):
        units.add(("tournament_answer",))
    return units


def _hamiltonian_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "hamiltonian_path":
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("hamiltonian", "path", "visit", "route", "order")):
        return set()
    units: set[tuple] = set()
    if any(cue in low for cue in ("each node exactly once", "every node exactly once", "visit all", "all nodes", "no repeat", "without repeating")):
        units.add(("hamiltonian_all_nodes_scope", int(len(spec.nodes))))
    if any(cue in low for cue in ("adjacent", "edge exists", "connected by an edge", "valid edge")):
        units.add(("hamiltonian_adjacency_rule",))
    if any(cue in low for cue in ("unvisited", "remaining", "unused")):
        units.add(("hamiltonian_unvisited_tracking",))
    for seq in _node_sequences(text, spec):
        seen: set[int] = set()
        valid_prefix: list[int] = []
        for node in seq:
            if node in seen:
                if any(cue in low for cue in ("repeat", "duplicate", "already visited")):
                    units.add(("hamiltonian_duplicate_rejected", int(node)))
                break
            if valid_prefix:
                edge = _norm_edge(valid_prefix[-1], node, spec.directed)
                if edge not in spec.edges:
                    break
                units.add(("hamiltonian_edge", edge))
                units.add(("hamiltonian_valid_extension", int(valid_prefix[-1]), int(node)))
            seen.add(node)
            valid_prefix.append(node)
            units.add(("hamiltonian_node", node))
        if valid_prefix:
            units.add(("hamiltonian_prefix_len", len(valid_prefix)))
        if len(valid_prefix) == len(spec.nodes):
            units.add(("hamiltonian_complete",))
    return units


def _pair_mentions(text: str, nodes: set[int]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for m in re.finditer(r"\((\d+)\s*,\s*(\d+)\)", text):
        u, v = int(m.group(1)), int(m.group(2))
        if u in nodes and v in nodes and u != v:
            pairs.add((u, v))
    patterns = (
        r"(?:from|between)\s+(?:node\s+)?(\d+)\D{0,30}(?:to|and)\s+(?:node\s+)?(\d+)",
        r"(?:node\s+)?(\d+)\s+(?:to|and)\s+(?:node\s+)?(\d+)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            u, v = int(m.group(1)), int(m.group(2))
            if u in nodes and v in nodes and u != v:
                pairs.add((u, v))
    return pairs


def _pair_distance_observations(text: str, spec: GraphLogicSpec) -> list[tuple[int, int, float]]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:distance|dist|shortest\s+path|path|length)\D{{0,50}}(?:from|between)?\s*(?:node\s+)?(\d+)\D{{0,35}}(?:to|and)\s*(?:node\s+)?(\d+)\D{{0,35}}(?:is|=|:|length|distance|steps?)\D{{0,12}}{number}",
        rf"(?:node\s+)?(\d+)\s+(?:to|and)\s+(?:node\s+)?(\d+)\D{{0,35}}(?:distance|dist|length|steps?|edges?|is|=|:)\D{{0,12}}{number}",
        rf"\((\d+)\s*,\s*(\d+)\)\D{{0,45}}(?:distance|dist|length|steps?|is|=|:)\D{{0,12}}{number}",
    )
    observations: list[tuple[int, int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u in spec.nodes and v in spec.nodes and u != v:
                observations.append((u, v, float(m.group(3))))
    return observations


def _global_efficiency_observations(text: str, spec: GraphLogicSpec) -> list[tuple[int, int, float]]:
    if spec.task != "global_efficiency":
        return []
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    patterns = (
        rf"(?:efficiency|contribution|reciprocal|1/d|1\s*/\s*d)\D{{0,50}}"
        rf"(?:from|between)?\s*(?:node\s+)?(\d+)\D{{0,30}}(?:to|and)\s*(?:node\s+)?(\d+)"
        rf"\D{{0,45}}(?:is|=|:)\D{{0,12}}{number}",
        rf"\((\d+)\s*,\s*(\d+)\)\D{{0,70}}(?:efficiency|contribution|reciprocal|1/d|1\s*/\s*d)"
        rf"\D{{0,30}}(?:is|=|:)\D{{0,12}}{number}",
    )
    observations: list[tuple[int, int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u in spec.nodes and v in spec.nodes and u != v:
                value = _parse_float_token(m.group(3))
                if value is not None:
                    observations.append((u, v, value))
    return observations


def _node_distance_sum_observations(text: str, spec: GraphLogicSpec) -> list[tuple[int, float]]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:node\s+)?(\d+).{{0,120}}?(?:sum|total)\s+(?:of\s+)?(?:distances?|shortest\s+paths?).{{0,45}}?{number}",
        rf"(?:sum|total)\s+(?:of\s+)?(?:distances?|shortest\s+paths?).{{0,90}}?(?:from|of|for)?\s*(?:node\s+)?(\d+).{{0,45}}?(?:is|=|:).{{0,20}}?{number}",
        rf"(?:sumdist|distance\s+sum|total\s+distance|S)\s*\(\s*(?:node\s+)?(\d+)\s*\)\s*(?:=|:|is)\s*{number}",
        rf"(?:node\s+)?(\d+)\s*[:=]\s*{number}\s*(?:total\s+distance|distance\s+sum|sum\s+distance)",
    )
    observations: list[tuple[int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            node = int(m.group(1))
            if node in spec.nodes:
                observations.append((node, float(m.group(2))))
    return observations


def _compact_pair_distance_observations(text: str, spec: GraphLogicSpec) -> list[tuple[int, int, float]]:
    low = text.lower()
    if not any(cue in low for cue in ("distance", "dist", "shortest", "matrix", "table", "wiener", "efficiency", "d(", "d[")):
        return []
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:d|dist|distance)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*(?:=|:|is)\s*{number}",
        rf"(?:d|dist|distance)\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]\s*(?:=|:|is)\s*{number}",
        rf"\((\d+)\s*,\s*(\d+)\)\s*(?:=|:)\s*{number}\s*(?:\b|,|;)",
        rf"\b(\d+)\s*[-–]\s*(\d+)\s*(?:=|:)\s*{number}\s*(?:\b|,|;)",
        rf"\b(\d+)\s*[-–]\s*(\d+)\b.{{0,36}}(?:distance|dist|length|steps?)\D{{0,12}}{number}",
    )
    observations: list[tuple[int, int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u in spec.nodes and v in spec.nodes and u != v:
                observations.append((u, v, float(m.group(3))))
    return observations


def _eccentricity_observations(text: str, spec: GraphLogicSpec) -> list[tuple[int, float]]:
    low = text.lower()
    if not any(cue in low for cue in ("eccentricity", "ecc", "farthest", "max distance", "maximum distance")):
        return []
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:ecc|eccentricity)\s*\(\s*(?:node\s+)?(\d+)\s*\)\s*(?:=|:|is)\s*{number}",
        rf"(?:node\s+)?(\d+)\s*(?:ecc|eccentricity)\s*(?:=|:|is)\s*{number}",
    )
    observations: list[tuple[int, float]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            node = int(m.group(1))
            if node in spec.nodes:
                observations.append((node, float(m.group(2))))
    return observations


def _named_metric_value_observations(text: str, names: Iterable[str]) -> list[float]:
    name_alt = "|".join(re.escape(name) for name in names)
    if not name_alt:
        return []
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    patterns = (
        rf"(?:{name_alt}).{{0,45}}?(?:is|=|:|becomes|equals).{{0,12}}?{number}",
        rf"{number}.{{0,45}}?(?:{name_alt})",
    )
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            raw = m.group(1)
            value = _parse_float_token(raw)
            if value is not None:
                values.append(value)
    return values


def _betweenness_value_observations(text: str, spec: GraphLogicSpec) -> list[float]:
    if spec.task != "betweenness_centrality":
        return []
    low = text.lower()
    if not any(cue in low for cue in ("betweenness", "centrality", "score", "value", "contribution")):
        return []
    values: list[float] = []
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    patterns = (
        rf"(?:betweenness|centrality|score|value|contribution).{{0,45}}?(?:is|=|:).{{0,12}}?{number}",
        rf"{number}.{{0,45}}?(?:betweenness|centrality)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            value = _parse_float_token(m.group(1))
            if value is not None:
                values.append(value)
    return values


def _betweenness_raw_score(spec: GraphLogicSpec, pivot: int) -> float:
    if pivot not in spec.nodes:
        return 0.0
    nodes = sorted(spec.nodes)
    total = 0.0
    for i, source in enumerate(nodes):
        if source == pivot:
            continue
        for target in nodes:
            if target == pivot or target == source:
                continue
            if not spec.directed and nodes.index(target) <= i:
                continue
            d_st = _shortest_distance(spec, source, target)
            d_sp = _shortest_distance(spec, source, pivot)
            d_pt = _shortest_distance(spec, pivot, target)
            if not math.isfinite(d_st) or not _close_number(d_sp + d_pt, d_st):
                continue
            total_paths = _shortest_path_count(spec, source, target)
            if total_paths <= 0:
                continue
            via_paths = _shortest_path_count(spec, source, pivot) * _shortest_path_count(spec, pivot, target)
            total += float(via_paths) / float(total_paths)
    return total


def _betweenness_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "betweenness_centrality" or spec.source is None:
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("between", "shortest path", "pass through", "passes through", "through node")):
        return set()
    pivot = int(spec.source)
    units: set[tuple] = set()
    pivot_ref = rf"(?:node\s*)?{pivot}\b"
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    certified_pair_patterns = (
        rf"\((\d+)\s*,\s*(\d+)\).{{0,120}}?(?:through|via|pass(?:es)?\s+through).{{0,40}}?{pivot_ref}.{{0,80}}?{number}?",
        rf"(?:from|between)\s+(?:node\s+)?(\d+)\D{{0,40}}(?:to|and)\s+(?:node\s+)?(\d+).{{0,120}}?(?:through|via|pass(?:es)?\s+through).{{0,40}}?{pivot_ref}.{{0,80}}?{number}?",
        rf"(?:through|via|pass(?:es)?\s+through).{{0,40}}?{pivot_ref}.{{0,100}}?(?:from|between)\s+(?:node\s+)?(\d+)\D{{0,40}}(?:to|and)\s+(?:node\s+)?(\d+).{{0,80}}?{number}?",
    )
    for pat in certified_pair_patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            try:
                u, v = int(m.group(1)), int(m.group(2))
            except (IndexError, ValueError):
                continue
            if u not in spec.nodes or v not in spec.nodes or pivot in {u, v} or u == v:
                continue
            context = m.group(0)
            a, b = (u, v) if spec.directed or u <= v else (v, u)
            d_uv = _shortest_distance(spec, u, v)
            d_up = _shortest_distance(spec, u, pivot)
            d_pv = _shortest_distance(spec, pivot, v)
            via_shortest = math.isfinite(d_uv) and _close_number(d_up + d_pv, d_uv)
            if not via_shortest:
                continue
            units.add(("betweenness_pair_via_cert", a, b, pivot))
            total = _shortest_path_count(spec, u, v)
            via = _shortest_path_count(spec, u, pivot) * _shortest_path_count(spec, pivot, v)
            fraction = float(via) / float(total) if total > 0 else 0.0
            context_values = _numeric_values(context)
            if total > 0 and any(
                _close_number(value, fraction) or _close_number(value, via) or _close_number(value, total)
                for value in context_values
            ):
                units.add(("betweenness_sigma_fraction_value", a, b, pivot, round(float(fraction), 6)))
    for u, v in _pair_mentions(text, spec.nodes):
        if pivot in {u, v}:
            continue
        a, b = (u, v) if spec.directed or u <= v else (v, u)
        units.add(("betweenness_pair_considered", a, b, pivot))
        d_uv = _shortest_distance(spec, u, v)
        d_up = _shortest_distance(spec, u, pivot)
        d_pv = _shortest_distance(spec, pivot, v)
        via_shortest = math.isfinite(d_uv) and abs((d_up + d_pv) - d_uv) <= 1e-9
        if via_shortest:
            units.add(("betweenness_pair_via", a, b, pivot))
        elif math.isfinite(d_uv):
            units.add(("betweenness_pair_checked", a, b, pivot))
        if math.isfinite(d_uv) and any(cue in low for cue in ("sigma", "fraction", "divide", "contribution")):
            total = _shortest_path_count(spec, u, v)
            via = 0
            if via_shortest:
                via = _shortest_path_count(spec, u, pivot) * _shortest_path_count(spec, pivot, v)
            if total > 0:
                fraction = float(via) / float(total)
                values = _numeric_values(text)
                if any(_close_number(value, fraction) or _close_number(value, via) or _close_number(value, total) for value in values):
                    units.add(("betweenness_sigma_fraction_value", a, b, pivot, round(float(fraction), 6)))
    for seq in _node_sequences(text, spec):
        if pivot not in seq[1:-1]:
            continue
        valid_edges = [_norm_edge(a, b, spec.directed) for a, b in zip(seq, seq[1:])]
        if all(edge in spec.edges for edge in valid_edges):
            u, v = seq[0], seq[-1]
            d_uv = _shortest_distance(spec, u, v)
            if math.isfinite(d_uv) and len(valid_edges) == int(d_uv):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("betweenness_path_via", a, b, pivot))
    return units


def _distance_pair_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
) -> set[tuple]:
    if spec.task not in DISTANCE_WITNESS_TASKS:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "distance",
            "dist",
            "shortest",
            "path",
            "reachable",
            "eccentricity",
            "length",
            "step",
            "steps",
            "directly connected",
            "maximum",
            "total",
            "sum",
            "d(",
            "d[",
            "matrix",
            "table",
        )
    ):
        return set()
    if spec.task == "weighted_shortest_path" and not _has_any(text, ACTION_CUES["weighted_path"]):
        return set()
    pairs: set[tuple[int, int]] = set()
    for u, v, _observed in _compact_pair_distance_observations(text, spec):
        pairs.add((u, v))
    patterns = (
        r"(?:distance|dist|shortest\s+path|path)\D{0,40}"
        r"(?:from|between)\s+(?:node\s+)?(\d+)\D{0,30}(?:to|and)\s+(?:node\s+)?(\d+)",
        r"(?:from)\s+(?:node\s+)?(\d+)\s+(?:to)\s+(?:node\s+)?(\d+)",
        r"(?:between)\s+(?:node\s+)?(\d+)\s+(?:and)\s+(?:node\s+)?(\d+)",
        r"(?:node\s+)?(\d+)\s+(?:to|and)\s+(?:node\s+)?(\d+).{0,40}(?:distance|dist|shortest|path)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u in spec.nodes and v in spec.nodes and u != v:
                pairs.add((u, v))
    for edge in valid_edges:
        u, v = edge
        if any(cue in low for cue in ("path", "shortest", "distance", "direct")):
            pairs.add((u, v))
    if spec.task in {"closeness_centrality", "harmonic_centrality"} and spec.source is not None:
        pairs = {pair for pair in pairs if spec.source in pair}
    if spec.task == "local_connectivity" and spec.source is not None and spec.target is not None:
        target_pair = {spec.source, spec.target}
        pairs = {pair for pair in pairs if set(pair) == target_pair}
    units = set()
    for u, v in pairs:
        if math.isfinite(_shortest_distance(spec, u, v)):
            a, b = (u, v) if spec.directed or u <= v else (v, u)
            units.add(("distance_pair", int(a), int(b)))
    return units


def _distance_metric_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task not in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "distance",
            "dist",
            "shortest",
            "path",
            "length",
            "step",
            "steps",
            "eccentricity",
            "maximum",
            "total",
            "sum",
            "barycenter",
            "center",
            "radius",
            "diameter",
            "periphery",
            "centrality",
            "efficiency",
            "wiener",
            "reciprocal",
            "directly connected",
        )
    ):
        return set()
    units: set[tuple] = set()
    number = r"([-+]?\d+(?:\.\d+)?)"
    pair_value_patterns = (
        rf"(?:distance|dist|shortest\s+path|path|length)\D{{0,50}}(?:from|between)?\s*(?:node\s+)?(\d+)\D{{0,35}}(?:to|and)\s*(?:node\s+)?(\d+)\D{{0,35}}(?:is|=|:|length|distance|steps?)\D{{0,12}}{number}",
        rf"(?:node\s+)?(\d+)\s+(?:to|and)\s+(?:node\s+)?(\d+)\D{{0,35}}(?:distance|dist|length|steps?|edges?|is|=|:)\D{{0,12}}{number}",
        rf"\((\d+)\s*,\s*(\d+)\)\D{{0,45}}(?:distance|dist|length|steps?|is|=|:)\D{{0,12}}{number}",
    )
    for pat in pair_value_patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u not in spec.nodes or v not in spec.nodes or u == v:
                continue
            observed = float(m.group(3))
            target = _shortest_distance(spec, u, v)
            if math.isfinite(target) and _close_number(observed, target):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("distance_value", int(a), int(b), round(float(target), 6)))
    for u, v, observed in _compact_pair_distance_observations(text, spec):
        target = _shortest_distance(spec, u, v)
        if math.isfinite(target) and _close_number(observed, target):
            a, b = (u, v) if spec.directed or u <= v else (v, u)
            units.add(("distance_value", int(a), int(b), round(float(target), 6)))

    if "direct" in low or "directly connected" in low:
        for u, v in _pair_mentions(text, spec.nodes):
            edge = _norm_edge(u, v, spec.directed)
            if edge in spec.edges and _close_number(_shortest_distance(spec, u, v), 1.0):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("distance_value", int(a), int(b), 1.0))

    mentioned_nodes = _numbers_in_text(text, spec.nodes)
    values = _numeric_values(text)
    if spec.source is not None and spec.task in {"closeness_centrality", "harmonic_centrality"}:
        if any(cue in low for cue in ("distance", "dist", "direct", "step", "steps", "path")):
            for node in sorted(mentioned_nodes - {spec.source}):
                target = _shortest_distance(spec, spec.source, node)
                if math.isfinite(target) and any(_close_number(value, target) for value in values):
                    a, b = (spec.source, node) if spec.directed or spec.source <= node else (node, spec.source)
                    units.add(("distance_value", int(a), int(b), round(float(target), 6)))

    eccentricity_patterns = (
        rf"(?:node\s+)?(\d+).{{0,180}}?(?:eccentricity|max(?:imum)?\s+distance|maximum\s+is|max\s+is).{{0,35}}?{number}",
        rf"(?:eccentricity|max(?:imum)?\s+distance).{{0,70}}?(?:from|of|for)?\s*(?:node\s+)?(\d+).{{0,45}}?(?:is|=|:).{{0,20}}?{number}",
    )
    if spec.task in {"center", "periphery", "radius", "diameter", "barycenter"}:
        if any(cue in low for cue in ("all nodes", "every node", "each node", "for every node")):
            units.add(("all_nodes_metric_scope", spec.task))
        if spec.task == "center" and any(cue in low for cue in ("minimum eccentricity", "min eccentricity", "smallest eccentricity", "radius")):
            units.add(("center_min_ecc_rule",))
        if spec.task == "periphery" and any(cue in low for cue in ("maximum eccentricity", "max eccentricity", "largest eccentricity", "diameter")):
            units.add(("periphery_max_ecc_rule",))
        if spec.task == "barycenter" and any(cue in low for cue in ("minimum distance sum", "smallest distance sum", "minimize", "minimum total distance")):
            units.add(("barycenter_min_sum_rule",))
        if any(cue in low for cue in ("tie", "ties", "all nodes", "all centers", "all peripheral", "all barycenter", "multiple")):
            units.add(("metric_tie_scope", spec.task))
        target_nodes = _target_answer_nodes(spec)
        if target_nodes and any(
            cue in low
            for cue in (
                "center",
                "periphery",
                "peripheral",
                "barycenter",
                "minimum eccentricity",
                "maximum eccentricity",
                "minimum distance sum",
                "smallest distance sum",
            )
        ):
            for node in sorted(mentioned_nodes & target_nodes):
                units.add(("metric_candidate_node", spec.task, int(node)))
        for pat in eccentricity_patterns:
            for m in re.finditer(pat, text, flags=re.I | re.S):
                node = int(m.group(1))
                if node not in spec.nodes:
                    continue
                observed = float(m.group(2))
                target = _eccentricity(spec, node)
                if _close_number(observed, target):
                    units.add(("eccentricity", int(node), round(float(target), 6)))
                elif any(cue in low for cue in ("eccentricity", "farthest", "maximum distance", "max distance")):
                    units.add(("eccentricity_node_considered", int(node)))
        if any(cue in low for cue in ("eccentricity", "maximum", "farthest", "max distance")):
            for node in sorted(mentioned_nodes):
                target = _eccentricity(spec, node)
                if any(_close_number(value, target) for value in values):
                    units.add(("eccentricity", int(node), round(float(target), 6)))
                else:
                    units.add(("eccentricity_node_considered", int(node)))
        for node, observed in _eccentricity_observations(text, spec):
            target = _eccentricity(spec, node)
            if _close_number(observed, target):
                units.add(("eccentricity", int(node), round(float(target), 6)))
            else:
                units.add(("eccentricity_node_considered", int(node)))

        if target_nodes and target_nodes <= mentioned_nodes and any(
            cue in low
            for cue in (
                "all centers",
                "all center",
                "all peripheral",
                "all periphery",
                "all barycenter",
                "all barycenters",
                "ties",
                "tie",
                "therefore",
                "answer",
            )
        ):
            units.add(("metric_target_set_complete", spec.task, tuple(sorted(target_nodes))))

    sum_patterns = (
        rf"(?:node\s+)?(\d+).{{0,120}}?(?:sum|total)\s+(?:of\s+)?(?:distances?|shortest\s+paths?).{{0,45}}?{number}",
        rf"(?:sum|total)\s+(?:of\s+)?(?:distances?|shortest\s+paths?).{{0,90}}?(?:from|of|for)?\s*(?:node\s+)?(\d+).{{0,45}}?(?:is|=|:).{{0,20}}?{number}",
    )
    if spec.task in {"barycenter", "wiener_index", "global_efficiency", "closeness_centrality"}:
        for node, observed in _node_distance_sum_observations(text, spec):
            target = _distance_sum(spec, node)
            if _close_number(observed, target):
                units.add(("distance_sum", int(node), round(float(target), 6)))
                if spec.task == "closeness_centrality" and spec.source is not None and int(node) == int(spec.source):
                    units.add(("closeness_distance_sum", int(spec.source), round(float(target), 6)))
            elif any(cue in low for cue in ("sum", "total", "distance")):
                units.add(("distance_sum_node_considered", int(node)))
        for pat in sum_patterns:
            for m in re.finditer(pat, text, flags=re.I | re.S):
                node = int(m.group(1))
                if node not in spec.nodes:
                    continue
                observed = float(m.group(2))
                target = _distance_sum(spec, node)
                if _close_number(observed, target):
                    units.add(("distance_sum", int(node), round(float(target), 6)))
                elif any(cue in low for cue in ("sum", "total", "distance")):
                    units.add(("distance_sum_node_considered", int(node)))
        if any(cue in low for cue in ("sum", "total")):
            for node in sorted(mentioned_nodes):
                target = _distance_sum(spec, node)
                if any(_close_number(value, target) for value in values):
                    units.add(("distance_sum", int(node), round(float(target), 6)))
                else:
                    units.add(("distance_sum_node_considered", int(node)))
    if spec.answer_numbers:
        metric_names = {
            "closeness_centrality": ("closeness centrality", "closeness"),
            "global_efficiency": ("global efficiency", "efficiency"),
            "wiener_index": ("wiener index", "wiener"),
            "radius": ("radius",),
            "diameter": ("diameter",),
        }.get(spec.task, ())
        if metric_names:
            target_value = float(spec.answer_numbers[0])
            for value in _named_metric_value_observations(text, metric_names):
                if _close_number(value, target_value):
                    unit_name = "global_metric_value" if spec.task in GLOBAL_WITNESS_TASKS else "metric_value"
                    units.add((unit_name, spec.task, round(float(target_value), 6)))
    return units


def _global_metric_formula_units(spec: GraphLogicSpec, text: str, answer_hit: bool) -> set[tuple]:
    if spec.task not in GLOBAL_WITNESS_TASKS:
        return set()
    low = text.lower()
    units: set[tuple] = set()
    n = len(spec.nodes)
    if spec.task == "global_efficiency":
        if any(cue in low for cue in ("efficiency", "reciprocal", "1/d", "1 / d", "inverse distance")):
            units.add(("global_efficiency_formula", int(n)))
        if any(cue in low for cue in ("ordered pairs", "n(n-1)", "divide", "average")) and (
            "distance" in low or "efficiency" in low
        ):
            units.add(("global_efficiency_normalizer", int(n)))
        if any(cue in low for cue in ("unreachable", "not reachable", "no path", "disconnected")):
            units.add(("global_unreachable_zero_contribution", spec.task))
        if any(cue in low for cue in ("denominator", "normaliz", "n(n-1)", "divide", "average", "ordered pairs")):
            values = _numeric_values(text)
            ordered_denominator = float(n * max(0, n - 1))
            if any(_close_number(value, ordered_denominator) for value in values):
                units.add(("global_efficiency_denominator_value", int(ordered_denominator)))
        units.update(_global_efficiency_contribution_units(spec, text))
        units.update(_global_pair_process_units(spec, text))
    elif spec.task == "wiener_index":
        if "wiener" in low or ("sum" in low and "distance" in low):
            units.add(("wiener_distance_sum_formula", int(n)))
        if any(cue in low for cue in ("unordered pairs", "all pairs", "pairwise")) and "distance" in low:
            units.add(("wiener_pair_scope", int(n)))
        if any(cue in low for cue in ("pair count", "number of pairs", "unordered pairs", "choose", "n(n-1)/2")):
            pair_count = float(n * max(0, n - 1) // 2)
            if any(_close_number(value, pair_count) for value in _numeric_values(text)):
                units.add(("wiener_pair_scope", int(n)))
        units.update(_global_pair_process_units(spec, text))
    if answer_hit:
        units.add(("global_metric_answer", spec.task))
    return units


def _global_pair_process_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task not in GLOBAL_WITNESS_TASKS:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "pair",
            "pairs",
            "distance",
            "shortest",
            "reciprocal",
            "1/d",
            "1 / d",
            "contribution",
            "wiener",
            "efficiency",
        )
    ):
        return set()
    units: set[tuple] = set()
    mentioned_pairs = set(_pair_mentions(text, spec.nodes))
    compact_distance_observations = _compact_pair_distance_observations(text, spec)
    mentioned_pairs.update((u, v) for u, v, _observed in compact_distance_observations)
    for u, v, observed in compact_distance_observations:
        distance = _shortest_distance(spec, u, v)
        if not math.isfinite(distance) or not _close_number(float(observed), float(distance)):
            continue
        a, b = (u, v) if spec.directed or u <= v else (v, u)
        if spec.task == "wiener_index":
            units.add(("wiener_pair_distance_value", int(a), int(b), round(float(distance), 6)))
        elif spec.task == "global_efficiency":
            units.add(("global_metric_pair_distance_value", int(a), int(b), round(float(distance), 6)))
    for u, v in mentioned_pairs:
        a, b = (u, v) if spec.directed or u <= v else (v, u)
        distance = _shortest_distance(spec, u, v)
        units.add(("all_pairs_pair", int(a), int(b)))
        if math.isfinite(distance):
            units.add(("global_pair_distance_scope", int(a), int(b)))
            if spec.task == "wiener_index":
                units.add(("wiener_pair_distance_scope", int(a), int(b)))
            elif any(cue in low for cue in ("reciprocal", "1/d", "1 / d", "efficiency", "contribution")):
                units.add(("global_efficiency_pair_scope", int(a), int(b)))
        elif any(cue in low for cue in ("unreachable", "not reachable", "no path", "zero")):
            units.add(("global_unreachable_pair", int(a), int(b)))
    return units


def _global_efficiency_contribution_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "global_efficiency":
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("efficiency", "contribution", "reciprocal", "1/d", "1 / d")):
        return set()
    number = r"([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)"
    patterns = (
        rf"(?:efficiency|contribution|reciprocal|1/d|1\s*/\s*d)\D{{0,50}}"
        rf"(?:from|between)?\s*(?:node\s+)?(\d+)\D{{0,30}}(?:to|and)\s*(?:node\s+)?(\d+)"
        rf"\D{{0,45}}(?:is|=|:)\D{{0,12}}{number}",
        rf"\((\d+)\s*,\s*(\d+)\)\D{{0,70}}(?:efficiency|contribution|reciprocal|1/d|1\s*/\s*d)"
        rf"\D{{0,30}}(?:is|=|:)\D{{0,12}}{number}",
    )
    units: set[tuple] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            u, v = int(m.group(1)), int(m.group(2))
            if u not in spec.nodes or v not in spec.nodes or u == v:
                continue
            observed = _parse_float_token(m.group(3))
            if observed is None:
                continue
            distance = _shortest_distance(spec, u, v)
            target = 0.0 if not math.isfinite(distance) or distance <= 0.0 else 1.0 / float(distance)
            if _close_number(observed, target):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("global_efficiency_pair_contribution", int(a), int(b), round(float(target), 6)))
    value_first_patterns = (
        rf"{number}\s*(?:for|as)?\s*(?:the\s*)?(?:efficiency|contribution|reciprocal|1\s*/\s*d)?\s*(?:for\s*)?(?:pair\s*)?\((\d+)\s*,\s*(\d+)\)",
        rf"(?:pair\s*)?\((\d+)\s*,\s*(\d+)\).{{0,45}}?{number}.{{0,35}}?(?:efficiency|contribution|reciprocal|1\s*/\s*d)",
    )
    for pat_index, pat in enumerate(value_first_patterns):
        for m in re.finditer(pat, text, flags=re.I | re.S):
            if len(m.groups()) != 3:
                continue
            if pat_index == 0:
                observed = _parse_float_token(m.group(1))
                u, v = int(m.group(2)), int(m.group(3))
            else:
                u, v = int(m.group(1)), int(m.group(2))
                observed = _parse_float_token(m.group(3))
            if observed is None or u not in spec.nodes or v not in spec.nodes or u == v:
                continue
            distance = _shortest_distance(spec, u, v)
            target = 0.0 if not math.isfinite(distance) or distance <= 0.0 else 1.0 / float(distance)
            if _close_number(observed, target):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("global_efficiency_pair_contribution", int(a), int(b), round(float(target), 6)))
    return units


def _distance_formula_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task not in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "unreachable",
            "not reachable",
            "no path",
            "infinite",
            "inf",
            "distance",
            "dist",
            "reachable",
            "normaliz",
            "reciprocal",
            "1/d",
            "1 / d",
            "all pairs",
            "ordered pairs",
            "unordered pairs",
            "exclude",
            "excluding",
            "sigma",
            "shortest paths",
            "directed",
            "undirected",
            "outgoing",
            "incoming",
            "edge direction",
            "direction",
        )
    ):
        return set()
    units: set[tuple] = set()

    if spec.task in STRICT_TEMPLATE_TASKS - {"maximum_flow", "weighted_shortest_path"}:
        if spec.directed and any(
            cue in low for cue in ("directed", "outgoing", "incoming", "edge direction", "respect direction")
        ):
            units.add(("directed_scope", spec.task))
        elif not spec.directed and any(cue in low for cue in ("undirected", "both directions", "unordered pairs")):
            units.add(("undirected_scope", spec.task))

    unreachable_cue = any(cue in low for cue in ("unreachable", "not reachable", "no path", "infinite", "inf"))
    if unreachable_cue:
        for u, v in _pair_mentions(text, spec.nodes):
            if not math.isfinite(_shortest_distance(spec, u, v)):
                a, b = (u, v) if spec.directed or u <= v else (v, u)
                units.add(("distance_unreachable_pair", int(a), int(b)))
        if spec.source is not None:
            for node in _numbers_in_text(text, spec.nodes) - {spec.source}:
                if not math.isfinite(_shortest_distance(spec, spec.source, node)):
                    units.add(("source_unreachable_node", int(spec.source), int(node)))

    if spec.source is not None and spec.task in {"closeness_centrality", "harmonic_centrality"}:
        finite = _finite_distances_from(spec, spec.source)
        reachable_count = len(finite)
        distance_sum = sum(finite)
        values = _numeric_values(text)
        if "reachable" in low and any(_close_number(value, reachable_count) for value in values):
            units.add(("source_reachable_count", int(spec.source), int(reachable_count)))
        elif "reachable" in low:
            units.add(("source_reachability_scope", int(spec.source)))
        if spec.task == "closeness_centrality":
            if any(cue in low for cue in ("closeness", "normaliz", "n-1", "reachable", "sum distance")):
                units.add(("closeness_formula", int(spec.source), int(reachable_count)))
            if any(cue in low for cue in ("wf", "wasserman", "faust", "reachable/(n-1)", "reachable / (n-1)")):
                units.add(("closeness_disconnected_scaling", int(spec.source)))
            if any(cue in low for cue in ("sum", "total", "denominator", "distance")) and any(
                _close_number(value, distance_sum) for value in values
            ):
                units.add(("closeness_distance_sum", int(spec.source), round(float(distance_sum), 6)))
            if any(cue in low for cue in ("normaliz", "denominator", "n-1", "reachable", "divide")) and (
                any(_close_number(value, reachable_count) for value in values)
                or any(_close_number(value, max(0, len(spec.nodes) - 1)) for value in values)
            ):
                units.add(("closeness_normalizer_value", int(spec.source), int(reachable_count)))
        else:
            if any(cue in low for cue in ("harmonic", "reciprocal", "1/d", "1 / d", "inverse distance")):
                units.add(("harmonic_formula", int(spec.source), int(reachable_count)))
                target = sum(1.0 / d for d in finite if d > 0.0)
                if any(_close_number(value, target) for value in values):
                    units.add(("harmonic_sum", int(spec.source), round(float(target), 6)))

    if spec.task == "betweenness_centrality" and spec.source is not None:
        if any(cue in low for cue in ("betweenness", "sigma", "shortest paths", "pass through", "through node")):
            units.add(("betweenness_formula", int(spec.source)))
        if any(cue in low for cue in ("sigma_st", "sigma st", "σ", "fraction", "divide by number of shortest paths")):
            units.add(("betweenness_sigma_fraction", int(spec.source)))
        if any(cue in low for cue in ("exclude", "excluding", "not include", "other nodes")):
            units.add(("betweenness_excludes_endpoints", int(spec.source)))
        if any(cue in low for cue in ("normaliz", "denominator", "divide", "(n-1)(n-2)", "n-1", "n-2")):
            values = _numeric_values(text)
            denominator = float(max(1, (len(spec.nodes) - 1) * max(1, len(spec.nodes) - 2)))
            factor = 1.0 / denominator if spec.directed else 2.0 / denominator
            if any(_close_number(value, denominator) or _close_number(value, factor) for value in values):
                units.add(("betweenness_normalizer_value", int(spec.source), round(float(factor), 6)))
        values = _numeric_values(text)
        raw_score = _betweenness_raw_score(spec, int(spec.source))
        if any(cue in low for cue in ("raw", "unnormalized", "un-normalized", "total contribution", "sum contribution")):
            if any(_close_number(value, raw_score) for value in values):
                units.add(("betweenness_raw_sum", int(spec.source), round(float(raw_score), 6)))
        if spec.answer_numbers and any(cue in low for cue in ("normaliz", "centrality", "betweenness value", "final value")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("betweenness_normalized_value", int(spec.source), round(float(target), 6)))

    if spec.task in {"diameter", "radius", "center", "periphery", "barycenter"}:
        if any(cue in low for cue in ("eccentricity", "farthest", "maximum distance", "max distance")):
            units.add(("eccentricity_formula", spec.task))
        if spec.task == "barycenter" and any(cue in low for cue in ("sum", "total", "distance sum")):
            units.add(("barycenter_formula",))

    if spec.task in GLOBAL_WITNESS_TASKS:
        if any(cue in low for cue in ("all pairs", "pairwise", "ordered pairs", "unordered pairs")):
            units.add(("all_pairs_scope", spec.task, int(len(spec.nodes))))
        if unreachable_cue:
            units.add(("global_unreachable_zero_contribution", spec.task))
    return units


def _distance_definition_invalid(spec: GraphLogicSpec, text: str) -> bool:
    low = text.lower()
    if not _has_answer_assertion_cue(spec, text):
        return False
    if spec.task == "closeness_centrality" and spec.source is not None:
        reachable_count = len(_finite_distances_from(spec, spec.source))
        disconnected = reachable_count < max(0, len(spec.nodes) - 1)
        mentions_average = any(cue in low for cue in ("average distance", "avg distance", "1/average", "1 / average"))
        handles_reachability = any(
            cue in low for cue in ("reachable", "unreachable", "not reachable", "normaliz", "n-1", "wf")
        )
        if disconnected and mentions_average and not handles_reachability:
            return True
    if spec.task == "global_efficiency":
        if any(cue in low for cue in ("denominator", "divide", "normaliz", "ordered pairs", "unordered pairs")):
            values = _numeric_values(text)
            n = len(spec.nodes)
            ordered_denominator = float(n * max(0, n - 1))
            unordered_denominator = ordered_denominator / 2.0
            if values and not any(
                _close_number(value, ordered_denominator) or _close_number(value, unordered_denominator)
                for value in values
            ):
                return True
    if spec.task == "betweenness_centrality":
        values = _final_answer_numeric_values(text)
        if values and spec.answer_numbers:
            target = float(spec.answer_numbers[0])
            # Common failure mode: report the raw number of paths through the
            # pivot (large integer) as the final normalized centrality.
            if abs(target) <= 1.0 and any(abs(value) > 1.0 and abs(value - round(value)) <= 1e-9 for value in values):
                return True
    return False


def _distance_process_invalid_count(spec: GraphLogicSpec, text: str) -> int:
    """Count local numeric/process contradictions for distance-style tasks.

    These checks intentionally avoid final-answer parsing. They only fire when a
    segment asserts an intermediate pair distance, pair contribution, node
    distance sum, or centrality value that can be checked against the graph.
    """
    if spec.task not in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return 0
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "distance",
            "dist",
            "shortest",
            "sum",
            "total",
            "efficiency",
            "wiener",
            "centrality",
            "betweenness",
            "center",
            "centre",
            "periphery",
            "peripheral",
            "barycenter",
            "barycentre",
            "eccentricity",
        )
    ):
        return 0
    invalid = 0
    seen_distance_obs: set[tuple[int, int, float]] = set()
    for u, v, observed in _pair_distance_observations(text, spec):
        key = (u, v, round(float(observed), 6))
        if key in seen_distance_obs:
            continue
        seen_distance_obs.add(key)
        target = _shortest_distance(spec, u, v)
        if math.isfinite(target) and not _close_number(observed, target):
            invalid += 1
    for u, v, observed in _compact_pair_distance_observations(text, spec):
        key = (u, v, round(float(observed), 6))
        if key in seen_distance_obs:
            continue
        seen_distance_obs.add(key)
        target = _shortest_distance(spec, u, v)
        if math.isfinite(target) and not _close_number(observed, target):
            invalid += 1
    seen_eff_obs: set[tuple[int, int, float]] = set()
    for u, v, observed in _global_efficiency_observations(text, spec):
        key = (u, v, round(float(observed), 6))
        if key in seen_eff_obs:
            continue
        seen_eff_obs.add(key)
        distance = _shortest_distance(spec, u, v)
        target = 0.0 if not math.isfinite(distance) or distance <= 0.0 else 1.0 / float(distance)
        if not _close_number(observed, target):
            invalid += 1
    if spec.task in {"barycenter", "wiener_index", "global_efficiency", "closeness_centrality"}:
        seen_sum_obs: set[tuple[int, float]] = set()
        for node, observed in _node_distance_sum_observations(text, spec):
            key = (node, round(float(observed), 6))
            if key in seen_sum_obs:
                continue
            seen_sum_obs.add(key)
            target = _distance_sum(spec, node)
            if not _close_number(observed, target):
                invalid += 1
    if spec.task in {"center", "periphery", "radius", "diameter", "barycenter"}:
        seen_ecc_obs: set[tuple[int, float]] = set()
        for node, observed in _eccentricity_observations(text, spec):
            key = (node, round(float(observed), 6))
            if key in seen_ecc_obs:
                continue
            seen_ecc_obs.add(key)
            target = _eccentricity(spec, node)
            if not _close_number(observed, target):
                invalid += 1
    if spec.task in {"center", "periphery", "barycenter"}:
        target_nodes = _target_answer_nodes(spec)
        if len(target_nodes) > 1:
            mentioned = _numbers_in_text(text, spec.nodes)
            if _has_strong_final_answer_cue(text) and mentioned and not target_nodes <= mentioned:
                invalid += 1
            only_one_claim = any(
                cue in low
                for cue in (
                    "only center",
                    "only one center",
                    "unique center",
                    "single center",
                    "only periphery",
                    "only one periphery",
                    "unique periphery",
                    "single periphery",
                    "only barycenter",
                    "only one barycenter",
                    "unique barycenter",
                    "single barycenter",
                )
            )
            if only_one_claim and not target_nodes <= mentioned:
                invalid += 1
    if spec.task == "closeness_centrality" and spec.source is not None:
        reachable_count = len(_finite_distances_from(spec, spec.source))
        disconnected = reachable_count < max(0, len(spec.nodes) - 1)
        mentions_average = any(cue in low for cue in ("average distance", "avg distance", "1/average", "1 / average"))
        handles_reachability = any(
            cue in low for cue in ("reachable", "unreachable", "not reachable", "normaliz", "n-1", "wf")
        )
        if disconnected and mentions_average and not handles_reachability:
            invalid += 1
        if spec.answer_numbers and not any(cue in low for cue in ("raw", "unnormalized", "un-normalized", "sum distance")):
            values = _named_metric_value_observations(text, ("closeness centrality", "closeness"))
            target = float(spec.answer_numbers[0])
            if values and not any(_close_number(value, target) for value in values):
                invalid += 1
    if spec.task in {"global_efficiency", "wiener_index"} and spec.answer_numbers:
        names = ("global efficiency", "efficiency") if spec.task == "global_efficiency" else ("wiener index", "wiener")
        if not any(cue in low for cue in ("raw", "unnormalized", "un-normalized", "total reciprocal", "sum of reciprocal")):
            values = _named_metric_value_observations(text, names)
            target = float(spec.answer_numbers[0])
            if values and not any(_close_number(value, target) for value in values):
                invalid += 1
    if spec.task == "betweenness_centrality" and spec.answer_numbers:
        target = float(spec.answer_numbers[0])
        if abs(target) <= 1.0:
            for value in _betweenness_value_observations(text, spec):
                if abs(value) > 1.0 and abs(value - round(value)) <= 1e-9:
                    invalid += 1
    return invalid


def _brace_substrings(text: str, max_len: int = 5000) -> list[str]:
    stack: list[int] = []
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if i + 1 - start <= max_len:
                out.append(text[start : i + 1])
    return out


def _int_mapping_dicts(text: str) -> list[dict[int, int]]:
    dicts: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for chunk in _brace_substrings(text):
        try:
            parsed = ast.literal_eval(chunk)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        out: dict[int, int] = {}
        ok = True
        for key, value in parsed.items():
            if type(key) is not int or type(value) is not int:
                ok = False
                break
            out[int(key)] = int(value)
        if not ok or not out:
            continue
        signature = tuple(sorted(out.items()))
        if signature not in seen:
            seen.add(signature)
            dicts.append(out)
    return dicts


def _isomorphism_edge_preservation_units(
    spec: GraphLogicSpec,
    mapping: dict[int, int],
) -> tuple[set[tuple], bool]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set(), False
    source_nodes = set(spec.answer_mapping.keys())
    target_nodes = set(spec.answer_mapping.values())
    keys = set(mapping.keys())
    values = list(mapping.values())
    if not keys or not keys <= source_nodes or any(value not in target_nodes for value in values):
        return set(), False
    if len(set(values)) != len(values):
        return set(), True

    units: set[tuple] = set()
    invalid = False
    source_edges = [edge for edge in spec.edges if edge[0] in source_nodes and edge[1] in source_nodes]
    checked = 0
    for u, v in source_edges:
        if u not in mapping or v not in mapping:
            continue
        checked += 1
        mapped_edge = _norm_edge(mapping[u], mapping[v], spec.directed)
        if mapped_edge in spec.edges:
            units.add(("mapping_preserves_edge", int(u), int(v), int(mapping[u]), int(mapping[v])))
        else:
            invalid = True
    complete = keys == source_nodes and set(values) == target_nodes
    nonedge_checked = 0
    if complete and not invalid:
        source_list = sorted(source_nodes)
        if spec.directed:
            candidate_pairs = ((u, v) for u in source_list for v in source_list if u != v)
        else:
            candidate_pairs = (
                (u, v)
                for idx, u in enumerate(source_list)
                for v in source_list[idx + 1 :]
                if u != v
            )
        for u, v in candidate_pairs:
            if _norm_edge(u, v, spec.directed) in spec.edges:
                continue
            mapped_edge = _norm_edge(mapping[u], mapping[v], spec.directed)
            if mapped_edge in spec.edges:
                invalid = True
                break
            nonedge_checked += 1
            if nonedge_checked <= 12:
                units.add(("mapping_preserves_nonedge", int(u), int(v), int(mapping[u]), int(mapping[v])))
    if complete and checked == len(source_edges) and not invalid:
        units.add(("mapping_valid_dict",))
    return units, invalid


def _mapping_induced_degree(spec: GraphLogicSpec, node: int, domain: set[int]) -> int:
    degree = 0
    for u, v in spec.edges:
        if u not in domain or v not in domain:
            continue
        if u == node:
            degree += 1
        if v == node and (spec.directed or u != v):
            degree += 1
    return degree


def _extract_mapping_candidate_pairs(spec: GraphLogicSpec, text: str) -> set[tuple[int, int]]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set()
    source_nodes = set(spec.answer_mapping.keys())
    target_nodes = set(spec.answer_mapping.values())
    candidate_pairs: set[tuple[int, int]] = set()
    for mapping in _int_mapping_dicts(text):
        for a, b in mapping.items():
            if a in source_nodes and b in target_nodes:
                candidate_pairs.add((a, b))
            elif b in source_nodes and a in target_nodes:
                candidate_pairs.add((b, a))
    patterns = (
        r"(?:node\s+)?(\d+)\s*(?:->|=>|:|maps?\s+to|corresponds?\s+to|is\s+mapped\s+to)\s*(?:node\s+)?(\d+)",
        r"(?:node\s+)?(\d+)\s+(?:in\s+g1\s+)?(?:corresponds?\s+to|maps?\s+to)\s+(?:node\s+)?(\d+)",
        r"g1\s*:?\s*(\d+).{0,90}?(?:image\s+is|maps?\s+to|mapped\s+to|corresponds?\s+to|is)\s+g2\s*:?\s*(\d+)",
        r"(?:g1|graph\s*1|graph1)\s*#?\s*(\d+).{0,90}?(?:g2|graph\s*2|graph2)\s*#?\s*(\d+)",
        r"(?:graph1|g1)\s*#?\s*(\d+)\s*(?:->|=>|:|maps?\s+to|corresponds?\s+to)\s*(?:graph2|g2)\s*#?\s*(\d+)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            a, b = int(m.group(1)), int(m.group(2))
            if a in source_nodes and b in target_nodes:
                candidate_pairs.add((a, b))
            elif b in source_nodes and a in target_nodes:
                candidate_pairs.add((b, a))

    # Many model traces state mappings as refinement evidence rather than as a
    # literal dict/arrow, e.g. "node 5 has unique degree 4, so it corresponds to
    # node 108".  Treat those as candidate pairs only when one source-side and
    # one target-side id co-occur in a small refinement window; executor checks
    # degree/bijection/adjacency before any strong reward is possible.
    low = text.lower()
    if any(cue in low for cue in ("degree", "unique", "correspond", "maps", "mapping", "image", "adjacency", "neighbor", "neighbour")):
        refinement_cue = re.compile(
            r"degree|deg\b|unique|correspond|maps?|mapping|image|adjacency|neighbors?|neighbours?|signature",
            flags=re.I,
        )
        for cue_match in refinement_cue.finditer(text):
            window = text[max(0, cue_match.start() - 120) : min(len(text), cue_match.end() + 120)]
            nums = [int(raw) for raw in re.findall(r"\b\d+\b", window) if len(raw) <= 12]
            srcs = [num for num in nums if num in source_nodes]
            dsts = [num for num in nums if num in target_nodes]
            if len(set(srcs)) == 1 and len(set(dsts)) == 1:
                candidate_pairs.add((int(srcs[0]), int(dsts[0])))
    return candidate_pairs


def _mapping_refinement_units(spec: GraphLogicSpec, text: str) -> tuple[set[tuple], int]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set(), 0
    low = text.lower()
    if not any(cue in low for cue in ("degree", "neighbor", "adjacency", "edge", "preserve", "bijection", "unique", "one-to-one", "refine", "partition", "map", "correspond")):
        return set(), 0
    source_nodes = set(spec.answer_mapping.keys())
    target_nodes = set(spec.answer_mapping.values())
    candidate_pairs = _extract_mapping_candidate_pairs(spec, text)
    units: set[tuple] = set()
    invalid = 0
    if any(cue in low for cue in ("same number of nodes", "same number of vertices", "node count", "vertex count")):
        if len(source_nodes) == len(target_nodes):
            units.add(("mapping_node_count_match", len(source_nodes)))
    if any(cue in low for cue in ("same number of edges", "edge count", "number of edges")):
        source_edge_count = sum(1 for u, v in spec.edges if u in source_nodes and v in source_nodes)
        target_edge_count = sum(1 for u, v in spec.edges if u in target_nodes and v in target_nodes)
        if source_edge_count == target_edge_count:
            units.add(("mapping_edge_count_match", int(source_edge_count)))
    if any(cue in low for cue in ("degree sequence", "degree multiset", "degree class", "partition by degree")):
        source_degrees = sorted(_mapping_induced_degree(spec, node, source_nodes) for node in source_nodes)
        target_degrees = sorted(_mapping_induced_degree(spec, node, target_nodes) for node in target_nodes)
        if source_degrees == target_degrees:
            units.add(("mapping_degree_partition", tuple(source_degrees[:16])))
    if any(cue in low for cue in ("degree", "same degree", "degree class", "partition", "refine")):
        for a, b in candidate_pairs:
            if _mapping_induced_degree(spec, a, source_nodes) == _mapping_induced_degree(spec, b, target_nodes):
                units.add(("mapping_degree_match", int(a), int(b)))
    if any(cue in low for cue in ("bijection", "one-to-one", "unique", "distinct", "no duplicate")):
        values = [b for _, b in candidate_pairs]
        if values and len(values) == len(set(values)):
            units.add(("mapping_partial_bijection", len(values)))
        elif values:
            invalid += 1
    if any(cue in low for cue in ("adjacency", "edge", "preserve", "neighbor")) and len(candidate_pairs) >= 2:
        pairs = sorted(candidate_pairs)
        for i, (a1, b1) in enumerate(pairs):
            for a2, b2 in pairs[i + 1 :]:
                src_edge = _norm_edge(a1, a2, spec.directed) in spec.edges
                dst_edge = _norm_edge(b1, b2, spec.directed) in spec.edges
                if src_edge == dst_edge:
                    units.add(("mapping_adjacency_consistent", int(a1), int(a2), int(b1), int(b2)))
                elif any(cue in low for cue in ("preserve", "adjacency", "edge")):
                    invalid += 1
    return units, invalid


def _mapping_format_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set()
    units: set[tuple] = set()
    source_nodes = set(spec.answer_mapping.keys())
    target_nodes = set(spec.answer_mapping.values())
    boxed = "\\boxed" in text
    for mapping in _int_mapping_dicts(text):
        keys = set(mapping.keys())
        values = list(mapping.values())
        in_domain = bool(keys) and keys <= source_nodes and all(value in target_nodes for value in values)
        if in_domain:
            units.add(("mapping_parseable_dict",))
            if boxed:
                units.add(("mapping_boxed_dict",))
            units.update(("mapping_key", int(key)) for key in keys)
        if in_domain and len(set(values)) == len(values):
            units.add(("mapping_unique_values", len(values)))
        for key, value in mapping.items():
            if spec.answer_mapping.get(key) == value:
                units.add(("map_pair", int(key), int(value)))
        if in_domain and keys == source_nodes and len(set(values)) == len(values):
            units.add(("mapping_complete_dict",))
        preserved_units, _ = _isomorphism_edge_preservation_units(spec, mapping)
        units.update(preserved_units)
    return units


def _mapping_format_invalid(spec: GraphLogicSpec, text: str) -> bool:
    if spec.task != "isomophic_mapping":
        return False
    low = text.lower()
    boxed = "\\boxed" in text
    final_cue = "final answer" in low or "the answer is" in low or "answer:" in low
    mapping_cue = "mapping:" in low
    if not (boxed or final_cue or mapping_cue):
        return False
    dicts = _int_mapping_dicts(text)
    if dicts:
        source_nodes = set((spec.answer_mapping or {}).keys())
        target_nodes = set((spec.answer_mapping or {}).values())
        for mapping in dicts:
            units, invalid = _isomorphism_edge_preservation_units(spec, mapping)
            complete = set(mapping.keys()) == source_nodes and set(mapping.values()) == target_nodes
            if invalid or ((boxed or final_cue or mapping_cue) and not complete):
                return True
            if any(unit and unit[0] == "mapping_valid_dict" for unit in units):
                return False
        return False
    if final_cue and not boxed:
        return True
    if any(cue in low for cue in ("graph1#", "graph2#", "g1#", "g2#", "maps to", "corresponds to")):
        return True
    return bool(re.search(r"\b\d+\s*:\s*\d+\s*(?:,|\n|$)", text)) or "{" in text or "}" in text


def _mapping_pair_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "map",
            "mapping",
            "mapped",
            "correspond",
            "isomorphic",
            "isomorphism",
            "image",
            "g1",
            "g2",
            "->",
            "=>",
            ":",
        )
    ):
        return set()
    pairs: set[tuple[int, int]] = set()
    candidate_pairs = _extract_mapping_candidate_pairs(spec, text)
    for a, b in candidate_pairs:
        if spec.answer_mapping.get(a) == b:
            pairs.add((a, b))
    units = {("mapping_candidate_pair", a, b) for a, b in candidate_pairs}
    units.update(("map_pair", a, b) for a, b in pairs)
    refinement_units, _ = _mapping_refinement_units(spec, text)
    units.update(refinement_units)
    return units


def _set_feasibility_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicPotentialState,
) -> set[tuple]:
    if spec.task not in SET_FEASIBILITY_TASKS or not _has_selection_cue(text):
        return set()
    adj = build_adj(spec)
    units: set[tuple] = set()
    selected_nodes = _selected_node_mentions(text, spec.nodes) or nodes
    selected_nodes &= spec.nodes
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching"}:
        explicit_edges = set(_edge_order_from_text(text, spec))
        if not explicit_edges:
            return set()
        used = set()
        accepted_edges: list[tuple[int, int]] = []
        for unit in state.seen_evidence:
            if len(unit) == 3 and unit[0] == "matching_edge":
                used.update(unit[1])
        for edge in sorted(explicit_edges):
            if edge[0] not in used and edge[1] not in used:
                units.add(("matching_edge", edge))
                accepted_edges.append(edge)
                used.update(edge)
        target_edge_count = len(spec.target_edges or set())
        if spec.task == "bipartite_maximum_matching" and target_edge_count and len(accepted_edges) >= target_edge_count and any(
            cue in text.lower() for cue in ("maximum", "max matching", "optimal", "size", "cardinality")
        ):
            units.add(("matching_size", int(target_edge_count)))
            units.add(("edge_set_answer_optimal_size", spec.task, int(target_edge_count)))
        elif spec.task == "max_weight_matching" and accepted_edges:
            total_weight = _edge_set_weight(spec, accepted_edges)
            target_weight = _max_weight_matching_target_weight(spec)
            total_weight_observed = any(_close_number(value, total_weight) for value in _numeric_values(text))
            if total_weight_observed:
                units.add(("max_weight_matching_weight", round(float(total_weight), 6)))
            if target_weight is not None and total_weight_observed and _close_number(total_weight, target_weight) and any(
                cue in text.lower() for cue in ("maximum", "max weight", "optimal", "total weight")
            ):
                units.add(("max_weight_matching_optimal_weight", round(float(total_weight), 6)))
    elif spec.task == "min_edge_covering":
        explicit_edges = set(_edge_order_from_text(text, spec))
        if not explicit_edges:
            return set()
        for u, v in explicit_edges:
            units.add(("edge_cover_node", int(u)))
            units.add(("edge_cover_node", int(v)))
            if "matching" in text.lower():
                units.add(("edge_cover_matching_edge", _norm_edge(u, v, spec.directed)))
        if any(cue in text.lower() for cue in ("unmatched", "not covered", "uncovered")):
            for node in selected_nodes:
                units.add(("edge_cover_unmatched_node", int(node)))
        target_edge_count = len(spec.target_edges or set())
        covered = {node for edge in explicit_edges for node in edge}
        if explicit_edges and set(spec.nodes) <= covered:
            units.add(("edge_set_answer_feasible", spec.task, tuple(sorted(explicit_edges))[:12]))
            if target_edge_count and len(explicit_edges) == target_edge_count:
                units.add(("edge_set_answer_optimal_size", spec.task, len(explicit_edges)))
    elif spec.task == "min_vertex_cover":
        if "matching" in text.lower() and any(cue in text.lower() for cue in ("lower bound", "bound", "at least")):
            units.add(("vertex_cover_matching_lower_bound",))
        for node in selected_nodes:
            for edge in incident_edges(spec, node):
                units.add(("vertex_cover_edge", edge))
        for edge in valid_edges:
            if edge[0] in selected_nodes or edge[1] in selected_nodes:
                units.add(("vertex_cover_edge", edge))
        if selected_nodes and any(cue in text.lower() for cue in ("cover all", "covered", "vertex cover", "covers every edge")):
            if all(u in selected_nodes or v in selected_nodes for u, v in spec.edges):
                units.add(("vertex_cover_all_edges_covered", len(selected_nodes)))
                target_node_count = len(spec.target_nodes or set())
                if target_node_count and len(selected_nodes) <= target_node_count:
                    units.add(("set_answer_optimal_size", spec.task, len(selected_nodes)))
    elif spec.task == "dominating_set":
        dominated = set(selected_nodes)
        for node in selected_nodes:
            units.add(("dominated_node", int(node)))
            for nbr in adj.get(node, []):
                dominated.add(nbr)
                units.add(("dominated_node", int(nbr)))
        if selected_nodes and set(spec.nodes) <= dominated:
            units.add(("dominating_all_nodes_covered", len(selected_nodes)))
            target_node_count = len(spec.target_nodes or set())
            if target_node_count and len(selected_nodes) <= target_node_count:
                units.add(("set_answer_optimal_size", spec.task, len(selected_nodes)))
    elif spec.task == "maximal_independent_set":
        chosen = {unit[1] for unit in state.seen_evidence if len(unit) == 2 and unit[0] == "independent_node"}
        for node in sorted(selected_nodes):
            if all(_norm_edge(node, other, spec.directed) not in spec.edges for other in chosen):
                units.add(("independent_node", int(node)))
                chosen.add(node)
        if any(cue in text.lower() for cue in ("maximal", "cannot add", "covered", "adjacent to selected", "neighbor of selected")):
            for u, v in valid_edges:
                if u in chosen and v not in chosen:
                    units.add(("maximal_independent_covered_node", int(v)))
                if v in chosen and u not in chosen:
                    units.add(("maximal_independent_covered_node", int(u)))
        if len(selected_nodes) >= 2 and any(cue in text.lower() for cue in ("not adjacent", "no edge", "independent")):
            for u in selected_nodes:
                for v in selected_nodes:
                    if u < v and _norm_edge(u, v, spec.directed) not in spec.edges:
                        units.add(("independent_nonedge_check", int(u), int(v)))
        if selected_nodes:
            independent = all(
                _norm_edge(u, v, spec.directed) not in spec.edges
                for u in selected_nodes
                for v in selected_nodes
                if u < v
            )
            maximal = independent and all(
                any(_norm_edge(node, chosen, spec.directed) in spec.edges for chosen in selected_nodes)
                for node in spec.nodes - selected_nodes
            )
            if maximal and any(cue in text.lower() for cue in ("maximal", "cannot add", "no more", "covered")):
                units.add(("maximal_independent_complete", len(selected_nodes)))
    return units


def _set_size_units(spec: GraphLogicSpec, text: str) -> set[tuple]:
    if spec.task not in SET_FEASIBILITY_TASKS:
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("matching", "cover", "dominating", "independent", "size", "cardinality")):
        return set()
    values = _numeric_values(text)
    if not values:
        return set()
    units: set[tuple] = set()
    target_edge_count = len(spec.target_edges or set())
    target_node_count = len(spec.target_nodes or set())
    if spec.task == "bipartite_maximum_matching" and target_edge_count:
        if "matching" in low and any(_close_number(v, target_edge_count) for v in values):
            units.add(("matching_size", int(target_edge_count)))
    elif spec.task == "max_weight_matching":
        # A bare "maximum weight is X" statement is not enough process
        # evidence. Max-weight matching reward is emitted only when the text
        # gives an explicit edge set whose weight can be checked.
        pass
    elif spec.task == "min_edge_covering" and target_edge_count:
        if "edge cover" in low and any(_close_number(v, target_edge_count) for v in values):
            units.add(("edge_cover_size", int(target_edge_count)))
        matching_size = max(0, len(spec.nodes) - target_edge_count)
        if "matching" in low and any(_close_number(v, matching_size) for v in values):
            units.add(("edge_cover_matching_size", int(matching_size)))
    elif spec.task == "min_vertex_cover" and target_node_count:
        if "vertex cover" in low and any(_close_number(v, target_node_count) for v in values):
            units.add(("vertex_cover_size", int(target_node_count)))
    elif spec.task == "dominating_set" and target_node_count:
        if "dominating" in low and any(_close_number(v, target_node_count) for v in values):
            units.add(("dominating_set_size", int(target_node_count)))
    elif spec.task == "maximal_independent_set" and target_node_count:
        if "independent" in low and any(_close_number(v, target_node_count) for v in values):
            units.add(("independent_set_size", int(target_node_count)))
    return units


def _bridge_units(spec: GraphLogicSpec, text: str, valid_edges: set[tuple[int, int]]) -> set[tuple]:
    if spec.task != "bridges" or not valid_edges:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "bridge",
            "cut edge",
            "critical edge",
            "remove",
            "removing",
            "removal",
            "disconnect",
            "split",
            "separate",
            "component",
            "alternate path",
            "alternative path",
            "cycle",
            "answer",
        )
    ):
        return set()
    target = set(spec.target_edges or bridge_edges(spec))
    units: set[tuple] = set()
    if any(cue in low for cue in ("lowlink", "low-link", "discovery time", "disc[", "tin[", "low[")):
        units.add(("bridge_lowlink_rule",))
    if any(cue in low for cue in ("remove edge", "removing edge", "delete edge", "disconnects", "components increase")):
        units.add(("bridge_removal_test",))
    positive_cues = (
        "bridge",
        "cut edge",
        "critical edge",
        "disconnect",
        "split",
        "separate",
        "component",
        "only connection",
        "no alternate path",
        "no alternative path",
    )
    negative_cues = (
        "not a bridge",
        "not bridge",
        "no bridge",
        "does not disconnect",
        "doesn't disconnect",
        "still connected",
        "remain connected",
        "alternate path",
        "alternative path",
        "in a cycle",
    )
    certificate_cues = (
        "remove",
        "removing",
        "removal",
        "delete",
        "disconnect",
        "components increase",
        "lowlink",
        "low-link",
        "disc[",
        "tin[",
        "low[",
        "no alternate path",
        "no alternative path",
        "only connection",
    )
    for edge in valid_edges:
        contexts = _edge_contexts(text, edge, spec.directed)
        ctxs = contexts or [low]
        has_positive = any(any(cue in ctx for cue in positive_cues) for ctx in ctxs)
        has_certificate = any(any(cue in ctx for cue in certificate_cues) for ctx in ctxs) or any(
            cue in low for cue in ("lowlink", "low-link", "disc[", "tin[", "low[")
        )
        if edge in target and has_positive and has_certificate:
            units.add(("bridge_edge", edge))
        elif edge not in target and any(any(cue in ctx for cue in negative_cues) for ctx in ctxs):
            units.add(("non_bridge_checked", edge))
    return units


def _component_units(spec: GraphLogicSpec, text: str, nodes: set[int], answer_hit: bool) -> set[tuple]:
    if spec.task not in {"connected_component_number", "strongly_connected_number"}:
        return set()
    low = text.lower()
    if "component" not in low and "scc" not in low and not answer_hit:
        return set()
    comps = _connected_components(spec)
    units: set[tuple] = set()
    if spec.task == "strongly_connected_number" and any(
        cue in low for cue in ("tarjan", "kosaraju", "lowlink", "low-link", "stack", "transpose", "finish order", "scc")
    ):
        units.add(("scc_algorithm_rule",))
    if spec.answer_numbers and any(_close_number(value, len(comps)) for value in _numeric_values(text)):
        units.add(("component_count", int(len(comps))))
    mentioned = sorted(nodes & spec.nodes)
    if len(mentioned) >= 2:
        mentioned_set = set(mentioned)
        for comp in comps:
            if mentioned_set <= comp:
                units.add(("component_subset", tuple(mentioned)))
                break
    elif len(mentioned) == 1 and any(cue in low for cue in ("isolated", "single node", "singleton")):
        node = mentioned[0]
        if any({node} == comp for comp in comps):
            units.add(("component_singleton", int(node)))
    return units


def _flow_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    check_final_answer: bool = False,
) -> set[tuple]:
    if spec.task != "maximum_flow":
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("flow", "capacity", "source", "sink", "augment", "path", "cut", "residual")):
        return set()
    units: set[tuple] = set()
    if spec.source is not None and spec.target is not None:
        reachable = _reachable_from(spec, spec.source)
        no_path_cue = any(
            cue in low
            for cue in (
                "no path",
                "not reachable",
                "cannot reach",
                "can't reach",
                "no connecting path",
                "isolated",
                "maximum flow would be zero",
                "maximum flow is zero",
            )
        )
        if spec.target not in reachable and no_path_cue:
            units.add(("flow_no_path", int(spec.source), int(spec.target)))
        for seq in _node_sequences(text, spec):
            if seq[0] != spec.source or seq[-1] != spec.target:
                continue
            path_edges = _valid_path_edges(spec, seq)
            if path_edges:
                units.add(("flow_path", tuple(path_edges)))
                for edge in path_edges:
                    units.add(("flow_path_edge", edge))
                bottleneck = min(float(spec.weights.get(edge, 1.0)) for edge in path_edges)
                if any(cue in low for cue in ("bottleneck", "augment", "send", "push", "capacity")) and any(
                    _close_number(value, bottleneck) for value in _numeric_values(text)
                ):
                    units.add(("flow_bottleneck", tuple(path_edges), round(float(bottleneck), 6)))
    if valid_edges and any(cue in low for cue in ("capacity", "flow", "residual", "bottleneck", "cut")):
        for edge in valid_edges:
            units.add(("flow_edge", edge))
            contexts = _edge_contexts(text, edge, spec.directed) or [text]
            weight = float(spec.weights.get(edge, 1.0))
            if any(any(_close_number(value, weight) for value in _numeric_values(ctx)) for ctx in contexts):
                units.add(("flow_capacity", edge, round(float(weight), 6)))
            if any("residual" in ctx for ctx in contexts):
                units.add(("flow_residual_edge", edge))
            if any(any(cue in ctx for cue in ("reverse edge", "back edge", "backward edge", "reverse capacity")) for ctx in contexts):
                units.add(("flow_reverse_edge", edge))
    if any(cue in low for cue in ("residual network", "residual graph", "reverse edge", "back edge", "backward edge")):
        units.add(("flow_residual_network",))
    if any(cue in low for cue in ("level graph", "bfs level", "blocking flow", "dinic")):
        units.add(("flow_level_graph",))
    if any(cue in low for cue in ("source side", "reachable from source", "s side", "cut capacity", "min cut")):
        units.add(("flow_cut_certificate_scope",))
    if check_final_answer and spec.answer_numbers and any(
        cue in low for cue in ("maximum flow", "max flow", "total flow", "flow value")
    ):
        values = _numeric_values(text)
        for target in spec.answer_numbers[:2]:
            if any(_close_number(value, target) for value in values):
                units.add(("flow_value", round(float(target), 6)))
    return units


def _edge_context_has_any(
    text: str,
    edge: tuple[int, int],
    spec: GraphLogicSpec,
    cues: Iterable[str],
) -> bool:
    contexts = _edge_contexts(text, edge, spec.directed)
    if not contexts:
        return False
    return any(any(cue in ctx for cue in cues) for ctx in contexts)


def _path_process_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
) -> set[tuple]:
    task = spec.task
    if task not in PATH_TASKS | TRAVERSAL_TASKS | {"topological_sort", "traveling_salesman_problem"}:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "path",
            "route",
            "tour",
            "order",
            "visit",
            "visited",
            "queue",
            "stack",
            "enqueue",
            "dequeue",
            "discover",
            "relax",
            "update",
            "distance",
            "dist",
            "cost",
            "next",
            "then",
        )
    ):
        return set()

    units: set[tuple] = set()
    if task in PATH_TASKS:
        if spec.directed and any(cue in low for cue in ("directed", "outgoing", "edge direction", "respect direction")):
            units.add(("path_directed_scope", task))
        elif not spec.directed and any(cue in low for cue in ("undirected", "both directions")):
            units.add(("path_undirected_scope", task))
        if task == "weighted_shortest_path" and any(
            cue in low for cue in ("weight", "weighted", "cost", "dijkstra", "priority queue", "heap")
        ):
            units.add(("weighted_path_uses_weights",))
    for seq in _node_sequences(text, spec):
        if task == "topological_sort":
            order = list(spec.target_node_order or [])
            if not order:
                continue
            rank = {node: i for i, node in enumerate(order)}
            prefix = []
            last_rank = -1
            for node in seq:
                if node not in rank or rank[node] <= last_rank:
                    break
                prefix.append(node)
                last_rank = rank[node]
                units.add(("topological_node", int(node)))
            if len(prefix) >= 2:
                units.add(("topological_prefix", tuple(prefix[: min(len(prefix), 8)])))
            continue

        path_edges = _valid_path_edges(spec, seq)
        if not path_edges:
            continue
        if task in PATH_TASKS and not _path_is_shortest_prefix(spec, seq, path_edges):
            continue
        if task == "weighted_shortest_path" and not _has_any(text, ACTION_CUES["weighted_path"]):
            continue
        for edge in path_edges:
            if task in TRAVERSAL_TASKS:
                units.add(("traversal_tree_edge", edge))
            elif task == "traveling_salesman_problem":
                units.add(("tour_edge", edge))
            else:
                units.add(("path_edge", edge))
        if spec.source is None or seq[0] == spec.source:
            units.add(("path_prefix", tuple(path_edges[: min(len(path_edges), 8)])))
        if spec.target is not None and seq[-1] == spec.target:
            units.add(("path_reaches_target", int(spec.target)))
        if task == "weighted_shortest_path":
            path_cost = _path_cost(spec, path_edges)
            if any(_close_number(value, path_cost) for value in _numeric_values(text)):
                units.add(
                    (
                        "dijkstra_candidate_path_cost",
                        tuple(path_edges[: min(len(path_edges), 8)]),
                        round(float(path_cost), 6),
                    )
                )

    if task in TRAVERSAL_TASKS:
        for node in nodes & set(spec.target_nodes or set()):
            if any(cue in low for cue in ("visit", "visited", "enqueue", "dequeue", "discover", "process", "mark")):
                units.add(("traversal_node", int(node)))
        for edge in valid_edges & set(spec.target_edges or set()):
            if any(cue in low for cue in ("visit", "discover", "tree", "parent", "neighbor", "explore")):
                units.add(("traversal_edge", edge))

    if task == "weighted_shortest_path" and spec.source is not None:
        relax_cues = ("relax", "update", "tentative", "shorter", "distance", "dist", "cost")
        for edge in valid_edges & set(spec.target_edges or set()):
            if _edge_context_has_any(text, edge, spec, relax_cues):
                units.add(("dijkstra_relax_edge", edge))

        number = r"([-+]?\d+(?:\.\d+)?)"
        dist_patterns = (
            rf"(?:dist(?:ance)?|cost)\s*(?:to|of|for)?\s*(?:node\s*)?(\d+)\D{{0,24}}"
            rf"(?:=|is|becomes|updated\s+to|set\s+to|to)\D{{0,12}}{number}",
            rf"d\s*\[\s*(\d+)\s*\]\s*(?:=|is|<-|becomes)\s*{number}",
        )
        for pat in dist_patterns:
            for m in re.finditer(pat, text, flags=re.I | re.S):
                node = int(m.group(1))
                if node not in spec.nodes:
                    continue
                observed = float(m.group(2))
                target = _shortest_distance(spec, spec.source, node)
                if math.isfinite(target) and _close_number(observed, target):
                    units.add(("source_distance", int(node), round(float(target), 6)))
        for child, parent in _predecessor_mentions(text, spec):
            target_parent = _shortest_path_parent(spec, child)
            if target_parent is not None and parent == target_parent:
                units.add(("dijkstra_predecessor", int(parent), int(child)))

    return units


def _mst_process_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
) -> set[tuple]:
    if spec.task not in MST_TASKS or not valid_edges:
        return set()
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "mst",
            "spanning",
            "kruskal",
            "prim",
            "edge",
            "weight",
            "lightest",
            "minimum",
            "select",
            "choose",
            "add",
            "include",
            "accept",
            "reject",
            "skip",
            "cycle",
            "component",
            "union",
            "safe",
        )
    ):
        return set()
    target = set(spec.target_edges or set())
    units: set[tuple] = set()
    accept_cues = ("add", "include", "select", "choose", "take", "pick", "accept", "safe edge", "union")
    cycle_reject_cues = (
        "cycle",
        "loop",
        "same component",
        "already connected",
        "forms a cycle",
        "form a cycle",
        "creates a cycle",
        "would create a cycle",
    )
    reject_cues = (
        "skip",
        "reject",
        "discard",
        *cycle_reject_cues,
    )
    consider_cues = ("consider", "check", "weight", "lightest", "minimum", "sorted", "candidate")
    if any(cue in low for cue in ("sort", "sorted", "ascending", "in increasing", "by weight", "lightest first")):
        units.add(("mst_weight_order",))
    for edge in sorted(valid_edges):
        contexts = _edge_contexts(text, edge, spec.directed) or [text]
        if any("weight" in ctx for ctx in contexts):
            units.add(("mst_edge_weight_checked", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
        if (
            spec.task != "weighted_minimum_spanning_tree"
            and edge in target
            and (_edge_context_has_any(text, edge, spec, accept_cues) or any(cue in low for cue in accept_cues))
        ):
            units.add(("mst_accept_edge", edge))
        elif (
            spec.task != "weighted_minimum_spanning_tree"
            and edge not in target
            and _edge_context_has_any(text, edge, spec, reject_cues)
        ):
            units.add(("mst_reject_edge", edge))
        elif _edge_context_has_any(text, edge, spec, consider_cues):
            units.add(("mst_consider_edge", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
    return units


ORDERING_STATE_TASKS = {"hamiltonian_path", "topological_sort", "traveling_salesman_problem"}
NEIGHBORHOOD_STATE_TASKS = {
    "neighbor",
    "common_neighbor",
    "jaccard_coefficient",
    "adamic_adar_index",
    "resource_allocation_index",
    "clustering_coefficient",
    "avg_neighbor_degree",
    "degree",
    "degree_centrality",
    "density",
    "edge_number",
    "node_number",
    "triangles",
}
BOOLEAN_STATE_TASKS = BOOL_WITNESS_TASKS | {
    "edge_existence",
    "connected_component_number",
    "strongly_connected_number",
    "bridges",
}
ISOMORPHISM_STATE_TASKS = {"isomophic_mapping"}
STATE_SHAPING_TASKS = (
    PATH_TASKS
    | TRAVERSAL_TASKS
    | MST_TASKS
    | {"maximum_flow"}
    | ORDERING_STATE_TASKS
    | DISTANCE_WITNESS_TASKS
    | GLOBAL_WITNESS_TASKS
    | NEIGHBORHOOD_STATE_TASKS
    | SET_FEASIBILITY_TASKS
    | BOOLEAN_STATE_TASKS
    | ISOMORPHISM_STATE_TASKS
)
COMPLETION_SENSITIVE_STATE_TASKS = (
    ANSWER_NODE_TASKS
    | ANSWER_EDGE_TASKS
    | SET_FEASIBILITY_TASKS
    | DISTANCE_WITNESS_TASKS
    | GLOBAL_WITNESS_TASKS
    | NEIGHBORHOOD_STATE_TASKS
    | BOOLEAN_STATE_TASKS
    | ISOMORPHISM_STATE_TASKS
    | ORDERING_STATE_TASKS
    | MST_TASKS
    | TRAVERSAL_TASKS
)
STRICT_COMPLETION_STATE_TASKS = (
    ANSWER_NODE_TASKS
    | ANSWER_EDGE_TASKS
    | SET_FEASIBILITY_TASKS
    | GLOBAL_WITNESS_TASKS
    | {"center", "periphery", "barycenter", "radius", "diameter", "betweenness_centrality"}
    | {"weighted_minimum_spanning_tree"}
)
ACCUMULATION_SENSITIVE_STATE_TASKS = {
    "weighted_shortest_path",
    "maximum_flow",
    "betweenness_centrality",
    "closeness_centrality",
    "global_efficiency",
    "wiener_index",
    "isomophic_mapping",
}

# These tasks showed the highest process-reward false positives offline: long
# explanations often mention the right vocabulary while failing to advance a
# verifiable algorithm state. Keep their weak units for diagnostics, but require
# concrete transitions/certificates before positive shaping is emitted.
FALSE_POSITIVE_SENSITIVE_STATE_TASKS = {
    "bfs",
    "dfs",
    "weighted_shortest_path",
    "weighted_minimum_spanning_tree",
    "maximum_flow",
    "betweenness_centrality",
    "closeness_centrality",
    "global_efficiency",
    "wiener_index",
    "isomophic_mapping",
    "min_vertex_cover",
    "min_edge_covering",
    "dominating_set",
    "maximal_independent_set",
    "bipartite_maximum_matching",
    "max_weight_matching",
    "hamiltonian_path",
    "traveling_salesman_problem",
}


STATE_ALGORITHM_OVERLAY_SCAFFOLD_UNIT_NAMES = {
    "state_dijkstra_init",
    "state_dijkstra_weighted_rule",
    "state_dijkstra_checked",
    "state_mst_consider",
    "mst_consider_edge",
    "state_mst_weight_order",
    "mst_weight_order",
    "state_mst_weighted_rule",
    "state_mst_edge_weight_checked",
    "mst_edge_weight_checked",
    "state_flow_capacity",
    "flow_capacity",
    "state_flow_candidate_path",
    "state_flow_residual_edge",
    "flow_residual_edge",
    "state_flow_reverse_edge",
    "flow_reverse_edge",
    "state_flow_residual_network",
    "flow_residual_network",
    "state_flow_level_graph",
    "flow_level_graph",
    "state_flow_cut_scope",
    "flow_cut_certificate_scope",
    "state_flow_residual_update",
    "state_flow_conservation_scope",
}


def _state_task_enabled(config: GraphLogicConfig, task: str) -> bool:
    if not bool(getattr(config, "state_shaping", False)):
        return False
    raw = str(getattr(config, "state_tasks", "") or "").strip().lower()
    if _legacy_core4_compat_enabled(config):
        if not raw or raw in {"all", "*"}:
            return task in STATE_SHAPING_TASKS
        requested = {item.strip() for item in re.split(r"[,;\s]+", raw) if item.strip()}
        return task in requested and task in STATE_SHAPING_TASKS
    if not raw or raw in {"all", "*"}:
        return task in STATE_SHAPING_TASKS
    requested = {item.strip() for item in re.split(r"[,;\s]+", raw) if item.strip()}
    return task in requested and task in STATE_SHAPING_TASKS


def _initial_residual(spec: GraphLogicSpec) -> dict[tuple[int, int], float]:
    residual: dict[tuple[int, int], float] = defaultdict(float)
    for edge in spec.edge_order:
        u, v = edge
        cap = max(0.0, float(spec.weights.get(edge, 1.0)))
        residual[(u, v)] += cap
        residual.setdefault((v, u), 0.0)
        if not spec.directed:
            residual[(v, u)] += cap
            residual.setdefault((u, v), residual[(u, v)])
    return dict(residual)


def _max_flow_value(spec: GraphLogicSpec) -> float:
    if spec.source is None or spec.target is None:
        return 0.0
    residual = _initial_residual(spec)
    source, sink = int(spec.source), int(spec.target)
    total = 0.0
    while True:
        parent: dict[int, int | None] = {source: None}
        queue = [source]
        head = 0
        while head < len(queue) and sink not in parent:
            u = queue[head]
            head += 1
            for (a, b), cap in list(residual.items()):
                if a != u or cap <= 1e-12 or b in parent:
                    continue
                parent[b] = u
                queue.append(b)
                if b == sink:
                    break
        if sink not in parent:
            break
        path: list[tuple[int, int]] = []
        cur = sink
        while parent[cur] is not None:
            prev = int(parent[cur])
            path.append((prev, cur))
            cur = prev
        path.reverse()
        bottleneck = min(residual.get(edge, 0.0) for edge in path)
        if bottleneck <= 1e-12:
            break
        for u, v in path:
            residual[(u, v)] = residual.get((u, v), 0.0) - bottleneck
            residual[(v, u)] = residual.get((v, u), 0.0) + bottleneck
        total += bottleneck
    return float(total)


def _init_algorithm_state(spec: GraphLogicSpec, config: GraphLogicConfig) -> GraphLogicAlgorithmState | None:
    if not _state_task_enabled(config, spec.task):
        return None
    state = GraphLogicAlgorithmState(task=spec.task)
    if spec.task in PATH_TASKS:
        if spec.source is not None:
            state.dist[int(spec.source)] = 0.0
            state.visited_nodes.add(int(spec.source))
    elif spec.task in TRAVERSAL_TASKS:
        if spec.source is not None:
            state.visited_nodes.add(int(spec.source))
    elif spec.task in MST_TASKS:
        state.dsu_parent = {int(node): int(node) for node in spec.nodes}
        state.dsu_rank = {int(node): 0 for node in spec.nodes}
    elif spec.task == "maximum_flow":
        state.residual = _initial_residual(spec)
        if bool(getattr(config, "final_answer_checks", False)) and spec.answer_numbers:
            state.max_flow_value = max(0.0, float(spec.answer_numbers[0]))
        else:
            state.max_flow_value = _max_flow_value(spec)
    return state


def _state_find(state: GraphLogicAlgorithmState, node: int) -> int:
    parent = state.dsu_parent
    parent.setdefault(node, node)
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _state_union(state: GraphLogicAlgorithmState, a: int, b: int) -> bool:
    ra, rb = _state_find(state, a), _state_find(state, b)
    if ra == rb:
        return False
    rank = state.dsu_rank
    rank.setdefault(ra, 0)
    rank.setdefault(rb, 0)
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    state.dsu_parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return True


def _context_values_for_edge(text: str, edge: tuple[int, int], spec: GraphLogicSpec) -> list[float]:
    values: list[float] = []
    for ctx in _edge_contexts(text, edge, spec.directed):
        values.extend(_numeric_values(ctx))
    return values


def _flow_amount_values(text: str) -> list[float]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:augment|send|push|increase|add\s+flow)\D{{0,35}}?(?:by|of|=|:|is)?\D{{0,12}}{number}",
        rf"(?:send|push|route|augment)\s+{number}\s+(?:units?\s+)?(?:of\s+)?flow",
        rf"{number}\s+(?:units?\s+)?(?:of\s+)?flow\s+(?:along|on|through|via)",
        rf"(?:by|with)\s+{number}\s+(?:units?\s+)?(?:along|on|through|via)",
        rf"(?:bottleneck|min(?:imum)?\s+capacity)\D{{0,35}}?(?:is|=|:)?\D{{0,12}}{number}",
        rf"(?:path\s+capacity|augmenting\s+amount|augment\s+amount)\D{{0,35}}?(?:is|=|:)?\D{{0,12}}{number}",
        rf"{number}\D{{0,24}}(?:units?\s+of\s+)?(?:flow|bottleneck)",
    )
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    return values


def _flow_value_update_values(text: str) -> list[float]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:total|current|new|overall)?\s*flow(?:\s+value)?\D{{0,40}}"
        rf"(?:becomes|is now|now|=|:|to)\D{{0,12}}{number}",
        rf"(?:flow\s+value|value\s+of\s+the\s+flow|max\s+flow\s+so\s+far)\D{{0,40}}"
        rf"(?:becomes|is|now|=|:|to)\D{{0,12}}{number}",
        rf"(?:flow|total\s+flow)\D{{0,40}}(?:increases?|rises?|goes)\D{{0,25}}(?:to|=|:)\D{{0,12}}{number}",
        rf"(?:after\s+(?:augmenting|sending|pushing)[^.,;]{{0,60}}flow[^0-9-+]*){number}",
    )
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    return values


def _distance_update_values_for_node(text: str, node: int) -> list[float]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:dist(?:ance)?|cost)\s*(?:to|of|for)?\s*(?:node\s*)?{node}\D{{0,24}}"
        rf"(?:=|is|becomes|updated\s+to|set\s+to|to)\D{{0,12}}{number}",
        rf"d\s*\[\s*{node}\s*\]\s*(?:=|is|<-|becomes)\s*{number}",
        rf"(?:update|set|improve|lower|decrease)\D{{0,35}}(?:node\s*)?{node}\D{{0,35}}"
        rf"(?:dist(?:ance)?|cost|value)?\D{{0,18}}(?:to|=|is|as|becomes)?\D{{0,12}}{number}",
        rf"(?:node\s*)?{node}\D{{0,35}}(?:gets|has|now has|becomes|is set to|updated to)\D{{0,24}}"
        rf"(?:dist(?:ance)?|cost)?\D{{0,12}}{number}",
    )
    values: list[float] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
    return values


def _dijkstra_relax_mentions(spec: GraphLogicSpec, text: str) -> list[tuple[int, int, float | None]]:
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        rf"(?:via|through|from)\s+(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:to|towards?|reach(?:ing)?|for)\s+(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:dist(?:ance)?|cost|total|value)\D{{0,18}}{number}",
        rf"(?:dist(?:ance)?|cost|total)\s*(?:to|of|for)?\s*(?:node\s*)?(\d+)\D{{0,24}}"
        rf"(?:=|is|becomes|updated\s+to|set\s+to|to)\D{{0,12}}{number}.{{0,70}}?"
        rf"(?:via|through|from)\s+(?:node\s+)?(\d+)",
        rf"(?:node\s+)?(\d+).{{0,50}}?(?:via|through)\s+(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:dist(?:ance)?|cost|total)\D{{0,18}}{number}",
        rf"(?:update|set|improve|lower|decrease).{{0,45}}?(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:via|through|from)\s+(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:dist(?:ance)?|cost|total|value)?\D{{0,18}}{number}",
        rf"(?:node\s+)?(\d+).{{0,55}}?(?:gets|has|now has|updated).{{0,55}}?"
        rf"(?:via|through|from)\s+(?:node\s+)?(\d+).{{0,70}}?"
        rf"(?:dist(?:ance)?|cost|total|value)?\D{{0,18}}{number}",
        rf"(?:node\s+)?(\d+)\s*(?:->|=>|{re.escape(chr(0x2192))}|to)\s*(?:node\s+)?(\d+)"
        rf".{{0,70}}?(?:dist(?:ance)?|cost|total|value)\D{{0,18}}{number}",
    )
    mentions: list[tuple[int, int, float | None]] = []
    for idx, pat in enumerate(patterns):
        for m in re.finditer(pat, text, flags=re.I | re.S):
            try:
                if idx == 0:
                    u, v, value = int(m.group(1)), int(m.group(2)), float(m.group(3))
                elif idx == 1:
                    v, value, u = int(m.group(1)), float(m.group(2)), int(m.group(3))
                elif idx == 2:
                    v, u, value = int(m.group(1)), int(m.group(2)), float(m.group(3))
                elif idx in {3, 4}:
                    v, u, value = int(m.group(1)), int(m.group(2)), float(m.group(3))
                else:
                    u, v, value = int(m.group(1)), int(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            if u in spec.nodes and v in spec.nodes and u != v:
                mentions.append((u, v, value))
    for m in re.finditer(
        r"(?:relax|update|improve|better|shorter|cheaper)\D{0,50}"
        rf"(?:node\s+)?(\d+)\s*(-|--|->|=>|{re.escape(chr(0x2192))}|to|and|with)\s*(?:node\s+)?(\d+)",
        text,
        flags=re.I,
    ):
        u, connector, v = int(m.group(1)), str(m.group(2)).lower(), int(m.group(3))
        match_text = m.group(0).lower()
        if connector == "to" and "edge" not in match_text and "relax" not in match_text:
            # Avoid reading "update distance to node 7 to 12" as an edge 7->12.
            continue
        if u in spec.nodes and v in spec.nodes and u != v:
            values = _distance_update_values_for_node(text, v)
            mentions.append((u, v, values[0] if values else None))
    return mentions


def _predecessor_mentions(text: str, spec: GraphLogicSpec) -> list[tuple[int, int]]:
    mentions: list[tuple[int, int]] = []
    patterns = (
        r"(?:predecessor|parent|prev(?:ious)?)\s*(?:of|for)?\s*(?:node\s*)?(\d+)\D{0,24}(?:is|=|<-|to)\D{0,12}(?:node\s*)?(\d+)",
        r"(?:node\s*)?(\d+)\D{0,24}(?:predecessor|parent|prev(?:ious)?)\D{0,18}(?:node\s*)?(\d+)",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            child, parent = int(m.group(1)), int(m.group(2))
            if child in spec.nodes and parent in spec.nodes and child != parent:
                mentions.append((child, parent))
    return mentions


def _shortest_path_parent(spec: GraphLogicSpec, child: int) -> int | None:
    if spec.source is None or child == spec.source or child not in spec.nodes:
        return None
    target = _shortest_distance(spec, spec.source, child)
    if not math.isfinite(target):
        return None
    best_parent: int | None = None
    best_dist = math.inf
    for edge in spec.edges:
        u, v = edge
        candidates = [(u, v)]
        if not spec.directed and u != v:
            candidates.append((v, u))
        for parent, node in candidates:
            if node != child:
                continue
            parent_dist = _shortest_distance(spec, spec.source, parent)
            cand = parent_dist + float(spec.weights.get(edge, 1.0))
            if math.isfinite(parent_dist) and _close_number(cand, target) and parent_dist < best_dist:
                best_parent = parent
                best_dist = parent_dist
    return best_parent


def _state_units_budget(spec: GraphLogicSpec, state: GraphLogicAlgorithmState) -> int:
    task = spec.task
    if task in TRAVERSAL_TASKS:
        return min(32, max(1, len(spec.target_edge_order or []) + len(spec.nodes)))
    if task == "shortest_path":
        return min(24, max(1, len(spec.target_edge_order or []) + 1))
    if task == "weighted_shortest_path":
        return min(32, max(1, len(spec.target_edges or set()) + len(spec.nodes)))
    if task in MST_TASKS:
        return min(32, max(1, len(spec.nodes) + len(spec.edges)))
    if task == "maximum_flow":
        return min(32, max(1, len(spec.edges) + 4))
    if task in ORDERING_STATE_TASKS:
        return min(32, max(1, len(spec.nodes) + len(spec.edges)))
    if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return min(32, _potential_evidence_budget(spec))
    if task in SET_FEASIBILITY_TASKS:
        return min(32, _potential_evidence_budget(spec))
    if task in NEIGHBORHOOD_STATE_TASKS | BOOLEAN_STATE_TASKS | ISOMORPHISM_STATE_TASKS:
        return min(32, max(1, _potential_evidence_budget(spec)))
    return min(32, max(1, _potential_evidence_budget(spec)))


def _state_unit_weight(spec: GraphLogicSpec, unit: tuple) -> float:
    if not unit:
        return 0.0
    name = str(unit[0])
    if spec.task in TRAVERSAL_TASKS and spec.target_edge_order:
        # In Erdos BFS/DFS are evaluated as traversal tree-edge lists. Node
        # sequences can be useful diagnostics, but rewarding them taught the
        # policy to answer with node orders instead of the required edge list.
        if name in {
            "parseable_node_list_answer",
            "schema_wrong_final_node_list",
            "state_traversal_order_node",
            "state_traversal_answer_prefix",
            "state_traversal_visit",
            "state_traversal_seen",
        }:
            return 0.0
    if spec.task == "min_edge_covering":
        # MEC had high correct recall but also high wrong/insensitive positives.
        # Keep complete edge-cover certificates meaningful, but make partial
        # node/size/matching hints weak so they cannot dominate the process
        # reward without a concrete covering edge set.
        if name == "edge_cover_node":
            return 0.10
        if name == "edge_cover_matching_edge":
            return 0.60
        if name in {"edge_cover_size", "edge_cover_matching_size"}:
            return 0.25
        if name == "edge_set_answer_feasible":
            return 1.25
        if name == "edge_set_answer_optimal_size":
            return 2.20
    if name in {"state_dijkstra_target_cert", "state_weighted_path_final_cert"}:
        return 3.00
    if name == "state_dijkstra_weighted_relax_cert":
        return 1.90
    if name in {"state_mst_spanning_nodes_covered", "state_mst_final_weight_checked"}:
        return 2.30
    if name == "state_flow_min_cut_cert":
        return 3.00
    if name == "state_flow_residual_bottleneck_cert":
        return 2.60
    if name == "state_flow_valid_augment_inferred":
        return 1.90
    if name == "state_flow_value_update_cert":
        return 0.80
    if name == "state_flow_candidate_bottleneck":
        return 0.10
    if name == "state_flow_candidate_path":
        return 0.10
    if name in STATE_ALGORITHM_OVERLAY_SCAFFOLD_UNIT_NAMES:
        return 0.10
    if name in {"parseable_node_list_answer", "parseable_edge_list_answer"}:
        if spec.task in MST_TASKS:
            return 0.60 if spec.task == "weighted_minimum_spanning_tree" else 0.45
        if spec.task in TRAVERSAL_TASKS | SET_FEASIBILITY_TASKS | {"topological_sort"}:
            return 0.65
        if spec.task in {"hamiltonian_path", "traveling_salesman_problem", "shortest_path", "weighted_shortest_path"}:
            return 0.55
        return 0.45
    if name == "parseable_scalar_answer":
        if spec.task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index", "maximum_flow"}:
            return 0.85
        return 0.35
    if name == "parseable_mapping_dict_answer":
        return 0.55
    if name == "complete_mapping_dict_answer":
        return 1.35
    if name == "schema_simple_node_sequence":
        return 0.35
    if name == "mst_answer_spanning_tree":
        if spec.task == "weighted_minimum_spanning_tree":
            return 1.50
        return 1.75
    if name in {"mst_answer_edge_count", "state_mst_kruskal_prefix_edge", "state_mst_n_minus_one_edges"}:
        return 2.00
    if name in {"state_mst_safe_edge", "state_mst_cycle_reject_cert"}:
        return 1.85
    if name in {"state_mst_connected_acyclic", "state_mst_optimal_weight_cert"}:
        return 3.20
    if name == "state_mst_target_edge":
        return 0.90
    if name in {"state_traversal_order_node", "state_traversal_answer_prefix", "state_traversal_answer_edge_prefix"}:
        return 1.25
    if name == "betweenness_raw_sum":
        return 1.25
    if name in {"distance_unreachable_pair", "global_unreachable_pair", "global_unreachable_zero_contribution"}:
        return 1.25
    if spec.task == "isomophic_mapping" and name == "mapping_partial_bijection":
        return 0.15
    if name in STATE_ISOMORPHISM_PROCESS_UNIT_NAMES:
        if name in {"mapping_preserves_edge", "mapping_preserves_nonedge", "mapping_adjacency_consistent", "mapping_complete_dict"}:
            return 1.50
        return 0.80
    if name in STATE_VERIFIED_PROCESS_UNIT_NAMES:
        if name in {
            "triangle",
            "regular_all_degrees",
            "regular_counterexample",
            "euler_all_even",
            "euler_odd_node",
            "tournament_pair",
            "tournament_violation",
            "cycle_edge",
            "cycle_witness",
            "bipartite_violation",
            "metric_candidate_node",
            "metric_target_set_complete",
            "betweenness_raw_sum",
            "betweenness_normalized_value",
            "metric_value",
            "global_metric_value",
            "state_mst_edge_count_target",
        }:
            return 1.50
        if name in {
            "state_dijkstra_init",
            "state_dijkstra_weighted_rule",
            "state_mst_weighted_rule",
            "state_mst_edge_weight_checked",
            "mst_edge_weight_checked",
            "state_flow_residual_update",
            "state_flow_conservation_scope",
            "metric_tie_scope",
            "vertex_cover_matching_lower_bound",
            "vertex_cover_all_edges_covered",
            "scc_algorithm_rule",
        }:
            return 0.85
        return 1.25
    if name in STATE_DEFINITION_PROCESS_UNIT_NAMES:
        return 0.65
    if name in {
        "state_dijkstra_scan_edge",
        "state_flow_capacity_checked",
        "state_flow_residual_available",
        "state_flow_conservation_check",
    }:
        return 0.80
    if name == "state_dijkstra_no_relax":
        return 1.00
    if name in {
        "global_efficiency_denominator_value",
        "betweenness_normalized_value",
        "metric_answer",
        "global_metric_answer",
    }:
        return 2.50
    if name in {
        "wiener_pair_distance_scope",
        "global_efficiency_pair_scope",
        "global_unreachable_zero_contribution",
    }:
        return 1.15
    if name in {
        "state_mst_consider",
        "state_dijkstra_checked",
        "state_flow_capacity",
        "distance_pair",
        "global_efficiency_formula",
        "global_efficiency_normalizer",
        "wiener_distance_sum_formula",
        "wiener_pair_scope",
        "all_pairs_scope",
        "directed_scope",
        "undirected_scope",
        "closeness_formula",
        "harmonic_formula",
        "betweenness_formula",
        "betweenness_excludes_endpoints",
        "eccentricity_formula",
        "barycenter_formula",
        "density_formula",
        "node_count",
        "edge_count",
        "mapping_parseable_dict",
        "mapping_boxed_dict",
        "mapping_unique_values",
        "mapping_candidate_pair",
        "mapping_key",
        "map_pair",
        "path_directed_scope",
        "path_undirected_scope",
        "weighted_path_uses_weights",
        "global_unreachable_zero_contribution",
        "source_reachability_scope",
        "closeness_disconnected_scaling",
        "betweenness_sigma_fraction",
        "all_nodes_metric_scope",
        "center_min_ecc_rule",
        "periphery_max_ecc_rule",
        "barycenter_min_sum_rule",
        "mst_weight_order",
        "state_mst_weight_order",
        "flow_residual_network",
        "flow_level_graph",
        "flow_cut_certificate_scope",
        "state_flow_residual_network",
        "state_flow_level_graph",
        "state_flow_cut_scope",
        "hamiltonian_all_nodes_scope",
        "hamiltonian_adjacency_rule",
        "hamiltonian_unvisited_tracking",
        "hamiltonian_duplicate_rejected",
        "vertex_cover_matching_lower_bound",
        "bridge_lowlink_rule",
        "bridge_removal_test",
    }:
        return 0.10
    if name in {
        "state_path_prefix",
        "state_flow_path_edge",
        "state_mst_component_count",
        "source_reachable_count",
        "closeness_normalizer_value",
        "neighbor_pair",
        "common_neighbor",
        "common_neighbor_count",
        "neighbor_union_count",
        "triangle_nonedge_checked",
        "component_subset",
        "component_singleton",
        "topological_node",
        "hamiltonian_node",
        "all_pairs_pair",
        "global_pair_distance_scope",
        "global_unreachable_pair",
        "eccentricity_node_considered",
        "distance_sum_node_considered",
        "betweenness_pair_considered",
        "mapping_degree_match",
        "mapping_node_count_match",
        "mapping_edge_count_match",
        "mapping_degree_partition",
        "mapping_partial_bijection",
        "mst_edge_weight_checked",
        "state_mst_edge_weight_checked",
        "flow_residual_edge",
        "flow_reverse_edge",
        "state_flow_residual_edge",
        "state_flow_reverse_edge",
        "hamiltonian_prefix_len",
        "edge_cover_unmatched_node",
        "state_dijkstra_scan_edge",
    }:
        return 0.40
    if name in {
        "state_path_edge",
        "state_path_feasible",
        "state_traversal_tree_edge",
        "state_flow_path",
        "distance_value",
        "distance_sum",
        "eccentricity",
        "closeness_distance_sum",
        "harmonic_sum",
        "global_efficiency_pair_contribution",
        "global_metric_pair_distance_value",
        "wiener_pair_distance_value",
        "betweenness_pair_via_cert",
        "betweenness_pair_via",
        "betweenness_path_via",
        "betweenness_pair_checked",
        "mapping_preserves_edge",
        "mapping_adjacency_consistent",
        "triangle",
        "neighborhood_metric_answer",
        "regular_all_degrees",
        "regular_counterexample",
        "euler_all_even",
        "euler_odd_node",
        "tournament_pair",
        "tournament_violation",
        "bridge_edge",
        "non_bridge_checked",
        "component_count",
        "matching_edge",
        "vertex_cover_edge",
        "vertex_cover_all_edges_covered",
        "edge_cover_node",
        "dominated_node",
        "independent_node",
        "topological_prefix",
        "hamiltonian_edge",
        "hamiltonian_valid_extension",
        "edge_cover_matching_edge",
        "maximal_independent_covered_node",
        "independent_nonedge_check",
    }:
        return 1.50
    if name in {
        "state_mst_accept",
        "state_mst_reject_cycle",
        "state_dijkstra_relax",
        "state_dijkstra_dist",
        "state_dijkstra_dist_observed",
        "state_dijkstra_settle",
        "state_dijkstra_no_relax",
        "state_dijkstra_decreased_dist",
        "state_dijkstra_relax_inferred",
        "state_traversal_order_edge",
        "state_flow_bottleneck",
        "state_flow_candidate_bottleneck",
        "state_flow_capacity",
        "flow_bottleneck",
        "flow_capacity",
        "source_distance",
        "state_dijkstra_candidate_path_cost",
        "dijkstra_candidate_path_cost",
        "state_dijkstra_predecessor",
        "dijkstra_predecessor",
        "state_flow_capacity_checked",
        "state_flow_residual_available",
        "state_flow_conservation_check",
        "state_flow_residual_bottleneck_cert",
        "betweenness_normalizer_value",
        "betweenness_sigma_fraction_value",
        "mapping_complete_dict",
        "matching_size",
        "max_weight_matching_weight",
        "max_weight_matching_optimal_weight",
        "edge_cover_size",
        "edge_cover_matching_size",
        "vertex_cover_size",
        "dominating_set_size",
        "independent_set_size",
    }:
        return 2.00
    if name in {
        "state_path_reaches_target",
        "state_flow_complete",
        "state_flow_valid_augment",
        "state_flow_residual_update_after_augment",
        "flow_value",
        "hamiltonian_complete",
        "metric_answer",
        "global_metric_answer",
        "metric_value",
        "global_metric_value",
        "mapping_valid_dict",
        "boolean_answer",
        "template_complete",
        "node_answer_valid",
        "mst_answer_valid",
        "ordering_answer_valid",
        "set_answer_optimal_size",
        "edge_answer_valid",
        "edge_set_answer_optimal_size",
        "dominating_all_nodes_covered",
        "maximal_independent_complete",
        "edge_exists",
        "edge_absent",
        "tournament_answer",
    }:
        return 3.00
    if name in {"set_answer_feasible", "edge_set_answer_feasible", "mst_answer_spanning_tree"}:
        return 1.80
    return 1.0


def _state_seen_weight(spec: GraphLogicSpec, units: Iterable[tuple]) -> float:
    return sum(_state_unit_weight(spec, unit) for unit in units)


STATE_PROGRESS_COMPONENTS = ("cov", "fea", "gap", "cert", "err", "can", "pos")


STATE_HARD_CERT_MILESTONE_UNIT_NAMES = {
    "state_path_cost_valid",
    "state_mst_n_minus_one_edges",
    "state_mst_connected_acyclic",
    "state_mst_optimal_weight_cert",
    "state_flow_valid_augment",
    "state_flow_residual_update_after_augment",
    "state_flow_value_update_cert",
    "state_flow_complete",
    "global_efficiency_pair_contribution",
    "global_metric_pair_distance_value",
    "wiener_pair_distance_value",
    "betweenness_pair_via_cert",
    "betweenness_sigma_fraction_value",
    "betweenness_raw_sum",
    "closeness_normalizer_value",
    "closeness_distance_sum",
    "distance_sum",
    "mapping_complete_dict",
    "mapping_valid_dict",
    "mapping_preserves_edge",
    "complete_mapping_dict_answer",
    "parseable_node_list_answer",
    "parseable_edge_list_answer",
    "set_answer_optimal_size",
    "edge_set_answer_optimal_size",
    "max_weight_matching_optimal_weight",
    "vertex_cover_all_edges_covered",
    "edge_set_answer_feasible",
    "dominating_all_nodes_covered",
    "maximal_independent_complete",
    "matching_size",
    "metric_target_set_complete",
    "eccentricity",
    "distance_value",
    "state_dijkstra_weighted_relax_cert",
    "state_dijkstra_target_cert",
    "state_weighted_path_final_cert",
    "state_mst_spanning_nodes_covered",
    "state_mst_final_weight_checked",
    "state_flow_valid_augment_inferred",
    "state_flow_min_cut_cert",
}


COMPONENT_PRECISION_DEFAULT_TASKS = {
    "maximum_flow",
    "global_efficiency",
    "wiener_index",
    "isomophic_mapping",
    "isomorphic_mapping",
    "min_edge_covering",
}


def _component_precision_gate_allowed(spec: GraphLogicSpec, cfg: GraphLogicConfig | None) -> bool:
    if cfg is None or not bool(getattr(cfg, "component_precision_gate", False)):
        return False
    allowlist = str(getattr(cfg, "component_precision_task_allowlist", "") or "")
    if allowlist.strip():
        return _csv_allows_task(spec.task, allowlist)
    return str(spec.task) in COMPONENT_PRECISION_DEFAULT_TASKS


def _component_precision_unit_filter_allowed(spec: GraphLogicSpec, cfg: GraphLogicConfig | None) -> bool:
    if cfg is None or not bool(getattr(cfg, "component_precision_filter_units", False)):
        return False
    allowlist = str(getattr(cfg, "component_precision_task_allowlist", "") or "")
    if allowlist.strip():
        return _csv_allows_task(spec.task, allowlist)
    return str(spec.task) in COMPONENT_PRECISION_DEFAULT_TASKS


def _unit_name_set_from_trace(state_step_trace: dict[str, Any] | None) -> set[str]:
    if not isinstance(state_step_trace, dict):
        return set()
    names: set[str] = set()
    for key in ("rewardable_unit_names", "new_unit_names", "unit_names", "oracle_unit_names"):
        value = state_step_trace.get(key)
        if isinstance(value, (list, tuple, set)):
            names.update(str(item) for item in value)
    return names


def _component_precision_verified_positive(spec: GraphLogicSpec, unit_names: set[str]) -> bool:
    """High-precision unit gate for extending core4 without admitting broad weak positives."""
    task = str(spec.task)
    if task == "maximum_flow":
        has_path = bool(unit_names & {"state_flow_path", "state_flow_valid_augment", "state_flow_valid_augment_inferred"})
        has_bottleneck = bool(
            unit_names
            & {
                "state_flow_bottleneck",
                "state_flow_candidate_bottleneck",
                "state_flow_residual_bottleneck_cert",
                "flow_bottleneck",
            }
        )
        has_update = bool(
            unit_names
            & {
                "state_flow_residual_update",
                "state_flow_residual_update_after_augment",
                "state_flow_value_update_cert",
                "state_flow_complete",
                "state_flow_min_cut_cert",
            }
        )
        return bool((has_path and has_bottleneck and has_update) or (unit_names & {"state_flow_complete", "state_flow_min_cut_cert"}))
    if task == "global_efficiency":
        return bool(unit_names & {"global_efficiency_pair_contribution", "global_metric_pair_distance_value"})
    if task == "wiener_index":
        return bool(unit_names & {"wiener_pair_distance_value"})
    if task in {"isomophic_mapping", "isomorphic_mapping"}:
        return bool(
            "mapping_valid_dict" in unit_names
            or (
                "mapping_complete_dict" in unit_names
                and bool(unit_names & {"mapping_preserves_edge", "mapping_preserves_nonedge", "mapping_adjacency_consistent"})
            )
        )
    if task == "min_edge_covering":
        return bool({"edge_set_answer_feasible", "edge_set_answer_optimal_size"} <= unit_names)
    return True


def _component_precision_is_vague_potential(
    *,
    potential_delta: float,
    state_delta: float,
    unit_names: set[str],
    hard_cert_units: set[str],
) -> bool:
    if potential_delta <= 0.0:
        return False
    if state_delta > 0.0:
        return False
    if hard_cert_units:
        return False
    if not unit_names:
        return True
    weak_markers = {
        "global_efficiency_formula",
        "global_efficiency_normalizer",
        "wiener_distance_sum_formula",
        "wiener_pair_scope",
        "all_pairs_scope",
        "distance_pair",
        "global_pair_distance_scope",
        "wiener_pair_distance_scope",
        "flow_path",
        "state_flow_path",
        "state_flow_path_edge",
        "flow_path_edge",
        "state_flow_candidate_path",
        "mapping_candidate_pair",
        "mapping_partial_bijection",
        "map_pair",
        "mapping_key",
        "edge_cover_node",
        "edge_cover_size",
        "edge_cover_matching_size",
        "parseable_scalar_answer",
        "parseable_edge_list_answer",
        "parseable_node_list_answer",
        "parseable_mapping_dict_answer",
    }
    return bool(unit_names and unit_names <= weak_markers)


def _component_precision_allowed_unit_names(spec: GraphLogicSpec, unit_names: set[str]) -> set[str]:
    """Strict per-task reward unit allowlist used before state reward is computed."""
    task = str(spec.task)
    if task == "maximum_flow":
        if not _component_precision_verified_positive(spec, unit_names):
            return set()
        return {
            "state_flow_valid_augment",
            "state_flow_valid_augment_inferred",
            "state_flow_residual_bottleneck_cert",
            "state_flow_residual_update_after_augment",
            "state_flow_value_update_cert",
            "state_flow_complete",
            "state_flow_min_cut_cert",
        }
    if task == "global_efficiency":
        return {
            "distance_value",
            "global_efficiency_pair_contribution",
            "global_metric_pair_distance_value",
            "global_unreachable_zero_contribution",
        }
    if task == "wiener_index":
        return {"distance_value", "wiener_pair_distance_value"}
    if task in {"isomophic_mapping", "isomorphic_mapping"}:
        if not _component_precision_verified_positive(spec, unit_names):
            return set()
        return {
            "mapping_valid_dict",
            "mapping_complete_dict",
            "mapping_preserves_edge",
            "mapping_preserves_nonedge",
            "mapping_adjacency_consistent",
            "complete_mapping_dict_answer",
        }
    if task == "min_edge_covering":
        if not _component_precision_verified_positive(spec, unit_names):
            return set()
        return {"edge_set_answer_feasible", "edge_set_answer_optimal_size"}
    return unit_names


def _component_precision_filter_units(
    spec: GraphLogicSpec,
    units: set[tuple],
) -> tuple[set[tuple], set[tuple]]:
    if not units:
        return units, set()
    unit_names = {str(unit[0]) for unit in units if unit}
    allowed_names = _component_precision_allowed_unit_names(spec, unit_names)
    if not allowed_names:
        return set(), set(units)
    kept = {unit for unit in units if unit and str(unit[0]) in allowed_names}
    return kept, set(units) - kept


def _state_task_component_weights(spec: GraphLogicSpec) -> dict[str, float]:
    """Task-specific potential template from graph-algorithm progress semantics.

    Components follow the report terminology:
    cov=coverage, fea=feasible partial solution, gap=bound/objective progress,
    cert=certificate/final witness, err=numeric error reduction, can=candidate
    quality, pos=scope/formula/belief evidence.
    """

    task = spec.task
    if task == "weighted_shortest_path":
        return {"cov": 0.12, "fea": 0.24, "gap": 0.42, "cert": 0.18, "can": 0.04}
    if task == "shortest_path":
        return {"cov": 0.22, "fea": 0.34, "gap": 0.24, "cert": 0.16, "can": 0.04}
    if task in TRAVERSAL_TASKS:
        return {"cov": 0.45, "fea": 0.25, "cert": 0.20, "pos": 0.10}
    if task in MST_TASKS:
        return {"can": 0.18, "fea": 0.34, "gap": 0.26, "cert": 0.22}
    if task == "maximum_flow":
        return {"can": 0.08, "fea": 0.30, "gap": 0.40, "cert": 0.22}
    if task in ORDERING_STATE_TASKS:
        return {"cov": 0.24, "fea": 0.38, "can": 0.12, "cert": 0.26}
    if task in SET_FEASIBILITY_TASKS:
        if task == "maximal_independent_set":
            return {"cov": 0.16, "fea": 0.42, "cert": 0.30, "can": 0.12}
        return {"cov": 0.10, "fea": 0.34, "gap": 0.28, "cert": 0.22, "can": 0.06}
    if task == "isomophic_mapping":
        return {"cov": 0.06, "can": 0.06, "fea": 0.24, "cert": 0.64}
    if task in GLOBAL_WITNESS_TASKS:
        return {"cov": 0.28, "err": 0.42, "cert": 0.20, "pos": 0.10}
    if task in DISTANCE_WITNESS_TASKS:
        return {"cov": 0.25, "err": 0.42, "cert": 0.22, "pos": 0.11}
    if task in NEIGHBORHOOD_STATE_TASKS:
        if task in COUNT_FORMULA_TASKS | DEGREE_WITNESS_TASKS:
            return {"cov": 0.40, "err": 0.25, "cert": 0.25, "pos": 0.10}
        return {"cov": 0.34, "err": 0.36, "cert": 0.20, "pos": 0.10}
    if task in BOOLEAN_STATE_TASKS:
        return {"cov": 0.25, "cert": 0.45, "pos": 0.30}
    return {"cov": 0.35, "fea": 0.20, "cert": 0.25, "pos": 0.20}


def _state_pair_budget(spec: GraphLogicSpec) -> int:
    n = len(spec.nodes)
    if spec.directed:
        return max(1, n * max(0, n - 1))
    return max(1, n * max(0, n - 1) // 2)


def _state_component_budget(spec: GraphLogicSpec, component: str) -> float:
    task = spec.task
    n = len(spec.nodes)
    m = len(spec.edges)
    if component == "cov":
        if task in PATH_TASKS:
            return float(min(12, max(1, len(spec.target_edge_order or []) + 1)))
        if task in TRAVERSAL_TASKS:
            return float(min(16, max(1, len(spec.target_nodes or spec.nodes))))
        if task in MST_TASKS:
            return float(min(14, max(1, n - 1)))
        if task == "maximum_flow":
            return float(min(12, max(1, len(spec.target_edge_order or []) or m)))
        if task in ORDERING_STATE_TASKS:
            return float(min(16, max(1, n)))
        if task in {"closeness_centrality", "harmonic_centrality"}:
            return float(min(16, max(1, n - 1)))
        if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
            return float(min(18, _state_pair_budget(spec)))
        if task in SET_FEASIBILITY_TASKS:
            return float(min(16, max(1, len(spec.target_nodes or set()) or len(spec.target_edges or set()) or n)))
        if task == "isomophic_mapping":
            return float(min(16, max(1, len(spec.answer_mapping or {}))))
        return float(min(16, max(1, n + min(m, n))))
    if component == "fea":
        if task in PATH_TASKS:
            return 3.0
        if task in MST_TASKS | SET_FEASIBILITY_TASKS | ORDERING_STATE_TASKS:
            return 5.0
        if task == "maximum_flow":
            return 5.0
        if task == "isomophic_mapping":
            return 5.0
        return 4.0
    if component == "gap":
        if task == "weighted_shortest_path":
            return 7.0
        if task in MST_TASKS | SET_FEASIBILITY_TASKS | {"maximum_flow"}:
            return 6.0
        if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
            return 6.0
        return 4.0
    if component == "cert":
        if task in STRICT_COMPLETION_STATE_TASKS | STRICT_TEMPLATE_TASKS:
            return 4.0
        return 3.0
    if component == "err":
        if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
            return float(min(18, max(4, _state_pair_budget(spec))))
        if task in NEIGHBORHOOD_STATE_TASKS:
            return 5.0
        return 4.0
    if component == "can":
        if task in MST_TASKS | PATH_TASKS | SET_FEASIBILITY_TASKS | ISOMORPHISM_STATE_TASKS:
            return 5.0
        return 4.0
    if component == "pos":
        return 3.0
    return float(max(1, _state_units_budget(spec, GraphLogicAlgorithmState(task=task))))


def _state_unit_component(spec: GraphLogicSpec, unit: tuple) -> str:
    if not unit:
        return "pos"
    name = str(unit[0])
    if spec.task == "isomophic_mapping" and name == "mapping_partial_bijection":
        return "cov"
    if name in {
        "state_path_edge",
        "path_edge",
        "state_traversal_visit",
        "state_traversal_seen",
        "state_traversal_tree_edge",
        "state_traversal_order_node",
        "traversal_node",
        "traversal_edge",
        "state_flow_path_edge",
        "state_flow_candidate_path",
        "flow_path_edge",
        "state_mst_component_count",
        "scc_algorithm_rule",
        "node_count",
        "edge_count",
        "degree",
        "neighbor_pair",
        "common_neighbor",
        "common_neighbor_count",
        "neighbor_union_count",
        "triangle",
        "triangle_nonedge_checked",
        "cycle_edge",
        "bipartite_partition_node",
        "distance_pair",
        "source_reachable_count",
        "component_subset",
        "component_singleton",
        "map_pair",
        "mapping_key",
        "hamiltonian_node",
        "hamiltonian_edge",
        "hamiltonian_prefix_len",
        "all_pairs_pair",
        "global_pair_distance_scope",
        "wiener_pair_distance_scope",
        "global_efficiency_pair_scope",
        "global_unreachable_pair",
        "eccentricity_node_considered",
        "distance_sum_node_considered",
        "betweenness_pair_considered",
        "edge_cover_unmatched_node",
        "topological_node",
    }:
        return "cov"
    if name in {
        "state_path_prefix",
        "state_path_feasible",
        "state_path_cost_valid",
        "state_dijkstra_target_cert",
        "state_weighted_path_final_cert",
        "path_prefix",
        "state_path_reaches_target",
        "path_reaches_target",
        "state_traversal_answer_prefix",
        "state_traversal_answer_edge_prefix",
        "state_mst_accept",
        "state_mst_safe_edge",
        "state_mst_target_edge",
        "mst_accept_edge",
        "mst_answer_spanning_tree",
        "mst_answer_edge_count",
        "state_flow_path",
        "flow_path",
        "state_flow_augment",
        "state_flow_valid_augment_inferred",
        "state_flow_residual_update",
        "state_flow_residual_available",
        "state_flow_conservation_check",
        "set_answer_feasible",
        "edge_set_answer_feasible",
        "parseable_node_list_answer",
        "parseable_edge_list_answer",
        "parseable_scalar_answer",
        "parseable_mapping_dict_answer",
        "schema_simple_node_sequence",
        "matching_edge",
        "vertex_cover_edge",
        "edge_cover_node",
        "dominated_node",
        "independent_node",
        "mapping_complete_dict",
        "complete_mapping_dict_answer",
        "mapping_partial_bijection",
        "mapping_adjacency_consistent",
        "topological_prefix",
        "hamiltonian_valid_extension",
        "edge_cover_matching_edge",
        "maximal_independent_covered_node",
        "independent_nonedge_check",
    }:
        return "fea"
    if name in {
        "state_dijkstra_relax",
        "state_dijkstra_weighted_relax_cert",
        "dijkstra_relax_edge",
        "state_dijkstra_dist",
        "state_dijkstra_dist_observed",
        "state_dijkstra_settle",
        "state_dijkstra_no_relax",
        "source_distance",
        "state_dijkstra_candidate_path_cost",
        "dijkstra_candidate_path_cost",
        "state_dijkstra_predecessor",
        "dijkstra_predecessor",
        "state_dijkstra_decreased_dist",
        "state_dijkstra_relax_inferred",
        "state_flow_bottleneck",
        "state_flow_candidate_bottleneck",
        "state_flow_capacity_checked",
        "flow_bottleneck",
        "state_flow_value",
        "state_flow_residual_bottleneck_cert",
        "flow_value",
        "set_answer_optimal_size",
        "edge_set_answer_optimal_size",
        "matching_size",
        "max_weight_matching_weight",
        "max_weight_matching_optimal_weight",
        "edge_cover_size",
        "edge_cover_matching_size",
        "vertex_cover_size",
        "dominating_set_size",
        "independent_set_size",
        "mst_answer_valid",
        "state_mst_kruskal_prefix_edge",
        "state_mst_spanning_nodes_covered",
        "state_flow_valid_augment",
        "state_flow_residual_bottleneck_cert",
        "state_flow_residual_update_after_augment",
    }:
        return "gap"
    if name in {
        "state_path_reaches_target",
        "state_flow_complete",
        "state_flow_min_cut_cert",
        "hamiltonian_complete",
        "metric_answer",
        "global_metric_answer",
        "metric_target_set_complete",
        "mapping_valid_dict",
        "mapping_preserves_edge",
        "boolean_answer",
        "template_complete",
        "node_answer_valid",
        "ordering_answer_valid",
        "edge_answer_valid",
        "edge_exists",
        "edge_absent",
        "component_count",
        "metric_candidate_node",
        "metric_target_set_complete",
        "betweenness_normalized_value",
        "state_mst_edge_count_target",
        "state_mst_n_minus_one_edges",
        "state_mst_connected_acyclic",
        "state_mst_optimal_weight_cert",
        "state_mst_final_weight_checked",
        "dominating_all_nodes_covered",
        "maximal_independent_complete",
        "cycle_witness",
        "acyclic_answer",
        "bipartite_violation",
        "bipartite_answer",
        "bridge_edge",
        "non_bridge_checked",
        "bridge_lowlink_rule",
        "bridge_removal_test",
        "tournament_answer",
    }:
        return "cert"
    if name in {
        "distance_value",
        "distance_sum",
        "eccentricity",
        "distance_unreachable_pair",
        "closeness_distance_sum",
        "closeness_normalizer_value",
        "harmonic_sum",
        "global_efficiency_pair_contribution",
        "global_metric_pair_distance_value",
        "wiener_pair_distance_value",
        "global_efficiency_denominator_value",
        "global_unreachable_zero_contribution",
        "global_unreachable_pair",
        "source_reachability_scope",
        "closeness_disconnected_scaling",
        "betweenness_pair_via_cert",
        "betweenness_pair_via",
        "betweenness_path_via",
        "betweenness_pair_checked",
        "betweenness_sigma_fraction",
        "betweenness_sigma_fraction_value",
        "betweenness_normalizer_value",
        "betweenness_raw_sum",
        "density_formula",
        "metric_tie_scope",
        "all_nodes_metric_scope",
        "center_min_ecc_rule",
        "periphery_max_ecc_rule",
        "barycenter_min_sum_rule",
        "neighborhood_metric_answer",
        "triangle_count_answer",
        "regular_all_degrees",
        "regular_counterexample",
        "euler_all_even",
        "euler_odd_node",
        "tournament_pair",
        "tournament_violation",
    }:
        return "err"
    if name in {
        "state_mst_consider",
        "mst_consider_edge",
        "state_mst_reject_cycle",
        "state_mst_cycle_reject_cert",
        "mst_reject_edge",
        "state_dijkstra_checked",
        "state_dijkstra_scan_edge",
        "state_dijkstra_init",
        "state_dijkstra_weighted_rule",
        "state_flow_capacity",
        "state_flow_candidate_path",
        "state_flow_candidate_bottleneck",
        "state_flow_conservation_check",
        "flow_capacity",
        "flow_edge",
        "mapping_parseable_dict",
        "mapping_boxed_dict",
        "mapping_unique_values",
        "mapping_candidate_pair",
        "mapping_degree_match",
        "mapping_node_count_match",
        "mapping_edge_count_match",
        "mapping_degree_partition",
        "mst_weight_order",
        "mst_edge_weight_checked",
        "state_mst_weight_order",
        "state_mst_edge_weight_checked",
        "state_mst_weighted_rule",
        "vertex_cover_matching_lower_bound",
        "flow_residual_edge",
        "flow_reverse_edge",
        "flow_residual_network",
        "flow_level_graph",
        "flow_cut_certificate_scope",
        "state_flow_residual_edge",
        "state_flow_reverse_edge",
        "state_flow_capacity_checked",
        "state_flow_residual_available",
        "state_flow_residual_network",
        "state_flow_level_graph",
        "state_flow_cut_scope",
        "state_flow_conservation_scope",
        "hamiltonian_all_nodes_scope",
        "hamiltonian_adjacency_rule",
        "hamiltonian_unvisited_tracking",
        "hamiltonian_duplicate_rejected",
        "vertex_cover_matching_lower_bound",
    }:
        return "can"
    return "pos"


def _algorithm_state_component_scores(
    spec: GraphLogicSpec,
    state: GraphLogicAlgorithmState,
) -> dict[str, float]:
    strict_rewardable_components = {
        "weighted_shortest_path",
        "shortest_path",
        "maximum_flow",
        "weighted_minimum_spanning_tree",
        "minimum_spanning_tree",
        "bipartite_maximum_matching",
        "max_weight_matching",
        "maximal_independent_set",
        "bridges",
        "center",
        "periphery",
        "barycenter",
        "betweenness_centrality",
        "closeness_centrality",
        "global_efficiency",
        "wiener_index",
        "isomophic_mapping",
    } | TRAVERSAL_TASKS
    weights = defaultdict(float)
    for unit in state.seen_units:
        if spec.task in strict_rewardable_components and not _state_rewardable_unit(spec, unit):
            continue
        component = _state_unit_component(spec, unit)
        weights[component] += _state_unit_weight(spec, unit)
    scores: dict[str, float] = {}
    for component in STATE_PROGRESS_COMPONENTS:
        budget = max(1e-6, _state_component_budget(spec, component))
        scores[component] = max(0.0, min(1.0, weights[component] / budget))
    if spec.task == "weighted_shortest_path" and spec.source is not None:
        reachable = [
            node
            for node in spec.nodes
            if node != spec.source and math.isfinite(_shortest_distance(spec, int(spec.source), int(node)))
        ]
        if reachable:
            correct_dist_nodes = [
                node
                for node in reachable
                if node in state.dist and _close_number(state.dist[node], _shortest_distance(spec, int(spec.source), int(node)))
            ]
            correct_settled = [
                node
                for node in state.settled_nodes
                if node in reachable
                and node in state.dist
                and _close_number(state.dist[node], _shortest_distance(spec, int(spec.source), int(node)))
            ]
            correct_parents = [
                node
                for node, parent in state.parent_node.items()
                if node in reachable and _shortest_path_parent(spec, node) == parent
            ]
            scores["gap"] = max(scores.get("gap", 0.0), min(1.0, len(correct_dist_nodes) / max(1.0, len(reachable))))
            scores["cov"] = max(scores.get("cov", 0.0), min(1.0, len(correct_settled) / max(1.0, len(reachable))))
            scores["fea"] = max(scores.get("fea", 0.0), min(1.0, len(correct_parents) / max(1.0, len(reachable))))
            if spec.target is not None and spec.target in state.dist:
                target_dist = _shortest_distance(spec, int(spec.source), int(spec.target))
                if math.isfinite(target_dist) and _close_number(state.dist[spec.target], target_dist):
                    scores["cert"] = max(scores.get("cert", 0.0), 0.60)
    if spec.task == "maximum_flow" and state.max_flow_value > 1e-12:
        flow_ratio = max(0.0, min(1.0, float(state.flow_value) / float(state.max_flow_value)))
        scores["gap"] = max(scores.get("gap", 0.0), flow_ratio)
        if flow_ratio > 0.0:
            scores["fea"] = max(scores.get("fea", 0.0), min(1.0, 0.35 + 0.65 * flow_ratio))
        if flow_ratio >= 1.0 - 1e-9:
            scores["cert"] = max(scores.get("cert", 0.0), 1.0)
    if spec.task in MST_TASKS:
        target_edge_count = max(0, len(spec.nodes) - 1)
        if target_edge_count > 0:
            accepted_ratio = min(1.0, len(state.accepted_edges) / float(target_edge_count))
            scores["fea"] = max(scores.get("fea", 0.0), accepted_ratio)
            scores["gap"] = max(scores.get("gap", 0.0), 0.75 * accepted_ratio)
            if len(state.accepted_edges) == target_edge_count:
                roots = {_state_find(state, int(node)) for node in spec.nodes}
                if len(roots) == 1:
                    scores["fea"] = max(scores.get("fea", 0.0), 1.0)
                    scores["gap"] = max(scores.get("gap", 0.0), 0.95)
                    scores["cert"] = max(scores.get("cert", 0.0), 0.90)
                    if spec.target_edges:
                        optimal_weight = _mst_optimal_weight(spec)
                        if _close_number(float(state.total_weight), float(optimal_weight)):
                            scores["cert"] = max(scores.get("cert", 0.0), 1.0)
    if spec.task == "isomophic_mapping" and spec.answer_mapping:
        source_nodes = set(spec.answer_mapping.keys())
        target_nodes = set(spec.answer_mapping.values())
        if source_nodes:
            mapped_items = {
                k: v
                for k, v in state.partial_mapping.items()
                if k in source_nodes and v in target_nodes
            }
            ratio = len(mapped_items) / max(1.0, len(source_nodes))
            edge_checked_sources: set[int] = set()
            for unit in state.seen_units:
                if unit and str(unit[0]) in {"mapping_adjacency_consistent", "mapping_preserves_edge"} and len(unit) >= 5:
                    edge_checked_sources.add(int(unit[1]))
                    edge_checked_sources.add(int(unit[2]))
            checked_ratio = len(edge_checked_sources & source_nodes) / max(1.0, len(source_nodes))
            scores["cov"] = max(scores.get("cov", 0.0), min(1.0, 0.35 * ratio + 0.65 * checked_ratio))
            if len(set(mapped_items.values())) == len(mapped_items) and checked_ratio > 0.0:
                scores["fea"] = max(scores.get("fea", 0.0), min(1.0, checked_ratio))
            if set(mapped_items.keys()) == source_nodes and set(mapped_items.values()) == target_nodes:
                scores["cert"] = max(scores.get("cert", 0.0), 0.85)
    return scores


def _state_mandatory_gate_factor(spec: GraphLogicSpec, scores: dict[str, float]) -> float:
    task = spec.task
    if task == "weighted_shortest_path":
        if scores.get("cert", 0.0) <= 0.0 and scores.get("gap", 0.0) < 0.25:
            return 0.50
    if task == "maximum_flow":
        if scores.get("cert", 0.0) <= 0.0 and scores.get("gap", 0.0) < 0.35:
            return 0.50
    if task == "weighted_minimum_spanning_tree":
        if scores.get("cert", 0.0) <= 0.0:
            return 0.65
    if task in {"min_vertex_cover", "min_edge_covering", "dominating_set", "maximal_independent_set"}:
        if scores.get("cert", 0.0) <= 0.0 and scores.get("fea", 0.0) < 0.50:
            return 0.50
    if task in TRAVERSAL_TASKS and scores.get("cert", 0.0) <= 0.0 and scores.get("fea", 0.0) < 0.35:
        return 0.55
    if task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index"}:
        if scores.get("cert", 0.0) <= 0.0 and scores.get("err", 0.0) < 0.20:
            return 0.50
    if task in PATH_TASKS | MST_TASKS | SET_FEASIBILITY_TASKS | ORDERING_STATE_TASKS | {"maximum_flow"}:
        if scores.get("fea", 0.0) <= 0.0 and scores.get("cert", 0.0) <= 0.0:
            return 0.65
        if task in STRICT_COMPLETION_STATE_TASKS and scores.get("cert", 0.0) <= 0.0:
            return 0.85
    if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        if scores.get("err", 0.0) <= 0.0 and scores.get("cert", 0.0) <= 0.0:
            return 0.65
    if task == "isomophic_mapping" and scores.get("fea", 0.0) <= 0.0 and scores.get("cert", 0.0) <= 0.0:
        return 0.60
    return 1.0


def _state_terminal_unit(unit: tuple) -> bool:
    if not unit:
        return False
    return str(unit[0]) in {
        "state_path_reaches_target",
        "state_flow_complete",
        "hamiltonian_complete",
        "metric_answer",
        "global_metric_answer",
        "mapping_valid_dict",
        "boolean_answer",
        "template_complete",
        "node_answer_valid",
        "mst_answer_valid",
        "ordering_answer_valid",
        "set_answer_optimal_size",
        "edge_answer_valid",
        "edge_set_answer_optimal_size",
        "edge_exists",
        "edge_absent",
        "component_count",
        "bridge_edge",
        "tournament_answer",
        "neighborhood_metric_answer",
        "triangle_count_answer",
        "state_dijkstra_target_cert",
        "state_weighted_path_final_cert",
        "state_mst_final_weight_checked",
        "state_flow_min_cut_cert",
    }


STATE_VERIFIED_PROCESS_UNIT_NAMES = {
    "node_count",
    "edge_count",
    "degree",
    "neighbor_pair",
    "common_neighbor",
    "common_neighbor_count",
    "neighbor_union_count",
    "triangle",
    "triangle_nonedge_checked",
    "regular_all_degrees",
    "regular_counterexample",
    "euler_all_even",
    "euler_odd_node",
    "tournament_pair",
    "tournament_violation",
    "cycle_edge",
    "cycle_witness",
    "acyclic_answer",
    "bipartite_partition_node",
    "bipartite_violation",
    "bipartite_answer",
    "state_dijkstra_init",
    "state_dijkstra_weighted_rule",
    "state_mst_weighted_rule",
    "state_mst_edge_weight_checked",
    "mst_edge_weight_checked",
    "state_mst_edge_count_target",
    "state_mst_n_minus_one_edges",
    "state_mst_connected_acyclic",
    "state_mst_optimal_weight_cert",
    "state_flow_residual_update",
    "state_flow_conservation_scope",
    "betweenness_raw_sum",
    "betweenness_normalized_value",
    "metric_value",
    "global_metric_value",
    "distance_unreachable_pair",
    "global_unreachable_pair",
    "global_unreachable_zero_contribution",
    "metric_candidate_node",
    "metric_target_set_complete",
    "metric_tie_scope",
    "vertex_cover_matching_lower_bound",
    "vertex_cover_all_edges_covered",
    "dominating_all_nodes_covered",
    "maximal_independent_complete",
    "scc_algorithm_rule",
    "state_dijkstra_weighted_relax_cert",
    "state_dijkstra_target_cert",
    "state_weighted_path_final_cert",
    "state_mst_spanning_nodes_covered",
    "state_mst_final_weight_checked",
    "state_flow_valid_augment_inferred",
    "state_flow_min_cut_cert",
}


STATE_DEFINITION_PROCESS_UNIT_NAMES = {
    "density_formula",
    "global_efficiency_formula",
    "global_efficiency_normalizer",
    "wiener_distance_sum_formula",
    "wiener_pair_scope",
    "all_pairs_scope",
    "directed_scope",
    "undirected_scope",
    "closeness_formula",
    "harmonic_formula",
    "betweenness_formula",
    "betweenness_excludes_endpoints",
    "betweenness_sigma_fraction",
    "eccentricity_formula",
    "barycenter_formula",
    "source_reachability_scope",
    "closeness_disconnected_scaling",
    "all_nodes_metric_scope",
    "center_min_ecc_rule",
    "periphery_max_ecc_rule",
    "barycenter_min_sum_rule",
}


STATE_ISOMORPHISM_PROCESS_UNIT_NAMES = {
    "mapping_partial_bijection",
    "mapping_adjacency_consistent",
    "mapping_preserves_edge",
    "mapping_preserves_nonedge",
    "mapping_complete_dict",
    "mapping_valid_dict",
}


STATE_SCAFFOLD_UNIT_NAMES = {
    "state_mst_consider",
    "state_mst_weight_order",
    "state_mst_edge_weight_checked",
    "mst_consider_edge",
    "mst_weight_order",
    "mst_edge_weight_checked",
    "state_dijkstra_checked",
    "distance_pair",
    "all_pairs_pair",
    "global_pair_distance_scope",
    "wiener_pair_distance_scope",
    "global_efficiency_pair_scope",
    "global_unreachable_pair",
    "all_pairs_scope",
    "directed_scope",
    "undirected_scope",
    "path_directed_scope",
    "path_undirected_scope",
    "weighted_path_uses_weights",
    "global_efficiency_formula",
    "global_efficiency_normalizer",
    "wiener_distance_sum_formula",
    "wiener_pair_scope",
    "closeness_formula",
    "harmonic_formula",
    "betweenness_formula",
    "betweenness_excludes_endpoints",
    "betweenness_sigma_fraction",
    "eccentricity_formula",
    "barycenter_formula",
    "density_formula",
    "source_reachability_scope",
    "closeness_disconnected_scaling",
    "all_nodes_metric_scope",
    "center_min_ecc_rule",
    "periphery_max_ecc_rule",
    "barycenter_min_sum_rule",
    "betweenness_pair_considered",
    "mapping_parseable_dict",
    "mapping_boxed_dict",
    "mapping_unique_values",
    "mapping_candidate_pair",
    "mapping_key",
    "map_pair",
    "mapping_node_count_match",
    "mapping_edge_count_match",
    "mapping_degree_partition",
    "mapping_degree_match",
    "flow_capacity",
    "state_flow_capacity",
    "flow_residual_edge",
    "flow_reverse_edge",
    "flow_residual_network",
    "flow_level_graph",
    "flow_cut_certificate_scope",
    "state_flow_residual_edge",
    "state_flow_reverse_edge",
    "state_flow_residual_network",
    "state_flow_level_graph",
    "state_flow_cut_scope",
    "hamiltonian_all_nodes_scope",
    "hamiltonian_adjacency_rule",
    "hamiltonian_unvisited_tracking",
    "hamiltonian_duplicate_rejected",
    "hamiltonian_prefix_len",
    "vertex_cover_matching_lower_bound",
    "bridge_lowlink_rule",
    "bridge_removal_test",
}


STATE_REWARDABLE_UNIT_NAMES = {
    "state_path_edge",
    "state_path_prefix",
    "state_path_feasible",
    "state_path_cost_valid",
    "state_path_reaches_target",
    "state_dijkstra_scan_edge",
    "state_dijkstra_no_relax",
    "state_dijkstra_relax",
    "state_dijkstra_dist",
    "state_dijkstra_dist_observed",
    "state_dijkstra_decreased_dist",
    "state_dijkstra_relax_inferred",
    "state_dijkstra_settle",
    "state_dijkstra_candidate_path_cost",
    "state_dijkstra_predecessor",
    "state_traversal_order_edge",
    "state_traversal_order_node",
    "state_traversal_answer_prefix",
    "state_traversal_answer_edge_prefix",
    "state_traversal_tree_edge",
    "state_traversal_visit",
    "state_mst_accept",
    "state_mst_safe_edge",
    "state_mst_target_edge",
    "state_mst_kruskal_prefix_edge",
    "state_mst_reject_cycle",
    "state_mst_cycle_reject_cert",
    "state_mst_component_count",
    "state_flow_path",
    "state_flow_path_edge",
    "state_flow_capacity_checked",
    "state_flow_residual_available",
    "state_flow_conservation_check",
    "state_flow_bottleneck",
    "state_flow_augment",
    "state_flow_valid_augment",
    "state_flow_residual_bottleneck_cert",
    "state_flow_residual_update_after_augment",
    "state_flow_value",
    "state_flow_complete",
    "distance_value",
    "global_metric_pair_distance_value",
    "wiener_pair_distance_value",
    "distance_sum",
    "eccentricity",
    "closeness_distance_sum",
    "closeness_normalizer_value",
    "harmonic_sum",
    "global_efficiency_pair_contribution",
    "global_efficiency_denominator_value",
    "distance_unreachable_pair",
    "global_unreachable_pair",
    "global_unreachable_zero_contribution",
    "betweenness_pair_via_cert",
    "betweenness_pair_via",
    "betweenness_path_via",
    "betweenness_pair_checked",
    "betweenness_sigma_fraction_value",
    "betweenness_normalizer_value",
    "mapping_partial_bijection",
    "mapping_adjacency_consistent",
    "mapping_preserves_edge",
    "mapping_preserves_nonedge",
    "mapping_complete_dict",
    "mapping_valid_dict",
    "matching_edge",
    "matching_size",
    "vertex_cover_edge",
    "vertex_cover_size",
    "edge_cover_node",
    "edge_cover_size",
    "edge_cover_matching_edge",
    "edge_cover_matching_size",
    "dominated_node",
    "dominating_set_size",
    "independent_node",
    "independent_set_size",
    "maximal_independent_covered_node",
    "maximal_independent_complete",
    "independent_nonedge_check",
    "topological_prefix",
    "topological_node",
    "hamiltonian_edge",
    "hamiltonian_node",
    "hamiltonian_valid_extension",
    "hamiltonian_complete",
    "bridge_edge",
    "non_bridge_checked",
    "component_count",
    "scc_algorithm_rule",
    "edge_exists",
    "edge_absent",
    "boolean_answer",
    "metric_answer",
    "global_metric_answer",
    "neighborhood_metric_answer",
    "triangle_count_answer",
    "node_answer_valid",
    "edge_answer_valid",
    "ordering_answer_valid",
    "mst_answer_valid",
    "mst_answer_edge_count",
    "set_answer_feasible",
    "set_answer_optimal_size",
    "edge_set_answer_feasible",
    "edge_set_answer_optimal_size",
    "mst_answer_spanning_tree",
    "parseable_node_list_answer",
    "parseable_edge_list_answer",
    "parseable_scalar_answer",
    "parseable_mapping_dict_answer",
    "complete_mapping_dict_answer",
    "schema_simple_node_sequence",
    "state_dijkstra_weighted_relax_cert",
    "state_dijkstra_target_cert",
    "state_weighted_path_final_cert",
    "state_mst_spanning_nodes_covered",
    "state_mst_final_weight_checked",
    "state_flow_valid_augment_inferred",
    "state_flow_min_cut_cert",
}


STATE_GRAPH_SENSITIVE_UNIT_NAMES = {
    # Weighted path: value-bearing relax/final certificates should change when
    # referenced weights are perturbed. Generic path/settle mentions are not
    # included because they survived counterfactual edge/weight audits.
    "state_dijkstra_weighted_relax_cert",
    "state_dijkstra_decreased_dist",
    "state_dijkstra_target_cert",
    "state_weighted_path_final_cert",
    "state_path_cost_valid",
    # MST/WMST: require graph-dependent component/cycle/weight certificates,
    # not just "consider edge" prose.
    "state_mst_accept",
    "state_mst_safe_edge",
    "state_mst_reject_cycle",
    "state_mst_cycle_reject_cert",
    "state_mst_connected_acyclic",
    "state_mst_optimal_weight_cert",
    "state_mst_final_weight_checked",
    "mst_answer_spanning_tree",
    "mst_answer_valid",
    # Flow: positive process should be tied to the residual ledger.
    "state_flow_valid_augment",
    "state_flow_valid_augment_inferred",
    "state_flow_residual_bottleneck_cert",
    "state_flow_residual_update_after_augment",
    "state_flow_value_update_cert",
    # Global numeric metrics: pair/table contributions are graph-dependent;
    # pure formula/final scalar mentions are intentionally excluded.
    "distance_value",
    "global_metric_pair_distance_value",
    "wiener_pair_distance_value",
    "global_efficiency_pair_contribution",
    "closeness_distance_sum",
    "closeness_normalizer_value",
    "betweenness_pair_via_cert",
    "betweenness_sigma_fraction_value",
    "betweenness_raw_sum",
    # Set/cover/matching: require explicit feasible/optimal edge-set or
    # matching evidence rather than a node/edge count alone.
    "edge_set_answer_feasible",
    "edge_set_answer_optimal_size",
    "matching_edge",
    "matching_size",
    "max_weight_matching_weight",
    "max_weight_matching_optimal_weight",
    "vertex_cover_all_edges_covered",
    "dominating_all_nodes_covered",
    "maximal_independent_complete",
    "set_answer_feasible",
    "set_answer_optimal_size",
    # Isomorphism: graph-sensitive certificates must preserve adjacency.
    "mapping_complete_dict",
    "mapping_valid_dict",
    "mapping_preserves_edge",
    "mapping_preserves_nonedge",
    "complete_mapping_dict_answer",
}


def _state_rewardable_unit(spec: GraphLogicSpec, unit: tuple) -> bool:
    """Whether a newly observed unit should create positive shaping signal.

    Scaffolding units such as "use Dijkstra", "sort by weight", formula names,
    or degree-partition hints are useful diagnostics, but by themselves they do
    not prove that the reasoning state moved forward. Rewardable units encode a
    concrete transition, numeric contribution, feasibility certificate, or
    checked witness.
    """
    if not unit:
        return False
    name = str(unit[0])
    if spec.task == "weighted_shortest_path":
        # Avoid rewarding arbitrary valid paths or local candidate costs. For a
        # weighted shortest path, positive process signal should come from
        # Dijkstra-style state updates or a weighted target-path certificate.
        # A valid node sequence that merely reaches the target is too weak: the
        # main WSP failure mode is outputting a plausible path with the wrong
        # weight or a non-optimal weighted route.
        return name in {
            "state_path_cost_valid",
            "state_dijkstra_relax",
            "state_dijkstra_weighted_relax_cert",
            "state_dijkstra_dist",
            "state_dijkstra_dist_observed",
            "state_dijkstra_decreased_dist",
            "state_dijkstra_settle",
            "state_dijkstra_no_relax",
            "state_dijkstra_predecessor",
            "state_dijkstra_target_cert",
            "state_weighted_path_final_cert",
        }
    if spec.task in TRAVERSAL_TASKS:
        # Node mentions and unordered visits were a large source of BFS/DFS
        # false positives. Reward ordered prefixes, verified tree edges, and
        # final parseable traversal objects rather than arbitrary node mentions.
        expects_edge_answer = bool(spec.target_edge_order)
        if name == "parseable_node_list_answer":
            return not expects_edge_answer
        if name == "parseable_edge_list_answer":
            return expects_edge_answer
        if expects_edge_answer:
            return name in {
                "state_traversal_order_edge",
                "state_traversal_answer_edge_prefix",
                "state_traversal_tree_edge",
                "ordering_answer_valid",
            }
        return name in {
            "state_traversal_order_node",
            "state_traversal_answer_prefix",
            "ordering_answer_valid",
        }
    if spec.task == "maximum_flow":
        # A path through the network is not enough. Greedy path-bottleneck
        # accumulation was the main false-positive mode in flow runs, so only
        # residual-aware updates and min-cut/completion certificates are
        # rewardable. Capacity/conservation mentions stay diagnostic.
        return name in {
            "state_flow_valid_augment",
            "state_flow_valid_augment_inferred",
            "state_flow_residual_bottleneck_cert",
            "state_flow_residual_update_after_augment",
            "state_flow_value_update_cert",
            "state_flow_complete",
            "state_flow_min_cut_cert",
        }
    if spec.task == "weighted_minimum_spanning_tree":
        return name in {
            "state_mst_weight_order",
            "state_mst_weighted_rule",
            "state_mst_edge_weight_checked",
            "state_mst_accept",
            "state_mst_safe_edge",
            "state_mst_target_edge",
            "state_mst_kruskal_prefix_edge",
            "state_mst_reject_cycle",
            "state_mst_cycle_reject_cert",
            "state_mst_edge_count_target",
            "state_mst_n_minus_one_edges",
            "state_mst_spanning_nodes_covered",
            "state_mst_connected_acyclic",
            "state_mst_optimal_weight_cert",
            "state_mst_final_weight_checked",
            "mst_answer_edge_count",
            "mst_answer_spanning_tree",
            "mst_answer_valid",
        }
    if spec.task == "isomophic_mapping":
        # The audit showed that this task can receive positive reward while
        # strict pass@1 is near zero. Treat degree/refinement/candidate-pair
        # evidence as diagnostics only; positive shaping requires a complete
        # dict or a verified edge/non-edge preservation certificate.
        return name in {
            "mapping_complete_dict",
            "mapping_valid_dict",
            "mapping_preserves_edge",
            "mapping_preserves_nonedge",
            "complete_mapping_dict_answer",
        }
    if spec.task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index"}:
        if spec.task == "global_efficiency":
            return name in {
                "distance_value",
                "global_metric_pair_distance_value",
                "global_efficiency_pair_contribution",
            }
        if spec.task == "wiener_index":
            return name in {
                "distance_value",
                "wiener_pair_distance_value",
            }
        if spec.task == "closeness_centrality":
            return name in {
                "distance_value",
                "closeness_distance_sum",
                "closeness_normalizer_value",
                "distance_unreachable_pair",
            }
        return name in {
            "betweenness_pair_via_cert",
            "betweenness_path_via",
            "betweenness_sigma_fraction_value",
            "betweenness_raw_sum",
        }
    if spec.task == "maximal_independent_set":
        return name in {
            "independent_nonedge_check",
            "maximal_independent_complete",
            "set_answer_feasible",
            "set_answer_optimal_size",
            "parseable_node_list_answer",
        }
    if spec.task in {"min_vertex_cover", "dominating_set"}:
        return name in {
            "vertex_cover_edge",
            "dominated_node",
            "dominating_all_nodes_covered",
            "vertex_cover_all_edges_covered",
            "vertex_cover_matching_lower_bound",
            "set_answer_feasible",
            "set_answer_optimal_size",
            "parseable_node_list_answer",
        }
    if spec.task == "min_edge_covering":
        return name in {
            "edge_set_answer_feasible",
            "edge_set_answer_optimal_size",
        }
    if spec.task == "global_efficiency":
        # Global efficiency reward should be tied to concrete pair-distance
        # evidence. Formula names, denominators, and final scalar mentions are
        # useful diagnostics but were too insensitive under graph perturbation.
        return name in {
            "distance_value",
            "global_metric_pair_distance_value",
            "global_efficiency_pair_contribution",
        }
    if spec.task == "wiener_index":
        # Wiener index is an all-pairs distance sum; only pair-level distance
        # values are treated as process reward. Aggregate sums/final scalars are
        # left to outcome reward and diagnostics.
        return name in {
            "distance_value",
            "wiener_pair_distance_value",
        }
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching"}:
        if spec.task == "max_weight_matching":
            return name in {
                "max_weight_matching_weight",
                "max_weight_matching_optimal_weight",
            }
        return name in {
            "matching_edge",
            "matching_size",
            "edge_set_answer_feasible",
            "edge_set_answer_optimal_size",
            "parseable_edge_list_answer",
        }
    if spec.task == "hamiltonian_path":
        return name in {
            "path_prefix",
            "path_reaches_target",
            "hamiltonian_valid_extension",
            "hamiltonian_complete",
            "ordering_answer_valid",
            "parseable_node_list_answer",
            "schema_simple_node_sequence",
        }
    if spec.task == "traveling_salesman_problem":
        return name in {
            "ordering_answer_valid",
        }
    if spec.task == "jaccard_coefficient":
        return name in {
            "neighborhood_metric_answer",
        }
    if spec.task in {"center", "periphery", "barycenter"}:
        # Candidate nodes and formula mentions were over-rewarded. Keep only
        # exact numeric table entries and complete final-set certificates.
        return name in {
            "distance_sum",
            "eccentricity",
            "metric_target_set_complete",
            "node_answer_valid",
        }
    if spec.task == "bridges":
        # Bridge process evidence is easy to over-detect from prose. Use it for
        # logging, but only the final exact edge-list certificate shapes reward.
        return name in {"edge_answer_valid"}
    if _state_terminal_unit(unit):
        return True
    if name in STATE_ALGORITHM_OVERLAY_SCAFFOLD_UNIT_NAMES:
        return False
    if name in STATE_VERIFIED_PROCESS_UNIT_NAMES:
        return True
    if (
        name in STATE_DEFINITION_PROCESS_UNIT_NAMES
        and spec.task
        in (
            NEIGHBORHOOD_STATE_TASKS
            | {"center", "periphery", "barycenter", "radius", "diameter"}
        )
    ):
        return True
    if name in STATE_SCAFFOLD_UNIT_NAMES:
        return False
    if spec.task in TRAVERSAL_TASKS and spec.target_edge_order and name in {
        "parseable_node_list_answer",
        "schema_wrong_final_node_list",
        "state_traversal_order_node",
        "state_traversal_answer_prefix",
        "state_traversal_visit",
        "state_traversal_seen",
    }:
        return False
    if spec.task == "isomophic_mapping":
        return name in {
            "mapping_adjacency_consistent",
            "mapping_preserves_edge",
            "mapping_complete_dict",
            "mapping_valid_dict",
        }
    if spec.task in MST_TASKS and name == "parseable_edge_list_answer":
        return False
    if name in STATE_REWARDABLE_UNIT_NAMES:
        return True
    component = _state_unit_component(spec, unit)
    weight = _state_unit_weight(spec, unit)
    if spec.task in TRAVERSAL_TASKS:
        if spec.target_edge_order:
            return name in {
                "state_traversal_order_edge",
                "state_traversal_answer_edge_prefix",
                "state_traversal_tree_edge",
                "ordering_answer_valid",
            }
        return name in {"state_traversal_order_node", "state_traversal_answer_prefix"}
    if spec.task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return component in {"err", "cert"} and weight >= 1.0
    if spec.task in NEIGHBORHOOD_STATE_TASKS | BOOLEAN_STATE_TASKS:
        return component in {"err", "cert", "fea"} and weight >= 1.0
    if spec.task == "isomophic_mapping":
        return component in {"fea", "cert"} and weight >= 1.0
    return component in {"fea", "gap", "cert"} and weight >= 1.0


def _state_delta_quality(
    spec: GraphLogicSpec,
    actions: int,
    invalid: int,
    new_unit_count: int,
    rewardable_new_unit_count: int | None = None,
    new_unit_weight: float = 0.0,
    rewardable_new_unit_weight: float = 0.0,
) -> float:
    if new_unit_count <= 0:
        return 0.0
    if rewardable_new_unit_count is not None and rewardable_new_unit_count <= 0:
        return 0.0
    if invalid <= 0:
        base = 1.0
    else:
        denom = max(1.0, float(max(actions, invalid)))
        ratio = min(1.0, float(invalid) / denom)
        if spec.task in PATH_TASKS | TRAVERSAL_TASKS | MST_TASKS | {"maximum_flow"}:
            base = max(0.25, 1.0 - 0.90 * ratio)
        elif spec.task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS | ISOMORPHISM_STATE_TASKS:
            base = max(0.35, 1.0 - 0.70 * ratio)
        else:
            base = max(0.40, 1.0 - 0.60 * ratio)
    if rewardable_new_unit_count is not None:
        if new_unit_weight > 1e-12 and rewardable_new_unit_weight > 0.0:
            rewardable_ratio = max(0.0, min(1.0, rewardable_new_unit_weight / new_unit_weight))
        else:
            rewardable_ratio = float(rewardable_new_unit_count) / max(1.0, float(new_unit_count))
        base *= max(0.25, min(1.0, rewardable_ratio))
    return base


def _state_invalid_penalty_scale(task: str) -> float:
    if task == "weighted_shortest_path":
        return 0.75
    if task == "maximum_flow":
        return 0.85
    if task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index"}:
        return 0.55
    if task == "isomophic_mapping":
        return 0.60
    if task in PATH_TASKS | TRAVERSAL_TASKS | MST_TASKS:
        return 0.55
    if task in ORDERING_STATE_TASKS | SET_FEASIBILITY_TASKS:
        return 0.45
    if task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS | ISOMORPHISM_STATE_TASKS:
        return 0.35
    return 0.30


def _state_negative_delta(
    spec: GraphLogicSpec,
    actions: int,
    invalid: int,
    completion_invalid: bool,
) -> float:
    """Potential-scale negative progress from illegal state transitions.

    This is deliberately separate from positive ``state_delta``. A segment can
    contain one valid update and one illegal update; those should not collapse
    into a single weak positive score. The magnitude is small but comparable to
    observed per-segment positive deltas, so invalid algorithm actions remain
    visible even when script-level ``state_invalid_penalty`` is conservative.
    """

    if invalid <= 0:
        return 0.0
    action_count = max(1.0, float(actions))
    invalid_ratio = min(1.0, float(invalid) / action_count)
    count_factor = 1.0 + 0.45 * min(2.0, max(0.0, float(invalid) - 1.0))
    task_scale = _state_invalid_penalty_scale(spec.task)
    sensitive_scale = 1.25 if spec.task in ACCUMULATION_SENSITIVE_STATE_TASKS else 1.0
    completion_scale = 1.50 if completion_invalid else 1.0
    base = 0.009 if spec.task in FALSE_POSITIVE_SENSITIVE_STATE_TASKS else 0.012
    value = base * task_scale * sensitive_scale * completion_scale * count_factor * (0.70 + 0.30 * invalid_ratio)
    return max(0.0, min(0.060, value))


def _state_positive_delta_scale(spec: GraphLogicSpec, state_step_stats: dict[str, float]) -> float:
    """Small confidence bonus for clean executable state progress.

    The previous stricter observer fixed reward leakage, but it also made many
    correct trajectories net-negative because positive deltas were too small
    relative to generic penalties. This multiplier only applies to clean,
    rewardable state transitions, so verbosity, duplicate evidence, and invalid
    actions do not benefit from it.
    """

    if float(state_step_stats.get("state_invalid_transitions", 0.0)) > 0.0:
        return 1.0
    if float(state_step_stats.get("state_rewardable_transitions", 0.0)) <= 0.0:
        return 1.0
    if float(state_step_stats.get("state_delta", 0.0)) <= 0.0:
        return 1.0
    scale = 1.55
    if spec.task in ACCUMULATION_SENSITIVE_STATE_TASKS:
        scale = 1.55
    if spec.task in {"weighted_shortest_path", "weighted_minimum_spanning_tree", "maximum_flow"}:
        scale = max(scale, 1.90)
    if spec.task in {"betweenness_centrality", "closeness_centrality", "global_efficiency", "wiener_index"}:
        scale = max(scale, 1.75)
    if spec.task in {
        "dominating_set",
        "min_vertex_cover",
        "maximal_independent_set",
        "bipartite_maximum_matching",
        "max_weight_matching",
        "min_edge_covering",
    }:
        scale = max(scale, 1.70)
    if spec.task in TRAVERSAL_TASKS:
        scale = max(scale, 1.75)
    if spec.task in {"hamiltonian_path", "traveling_salesman_problem"}:
        scale = max(scale, 1.80)
    if spec.task == "topological_sort":
        scale = max(scale, 1.65)
    if float(state_step_stats.get("state_terminal_units", 0.0)) > 0.0:
        scale = max(scale, 2.10)
    return scale


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _response_positive_reward_budget(
    spec: GraphLogicSpec,
    cfg: GraphLogicConfig,
    stats: dict[str, float],
    base_cap: float,
) -> tuple[float, dict[str, float]]:
    """Dynamic positive-reward budget for a full response.

    The budget is intentionally based on length-normalized *net* state progress,
    not on the number of rewarded events. This keeps correct concise reasoning
    positive while preventing long wrong trajectories from accumulating positive
    process reward through many shallow or repetitive segments.
    """

    if base_cap <= 0.0:
        return 0.0, {}
    budget_mode = str(getattr(cfg, "positive_reward_budget_mode", "quality") or "quality").lower()
    if budget_mode not in {"quality", "net_progress"}:
        budget_mode = "quality"
    state_supported = float(stats.get("state_supported", 0.0)) > 0.0
    state_segments = max(1.0, float(stats.get("state_supported_segments", 0.0)))
    action_segments = max(1.0, float(stats.get("state_action_segments", 0.0)))
    state_positive_delta = float(stats.get("state_positive_delta", stats.get("state_delta", 0.0)))
    potential_positive_delta = float(stats.get("potential_delta", 0.0))
    positive_delta = state_positive_delta if state_supported else potential_positive_delta
    negative_delta = float(stats.get("state_negative_delta", 0.0)) if state_supported else 0.0
    net_delta = positive_delta - 1.50 * negative_delta
    net_rate = net_delta / state_segments
    if spec.task in FALSE_POSITIVE_SENSITIVE_STATE_TASKS:
        quality = _clamp_float((net_rate - 0.0035) / 0.0045, 0.0, 1.0)
    else:
        quality = _clamp_float((net_rate - 0.0025) / 0.0035, 0.0, 1.0)

    invalid_segment_rate = float(stats.get("state_invalid_action_segments", 0.0)) / state_segments
    invalid_transition_rate = float(stats.get("state_invalid_transitions", 0.0)) / action_segments
    duplicate_rate = float(stats.get("state_duplicate_no_progress_segments", 0.0)) / state_segments
    weak_only_rate = float(stats.get("state_weak_only_transition", 0.0)) / state_segments

    if spec.task in FALSE_POSITIVE_SENSITIVE_STATE_TASKS:
        invalid_slope = 1.45 + 1.35 * (1.0 - quality)
        duplicate_slope = 0.30 + 0.45 * (1.0 - quality)
        weak_slope = 0.95
        min_invalid_factor = 0.12
        min_repetition_factor = 0.35
    else:
        invalid_slope = 1.15 + 1.15 * (1.0 - quality)
        duplicate_slope = 0.18 + 0.32 * (1.0 - quality)
        weak_slope = 0.75
        min_invalid_factor = 0.20
        min_repetition_factor = 0.55
    invalid_factor = _clamp_float(
        1.0 - invalid_slope * invalid_segment_rate - 0.20 * invalid_transition_rate,
        min_invalid_factor,
        1.0,
    )
    repetition_factor = _clamp_float(
        1.0 - duplicate_slope * duplicate_rate - weak_slope * weak_only_rate,
        min_repetition_factor,
        1.0,
    )

    state_net_progress = max(
        0.0,
        float(stats.get("state_potential_final", 0.0)) - float(stats.get("state_potential_initial", 0.0)),
    )
    potential_net_progress = max(
        0.0,
        float(stats.get("potential_final", 0.0)) - float(stats.get("potential_initial", 0.0)),
    )
    if state_supported:
        net_progress = state_net_progress
        progress_source = 1.0
    else:
        net_progress = potential_net_progress
        progress_source = 0.0
    if net_progress <= 0.0 and positive_delta > 0.0:
        # Fallback for tasks whose canonical potential is intentionally sparse:
        # keep the value bounded so event count still cannot set the response budget.
        net_progress = min(1.0, max(0.0, net_delta))
        progress_source = -1.0
    net_progress = _clamp_float(net_progress, 0.0, 1.0)

    if budget_mode == "net_progress":
        budget = base_cap * net_progress * invalid_factor * repetition_factor
        if positive_delta <= 0.0:
            budget = 0.0
        diagnostics = {
            "positive_reward_budget_quality": float(quality),
            "positive_reward_budget_net_rate": float(net_rate),
            "positive_reward_budget_invalid_factor": float(invalid_factor),
            "positive_reward_budget_repetition_factor": float(repetition_factor),
            "positive_reward_budget_multiplier": float(budget / max(1e-12, base_cap)),
            "positive_reward_budget_target_fraction": 1.0,
            "positive_reward_budget_mode_net_progress": 1.0,
            "positive_reward_budget_net_progress": float(net_progress),
            "positive_reward_budget_state_net_progress": float(state_net_progress),
            "positive_reward_budget_potential_net_progress": float(potential_net_progress),
            "positive_reward_budget_progress_source": float(progress_source),
            "positive_reward_budget_positive_delta": float(positive_delta),
            "positive_reward_budget_state_positive_delta": float(state_positive_delta),
            "positive_reward_budget_potential_positive_delta": float(potential_positive_delta),
        }
        return max(0.0, budget), diagnostics

    if spec.task in FALSE_POSITIVE_SENSITIVE_STATE_TASKS:
        min_multiplier = 0.02
        max_multiplier = 1.15
    else:
        min_multiplier = 0.08 if spec.task in ACCUMULATION_SENSITIVE_STATE_TASKS else 0.12
        max_multiplier = 1.60
    multiplier = (min_multiplier + (max_multiplier - min_multiplier) * quality) * invalid_factor * repetition_factor
    if positive_delta <= 0.0:
        multiplier = 0.0
    budget = base_cap * _clamp_float(multiplier, 0.0, max_multiplier)
    diagnostics = {
        "positive_reward_budget_quality": float(quality),
        "positive_reward_budget_net_rate": float(net_rate),
        "positive_reward_budget_invalid_factor": float(invalid_factor),
        "positive_reward_budget_repetition_factor": float(repetition_factor),
        "positive_reward_budget_multiplier": float(multiplier),
        "positive_reward_budget_target_fraction": float(0.55 + 0.30 * quality),
        "positive_reward_budget_mode_net_progress": 0.0,
        "positive_reward_budget_net_progress": float(net_progress),
        "positive_reward_budget_state_net_progress": float(state_net_progress),
        "positive_reward_budget_potential_net_progress": float(potential_net_progress),
        "positive_reward_budget_progress_source": float(progress_source),
        "positive_reward_budget_positive_delta": float(positive_delta),
        "positive_reward_budget_state_positive_delta": float(state_positive_delta),
        "positive_reward_budget_potential_positive_delta": float(potential_positive_delta),
    }
    return max(0.0, budget), diagnostics


def _calibrated_potential_delta(
    spec: GraphLogicSpec,
    potential_delta: float,
    state_delta: float,
    state_step_stats: dict[str, float],
    cfg: GraphLogicConfig,
) -> float:
    if potential_delta <= 0.0 or not bool(getattr(cfg, "state_calibrate_potential", True)):
        return potential_delta
    actions = float(state_step_stats.get("state_actions", 0.0))
    if actions <= 0.0:
        return 0.0
    invalid = float(state_step_stats.get("state_invalid_transitions", 0.0))
    completion_invalid = float(state_step_stats.get("state_completion_invalid", 0.0)) > 0.0
    rewardable = float(state_step_stats.get("state_rewardable_transitions", 0.0)) > 0.0
    valid = rewardable and float(state_step_stats.get("state_valid_transitions", 0.0)) > 0.0 and state_delta > 0.0
    if completion_invalid or not valid:
        return 0.0

    # In state-shaping mode, generic edge/node/evidence potential may not create
    # reward by itself. It can only mirror executable state progress, which
    # removes the old length bias where verbose generations accumulated many
    # shallow evidence units.
    calibrated = min(potential_delta, state_delta)
    if invalid > 0.0:
        invalid_ratio = min(1.0, invalid / max(1.0, actions))
        calibrated *= max(0.0, 1.0 - 1.25 * invalid_ratio)
    return max(0.0, calibrated)


def _algorithm_state_potential(spec: GraphLogicSpec, state: GraphLogicAlgorithmState) -> float:
    scores = _algorithm_state_component_scores(spec, state)
    profile = _state_task_component_weights(spec)
    denom = max(1e-6, sum(max(0.0, weight) for weight in profile.values()))
    value = sum(max(0.0, weight) * scores.get(component, 0.0) for component, weight in profile.items()) / denom
    return max(0.0, min(1.0, value * _state_mandatory_gate_factor(spec, scores)))


def _valid_state_flow_path_edges(spec: GraphLogicSpec, seq: list[int], state: GraphLogicAlgorithmState) -> list[tuple[int, int]]:
    if len(seq) < 2:
        return []
    path = [(u, v) if spec.directed else (u, v) for u, v in zip(seq, seq[1:])]
    if all(state.residual.get(edge, 0.0) > 1e-12 for edge in path):
        return path
    if not spec.directed:
        rev_ok = all(state.residual.get(edge, 0.0) > 1e-12 for edge in path)
        if rev_ok:
            return path
    return []


def _dijkstra_relax_orientation(
    spec: GraphLogicSpec,
    state: GraphLogicAlgorithmState,
    edge: tuple[int, int],
) -> tuple[int, int] | None:
    u, v = edge
    candidates = [(u, v)]
    if not spec.directed and u != v:
        candidates.append((v, u))
    finite = [(a, b) for a, b in candidates if math.isfinite(state.dist.get(a, math.inf))]
    if finite:
        return min(finite, key=lambda item: state.dist.get(item[0], math.inf))
    if spec.source is not None and u == spec.source:
        return u, v
    if spec.source is not None and not spec.directed and v == spec.source:
        return v, u
    return None


def _dijkstra_settle_units(
    spec: GraphLogicSpec,
    text: str,
    nodes: set[int],
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ("settle", "settled", "extract", "pop", "minimum", "smallest", "least")):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    unsettled = {
        node: dist for node, dist in state.dist.items() if node in spec.nodes and node not in state.settled_nodes
    }
    if not unsettled:
        return units, actions, invalid
    min_dist = min(unsettled.values())
    for node in sorted(nodes & set(unsettled)):
        actions += 1
        if _close_number(unsettled[node], min_dist):
            state.settled_nodes.add(node)
            units.add(("state_dijkstra_settle", int(node), round(float(unsettled[node]), 6)))
        else:
            invalid += 1
    return units, actions, invalid


def _state_step_path(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    if not any(cue in low for cue in ACTION_CUES["path"] + ACTION_CUES["weighted_path"]):
        return units, actions, invalid

    strict_path_segment = (
        bool(check_final_answer) and _has_answer_assertion_cue(spec, text)
    ) or any(cue in low for cue in ("shortest path", "optimal path", "minimum path", "best path", "dijkstra"))
    weighted_task = spec.task == "weighted_shortest_path"
    weighted_path_evidence = _has_any(text, ACTION_CUES["weighted_path"])
    for seq in _node_sequences(text, spec):
        path_edges = _valid_path_edges(spec, seq)
        if not path_edges:
            invalid += int(strict_path_segment)
            continue
        actions += 1
        path_cost = sum(float(spec.weights.get(edge, 1.0)) for edge in path_edges)
        path_cost_observed = any(_close_number(value, path_cost) for value in _numeric_values(text))
        if weighted_task and path_cost_observed:
            units.add(
                (
                    "state_dijkstra_candidate_path_cost",
                    tuple(path_edges[: min(8, len(path_edges))]),
                    round(float(path_cost), 6),
                )
            )
        if weighted_task and not (weighted_path_evidence or path_cost_observed):
            invalid += int(strict_path_segment or _has_answer_assertion_cue(spec, text))
            continue
        if not _path_is_shortest_prefix(spec, seq, path_edges):
            # For weighted shortest path, a valid but non-shortest candidate path
            # is a common failure mode. Keep the earlier candidate-cost diagnostic
            # if present, but do not turn it into path progress.
            invalid += int(
                strict_path_segment
                or any(cue in low for cue in ("shortest", "best", "minimum", "optimal", "dijkstra"))
            )
            continue
        units.add(("state_path_feasible", tuple(path_edges[: min(8, len(path_edges))])))
        if weighted_task and spec.source is not None and path_cost_observed:
            target_cost = _shortest_distance(spec, int(spec.source), int(seq[-1]))
            if math.isfinite(target_cost) and _close_number(path_cost, target_cost):
                units.add(("state_path_cost_valid", tuple(path_edges[: min(8, len(path_edges))]), round(float(path_cost), 6)))
        for edge in path_edges:
            units.add(("state_path_edge", edge))
        units.add(("state_path_prefix", tuple(path_edges[: min(8, len(path_edges))])))
        if spec.target is not None and seq[-1] == spec.target:
            units.add(("state_path_reaches_target", int(spec.target)))
            if weighted_task and spec.source is not None:
                target_cost = _shortest_distance(spec, int(spec.source), int(spec.target))
                if math.isfinite(target_cost) and _close_number(path_cost, target_cost) and path_cost_observed:
                    units.add(("state_dijkstra_target_cert", int(spec.target), round(float(target_cost), 6)))
                    units.add(("state_weighted_path_final_cert", tuple(path_edges[: min(8, len(path_edges))]), round(float(target_cost), 6)))

    if spec.task == "weighted_shortest_path":
        if any(cue in low for cue in ("dijkstra", "priority queue", "heap", "tentative distance", "initialize", "initialise")):
            actions += 1
            units.add(("state_dijkstra_weighted_rule",))
            if spec.source is not None and any(cue in low for cue in ("source", "start", "0", "infinity", "inf")):
                units.add(("state_dijkstra_init", int(spec.source)))
        if any(cue in low for cue in ("bfs", "fewest edges", "fewest hops", "number of hops", "ignore weight", "unweighted")):
            if not any(cue in low for cue in ("not bfs", "not use bfs", "cannot use bfs", "must use weights", "use weights")):
                invalid += 1

        settle_units, settle_actions, settle_invalid = _dijkstra_settle_units(spec, text, nodes, state)
        units.update(settle_units)
        actions += settle_actions
        invalid += settle_invalid

        relax_cues = ("relax", "update", "tentative", "shorter", "distance", "dist", "cost")

        def apply_relax(u: int, v: int, edge: tuple[int, int], observed_value: float | None) -> None:
            nonlocal actions, invalid
            actions += 1
            base = state.dist.get(u, math.inf)
            if not math.isfinite(base):
                invalid += 1
                return
            cand = base + float(spec.weights.get(edge, 1.0))
            old = state.dist.get(v, math.inf)
            explicit_value_ok = observed_value is not None and _close_number(observed_value, cand)
            if cand + 1e-9 < old and explicit_value_ok:
                units.add(("state_dijkstra_scan_edge", int(u), int(v), round(float(spec.weights.get(edge, 1.0)), 6)))
                state.dist[v] = cand
                state.parent_node[v] = u
                units.add(("state_dijkstra_decreased_dist", int(u), int(v), round(float(cand), 6)))
                units.add(("state_dijkstra_relax", int(u), int(v), round(float(cand), 6)))
                units.add(("state_dijkstra_weighted_relax_cert", int(u), int(v), round(float(spec.weights.get(edge, 1.0)), 6), round(float(cand), 6)))
                units.add(("state_dijkstra_dist", int(v), round(float(cand), 6)))
                if spec.target is not None and int(v) == int(spec.target):
                    target_cost = _shortest_distance(spec, int(spec.source), int(spec.target)) if spec.source is not None else math.inf
                    if math.isfinite(target_cost) and _close_number(cand, target_cost):
                        units.add(("state_dijkstra_target_cert", int(v), round(float(cand), 6)))
            elif old < math.inf and cand + 1e-9 >= old and observed_value is not None and (
                _close_number(observed_value, old) or _close_number(observed_value, cand)
            ):
                units.add(("state_dijkstra_scan_edge", int(u), int(v), round(float(spec.weights.get(edge, 1.0)), 6)))
                units.add(("state_dijkstra_checked", int(u), int(v)))
                units.add(("state_dijkstra_no_relax", int(u), int(v), round(float(old), 6)))
            elif old < math.inf and _close_number(
                old,
                _shortest_distance(spec, spec.source, v) if spec.source is not None else old,
            ) and observed_value is None:
                units.add(("state_dijkstra_scan_edge", int(u), int(v), round(float(spec.weights.get(edge, 1.0)), 6)))
                units.add(("state_dijkstra_checked", int(u), int(v)))
            else:
                invalid += 1

        for edge in sorted(valid_edges):
            contexts = _edge_contexts(text, edge, spec.directed)
            context_has_relax = any(any(cue in ctx for cue in relax_cues) for ctx in contexts)
            if not context_has_relax:
                continue
            oriented = _dijkstra_relax_orientation(spec, state, edge)
            if oriented is None:
                invalid += 1
                continue
            u, v = oriented
            context_text = "\n".join(contexts)
            update_values = _distance_update_values_for_node(context_text, v)
            observed = update_values[0] if update_values else None
            apply_relax(u, v, edge, observed)

        for u, v, observed in _dijkstra_relax_mentions(spec, text):
            edge = _norm_edge(u, v, spec.directed)
            if edge not in spec.edges:
                invalid += 1
                continue
            apply_relax(u, v, edge, observed)

        if any(cue in low for cue in ("distance", "dist", "cost", "best", "shortest", "tentative")):
            for node in sorted(nodes - ({spec.source} if spec.source is not None else set())):
                values = _distance_update_values_for_node(text, node)
                if not values or spec.source is None:
                    continue
                actions += 1
                target_dist = _shortest_distance(spec, spec.source, node)
                if math.isfinite(target_dist) and any(_close_number(value, target_dist) for value in values):
                    old = state.dist.get(node, math.inf)
                    if target_dist + 1e-9 < old:
                        state.dist[node] = target_dist
                    units.add(("state_dijkstra_dist_observed", int(node), round(float(target_dist), 6)))
                else:
                    invalid += 1
        if any(cue in low for cue in ("predecessor", "parent", "prev", "previous")):
            for child, parent in _predecessor_mentions(text, spec):
                actions += 1
                target_parent = _shortest_path_parent(spec, child)
                if target_parent is not None and parent == target_parent:
                    state.parent_node[child] = parent
                    units.add(("state_dijkstra_predecessor", int(parent), int(child)))
                else:
                    invalid += 1
    else:
        for edge in sorted(valid_edges & set(spec.target_edge_order or [])):
            actions += 1
            units.add(("state_path_edge", edge))
    return units, actions, invalid


def _state_step_traversal(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["traversal"]):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    order = list(spec.target_edge_order or [])
    for edge in sorted(valid_edges):
        if edge not in spec.edges:
            continue
        actions += 1
        if state.order_pos < len(order) and edge == order[state.order_pos]:
            state.order_pos += 1
            state.visited_nodes.update(edge)
            units.add(("state_traversal_order_edge", edge, state.order_pos))
        elif edge in set(order):
            state.visited_nodes.update(edge)
            units.add(("state_traversal_tree_edge", edge))
        else:
            invalid += 1
    if any(cue in low for cue in ("visit", "visited", "discover", "enqueue", "dequeue", "push", "pop", "process", "mark")):
        target_node_order = list(spec.target_node_order or [])
        for node in sorted(nodes & set(spec.target_nodes or spec.nodes)):
            actions += 1
            if node not in state.visited_nodes:
                state.visited_nodes.add(node)
                if target_node_order and state.node_order_pos < len(target_node_order):
                    expected = int(target_node_order[state.node_order_pos])
                    if int(node) == expected:
                        state.node_order_pos += 1
                        units.add(("state_traversal_order_node", spec.task, int(node), int(state.node_order_pos)))
                    else:
                        units.add(("state_traversal_visit", int(node)))
                else:
                    units.add(("state_traversal_visit", int(node)))
            else:
                units.add(("state_traversal_seen", int(node)))
    order_units, order_actions, order_invalid = _traversal_parseable_and_order_units(spec, text, state)
    units.update(order_units)
    actions += order_actions
    invalid += order_invalid
    return units, actions, invalid


def _mst_edge_is_current_safe(spec: GraphLogicSpec, state: GraphLogicAlgorithmState, edge: tuple[int, int]) -> bool:
    u, v = edge
    if _state_find(state, u) == _state_find(state, v):
        return False
    candidates = [
        e
        for e in spec.edges
        if e not in state.accepted_edges and _state_find(state, e[0]) != _state_find(state, e[1])
    ]
    if not candidates:
        return False
    min_weight = min(float(spec.weights.get(e, 1.0)) for e in candidates)
    return _close_number(float(spec.weights.get(edge, 1.0)), min_weight)


def _state_step_mst(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["mst"]) and not _has_answer_assertion_cue(spec, text):
        return set(), 0, 0
    accept_cues = ("add", "include", "select", "choose", "take", "pick", "accept", "safe edge", "union")
    cycle_reject_cues = (
        "cycle",
        "loop",
        "same component",
        "already connected",
        "forms a cycle",
        "form a cycle",
        "creates a cycle",
        "would create a cycle",
    )
    reject_cues = (
        "skip",
        "reject",
        "discard",
        *cycle_reject_cues,
    )
    consider_cues = ("consider", "check", "weight", "lightest", "minimum", "sorted", "candidate")
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    if any(cue in low for cue in ("sort", "sorted", "ascending", "in increasing", "by weight", "lightest first")):
        actions += 1
        units.add(("state_mst_weight_order",))
    if spec.task == "weighted_minimum_spanning_tree":
        if any(cue in low for cue in ("weight", "weighted", "kruskal", "prim", "lightest", "cheapest")):
            actions += 1
            units.add(("state_mst_weighted_rule",))
        if len(spec.nodes) > 0 and any(cue in low for cue in ("n-1", "|v|-1", "number of edges", "edges in the tree")):
            if any(_close_number(value, max(0, len(spec.nodes) - 1)) for value in _numeric_values(text)):
                units.add(("state_mst_edge_count_target", max(0, len(spec.nodes) - 1)))
    wrong_spanning_tree_claim = any(
        cue in low for cue in ("bfs tree", "dfs tree", "any spanning tree", "any tree", "ignore weight", "unweighted")
    )
    if wrong_spanning_tree_claim and not any(
        cue in low for cue in ("not any", "not just", "must use weight", "by weight", "minimum weight")
    ):
        # Frequent false-positive mode: the response claims a traversal tree or
        # arbitrary spanning tree is enough, instead of producing a verifiable
        # MST edge-list certificate. Valid final edge-list certificates can
        # still earn positive units later in this function.
        if spec.task == "weighted_minimum_spanning_tree" or _has_answer_assertion_cue(spec, text):
            actions += 1
            invalid += 1
            units.add(("state_mst_any_spanning_tree_claim", spec.task))
    final_units_early: set[tuple] = set()
    final_invalid_early = 0
    final_answer_edge_order = _final_answer_edge_order(text, spec)
    final_edge_list_only = bool(final_answer_edge_order and _has_answer_assertion_cue(spec, text)) and not any(
        cue in low
        for cue in (
            "consider",
            "check",
            "skip",
            "reject",
            "same component",
            "forms a cycle",
            "would create a cycle",
            "union",
            "safe edge",
            "next edge",
            "then add",
            "process",
        )
    )
    if final_edge_list_only:
        # A final MST edge list is a certificate object, not necessarily a
        # chronological Kruskal/Prim trace. Do not replay each edge as an
        # accept action; otherwise correct final lists can be over-penalized
        # for lacking local safe-edge wording.
        final_units_early, final_invalid_early = _mst_parseable_object_units(spec, text, check_final_answer)
        units.update(final_units_early)
        invalid += final_invalid_early
        if final_units_early or final_invalid_early:
            actions = max(actions, 1)
            return units, actions, invalid
    ordered_valid_edges = [edge for edge in _edge_order_from_text(text, spec) if edge in valid_edges]
    ordered_valid_edges.extend(edge for edge in sorted(valid_edges) if edge not in set(ordered_valid_edges))
    list_accept_cue = bool(
        len(valid_edges) > 1
        and any(
            cue in low
            for cue in (
                "selected edges",
                "accepted edges",
                "mst edges",
                "tree edges",
                "spanning tree edges",
                "choose edges",
                "choose the edges",
                "select edges",
                "select the edges",
                "take edges",
                "take the edges",
                "add edges",
                "add the edges",
                "include edges",
                "include the edges",
            )
        )
        and not any(cue in low for cue in ("candidate edges", "consider the edges", "sort the edges", "reject", "skip"))
    )
    for edge in ordered_valid_edges:
        if edge not in spec.edges:
            continue
        contexts = _edge_contexts(text, edge, spec.directed) or [text]
        edge_weight_checked = any("weight" in ctx for ctx in contexts)
        if edge_weight_checked:
            units.add(("state_mst_edge_weight_checked", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
        context_accept = _edge_context_has_any(text, edge, spec, accept_cues)
        global_single_edge_accept = len(valid_edges) == 1 and any(cue in low for cue in accept_cues)
        has_accept = context_accept or global_single_edge_accept or (list_accept_cue and edge in valid_edges)
        has_reject = _edge_context_has_any(text, edge, spec, reject_cues)
        has_consider = _edge_context_has_any(text, edge, spec, consider_cues)
        if has_accept:
            actions += 1
            target_edges = set(spec.target_edges or set())
            safe_now = _mst_edge_is_current_safe(spec, state, edge)
            is_target_forest_edge = edge in target_edges and _state_find(state, edge[0]) != _state_find(state, edge[1])
            weighted_accept_evidence = (
                edge_weight_checked
                or "state_mst_weight_order" in {str(unit[0]) for unit in state.seen_units}
                or "state_mst_weighted_rule" in {str(unit[0]) for unit in state.seen_units}
                or any(cue in low for cue in ("weight", "weighted", "lightest", "cheapest", "minimum", "kruskal", "prim", "safe"))
            )
            if spec.task == "weighted_minimum_spanning_tree":
                safe_edge_allowed = safe_now and weighted_accept_evidence
                target_edge_allowed = False
            else:
                safe_edge_allowed = safe_now
                target_edge_allowed = is_target_forest_edge
            if edge not in state.accepted_edges and (safe_edge_allowed or target_edge_allowed):
                _state_union(state, edge[0], edge[1])
                state.accepted_edges.add(edge)
                state.total_weight += float(spec.weights.get(edge, 1.0))
                units.add(("state_mst_accept", edge, round(float(state.total_weight), 6)))
                if safe_now:
                    units.add(("state_mst_safe_edge", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
                if is_target_forest_edge:
                    units.add(("state_mst_target_edge", edge))
                order = list(spec.target_edge_order or [])
                if state.order_pos < len(order) and edge == order[state.order_pos]:
                    state.order_pos += 1
                    units.add(("state_mst_kruskal_prefix_edge", edge, int(state.order_pos)))
                units.add(("state_mst_component_count", len({_state_find(state, n) for n in spec.nodes})))
            else:
                invalid += 1
            if spec.task == "weighted_minimum_spanning_tree" and not weighted_accept_evidence:
                invalid += 1
        elif has_reject:
            actions += 1
            has_cycle_reject = _edge_context_has_any(text, edge, spec, cycle_reject_cues)
            if _state_find(state, edge[0]) == _state_find(state, edge[1]) and has_cycle_reject:
                state.rejected_edges.add(edge)
                units.add(("state_mst_reject_cycle", edge))
                units.add(("state_mst_cycle_reject_cert", edge))
            elif has_cycle_reject:
                invalid += 1
        elif has_consider:
            actions += 1
            units.add(("state_mst_consider", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
        elif spec.task == "weighted_minimum_spanning_tree" and edge_weight_checked:
            actions += 1
            units.add(("state_mst_consider", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
        elif len(valid_edges) <= 3 and any(cue in low for cue in consider_cues + ("edge", "connection", "link")):
            actions += 1
            units.add(("state_mst_consider", edge, round(float(spec.weights.get(edge, 1.0)), 6)))
    target_edge_count = max(0, len(spec.nodes) - 1)
    if target_edge_count > 0 and len(state.accepted_edges) == target_edge_count:
        roots = {_state_find(state, int(node)) for node in spec.nodes}
        if len(roots) == 1:
            units.add(("state_mst_n_minus_one_edges", target_edge_count))
            units.add(("state_mst_connected_acyclic", tuple(sorted(state.accepted_edges))[:12]))
            units.add(("state_mst_spanning_nodes_covered", tuple(sorted(spec.nodes))[:12]))
            if spec.task == "weighted_minimum_spanning_tree" and any(_close_number(value, float(state.total_weight)) for value in _numeric_values(text)):
                units.add(("state_mst_final_weight_checked", round(float(state.total_weight), 6)))
            if spec.task == "weighted_minimum_spanning_tree" and spec.target_edges:
                optimal_weight = _mst_optimal_weight(spec)
                if _close_number(float(state.total_weight), float(optimal_weight)):
                    units.add(("state_mst_optimal_weight_cert", round(float(state.total_weight), 6)))
    if any(
        cue in low
        for cue in (
            "mst edges",
            "tree edges",
            "spanning tree edges",
            "selected edges",
            "accepted edges",
            "current tree",
            "final tree",
            "the tree is",
            "spanning tree is",
            "final answer",
            "answer is",
            "\\boxed",
        )
    ):
        cert_order = _edge_order_from_text(text, spec)
        cert_edges = set(cert_order)
        target_edge_count = max(0, len(spec.nodes) - 1)
        if cert_order:
            units.add(("parseable_edge_list_answer", spec.task, tuple(cert_order[:12])))
        if cert_edges and len(cert_order) == target_edge_count and len(cert_edges) == target_edge_count:
            valid_cert, cert_weight = _spanning_tree_valid(spec, cert_edges)
            if valid_cert:
                units.add(("state_mst_n_minus_one_edges", target_edge_count))
                units.add(("state_mst_connected_acyclic", tuple(sorted(cert_edges))[:12]))
                units.add(("state_mst_spanning_nodes_covered", tuple(sorted({node for edge in cert_edges for node in edge}))[:12]))
                if spec.task == "weighted_minimum_spanning_tree" and any(_close_number(value, float(cert_weight)) for value in _numeric_values(text)):
                    units.add(("state_mst_final_weight_checked", round(float(cert_weight), 6)))
                if spec.task == "weighted_minimum_spanning_tree" and spec.target_edges:
                    optimal_weight = _mst_optimal_weight(spec)
                    if _close_number(float(cert_weight), float(optimal_weight)):
                        units.add(("state_mst_optimal_weight_cert", round(float(cert_weight), 6)))

    final_units, final_invalid = _mst_parseable_object_units(spec, text, check_final_answer)
    units.update(final_units)
    invalid += final_invalid
    if (final_units or final_invalid) and actions <= 0:
        actions = 1
    return units, actions, invalid


def _state_step_flow(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if spec.source is None or spec.target is None:
        return set(), 0, 0
    if not any(cue in low for cue in ACTION_CUES["flow"] + ACTION_CUES["path"]):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    augment_cue = any(
        cue in low
        for cue in (
            "augment",
            "augmenting",
            "send",
            "send flow",
            "push",
            "push flow",
            "increase",
            "increase flow",
            "add flow",
            "route flow",
            "augment path",
            "augment the path",
            "send along",
            "push along",
            "send through",
            "push through",
        )
    )
    bottleneck_cue = any(
        cue in low
        for cue in (
            "bottleneck",
            "minimum capacity",
            "min capacity",
            "minimum residual",
            "min residual",
            "limiting capacity",
            "limited by",
        )
    )
    residual_update_cue = any(
        cue in low
        for cue in (
            "residual",
            "remaining capacity",
            "remaining residual",
            "remaining cap",
            "capacity left",
            "capacity remains",
            "capacity leftover",
            "capacity after",
            "capacity becomes",
            "capacity is now",
            "capacity now",
            "leftover capacity",
            "reduce capacity",
            "decrease capacity",
            "reduced capacity",
            "decreased capacity",
            "reverse edge",
            "reverse capacity",
            "reverse residual",
            "reverse arc",
            "backward arc",
            "back edge",
            "backward edge",
            "backward residual",
            "return edge",
            "update residual",
            "residual update",
            "updated residual",
            "update capacities",
            "updated capacities",
            "residual becomes",
            "residual is now",
            "residual now",
            "residual left",
            "add reverse",
            "add back",
            "increase reverse",
            "increase backward",
            "subtract",
            "subtracting",
            "after sending",
            "after pushing",
            "after augmenting",
        )
    )
    no_path_cue = any(
        cue in low
        for cue in (
            "no path",
            "no s-t path",
            "no st path",
            "not reachable",
            "cannot reach",
            "can't reach",
            "no more path",
            "no more paths",
            "no additional path",
            "no other path",
            "no augmenting path",
            "no augmenting paths",
            "no residual path",
            "residual graph has no path",
        )
    )
    if no_path_cue and state.max_flow_value <= 1e-12:
        actions += 1
        units.add(("state_flow_complete", 0.0))
        units.add(("state_flow_min_cut_cert", 0.0))
    if any(cue in low for cue in ("residual capacity", "residual network", "residual graph", "reverse edge", "back edge")):
        actions += 1
        units.add(("state_flow_residual_network",))
    if any(cue in low for cue in ("subtract", "decrease residual", "add reverse", "increase reverse", "update residual")):
        actions += 1
        units.add(("state_flow_residual_update",))
    if any(cue in low for cue in ("flow conservation", "incoming flow", "outgoing flow", "conservation")):
        actions += 1
        units.add(("state_flow_conservation_scope",))
        for node in sorted(_numbers_in_text(text, spec.nodes) - {int(spec.source), int(spec.target)}):
            units.add(("state_flow_conservation_check", int(node)))
    if any(cue in low for cue in ("edge-disjoint", "edge disjoint", "disjoint paths", "just add path", "sum the bottleneck")):
        if ("residual" not in low and "reverse" not in low) or any(
            cue in low for cue in ("without residual", "ignore residual", "no residual")
        ):
            invalid += 1
    for seq in _node_sequences(text, spec):
        if seq[0] != spec.source or seq[-1] != spec.target:
            continue
        actions += 1
        path = _valid_state_flow_path_edges(spec, seq, state)
        original_path = _valid_path_edges(spec, seq)
        amount_values = _flow_amount_values(text)
        explicit_bottleneck_value = bool(
            re.search(r"(?:bottleneck|min(?:imum)?\s+capacity)\D{0,30}(?:is|=|:)\D{0,12}[-+]?\d", text, flags=re.I)
        )
        if not path:
            if original_path:
                if augment_cue and (
                    not residual_update_cue
                    or any(cue in low for cue in ("without residual", "ignore residual", "no residual"))
                ):
                    invalid += 1
                continue
            invalid += 1
            continue
        bottleneck = min(state.residual.get(edge, 0.0) for edge in path)
        units.add(("state_flow_path", tuple(path)))
        for edge in path:
            units.add(("state_flow_path_edge", edge))
        amount_ok = any(_close_number(value, bottleneck) for value in amount_values)
        flow_after = float(state.flow_value) + float(bottleneck)
        flow_value_update_values = _flow_value_update_values(text)
        flow_value_update_ok = any(_close_number(value, flow_after) for value in flow_value_update_values)
        if bottleneck_cue and (amount_ok or not explicit_bottleneck_value):
            units.add(("state_flow_bottleneck", tuple(path), round(float(bottleneck), 6)))
        elif bottleneck_cue and explicit_bottleneck_value and not amount_ok:
            invalid += 1
        inferred_residual_augment = (
            augment_cue
            and bottleneck_cue
            and residual_update_cue
            and not explicit_bottleneck_value
            and bottleneck > 1e-12
        )
        # A total-flow statement alone is too weak: previous runs showed that
        # LLMs can guess/increment the scalar while not maintaining the residual
        # ledger. Treat it as strong evidence only when paired with either an
        # explicit bottleneck amount or a residual update cue.
        strong_flow_value_update_ok = flow_value_update_ok and (amount_ok or residual_update_cue)
        amount_or_value_ok = amount_ok or inferred_residual_augment or strong_flow_value_update_ok
        update_cert_ok = residual_update_cue or strong_flow_value_update_ok
        if augment_cue and update_cert_ok and amount_or_value_ok and bottleneck > 1e-12:
            for u, v in path:
                state.residual[(u, v)] = state.residual.get((u, v), 0.0) - bottleneck
                state.residual[(v, u)] = state.residual.get((v, u), 0.0) + bottleneck
            state.flow_value += bottleneck
            units.add(("state_flow_valid_augment", tuple(path), round(float(bottleneck), 6)))
            if residual_update_cue and (amount_ok or inferred_residual_augment or flow_value_update_ok):
                units.add(("state_flow_residual_bottleneck_cert", tuple(path), round(float(bottleneck), 6)))
            if inferred_residual_augment or (strong_flow_value_update_ok and not residual_update_cue):
                units.add(("state_flow_valid_augment_inferred", tuple(path), round(float(bottleneck), 6)))
            if residual_update_cue:
                units.add(("state_flow_residual_update_after_augment", tuple(path), round(float(bottleneck), 6)))
            if strong_flow_value_update_ok:
                units.add(("state_flow_value_update_cert", round(float(state.flow_value), 6)))
            units.add(("state_flow_augment", tuple(path), round(float(bottleneck), 6)))
            units.add(("state_flow_value", round(float(state.flow_value), 6)))
        elif augment_cue and not residual_update_cue:
            invalid += 1
        elif augment_cue:
            invalid += 1
    if any(
        cue in low
        for cue in (
            "current flow",
            "flow value becomes",
            "flow value is now",
            "total flow so far",
            "new flow value",
        )
    ):
        actions += 1
        values = _numeric_values(text)
        if any(_close_number(value, state.flow_value) for value in values):
            units.add(("state_flow_value", round(float(state.flow_value), 6)))
        elif values:
            invalid += 1
    if valid_edges and any(cue in low for cue in ("capacity", "residual", "cut", "flow")):
        for edge in valid_edges:
            actions += 1
            cap = float(spec.weights.get(edge, 1.0))
            ctx_values = _context_values_for_edge(text, edge, spec)
            if not ctx_values or any(_close_number(value, cap) for value in ctx_values):
                units.add(("state_flow_capacity", edge, round(float(cap), 6)))
                if ctx_values and any(_close_number(value, cap) for value in ctx_values):
                    units.add(("state_flow_capacity_checked", edge, round(float(cap), 6)))
            contexts = _edge_contexts(text, edge, spec.directed) or [text]
            if any("residual" in ctx for ctx in contexts):
                units.add(("state_flow_residual_edge", edge))
                residual_cap = float(state.residual.get(edge, 0.0))
                if residual_cap > 1e-12:
                    units.add(("state_flow_residual_available", edge, round(residual_cap, 6)))
            if any(any(cue in ctx for cue in ("reverse edge", "back edge", "backward edge", "reverse capacity")) for ctx in contexts):
                units.add(("state_flow_reverse_edge", edge))
                rev_edge = (edge[1], edge[0])
                reverse_residual = float(state.residual.get(rev_edge, 0.0))
                if reverse_residual > 1e-12:
                    units.add(("state_flow_residual_available", rev_edge, round(reverse_residual, 6)))
    if any(cue in low for cue in ("residual network", "residual graph", "reverse edge", "back edge", "backward edge")):
        actions += 1
        units.add(("state_flow_residual_network",))
    if any(cue in low for cue in ("level graph", "bfs level", "blocking flow", "dinic")):
        actions += 1
        units.add(("state_flow_level_graph",))
    if any(cue in low for cue in ("source side", "reachable from source", "s side", "cut capacity", "min cut", "minimum cut")):
        actions += 1
        units.add(("state_flow_cut_scope",))
        values = _numeric_values(text)
        if state.max_flow_value > 1e-12 and any(_close_number(value, state.max_flow_value) for value in values):
            units.add(("state_flow_complete", round(float(state.max_flow_value), 6)))
            units.add(("state_flow_min_cut_cert", round(float(state.max_flow_value), 6)))
    if no_path_cue:
        actions += 1
        if _close_number(state.flow_value, state.max_flow_value):
            units.add(("state_flow_complete", round(float(state.max_flow_value), 6)))
            units.add(("state_flow_min_cut_cert", round(float(state.max_flow_value), 6)))
        elif state.max_flow_value > 1e-12 and any(_close_number(value, state.max_flow_value) for value in _numeric_values(text)):
            units.add(("state_flow_complete", round(float(state.max_flow_value), 6)))
            units.add(("state_flow_min_cut_cert", round(float(state.max_flow_value), 6)))
        else:
            invalid += 1
    if check_final_answer and spec.answer_numbers and any(
        cue in low for cue in ("maximum flow", "max flow", "total flow", "flow value")
    ):
        values = _numeric_values(text)
        target = float(spec.answer_numbers[0])
        actions += 1
        if any(_close_number(value, target) for value in values):
            units.add(("state_flow_complete", round(float(target), 6)))
        elif values:
            invalid += 1
    return units, actions, invalid


def _state_potential_view(state: GraphLogicAlgorithmState) -> GraphLogicPotentialState:
    return GraphLogicPotentialState(
        seen_edges=set(),
        seen_nodes=set(),
        edge_positions=[],
        seen_evidence=set(state.seen_units),
    )


def _trace_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_trace_value(item) for item in value]
    if isinstance(value, list):
        return [_trace_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_trace_value(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {str(key): _trace_value(item) for key, item in value.items()}
    return value


def _trace_units(units: Iterable[tuple]) -> list:
    return sorted((_trace_value(unit) for unit in units), key=repr)


def _trace_unit_names(state_step_trace: dict[str, Any], key: str) -> set[str]:
    values = state_step_trace.get(key) if isinstance(state_step_trace, dict) else None
    if not isinstance(values, list):
        return set()
    names: set[str] = set()
    for item in values:
        if isinstance(item, (list, tuple)) and item:
            names.add(str(item[0]))
        elif isinstance(item, str):
            names.add(item)
    return names


def _freeze_trace_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_trace_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_trace_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_trace_value(item) for item in value), key=repr))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_trace_value(item)) for key, item in value.items()))
    return value


def _trace_units_for_keys(state_step_trace: dict[str, Any], *keys: str) -> set[tuple]:
    out: set[tuple] = set()
    if not isinstance(state_step_trace, dict):
        return out
    for key in keys:
        values = state_step_trace.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, (list, tuple)) and item:
                frozen = _freeze_trace_value(item)
                if isinstance(frozen, tuple) and frozen:
                    out.add(frozen)
    return out


def _state_segment_hard_cert_units(state_step_trace: dict[str, Any]) -> set[str]:
    names = _trace_unit_names(state_step_trace, "rewardable_unit_names") | _trace_unit_names(
        state_step_trace, "new_unit_names"
    )
    return names & STATE_HARD_CERT_MILESTONE_UNIT_NAMES


def _state_segment_graph_sensitive_units(state_step_trace: dict[str, Any]) -> set[str]:
    # This is intentionally stricter than a name-only gate: a segment should
    # pass the graph-sensitivity gate only when the trace contains a concrete
    # graph payload (edge/path/pair/value/signature), not merely a rewardable
    # unit name. The counterfactual audit still measures the true sensitivity,
    # but this prevents name-only scaffold events from bypassing the gate.
    sensitive: set[str] = set()
    for unit in _trace_units_for_keys(state_step_trace, "rewardable_units", "new_units"):
        name = str(unit[0])
        if name not in STATE_GRAPH_SENSITIVE_UNIT_NAMES:
            continue
        if len(unit) <= 1:
            continue
        sensitive.add(name)
    return sensitive


def _count_trace_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _state_segment_ambiguity_score(
    segment_grounded: bool,
    state_mention: bool,
    candidate_edges: set[tuple[int, int]],
    candidate_nodes: set[int],
    state_step_stats: dict[str, float],
    state_step_trace: dict[str, Any],
    duplicate_no_progress: bool,
) -> float:
    """Executor-visible ambiguity proxy used to suppress weak positive reward.

    This intentionally avoids lexical confidence cues. It measures whether the
    segment maps to too many possible actions, mostly weak evidence, invalid
    transitions, or repeated no-progress units.
    """

    actions = float(state_step_stats.get("state_actions", 0.0))
    invalid = float(state_step_stats.get("state_invalid_transitions", 0.0))
    rewardable = float(state_step_stats.get("state_rewardable_new_units", 0.0))
    weak = float(state_step_stats.get("state_weak_new_units", 0.0))
    event_count = _count_trace_items(state_step_trace.get("action_events") if isinstance(state_step_trace, dict) else [])
    candidate_count = len(candidate_edges) + len(candidate_nodes)
    weak_only = rewardable <= 0.0 and weak > 0.0

    ambiguity = 0.0
    ambiguity += 0.16 * max(0.0, actions - 1.0)
    ambiguity += 0.10 * max(0, candidate_count - 1)
    ambiguity += 0.08 * max(0, event_count - 2)
    ambiguity += 0.30 * float(weak_only)
    ambiguity += 0.45 * min(1.0, invalid)
    ambiguity += 0.20 * float(duplicate_no_progress)
    if (not segment_grounded) and state_mention:
        ambiguity += 0.30
    return max(0.0, min(1.0, ambiguity))


def _state_executable_commitment_score(
    segment_grounded: bool,
    executor_valid: bool,
    state_consistent: bool,
    state_reward_consistent: bool,
    state_step_stats: dict[str, float],
    state_step_trace: dict[str, Any],
    ambiguity: float,
) -> float:
    """Commitment score based on executable evidence rather than wording."""

    rewardable = float(state_step_stats.get("state_rewardable_new_units", 0.0))
    state_delta = max(0.0, float(state_step_stats.get("state_delta", 0.0)))
    negative_delta = max(0.0, float(state_step_stats.get("state_negative_delta", 0.0)))
    action_events = _count_trace_items(state_step_trace.get("action_events") if isinstance(state_step_trace, dict) else [])
    hard_cert = bool(_state_segment_hard_cert_units(state_step_trace))

    score = 0.0
    score += 0.12 * float(segment_grounded)
    score += 0.12 * float(executor_valid)
    score += 0.16 * float(state_consistent)
    score += 0.16 * float(state_reward_consistent)
    score += 0.18 * min(1.0, rewardable / 2.0)
    score += 0.18 * min(1.0, state_delta / 0.01)
    score += 0.12 * float(hard_cert)
    score += 0.08 * min(1.0, action_events / 2.0)
    score -= 0.45 * max(0.0, min(1.0, ambiguity))
    score -= 0.20 * min(1.0, negative_delta / 0.01)
    return max(0.0, min(1.0, score))


def _state_step_ordering(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["ordering"] + ACTION_CUES["path"]):
        return set(), 0, 0
    if spec.task == "traveling_salesman_problem":
        if not check_final_answer:
            return set(), 0, 0
        final_units, final_invalid = _final_answer_validation_units(spec, text)
        actions = 1 if final_units or final_invalid else 0
        return final_units, actions, final_invalid
    units = set()
    units.update(_path_process_units(spec, text, valid_edges, nodes))
    units.update(_hamiltonian_units(spec, text))
    actions = len(_node_sequences(text, spec)) + len(valid_edges)
    invalid = 0
    if spec.task == "topological_sort" and spec.target_node_order:
        order = list(spec.target_node_order)
        rank = {node: i for i, node in enumerate(order)}
        for seq in _node_sequences(text, spec):
            last = -1
            for node in seq:
                if node not in rank or rank[node] <= last:
                    invalid += 1
                    break
                last = rank[node]
    elif spec.task in {"hamiltonian_path", "traveling_salesman_problem"}:
        for seq in _node_sequences(text, spec):
            if len(set(seq)) != len(seq):
                invalid += 1
            path_edges = _valid_path_edges(spec, seq)
            if len(seq) >= 2 and not path_edges:
                invalid += 1
    if spec.task == "topological_sort":
        final_units, final_invalid = _topological_parseable_object_units(spec, text)
    else:
        final_units, final_invalid = _final_answer_validation_units(spec, text) if check_final_answer else (set(), 0)
    units.update(final_units)
    invalid += final_invalid
    if (final_units or final_invalid) and actions <= 0:
        actions = 1
    return units, actions, invalid


def _state_step_distance(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "distance",
            "dist",
            "shortest",
            "path",
            "reachable",
            "unreachable",
            "eccentricity",
            "centrality",
            "efficiency",
            "wiener",
            "sum",
            "total",
            "radius",
            "diameter",
            "center",
            "periphery",
            "barycenter",
            "normaliz",
            "reciprocal",
            "1/d",
        )
    ):
        return set(), 0, 0
    answer_hit = _scalar_answer_hit(text, spec) if check_final_answer else False
    units = set()
    units.update(_distance_pair_units(spec, text, valid_edges, nodes))
    units.update(_distance_metric_units(spec, text))
    units.update(_distance_formula_units(spec, text))
    units.update(_global_metric_formula_units(spec, text, answer_hit))
    units.update(_betweenness_units(spec, text))
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching", "min_edge_covering"}:
        final_units, final_invalid = _set_parseable_edge_answer_units(spec, text, check_final_answer)
    else:
        final_units, final_invalid = _final_answer_validation_units(spec, text) if check_final_answer else (set(), 0)
    units.update(final_units)
    actions = len(units)
    if not actions and (nodes or valid_edges or answer_hit):
        actions = 1
    if (final_units or final_invalid) and actions <= 0:
        actions = 1
    invalid = _distance_process_invalid_count(spec, text)
    if check_final_answer:
        invalid += int(
            _numeric_final_invalid(spec, text)
            or _numeric_final_missing_invalid(spec, text)
            or _distance_definition_invalid(spec, text)
        )
    invalid += final_invalid
    return units, actions, invalid


def _state_step_neighborhood(
    spec: GraphLogicSpec,
    text: str,
    nodes: set[int],
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "neighbor",
            "neighbour",
            "common",
            "degree",
            "coefficient",
            "index",
            "jaccard",
            "adar",
            "allocation",
            "triangle",
            "density",
            "count",
            "number",
            "edge",
            "node",
            "vertex",
        )
    ):
        return set(), 0, 0
    answer_hit = _scalar_answer_hit(text, spec) if check_final_answer else False
    units = set()
    units.update(_neighborhood_units(spec, text, nodes))
    units.update(_degree_units(spec, text))
    units.update(_count_formula_units(spec, text, answer_hit))
    units.update(_triangle_units(spec, text))
    actions = len(units)
    if not actions and (nodes or answer_hit):
        actions = 1
    invalid = (
        int(_numeric_final_invalid(spec, text) or _numeric_final_missing_invalid(spec, text))
        if check_final_answer
        else 0
    )
    return units, actions, invalid


def _state_step_set(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["set"]):
        return set(), 0, 0
    view = _state_potential_view(state)
    units = set()
    units.update(_set_feasibility_units(spec, text, valid_edges, nodes, view))
    units.update(_set_size_units(spec, text))
    final_units, final_invalid = _final_answer_validation_units(spec, text) if check_final_answer else (set(), 0)
    units.update(final_units)
    actions = len(valid_edges) + len(nodes & spec.nodes) if _has_selection_cue(text) else len(units)
    invalid = 0
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching"}:
        selected_edges = [unit[1] for unit in units if unit and unit[0] == "matching_edge"]
        endpoints: set[int] = set()
        for edge in selected_edges:
            if edge[0] in endpoints or edge[1] in endpoints:
                invalid += 1
            endpoints.update(edge)
    if spec.task == "min_vertex_cover" and any(cue in low for cue in ("cover", "covered", "vertex cover")):
        selected_nodes = (_selected_node_mentions(text, spec.nodes) or nodes) & spec.nodes
        if selected_nodes:
            uncovered = [edge for edge in spec.edges if edge[0] not in selected_nodes and edge[1] not in selected_nodes]
            complete_cover_cues = (
                "cover all",
                "covers all",
                "covered all",
                "covers every",
                "cover every",
                "valid cover",
                "complete cover",
                "all edges are covered",
                "every edge",
            )
            if uncovered and any(cue in low for cue in complete_cover_cues):
                invalid += min(3, len(uncovered))
            target_size = len(spec.target_nodes or set())
            if target_size and len(selected_nodes) > target_size and any(
                cue in low for cue in ("minimum", "minimum vertex cover", "min vertex cover", "smallest", "optimal")
            ):
                invalid += 1
    if spec.task == "min_edge_covering" and valid_edges:
        target_size = len(spec.target_edges or set())
        if target_size and len(valid_edges) > target_size and any(
            cue in low for cue in ("minimum", "minimum edge cover", "min edge cover", "smallest", "optimal")
        ):
            invalid += 1
        if any(cue in low for cue in ("cover all", "covers all", "covered all", "every vertex", "all nodes")):
            covered = {node for edge in valid_edges for node in edge}
            if set(spec.nodes) <= covered:
                units.add(("edge_set_answer_feasible", spec.task, tuple(sorted(valid_edges))[:12]))
            else:
                invalid += 1
    if spec.task == "dominating_set":
        selected_nodes = (_selected_node_mentions(text, spec.nodes) or nodes) & spec.nodes
        if selected_nodes:
            adj = build_adj(spec)
            dominated = set(selected_nodes)
            for node in selected_nodes:
                dominated.update(adj.get(node, []))
            all_domination_cues = ("dominate all", "dominates all", "all nodes are dominated", "every node", "valid dominating")
            if any(cue in low for cue in all_domination_cues):
                if set(spec.nodes) <= dominated:
                    units.add(("dominating_all_nodes_covered", len(selected_nodes)))
                else:
                    invalid += min(3, len(set(spec.nodes) - dominated))
            target_size = len(spec.target_nodes or set())
            if target_size and len(selected_nodes) > target_size and any(
                cue in low for cue in ("minimum", "minimum dominating", "smallest", "optimal")
            ):
                invalid += 1
    if spec.task == "maximal_independent_set":
        selected_nodes = (_selected_node_mentions(text, spec.nodes) or nodes) & spec.nodes
        if selected_nodes:
            conflict = any(
                _norm_edge(u, v, spec.directed) in spec.edges
                for u in selected_nodes
                for v in selected_nodes
                if u < v
            )
            if conflict and any(cue in low for cue in ("independent", "valid", "answer", "maximal")):
                invalid += 1
            maximal_cues = ("maximal", "cannot add", "no more", "every outside", "covered")
            if any(cue in low for cue in maximal_cues):
                maximal = (not conflict) and all(
                    any(_norm_edge(node, chosen, spec.directed) in spec.edges for chosen in selected_nodes)
                    for node in spec.nodes - selected_nodes
                )
                if maximal:
                    units.add(("maximal_independent_complete", len(selected_nodes)))
                else:
                    invalid += 1
    invalid += final_invalid
    if (final_units or final_invalid) and actions <= 0:
        actions = 1
    return units, actions, invalid


def _state_step_boolean(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["bool"] + ("bridge", "component", "scc", "exist", "edge")):
        return set(), 0, 0
    answer_hit = _scalar_answer_hit(text, spec) if check_final_answer else False
    units = set()
    units.update(_boolean_witness_units(spec, text, valid_edges, nodes, answer_hit))
    units.update(_degree_units(spec, text))
    units.update(_tournament_units(spec, text))
    units.update(_bridge_units(spec, text, valid_edges))
    units.update(_component_units(spec, text, nodes, answer_hit))
    final_units, final_invalid = _final_answer_validation_units(spec, text) if check_final_answer else (set(), 0)
    units.update(final_units)
    if spec.task == "edge_existence" and spec.source is not None and spec.target is not None:
        edge = _norm_edge(spec.source, spec.target, spec.directed)
        neg = any(cue in low for cue in ("no edge", "not connected", "does not exist", "doesn't exist", "absent"))
        pos = edge in valid_edges or any(cue in low for cue in ("edge exists", "connected", "adjacent", "there is an edge"))
        if edge in spec.edges and pos:
            units.add(("edge_exists", edge))
        elif edge not in spec.edges and neg:
            units.add(("edge_absent", edge))
    actions = len(units)
    if not actions and (nodes or valid_edges or answer_hit):
        actions = 1
    if (final_units or final_invalid) and actions <= 0:
        actions = 1
    invalid = (
        int(_numeric_final_invalid(spec, text) or _numeric_final_missing_invalid(spec, text))
        if check_final_answer
        else 0
    )
    return units, actions, final_invalid + invalid


def _state_step_isomorphism(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
    check_final_answer: bool = False,
) -> tuple[set[tuple], int, int]:
    low = text.lower()
    has_mapping_dict = bool(_int_mapping_dicts(text))
    if not has_mapping_dict and not any(
        cue in low for cue in ("map", "mapping", "mapped", "correspond", "isomorphic", "g1", "g2", "graph1", "graph2", "image")
    ):
        return set(), 0, 0
    format_bad = _mapping_format_invalid(spec, text)
    units = set()
    units.update(_mapping_format_units(spec, text))
    if not format_bad:
        units.update(_mapping_pair_units(spec, text))
        refinement_units, refinement_invalid = _mapping_refinement_units(spec, text)
        units.update(refinement_units)
    else:
        refinement_invalid = 0
    # Keep format errors as a negative schema signal, but still inspect the
    # mapping pairs. Many current rollouts use Graph1#u:Graph2#v syntax; that
    # is verifier-incompatible, yet its pairwise edge-preservation evidence is
    # useful process signal for learning the actual isomorphism reasoning.
    invalid = 6 if format_bad else 0
    invalid += refinement_invalid
    actions = len(units) if units else int(bool(re.search(r"\b\d+\s*(?:->|=>|:|maps?\s+to)\s*\d+", text)))
    if spec.answer_mapping:
        source_nodes = set(spec.answer_mapping.keys())
        target_nodes = set(spec.answer_mapping.values())
        candidate_pairs = _extract_mapping_candidate_pairs(spec, text)
        for a, b in sorted(candidate_pairs):
            if a not in source_nodes or b not in target_nodes:
                continue
            actions += 1
            units.add(("mapping_candidate_pair", int(a), int(b)))
            if spec.answer_mapping.get(a) == b:
                units.add(("map_pair", int(a), int(b)))
            if _mapping_induced_degree(spec, a, source_nodes) != _mapping_induced_degree(spec, b, target_nodes):
                invalid += 1
                continue
            if a in state.partial_mapping:
                if state.partial_mapping[a] == b:
                    units.add(("mapping_partial_bijection", len(state.partial_mapping)))
                else:
                    invalid += 1
                continue
            if b in state.partial_mapping.values():
                invalid += 1
                continue
            edge_consistent = True
            for old_a, old_b in state.partial_mapping.items():
                src_edge = _norm_edge(a, old_a, spec.directed) in spec.edges
                dst_edge = _norm_edge(b, old_b, spec.directed) in spec.edges
                if src_edge != dst_edge:
                    edge_consistent = False
                    break
                if src_edge:
                    units.add(("mapping_adjacency_consistent", int(a), int(old_a), int(b), int(old_b)))
                    units.add(("mapping_preserves_edge", int(a), int(old_a), int(b), int(old_b)))
            if not edge_consistent:
                invalid += 1
                continue
            state.partial_mapping[a] = b
            units.add(("mapping_partial_bijection", len(state.partial_mapping)))
        if source_nodes and set(state.partial_mapping.keys()) == source_nodes and set(state.partial_mapping.values()) == target_nodes:
            preserved_units, preserved_invalid = _isomorphism_edge_preservation_units(spec, state.partial_mapping)
            preserved_units = {unit for unit in preserved_units if unit and unit[0] != "mapping_valid_dict"}
            units.update(preserved_units)
            invalid += int(preserved_invalid)
    return units, actions, invalid


def _oracle_shortest_path_nodes(spec: GraphLogicSpec, source: int, target: int) -> list[int]:
    if source not in spec.nodes or target not in spec.nodes:
        return []
    if source == target:
        return [source]
    adj = build_adj(spec)
    weighted = any(abs(float(w) - 1.0) > 1e-9 for w in spec.weights.values())
    parent: dict[int, int | None] = {source: None}
    if weighted:
        import heapq

        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            if u == target:
                break
            for v in adj.get(u, []):
                edge = _norm_edge(u, v, spec.directed)
                nd = d + float(spec.weights.get(edge, 1.0))
                if nd + 1e-12 < dist.get(v, math.inf):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
    else:
        queue = [source]
        seen = {source}
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if u == target:
                break
            for v in adj.get(u, []):
                if v in seen:
                    continue
                seen.add(v)
                parent[v] = u
                queue.append(v)
    if target not in parent:
        return []
    path = []
    cur: int | None = target
    while cur is not None:
        path.append(int(cur))
        cur = parent.get(cur)
    path.reverse()
    return path if path and path[0] == source else []


def _oracle_kruskal_sets(spec: GraphLogicSpec) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], float]:
    parent = {int(node): int(node) for node in spec.nodes}
    rank = {int(node): 0 for node in spec.nodes}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        rank.setdefault(ra, 0)
        rank.setdefault(rb, 0)
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    index = {edge: pos for pos, edge in enumerate(spec.edge_order)}
    ordered = sorted(spec.edge_order, key=lambda edge: (float(spec.weights.get(edge, 1.0)), index.get(edge, 0), edge))
    accepted: list[tuple[int, int]] = []
    rejected: set[tuple[int, int]] = set()
    total = 0.0
    for edge in ordered:
        u, v = edge
        if union(u, v):
            accepted.append(edge)
            total += float(spec.weights.get(edge, 1.0))
            if len(accepted) >= max(0, len(spec.nodes) - 1):
                break
        else:
            rejected.add(edge)
    return accepted, set(accepted), rejected, total


def _mst_optimal_weight(spec: GraphLogicSpec) -> float:
    """Return the MST/MSF weight used by the executable observer.

    ``spec.target_edges`` may contain the union of multiple valid executor
    traces, e.g. Kruskal and Prim alternatives. Summing that union is not a
    valid optimum and can falsely reject equivalent MST answers. Use the
    deterministic Kruskal oracle weight instead.
    """

    if spec.task not in MST_TASKS:
        return 0.0
    _accepted_order, _accepted, _rejected, total = _oracle_kruskal_sets(spec)
    return float(total)


def _csv_allows_task(task: str, raw: str | None) -> bool:
    value = str(raw or "").strip().lower()
    if not value or value in {"*", "all"}:
        return True
    allowed = {part.strip().lower() for part in re.split(r"[,;\s]+", value) if part.strip()}
    return str(task or "").strip().lower() in allowed


def _action_router_mode(cfg: GraphLogicConfig) -> str:
    mode = str(getattr(cfg, "action_router_mode", "shadow") or "shadow").strip().lower()
    aliases = {
        "true": "grounding",
        "on": "grounding",
        "heuristic": "grounding",
        "positive": "positive_only",
        "positive-only": "positive_only",
        "booster": "positive_only",
    }
    return aliases.get(mode, mode)


def _action_router_active(spec: GraphLogicSpec, cfg: GraphLogicConfig) -> bool:
    return bool(
        getattr(cfg, "action_router_enable", False)
        and _csv_allows_task(spec.task, getattr(cfg, "action_router_task_allowlist", ""))
    )


def _action_router_can_ground(cfg: GraphLogicConfig) -> bool:
    return _action_router_mode(cfg) in {"grounding", "positive_only", "replace_parser"}


def _action_router_can_reward(cfg: GraphLogicConfig) -> bool:
    return _action_router_mode(cfg) in {"positive_only", "replace_parser"}


def _action_router_segment_priority(
    spec: GraphLogicSpec,
    seg_text: str,
    *,
    candidate: bool,
    family_candidate: bool,
    answer_hit: bool,
    state_mention: bool,
) -> int:
    if not candidate:
        return 0
    text_l = str(seg_text or "").lower()
    score = 0
    if state_mention:
        score += 3
    if answer_hit or _answer_like(seg_text):
        score += 2
    if family_candidate:
        score += 1
    if re.search(
        r"\b("
        r"relax|update|settle|parent|distance|dist|shortest|"
        r"add|accept|reject|cycle|component|union|mst|tree|"
        r"augment|bottleneck|residual|capacity|flow|cut|"
        r"pair|sum|denominator|normaliz|efficiency|wiener|centrality|"
        r"map|mapping|bijection|preserve|edge"
        r")\b",
        text_l,
    ):
        score += 2
    if re.search(r"\d", text_l):
        score += 1
    if spec.task in {"maximum_flow", "weighted_shortest_path", "weighted_minimum_spanning_tree"}:
        score += 1
    return score


def _route_action_segment(
    spec: GraphLogicSpec,
    seg_text: str,
    state: GraphLogicAlgorithmState | None,
    cfg: GraphLogicConfig,
) -> tuple[RoutedAction | None, ActionCandidate | None, list[ActionCandidate], RouterPayload | None]:
    if state is None or not _action_router_active(spec, cfg):
        return None, None, [], None
    candidates = enumerate_action_candidates(
        spec.task,
        spec,
        state,
        seg_text,
        max_candidates=max(2, int(getattr(cfg, "action_router_max_candidates", 16))),
    )
    if not candidates:
        return None, None, [], None
    payload = build_router_payload(
        task=spec.task,
        spec=spec,
        state=state,
        segment_text=seg_text,
        candidates=candidates,
    )
    routed = route_action_from_payload(
        payload,
        candidates,
        backend=str(getattr(cfg, "action_router_backend", "heuristic") or "heuristic"),
        base_url=str(getattr(cfg, "action_router_base_url", "") or ""),
        model=str(getattr(cfg, "action_router_model", "qwen3-0.6b-action-router") or ""),
        timeout=float(getattr(cfg, "action_router_timeout", 2.0)),
        max_tokens=int(getattr(cfg, "action_router_max_tokens", 96)),
        heuristic_fallback=bool(getattr(cfg, "action_router_heuristic_fallback", False)),
        heuristic_fallback_threshold=float(getattr(cfg, "action_router_heuristic_fallback_threshold", 0.75)),
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = by_id.get(routed.selected_candidate_id)
    return routed, selected, candidates, payload


def _action_router_confident(routed: RoutedAction | None, cfg: GraphLogicConfig) -> bool:
    if routed is None or routed.abstain or routed.ambiguous:
        return False
    return float(routed.confidence) >= float(getattr(cfg, "action_router_conf_threshold", 0.75))


def _action_router_edges_nodes(
    routed: RoutedAction | None,
    selected: ActionCandidate | None,
    cfg: GraphLogicConfig,
) -> tuple[set[tuple[int, int]], set[int]]:
    if selected is None or not _action_router_confident(routed, cfg):
        return set(), set()
    return {tuple(edge) for edge in selected.edges}, {int(node) for node in selected.nodes}


def _action_router_acceptance(
    routed: RoutedAction | None,
    selected: ActionCandidate | None,
    state_step_trace: dict[str, Any],
) -> bool:
    if routed is None or selected is None or routed.abstain:
        return False
    unit_names = set(str(name) for name in selected.unit_names)
    if not unit_names:
        return False
    trace_names = set(str(name) for name in state_step_trace.get("rewardable_unit_names", []) or [])
    trace_names.update(str(name) for name in state_step_trace.get("new_unit_names", []) or [])
    return bool(unit_names & trace_names)


def _action_router_event(
    routed: RoutedAction,
    selected: ActionCandidate,
) -> dict[str, Any] | None:
    if routed.abstain or routed.evidence_span is None:
        return None
    span = routed.evidence_span
    unit_name = selected.unit_names[0] if selected.unit_names else selected.action_type
    unit = (str(unit_name), str(selected.canonical_key))
    return {
        "kind": str(selected.action_type),
        "unit": _trace_value(unit),
        "char_span": [int(span[0]), int(span[1])],
        "confidence": float(routed.confidence),
        "rewardable": True,
        "new": True,
        "source": "semantic_action_router",
        "candidate_id": selected.candidate_id,
    }


def _trace_action_router(
    routed: RoutedAction | None,
    selected: ActionCandidate | None,
    candidates: list[ActionCandidate],
    accepted: bool = False,
    payload: RouterPayload | None = None,
    include_payload: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "enabled": bool(routed is not None or candidates),
        "candidate_count": int(len(candidates)),
        "accepted": bool(accepted),
    }
    if routed is not None:
        out.update(
            {
                "selected_candidate_id": routed.selected_candidate_id,
                "action_type": routed.action_type,
                "confidence": float(routed.confidence),
                "abstain": bool(routed.abstain),
                "ambiguous": bool(routed.ambiguous),
                "evidence_span": list(routed.evidence_span) if routed.evidence_span else None,
                "scores": [[cid, float(score)] for cid, score in routed.candidate_scores],
            }
        )
    if selected is not None:
        out["selected"] = {
            "candidate_id": selected.candidate_id,
            "action_type": selected.action_type,
            "args": _trace_value(selected.args),
            "unit_names": list(selected.unit_names),
            "edges": _trace_value(selected.edges),
            "nodes": _trace_value(selected.nodes),
            "reason": selected.reason,
        }
    if include_payload and payload is not None:
        out["payload"] = {
            "instruction": payload.instruction,
            "task": payload.task,
            "segment": payload.segment,
            "state_summary": _trace_value(payload.state_summary),
            "candidate_actions": _trace_value(payload.candidate_actions),
        }
    return out


def _oracle_wsp_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    if spec.task != "weighted_shortest_path" or spec.source is None:
        return set(), 0, 0
    low = text.lower()
    final_or_path_object = bool(_has_answer_assertion_cue(spec, text) and _node_sequences(_final_answer_text(text) or text, spec))
    if not any(cue in low for cue in ACTION_CUES["weighted_path"] + ACTION_CUES["path"]) and not final_or_path_object:
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    relax_cues = ("relax", "update", "tentative", "shorter", "decrease", "distance", "dist", "cost", "weight")
    no_relax_cues = ("no relax", "not relax", "keep", "unchanged", "not smaller", "no update")
    settle_cues = ("settle", "settled", "extract", "pop", "minimum", "smallest", "least")
    if any(cue in low for cue in ("dijkstra", "priority queue", "heap", "tentative distance", "initialize", "initialise")):
        actions += 1
        units.add(("state_dijkstra_weighted_rule",))
        units.add(("state_dijkstra_init", int(spec.source)))
    for node in sorted(nodes & spec.nodes):
        if node == spec.source:
            continue
        if not _node_context_has_any(text, int(node), settle_cues, window=72):
            continue
        oracle_dist = _shortest_distance(spec, int(spec.source), int(node))
        if not math.isfinite(oracle_dist):
            continue
        actions += 1
        current_min = min(
            (
                _shortest_distance(spec, int(spec.source), int(other))
                for other in spec.nodes
                if other not in state.settled_nodes and math.isfinite(_shortest_distance(spec, int(spec.source), int(other)))
            ),
            default=oracle_dist,
        )
        if _close_number(oracle_dist, current_min):
            state.settled_nodes.add(int(node))
            state.dist[int(node)] = float(oracle_dist)
            units.add(("state_dijkstra_settle", int(node), round(float(oracle_dist), 6)))
        else:
            invalid += 1
    for edge in sorted(valid_edges):
        contexts = _edge_contexts(text, edge, spec.directed)
        implicit = len(valid_edges) <= 2 and any(cue in low for cue in relax_cues + no_relax_cues)
        if not contexts and not implicit:
            continue
        has_relax = _edge_context_has_any(text, edge, spec, relax_cues) or implicit
        has_no_relax = _edge_context_has_any(text, edge, spec, no_relax_cues)
        if not (has_relax or has_no_relax):
            continue
        orientations = [edge]
        if not spec.directed and edge[0] != edge[1]:
            orientations.append((edge[1], edge[0]))
        matched = False
        for u, v in orientations:
            du = _shortest_distance(spec, int(spec.source), int(u))
            dv = _shortest_distance(spec, int(spec.source), int(v))
            w = float(spec.weights.get(_norm_edge(u, v, spec.directed), 1.0))
            if not math.isfinite(du) or not math.isfinite(dv):
                continue
            if _close_number(du + w, dv):
                actions += 1
                units.add(("state_dijkstra_scan_edge", int(u), int(v), round(float(w), 6)))
                old = state.dist.get(v, math.inf)
                if has_no_relax and old < math.inf and du + w >= old - 1e-9:
                    units.add(("state_dijkstra_no_relax", int(u), int(v), round(float(old), 6)))
                    matched = True
                    break
                observed = _distance_update_values_for_node(text, int(v))
                if observed and any(_close_number(value, dv) for value in observed):
                    if dv + 1e-9 < old:
                        state.dist[int(v)] = float(dv)
                        state.parent_node[int(v)] = int(u)
                    units.add(("state_dijkstra_decreased_dist", int(u), int(v), round(float(dv), 6)))
                    units.add(("state_dijkstra_relax", int(u), int(v), round(float(dv), 6)))
                    units.add(("state_dijkstra_dist", int(v), round(float(dv), 6)))
                    units.add(("state_dijkstra_weighted_relax_cert", int(u), int(v), round(float(w), 6), round(float(dv), 6)))
                    if spec.target is not None and int(v) == int(spec.target):
                        units.add(("state_dijkstra_target_cert", int(v), round(float(dv), 6)))
                    matched = True
                    break
                invalid += 1
                matched = True
                break
        if not matched and any(cue in low for cue in ("shortest", "dijkstra", "relax", "update")):
            invalid += 1
    for seq in _node_sequences(text, spec):
        if spec.source is None or spec.target is None or seq[0] != spec.source or seq[-1] != spec.target:
            continue
        path_edges = _valid_path_edges(spec, seq)
        if not path_edges:
            invalid += 1
            continue
        cost = sum(float(spec.weights.get(edge, 1.0)) for edge in path_edges)
        target = _shortest_distance(spec, int(spec.source), int(spec.target))
        cost_observed = any(_close_number(value, cost) for value in _numeric_values(text))
        if math.isfinite(target) and _close_number(cost, target) and cost_observed:
            actions += 1
            units.add(("state_path_feasible", tuple(path_edges[: min(8, len(path_edges))])))
            units.add(("state_path_cost_valid", tuple(path_edges[: min(8, len(path_edges))]), round(float(cost), 6)))
            units.add(("state_path_reaches_target", int(spec.target)))
            units.add(("state_dijkstra_target_cert", int(spec.target), round(float(target), 6)))
            units.add(("state_weighted_path_final_cert", tuple(path_edges[: min(8, len(path_edges))]), round(float(target), 6)))
            for edge in path_edges[: min(8, len(path_edges))]:
                units.add(("state_path_edge", edge))
    return units, actions, invalid


def _oracle_mst_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    if spec.task not in MST_TASKS:
        return set(), 0, 0
    low = text.lower()
    final_edge_order = _final_answer_edge_order(text, spec) if _has_answer_assertion_cue(spec, text) else []
    if not any(cue in low for cue in ACTION_CUES["mst"]) and not final_edge_order:
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    accepted_order, accepted, rejected, optimal_weight = _oracle_kruskal_sets(spec)
    accepted_rank = {edge: idx + 1 for idx, edge in enumerate(accepted_order)}
    accept_cues = ("accept", "add", "include", "take", "choose", "select", "safe", "connects components", "different components")
    reject_cues = ("reject", "skip", "discard", "cycle", "would form a cycle", "same component")
    if any(cue in low for cue in ("sort", "sorted", "increasing weight", "nondecreasing", "kruskal", "lightest", "smallest weight")):
        actions += 1
        units.add(("state_mst_weight_order", len(spec.edge_order)))
        units.add(("state_mst_weighted_rule",))
    list_accept_cue = (
        any(cue in low for cue in ("mst edges", "tree edges", "spanning tree edges", "selected edges", "accepted edges"))
        and not any(cue in low for cue in ("candidate", "consider", "reject", "skip"))
    )
    for edge in sorted(valid_edges):
        contexts = _edge_contexts(text, edge, spec.directed)
        edge_weight = float(spec.weights.get(edge, 1.0))
        if contexts and any("weight" in ctx or "cost" in ctx for ctx in contexts):
            units.add(("state_mst_edge_weight_checked", edge, round(float(edge_weight), 6)))
        has_accept = _edge_context_has_any(text, edge, spec, accept_cues) or (list_accept_cue and edge in accepted)
        has_reject = _edge_context_has_any(text, edge, spec, reject_cues)
        same_component = _state_find(state, edge[0]) == _state_find(state, edge[1])
        if has_accept:
            actions += 1
            if edge in state.accepted_edges:
                units.add(("state_mst_accept", edge, round(float(state.total_weight), 6)))
                continue
            if same_component:
                invalid += 1
                continue
            safe_now = _mst_edge_is_current_safe(spec, state, edge)
            if spec.task == "weighted_minimum_spanning_tree" and not (safe_now or edge in accepted):
                invalid += 1
                continue
            _state_union(state, edge[0], edge[1])
            state.accepted_edges.add(edge)
            state.total_weight += edge_weight
            units.add(("state_mst_accept", edge, round(float(state.total_weight), 6)))
            if safe_now or edge in accepted:
                units.add(("state_mst_safe_edge", edge, round(float(edge_weight), 6)))
            if edge in accepted:
                units.add(("state_mst_target_edge", edge))
            if edge in accepted_rank:
                units.add(("state_mst_kruskal_prefix_edge", edge, int(accepted_rank[edge])))
            units.add(("state_mst_component_count", len({_state_find(state, n) for n in spec.nodes})))
        elif has_reject:
            actions += 1
            if same_component or edge in rejected:
                state.rejected_edges.add(edge)
                units.add(("state_mst_reject_cycle", edge))
                units.add(("state_mst_cycle_reject_cert", edge))
            elif any(cue in " ".join(contexts) for cue in ("cycle", "same component", "loop")):
                invalid += 1
    cert_edges = set(final_edge_order or _edge_order_from_text(text, spec))
    if cert_edges:
        actions += 1
        target_count = max(0, len(spec.nodes) - 1)
        valid_cert, cert_weight = _spanning_tree_valid(spec, cert_edges)
        if len(cert_edges) == target_count and valid_cert:
            signature = tuple(sorted(cert_edges))[:12]
            units.add(("state_mst_n_minus_one_edges", int(target_count)))
            units.add(("state_mst_connected_acyclic", signature))
            units.add(("state_mst_spanning_nodes_covered", tuple(sorted({node for edge in cert_edges for node in edge}))[:12]))
            units.add(("mst_answer_edge_count", spec.task, int(target_count)))
            units.add(("mst_answer_spanning_tree", signature))
            if spec.task == "weighted_minimum_spanning_tree":
                if _close_number(cert_weight, optimal_weight):
                    units.add(("state_mst_optimal_weight_cert", round(float(optimal_weight), 6)))
                    units.add(("mst_answer_valid", signature, round(float(cert_weight), 6)))
                if any(_close_number(value, cert_weight) for value in _numeric_values(text)):
                    units.add(("state_mst_final_weight_checked", round(float(cert_weight), 6)))
        elif any(cue in low for cue in ("mst", "spanning tree", "final", "answer")):
            invalid += 1
    target_count = max(0, len(spec.nodes) - 1)
    if target_count > 0 and len(state.accepted_edges) == target_count:
        roots = {_state_find(state, int(node)) for node in spec.nodes}
        if len(roots) == 1:
            units.add(("state_mst_n_minus_one_edges", int(target_count)))
            units.add(("state_mst_connected_acyclic", tuple(sorted(state.accepted_edges))[:12]))
            units.add(("state_mst_spanning_nodes_covered", tuple(sorted(spec.nodes))[:12]))
            if spec.task == "weighted_minimum_spanning_tree" and _close_number(state.total_weight, optimal_weight):
                units.add(("state_mst_optimal_weight_cert", round(float(optimal_weight), 6)))
    return units, actions, invalid


def _oracle_flow_units(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    if spec.task != "maximum_flow" or spec.source is None or spec.target is None:
        return set(), 0, 0
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["flow"] + ACTION_CUES["path"]):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    augment_cue = any(
        cue in low
        for cue in (
            "augment",
            "augmenting",
            "send",
            "send flow",
            "push",
            "push flow",
            "increase",
            "add flow",
            "route flow",
            "bottleneck",
            "augment path",
            "augment the path",
            "send along",
            "push along",
            "send through",
            "push through",
        )
    )
    residual_cue = any(
        cue in low
        for cue in (
            "residual",
            "remaining capacity",
            "remaining residual",
            "remaining cap",
            "capacity left",
            "capacity remains",
            "capacity leftover",
            "capacity after",
            "capacity becomes",
            "capacity is now",
            "capacity now",
            "reduce capacity",
            "decrease capacity",
            "reduced capacity",
            "decreased capacity",
            "reverse edge",
            "reverse capacity",
            "reverse residual",
            "reverse arc",
            "backward arc",
            "back edge",
            "backward edge",
            "backward residual",
            "return edge",
            "update residual",
            "residual update",
            "updated residual",
            "update capacities",
            "updated capacities",
            "residual becomes",
            "residual is now",
            "residual now",
            "residual left",
            "subtract",
            "subtracting",
            "after sending",
            "after pushing",
            "after augmenting",
        )
    )
    for seq in _node_sequences(text, spec):
        if not seq or seq[0] != spec.source or seq[-1] != spec.target:
            continue
        path = _valid_state_flow_path_edges(spec, seq, state)
        actions += 1
        if not path:
            original_path = _valid_path_edges(spec, seq)
            if original_path:
                if augment_cue and (
                    not residual_cue
                    or any(cue in low for cue in ("without residual", "ignore residual", "no residual"))
                ):
                    invalid += 1
                continue
            invalid += 1
            continue
        bottleneck = min(state.residual.get(edge, 0.0) for edge in path)
        units.add(("state_flow_path", tuple(path)))
        units.add(("state_flow_bottleneck", tuple(path), round(float(bottleneck), 6)))
        amount_values = _flow_amount_values(text)
        amount_ok = not amount_values or any(_close_number(value, bottleneck) for value in amount_values)
        flow_after = float(state.flow_value) + float(bottleneck)
        flow_value_update_ok = any(_close_number(value, flow_after) for value in _flow_value_update_values(text))
        strong_flow_value_update_ok = flow_value_update_ok and (residual_cue or bool(amount_values))
        update_cert_ok = residual_cue or strong_flow_value_update_ok
        if augment_cue and update_cert_ok and (amount_ok or strong_flow_value_update_ok):
            for u, v in path:
                state.residual[(u, v)] = state.residual.get((u, v), 0.0) - bottleneck
                state.residual[(v, u)] = state.residual.get((v, u), 0.0) + bottleneck
            state.flow_value += bottleneck
            units.add(("state_flow_valid_augment", tuple(path), round(float(bottleneck), 6)))
            if residual_cue:
                units.add(("state_flow_residual_bottleneck_cert", tuple(path), round(float(bottleneck), 6)))
                units.add(("state_flow_valid_augment_inferred", tuple(path), round(float(bottleneck), 6)))
                units.add(("state_flow_residual_update_after_augment", tuple(path), round(float(bottleneck), 6)))
            elif strong_flow_value_update_ok:
                units.add(("state_flow_valid_augment_inferred", tuple(path), round(float(bottleneck), 6)))
            if strong_flow_value_update_ok:
                units.add(("state_flow_value_update_cert", round(float(state.flow_value), 6)))
            units.add(("state_flow_augment", tuple(path), round(float(bottleneck), 6)))
            units.add(("state_flow_value", round(float(state.flow_value), 6)))
        elif augment_cue and not residual_cue:
            invalid += 1
        elif augment_cue:
            invalid += 1
    if any(
        cue in low
        for cue in (
            "no augmenting path",
            "no augmenting paths",
            "no residual path",
            "no more path",
            "no more paths",
            "no additional path",
            "no other path",
            "min cut",
            "minimum cut",
            "cut capacity",
        )
    ):
        actions += 1
        values = _numeric_values(text)
        if state.max_flow_value > 1e-12 and any(_close_number(value, state.max_flow_value) for value in values):
            units.add(("state_flow_complete", round(float(state.max_flow_value), 6)))
            units.add(("state_flow_min_cut_cert", round(float(state.max_flow_value), 6)))
        elif _close_number(state.flow_value, state.max_flow_value):
            units.add(("state_flow_complete", round(float(state.max_flow_value), 6)))
        else:
            invalid += int("no augmenting" in low or "minimum cut" in low or "min cut" in low)
    return units, actions, invalid


def _oracle_metric_units(spec: GraphLogicSpec, text: str) -> tuple[set[tuple], int, int]:
    if spec.task not in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return set(), 0, 0
    low = text.lower()
    if not any(
        cue in low
        for cue in (
            "distance",
            "shortest",
            "sum",
            "total",
            "centrality",
            "efficiency",
            "wiener",
            "eccentricity",
            "radius",
            "diameter",
            "center",
            "periphery",
            "barycenter",
            "normaliz",
            "denominator",
        )
    ):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = _distance_process_invalid_count(spec, text)
    values = _numeric_values(text)
    n = len(spec.nodes)
    if spec.task == "global_efficiency":
        if any(cue in low for cue in ("1/d", "reciprocal", "inverse distance", "efficiency")):
            units.add(("global_efficiency_formula", int(n)))
        denominator = float(n * max(0, n - 1))
        if any(cue in low for cue in ("denominator", "n(n-1)", "ordered pairs", "divide", "average")):
            actions += 1
            if any(_close_number(value, denominator) for value in values):
                units.add(("global_efficiency_denominator_value", int(denominator)))
        if spec.answer_numbers and any(cue in low for cue in ("global efficiency", "final efficiency", "therefore")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("global_metric_answer", spec.task))
    elif spec.task == "wiener_index":
        pair_count = float(n * max(0, n - 1) // 2)
        if any(cue in low for cue in ("unordered pairs", "pair count", "n(n-1)/2", "choose")):
            actions += 1
            if any(_close_number(value, pair_count) for value in values):
                units.add(("wiener_pair_scope", int(n)))
        if spec.answer_numbers and any(cue in low for cue in ("wiener", "distance sum", "total distance")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("global_metric_answer", spec.task))
    elif spec.task == "closeness_centrality" and spec.source is not None:
        source = int(spec.source)
        dist_sum = _distance_sum(spec, source)
        reachable = len(_finite_distances_from(spec, source))
        if any(cue in low for cue in ("distance sum", "sum of distances", "total distance")):
            actions += 1
            if any(_close_number(value, dist_sum) for value in values):
                units.add(("closeness_distance_sum", source, round(float(dist_sum), 6)))
        if any(cue in low for cue in ("normaliz", "reachable", "unreachable", "wf", "n-1")):
            units.add(("closeness_normalizer_value", source, int(reachable), max(0, n - 1)))
        if spec.answer_numbers and any(cue in low for cue in ("closeness", "centrality", "final value")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("metric_answer", spec.task))
    elif spec.task == "betweenness_centrality" and spec.source is not None:
        pivot = int(spec.source)
        raw = _betweenness_raw_score(spec, pivot)
        if any(cue in low for cue in ("sigma", "fraction", "shortest paths", "through")):
            units.add(("betweenness_sigma_fraction", pivot))
        if any(cue in low for cue in ("raw", "unnormalized", "sum contribution", "total contribution")):
            actions += 1
            if any(_close_number(value, raw) for value in values):
                units.add(("betweenness_raw_sum", pivot, round(float(raw), 6)))
        if any(cue in low for cue in ("normaliz", "denominator", "(n-1)(n-2)", "n-1", "n-2")):
            denom = float(max(1, (n - 1) * max(1, n - 2)))
            factor = 1.0 / denom if spec.directed else 2.0 / denom
            if any(_close_number(value, denom) or _close_number(value, factor) for value in values):
                units.add(("betweenness_normalizer_value", pivot, round(float(factor), 6)))
        if spec.answer_numbers and any(cue in low for cue in ("betweenness", "centrality", "final value", "normalized")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("betweenness_normalized_value", pivot, round(float(target), 6)))
    elif spec.task in {"center", "periphery", "barycenter", "radius", "diameter"}:
        target_nodes = _target_answer_nodes(spec)
        if target_nodes and any(node in _numbers_in_text(text, spec.nodes) for node in target_nodes):
            if spec.task in {"center", "periphery", "barycenter"} and any(cue in low for cue in (spec.task, "tie", "all")):
                units.add(("metric_target_set_complete", spec.task, tuple(sorted(target_nodes))))
        if spec.answer_numbers and any(cue in low for cue in (spec.task, "eccentricity", "distance")):
            target = float(spec.answer_numbers[0])
            if any(_close_number(value, target) for value in values):
                units.add(("metric_answer", spec.task))
    actions += len(units)
    return units, actions, invalid


def _oracle_set_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
) -> tuple[set[tuple], int, int]:
    if spec.task not in SET_FEASIBILITY_TASKS:
        return set(), 0, 0
    low = text.lower()
    if not any(cue in low for cue in ACTION_CUES["set"]):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    values = _numeric_values(text)
    selected_nodes = (_selected_node_mentions(text, spec.nodes) or nodes) & spec.nodes
    selected_edges = set(_edge_order_from_text(text, spec))
    if spec.task in {"bipartite_maximum_matching", "max_weight_matching"} and selected_edges:
        endpoints: set[int] = set()
        feasible = True
        for edge in selected_edges:
            if edge[0] in endpoints or edge[1] in endpoints:
                feasible = False
                break
            endpoints.update(edge)
            units.add(("matching_edge", edge))
        actions += len(selected_edges)
        target_size = len(spec.target_edges or set())
        if spec.task == "bipartite_maximum_matching" and feasible and target_size and len(selected_edges) >= target_size and any(cue in low for cue in ("maximum", "optimal", "size", "cardinality")):
            units.add(("matching_size", int(target_size)))
            units.add(("edge_set_answer_optimal_size", spec.task, int(target_size)))
        elif spec.task == "max_weight_matching" and feasible:
            total_weight = _edge_set_weight(spec, selected_edges)
            target_weight = _max_weight_matching_target_weight(spec)
            total_weight_observed = any(_close_number(value, total_weight) for value in values)
            if total_weight_observed:
                units.add(("max_weight_matching_weight", round(float(total_weight), 6)))
            if target_weight is not None and total_weight_observed and _close_number(total_weight, target_weight) and any(
                cue in low for cue in ("maximum", "max weight", "optimal", "total weight")
            ):
                units.add(("max_weight_matching_optimal_weight", round(float(total_weight), 6)))
            elif target_weight is not None and any(cue in low for cue in ("maximum", "max weight", "optimal")):
                invalid += 1
        elif not feasible:
            invalid += 1
    elif spec.task == "min_edge_covering" and selected_edges:
        covered = {node for edge in selected_edges for node in edge}
        for node in sorted(covered & spec.nodes):
            units.add(("edge_cover_node", int(node)))
        actions += len(selected_edges)
        target_size = len(spec.target_edges or set())
        if set(spec.nodes) <= covered:
            units.add(("edge_set_answer_feasible", spec.task, tuple(sorted(selected_edges))[:12]))
            if target_size and len(selected_edges) == target_size:
                units.add(("edge_set_answer_optimal_size", spec.task, len(selected_edges)))
        elif any(cue in low for cue in ("cover all", "every vertex", "all nodes")):
            invalid += 1
    elif spec.task == "min_vertex_cover" and selected_nodes:
        uncovered = [edge for edge in spec.edges if edge[0] not in selected_nodes and edge[1] not in selected_nodes]
        for edge in spec.edges:
            if edge[0] in selected_nodes or edge[1] in selected_nodes:
                units.add(("vertex_cover_edge", edge))
        actions += len(selected_nodes)
        target_size = len(spec.target_nodes or set())
        if not uncovered:
            units.add(("vertex_cover_all_edges_covered", len(selected_nodes)))
            units.add(("set_answer_feasible", spec.task, tuple(sorted(selected_nodes))[:12]))
            if target_size and len(selected_nodes) <= target_size:
                units.add(("set_answer_optimal_size", spec.task, len(selected_nodes)))
        elif any(cue in low for cue in ("cover all", "every edge", "valid cover")):
            invalid += 1
    elif spec.task == "dominating_set" and selected_nodes:
        adj = build_adj(spec)
        dominated = set(selected_nodes)
        for node in selected_nodes:
            for nbr in adj.get(node, []):
                dominated.add(nbr)
        for node in sorted(dominated & spec.nodes):
            units.add(("dominated_node", int(node)))
        actions += len(selected_nodes)
        target_size = len(spec.target_nodes or set())
        if set(spec.nodes) <= dominated:
            units.add(("dominating_all_nodes_covered", len(selected_nodes)))
            units.add(("set_answer_feasible", spec.task, tuple(sorted(selected_nodes))[:12]))
            if target_size and len(selected_nodes) <= target_size:
                units.add(("set_answer_optimal_size", spec.task, len(selected_nodes)))
        elif any(cue in low for cue in ("dominate all", "every node", "all nodes")):
            invalid += 1
    elif spec.task == "maximal_independent_set" and selected_nodes:
        conflict = any(
            _norm_edge(u, v, spec.directed) in spec.edges
            for u in selected_nodes
            for v in selected_nodes
            if u < v
        )
        actions += len(selected_nodes)
        if not conflict:
            for node in sorted(selected_nodes):
                units.add(("independent_node", int(node)))
            maximal = all(
                any(_norm_edge(node, chosen, spec.directed) in spec.edges for chosen in selected_nodes)
                for node in spec.nodes - selected_nodes
            )
            if maximal and any(cue in low for cue in ("maximal", "cannot add", "no more", "every outside", "covered")):
                units.add(("maximal_independent_complete", len(selected_nodes)))
        elif any(cue in low for cue in ("independent", "valid", "maximal")):
            invalid += 1
    return units, actions, invalid


def _oracle_isomorphism_units(
    spec: GraphLogicSpec,
    text: str,
    state: GraphLogicAlgorithmState,
) -> tuple[set[tuple], int, int]:
    if spec.task != "isomophic_mapping" or not spec.answer_mapping:
        return set(), 0, 0
    low = text.lower()
    if not any(cue in low for cue in ("mapping", "isomorphism", "maps", "bijection", "preserve", "adjacency", "graph1", "graph2")):
        return set(), 0, 0
    units: set[tuple] = set()
    actions = 0
    invalid = 0
    source_nodes = set(spec.answer_mapping.keys())
    target_nodes = set(spec.answer_mapping.values())
    for a, b in sorted(_extract_mapping_candidate_pairs(spec, text)):
        if a not in source_nodes or b not in target_nodes:
            invalid += 1
            continue
        actions += 1
        if a in state.partial_mapping and state.partial_mapping[a] != b:
            invalid += 1
            continue
        if b in state.partial_mapping.values() and state.partial_mapping.get(a) != b:
            invalid += 1
            continue
        consistent = True
        for old_a, old_b in state.partial_mapping.items():
            src_edge = _norm_edge(a, old_a, spec.directed) in spec.edges
            dst_edge = _norm_edge(b, old_b, spec.directed) in spec.edges
            if src_edge != dst_edge:
                consistent = False
                break
            if src_edge:
                units.add(("mapping_preserves_edge", int(a), int(old_a), int(b), int(old_b)))
        if consistent:
            state.partial_mapping[a] = b
            units.add(("mapping_partial_bijection", len(state.partial_mapping)))
        else:
            invalid += 1
    for mapping in _int_mapping_dicts(text):
        actions += 1
        preserved_units, preserved_invalid = _isomorphism_edge_preservation_units(spec, mapping)
        units.update(preserved_units)
        invalid += int(preserved_invalid)
        if not preserved_invalid and set(mapping.keys()) == source_nodes and set(mapping.values()) == target_nodes:
            units.add(("mapping_complete_dict", len(mapping)))
            state.partial_mapping.update(mapping)
    return units, actions, invalid


def _oracle_step_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState,
    max_units: int,
) -> tuple[set[tuple], int, int]:
    if max_units <= 0:
        return set(), 0, 0
    if spec.task == "weighted_shortest_path":
        units, actions, invalid = _oracle_wsp_units(spec, text, valid_edges, nodes, state)
    elif spec.task in MST_TASKS:
        units, actions, invalid = _oracle_mst_units(spec, text, valid_edges, state)
    elif spec.task == "maximum_flow":
        units, actions, invalid = _oracle_flow_units(spec, text, state)
    elif spec.task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        units, actions, invalid = _oracle_metric_units(spec, text)
    elif spec.task in SET_FEASIBILITY_TASKS:
        units, actions, invalid = _oracle_set_units(spec, text, valid_edges, nodes)
    elif spec.task == "isomophic_mapping":
        units, actions, invalid = _oracle_isomorphism_units(spec, text, state)
    else:
        return set(), 0, 0
    if len(units) > max_units:
        units = set(sorted(units, key=repr)[:max_units])
    # Oracle milestones are a recall booster for verified positive progress.
    # Keep negative process rewards in the original executor branch, where
    # action legality is tied to the current observed state. This avoids
    # double-penalizing verbose but partially correct reasoning.
    return units, actions, 0


def _legacy_core4_filter_state_units(
    spec: GraphLogicSpec,
    config: GraphLogicConfig | None,
    units: set[tuple],
) -> tuple[set[tuple], set[tuple]]:
    """Restrict legacy/core4 reproduction to the old narrow state surface.

    ``legacy_core4_compat`` narrows template/potential evidence, but the newer
    algorithm-state executor can still emit many post-core4 units. Keep the
    legacy core4 state surface narrow while preserving the old protective state
    checks for stable traversal/connectivity tasks. Other hard-task state units
    stay disabled in compatibility mode.
    """
    if not bool(config is not None and getattr(config, "legacy_core4_compat", False)):
        return set(units), set()
    if spec.task in LEGACY_CORE4_PROTECTED_TASKS:
        return set(units), set()
    if spec.task not in LEGACY_CORE4_STATE_TASKS:
        return set(), set(units)
    allowed = LEGACY_CORE4_STATE_UNIT_NAMES_BY_TASK.get(spec.task, set())
    if not allowed:
        return set(), set(units)
    kept = {unit for unit in units if unit and str(unit[0]) in allowed}
    dropped = set(units) - kept
    return kept, dropped


def _algorithm_state_step(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicAlgorithmState | None,
    config: GraphLogicConfig | None = None,
    check_final_answer: bool = False,
    return_trace: bool = False,
    return_action_events: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], dict[str, Any]]:
    wants_aux_trace = bool(return_trace or return_action_events)
    if state is None:
        if wants_aux_trace:
            return 0.0, {}, {"state_supported": False, "action_events": []}
        return 0.0, {}
    before = _algorithm_state_potential(spec, state)
    before_components = _algorithm_state_component_scores(spec, state)
    if spec.task in PATH_TASKS:
        units, actions, invalid = _state_step_path(spec, text, valid_edges, nodes, state, check_final_answer)
    elif spec.task in TRAVERSAL_TASKS:
        units, actions, invalid = _state_step_traversal(spec, text, valid_edges, nodes, state)
    elif spec.task in MST_TASKS:
        units, actions, invalid = _state_step_mst(spec, text, valid_edges, state, check_final_answer)
    elif spec.task == "maximum_flow":
        units, actions, invalid = _state_step_flow(spec, text, valid_edges, state, check_final_answer)
    elif spec.task in ORDERING_STATE_TASKS:
        units, actions, invalid = _state_step_ordering(spec, text, valid_edges, nodes, state, check_final_answer)
    elif spec.task in DISTANCE_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        units, actions, invalid = _state_step_distance(spec, text, valid_edges, nodes, check_final_answer)
    elif spec.task in NEIGHBORHOOD_STATE_TASKS:
        units, actions, invalid = _state_step_neighborhood(spec, text, nodes, check_final_answer)
    elif spec.task in SET_FEASIBILITY_TASKS:
        units, actions, invalid = _state_step_set(spec, text, valid_edges, nodes, state, check_final_answer)
    elif spec.task in BOOLEAN_STATE_TASKS:
        units, actions, invalid = _state_step_boolean(spec, text, valid_edges, nodes, check_final_answer)
    elif spec.task in ISOMORPHISM_STATE_TASKS:
        units, actions, invalid = _state_step_isomorphism(spec, text, state, check_final_answer)
    else:
        if wants_aux_trace:
            return 0.0, {}, {"state_supported": False, "action_events": []}
        return 0.0, {}
    oracle_units: set[tuple] = set()
    oracle_actions = 0
    oracle_invalid = 0
    oracle_task_allowed = bool(
        config is not None
        and bool(getattr(config, "oracle_milestones", False))
        and _csv_allows_task(spec.task, getattr(config, "oracle_milestone_task_allowlist", ""))
    )
    if oracle_task_allowed:
        max_units = max(0, int(getattr(config, "oracle_milestone_max_units", 96)))
        oracle_units, oracle_actions, oracle_invalid = _oracle_step_units(
            spec,
            text,
            valid_edges,
            nodes,
            state,
            max_units,
        )
        units.update(oracle_units)
        if bool(getattr(config, "oracle_milestone_count_actions", True)):
            actions += int(oracle_actions)
        if not bool(getattr(config, "oracle_milestone_positive_only", False)):
            invalid += int(oracle_invalid)
    legacy_core4_state_unit_filter = bool(config is not None and getattr(config, "legacy_core4_compat", False))
    legacy_filtered_units: set[tuple] = set()
    pre_legacy_filter_unit_count = len(units)
    if legacy_core4_state_unit_filter:
        units, legacy_filtered_units = _legacy_core4_filter_state_units(spec, config, units)
        if oracle_units:
            oracle_units = {unit for unit in oracle_units if unit in units}
    component_precision_unit_filter = bool(_component_precision_unit_filter_allowed(spec, config))
    filtered_units: set[tuple] = set()
    pre_filter_unit_count = len(units)
    if component_precision_unit_filter:
        units, filtered_units = _component_precision_filter_units(spec, units)
        if oracle_units:
            oracle_units = {unit for unit in oracle_units if unit in units}
    old_units = set(state.seen_units)
    state.seen_units.update(units)
    new_units = state.seen_units - old_units
    after = _algorithm_state_potential(spec, state)
    raw_delta = max(0.0, after - before)
    unit_count = len(units)
    new_unit_count = len(new_units)
    duplicate_unit_count = len(units & old_units)
    new_unit_weight = _state_seen_weight(spec, new_units)
    rewardable_new_units = {unit for unit in new_units if _state_rewardable_unit(spec, unit)}
    rewardable_new_unit_count = len(rewardable_new_units)
    rewardable_new_unit_weight = _state_seen_weight(spec, rewardable_new_units)
    weak_new_unit_count = max(0, new_unit_count - rewardable_new_unit_count)
    terminal_count = sum(1 for unit in new_units if _state_terminal_unit(unit))
    completion_invalid = int(
        invalid > 0
        and _has_answer_assertion_cue(spec, text)
        and spec.task in COMPLETION_SENSITIVE_STATE_TASKS
    )
    if invalid > 0:
        invalid_key = (
            "invalid",
            spec.task,
            tuple(sorted(str(unit[0]) for unit in units if unit)[:8]),
            tuple(sorted(valid_edges))[:4] if "valid_edges" in locals() else (),
            min(3, int(invalid)),
        )
        if invalid_key not in state.invalid_steps:
            state.invalid_steps.add(invalid_key)
            state.invalid_weight += _state_negative_delta(spec, actions, invalid, bool(completion_invalid))
    delta_quality = _state_delta_quality(
        spec,
        actions,
        invalid,
        new_unit_count,
        rewardable_new_unit_count,
        new_unit_weight,
        rewardable_new_unit_weight,
    )
    delta = raw_delta * delta_quality
    state_negative_delta = _state_negative_delta(spec, actions, invalid, bool(completion_invalid))
    state_net_delta = delta - state_negative_delta
    after_components = _algorithm_state_component_scores(spec, state)
    component_stats = {}
    for component in STATE_PROGRESS_COMPONENTS:
        component_stats[f"state_component_{component}"] = float(after_components.get(component, 0.0))
        component_stats[f"state_component_delta_{component}"] = float(
            max(0.0, after_components.get(component, 0.0) - before_components.get(component, 0.0))
        )
    stats = {
        "state_actions": float(actions),
        "state_units": float(unit_count),
        "legacy_core4_state_unit_filter_segments": float(legacy_core4_state_unit_filter),
        "legacy_core4_state_unit_filter_input_units": float(pre_legacy_filter_unit_count),
        "legacy_core4_state_unit_filter_kept_units": float(unit_count),
        "legacy_core4_state_unit_filter_dropped_units": float(len(legacy_filtered_units)),
        "legacy_core4_state_unit_filter_drop_segments": float(
            legacy_core4_state_unit_filter and bool(legacy_filtered_units)
        ),
        "component_precision_unit_filter_segments": float(component_precision_unit_filter),
        "component_precision_unit_filter_input_units": float(pre_filter_unit_count),
        "component_precision_unit_filter_kept_units": float(unit_count),
        "component_precision_unit_filter_dropped_units": float(len(filtered_units)),
        "component_precision_unit_filter_drop_segments": float(component_precision_unit_filter and bool(filtered_units)),
        "state_oracle_units": float(len(oracle_units)),
        "state_oracle_actions": float(oracle_actions),
        "state_oracle_invalid": float(oracle_invalid),
        "state_oracle_action_segments": float(oracle_actions > 0),
        "state_oracle_positive_only": float(
            bool(config is not None and getattr(config, "oracle_milestone_positive_only", False))
        ),
        "state_oracle_count_actions": float(
            bool(config is not None and getattr(config, "oracle_milestone_count_actions", True))
        ),
        "state_oracle_task_allowed": float(oracle_task_allowed),
        "state_new_units": float(new_unit_count),
        "state_duplicate_units": float(duplicate_unit_count),
        "state_weighted_new_units": float(new_unit_weight),
        "state_rewardable_new_units": float(rewardable_new_unit_count),
        "state_rewardable_weighted_new_units": float(rewardable_new_unit_weight),
        "state_weak_new_units": float(weak_new_unit_count),
        "state_terminal_units": float(terminal_count),
        "state_completion_invalid": float(completion_invalid),
        "state_valid_transitions": float(new_unit_count > 0),
        "state_rewardable_transitions": float(rewardable_new_unit_count > 0),
        "state_weak_only_transition": float(new_unit_count > 0 and rewardable_new_unit_count <= 0),
        "state_invalid_transitions": float(invalid),
        "state_progressive": float(delta > 0.0),
        "state_delta": float(delta),
        "state_positive_delta": float(delta),
        "state_negative_delta": float(state_negative_delta),
        "state_net_delta": float(state_net_delta),
        "state_raw_delta": float(raw_delta),
        "state_delta_quality": float(delta_quality),
        "state_action_segments": float(actions > 0),
        "state_no_action_segments": float(actions <= 0),
        "state_invalid_action_segments": float(invalid > 0),
        "state_duplicate_no_progress_segments": float(
            actions > 0 and unit_count > 0 and new_unit_count == 0 and invalid == 0
        ),
        "state_valid_zero_delta_segments": float(new_unit_count > 0 and delta <= 0.0),
        "state_action_no_progress_segments": float(actions > 0 and delta <= 0.0),
        "state_no_valid_units_segments": float(actions > 0 and unit_count == 0),
        **component_stats,
    }
    if not wants_aux_trace:
        return delta, stats

    weak_new_units = new_units - rewardable_new_units
    event_units = set(rewardable_new_units)
    event_units.update(new_units)
    if invalid > 0:
        # These pseudo-units are localization-only. They do not enter the
        # algorithm state and are never rewardable; they just keep negative
        # process rewards anchored to the invalid edge/node that triggered the
        # executor failure instead of smearing the penalty over the full segment.
        for edge in sorted(valid_edges)[:6]:
            event_units.add(("state_invalid_edge_action", edge))
        if not valid_edges:
            for node in sorted(nodes)[:4]:
                event_units.add(("state_invalid_node_action", int(node)))
    if not event_units:
        event_units.update(units & old_units)
    if not event_units:
        event_units.update(units)
    action_events = _state_action_events_for_units(
        spec,
        text,
        event_units,
        new_units,
        rewardable_new_units,
    )
    if return_action_events and not return_trace:
        return delta, stats, {
            "state_supported": True,
            "actions": int(actions),
            "invalid": int(invalid),
            "unit_names": sorted({str(unit[0]) for unit in units if unit}),
            "new_unit_names": sorted({str(unit[0]) for unit in new_units if unit}),
            "rewardable_unit_names": sorted({str(unit[0]) for unit in rewardable_new_units if unit}),
            "component_precision_unit_filter": bool(component_precision_unit_filter),
            "component_precision_filtered_unit_names": sorted({str(unit[0]) for unit in filtered_units if unit}),
            "legacy_core4_state_unit_filter": bool(legacy_core4_state_unit_filter),
            "legacy_core4_filtered_unit_names": sorted({str(unit[0]) for unit in legacy_filtered_units if unit}),
            "action_events": _trace_action_events(action_events),
        }
    trace = {
        "state_supported": True,
        "phi_before": float(before),
        "phi_after": float(after),
        "raw_delta": float(raw_delta),
        "delta": float(delta),
        "positive_delta": float(delta),
        "negative_delta": float(state_negative_delta),
        "net_delta": float(state_net_delta),
        "delta_quality": float(delta_quality),
        "actions": int(actions),
        "invalid": int(invalid),
        "units": _trace_units(units),
        "oracle_units": _trace_units(oracle_units),
        "new_units": _trace_units(new_units),
        "duplicate_units": _trace_units(units & old_units),
        "rewardable_units": _trace_units(rewardable_new_units),
        "weak_units": _trace_units(weak_new_units),
        "unit_names": sorted({str(unit[0]) for unit in units if unit}),
        "oracle_unit_names": sorted({str(unit[0]) for unit in oracle_units if unit}),
        "component_precision_unit_filter": bool(component_precision_unit_filter),
        "component_precision_filtered_units": _trace_units(filtered_units),
        "component_precision_filtered_unit_names": sorted({str(unit[0]) for unit in filtered_units if unit}),
        "legacy_core4_state_unit_filter": bool(legacy_core4_state_unit_filter),
        "legacy_core4_filtered_units": _trace_units(legacy_filtered_units),
        "legacy_core4_filtered_unit_names": sorted({str(unit[0]) for unit in legacy_filtered_units if unit}),
        "new_unit_names": sorted({str(unit[0]) for unit in new_units if unit}),
        "rewardable_unit_names": sorted({str(unit[0]) for unit in rewardable_new_units if unit}),
        "weak_unit_names": sorted({str(unit[0]) for unit in weak_new_units if unit}),
        "action_events": _trace_action_events(action_events),
        "component_before": {
            component: float(before_components.get(component, 0.0)) for component in STATE_PROGRESS_COMPONENTS
        },
        "component_after": {
            component: float(after_components.get(component, 0.0)) for component in STATE_PROGRESS_COMPONENTS
        },
        "component_delta": {
            component: float(component_stats.get(f"state_component_delta_{component}", 0.0))
            for component in STATE_PROGRESS_COMPONENTS
        },
    }
    return delta, stats, trace


def _neighborhood_units(spec: GraphLogicSpec, text: str, nodes: set[int]) -> set[tuple]:
    if spec.task not in {
        "neighbor",
        "common_neighbor",
        "jaccard_coefficient",
        "adamic_adar_index",
        "resource_allocation_index",
        "clustering_coefficient",
        "avg_neighbor_degree",
    }:
        return set()
    low = text.lower()
    if not any(cue in low for cue in ("neighbor", "neighbour", "common", "adjacent", "degree", "coefficient")):
        return set()
    adj = {node: set(nbrs) for node, nbrs in build_adj(spec).items()}
    units: set[tuple] = set()

    for m in re.finditer(
        r"(?:neighbors?|neighbours?)\s+of\s+(?:node\s+)?(\d+)\s*(?:are|is|:|=)\s*[\{\[\(]?\s*([0-9,\sand]+)",
        text,
        flags=re.I,
    ):
        node = int(m.group(1))
        if node not in spec.nodes:
            continue
        for raw in re.findall(r"\d+", m.group(2)):
            nbr = int(raw)
            if nbr in adj.get(node, set()):
                a, b = (node, nbr) if spec.directed or node <= nbr else (nbr, node)
                units.add(("neighbor_pair", int(a), int(b)))

    for m in re.finditer(
        r"(?:N|Γ|gamma)\s*\(\s*(\d+)\s*\)\s*(?:=|:)\s*[\{\[\(]?\s*([0-9,\sand]+)",
        text,
        flags=re.I,
    ):
        node = int(m.group(1))
        if node not in spec.nodes:
            continue
        for raw in re.findall(r"\d+", m.group(2)):
            nbr = int(raw)
            if nbr in adj.get(node, set()):
                a, b = (node, nbr) if spec.directed or node <= nbr else (nbr, node)
                units.add(("neighbor_pair", int(a), int(b)))

    if spec.source is not None and spec.target is not None:
        common = adj.get(spec.source, set()) & adj.get(spec.target, set())
        union = adj.get(spec.source, set()) | adj.get(spec.target, set())
        for node in (nodes & common):
            if any(cue in low for cue in ("common", "shared", "both", "neighbor")):
                units.add(("common_neighbor", int(node)))
        for m in re.finditer(r"(?:common|shared|intersection)[^:={]{0,40}(?:are|is|:|=)\s*[\{\[\(]?\s*([0-9,\sand]+)", text, flags=re.I):
            for raw in re.findall(r"\d+", m.group(1)):
                node = int(raw)
                if node in common:
                    units.add(("common_neighbor", int(node)))
        if spec.task in {"jaccard_coefficient", "adamic_adar_index", "resource_allocation_index"}:
            if common and any(cue in low for cue in ("common", "intersection", "shared")) and _mentions_number_value(
                text, len(common)
            ):
                units.add(("common_neighbor_count", int(len(common))))
            if (
                spec.task == "jaccard_coefficient"
                and union
                and any(cue in low for cue in ("union", "jaccard"))
                and _mentions_number_value(text, len(union))
            ):
                units.add(("neighbor_union_count", int(len(union))))
            metric_cues = {
                "jaccard_coefficient": ("jaccard", "coefficient"),
                "adamic_adar_index": ("adamic", "adar"),
                "resource_allocation_index": ("resource allocation", "allocation index"),
            }.get(spec.task, ())
            if spec.answer_numbers and (
                _has_answer_assertion_cue(spec, text) or any(cue in low for cue in metric_cues)
            ):
                target = float(spec.answer_numbers[0])
                final_text = _final_answer_text(text)
                search_text = final_text if final_text else text
                if any(_close_number(value, target) for value in _numeric_values(search_text)):
                    units.add(("neighborhood_metric_answer", spec.task, round(float(target), 6)))

    return units


def _boolean_witness_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    answer_hit: bool,
) -> set[tuple]:
    if spec.task not in BOOL_WITNESS_TASKS:
        return set()
    low = text.lower()
    units: set[tuple] = set()
    if spec.task == "has_cycle" and any(cue in low for cue in ("cycle", "back edge", "already visited")):
        for seq in _node_sequences(text, spec):
            cycle_nodes = seq[:-1] if len(seq) >= 2 and seq[0] == seq[-1] else seq
            if len(cycle_nodes) < 3 or len(set(cycle_nodes)) != len(cycle_nodes):
                continue
            cycle_edges = [_norm_edge(u, v, spec.directed) for u, v in zip(cycle_nodes, cycle_nodes[1:])]
            cycle_edges.append(_norm_edge(cycle_nodes[-1], cycle_nodes[0], spec.directed))
            if all(edge in spec.edges for edge in cycle_edges):
                units.add(("cycle_witness", tuple(cycle_edges[: min(len(cycle_edges), 8)])))
                for edge in cycle_edges:
                    units.add(("cycle_edge", edge))
        if valid_edges & set(spec.target_edges or set()):
            for edge in valid_edges & set(spec.target_edges or set()):
                units.add(("cycle_edge", edge))
        if spec.answer_bool is False and answer_hit:
            units.add(("acyclic_answer",))

    if spec.task == "is_bipartite":
        if any(cue in low for cue in ("partition", "color", "left", "right", "bipartition")):
            for node in nodes & spec.nodes:
                units.add(("bipartite_partition_node", int(node)))
        if spec.answer_bool is False and any(cue in low for cue in ("odd cycle", "same color", "not bipartite")):
            units.add(("bipartite_violation",))
        if spec.answer_bool is True and answer_hit:
            units.add(("bipartite_answer",))

    if spec.task in {"is_regular", "is_eularian", "is_tournament"} and answer_hit:
        units.add(("boolean_answer", spec.task, bool(spec.answer_bool)))
    return units


def _ordered_progress_value(
    edge_positions: list[int],
    edge_orders: list[list[tuple[int, int]]],
    node_pos: int,
    node_order: list[int],
) -> float:
    values = []
    for pos, order in zip(edge_positions, edge_orders):
        if order:
            values.append(float(pos) / max(1.0, float(len(order))))
    if node_order:
        values.append(float(node_pos) / max(1.0, float(len(node_order))))
    return max(values) if values else 0.0


def _potential_weights(
    spec: GraphLogicSpec,
    edge_orders: list[list[tuple[int, int]]],
    node_order: list[int],
) -> tuple[float, float, float, float]:
    has_order = bool(edge_orders or node_order)
    has_edges = bool(spec.target_edges)
    has_nodes = bool(spec.target_nodes)
    has_answer = bool(spec.answer_numbers or spec.answer_bool is not None)
    if has_order:
        weights = (0.70, 0.20 if has_edges else 0.0, 0.10 if has_nodes else 0.0, 0.10 if has_answer else 0.0)
    elif has_edges and has_nodes:
        weights = (0.0, 0.55, 0.35, 0.10 if has_answer else 0.0)
    elif has_edges:
        weights = (0.0, 0.85, 0.0, 0.15 if has_answer else 0.0)
    elif has_nodes:
        weights = (0.0, 0.0, 0.85, 0.15 if has_answer else 0.0)
    else:
        weights = (0.0, 0.0, 0.0, 1.0 if has_answer else 0.0)
    total = sum(weights)
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(float(w) / float(total) for w in weights)


def _generic_node_evidence_tasks() -> set[str]:
    return {
        "node_number",
        "degree",
        "degree_centrality",
        "avg_neighbor_degree",
        "neighbor",
        "common_neighbor",
        "jaccard_coefficient",
        "adamic_adar_index",
        "resource_allocation_index",
        "closeness_centrality",
        "harmonic_centrality",
        "betweenness_centrality",
        "barycenter",
        "center",
        "periphery",
        "radius",
        "diameter",
        "dominating_set",
        "maximal_independent_set",
        "min_vertex_cover",
        "local_connectivity",
    }


def _potential_evidence_universe(spec: GraphLogicSpec) -> tuple[set[tuple[int, int]], set[int]]:
    task = spec.task
    edge_universe = set(spec.target_edges or set())
    node_universe = set(spec.target_nodes or set())
    if task == "hamiltonian_path":
        edge_universe.update(spec.edges)
        node_universe.update(spec.nodes)
    elif task == "isomophic_mapping":
        node_universe.update(spec.answer_mapping or {})
        node_universe.update((spec.answer_mapping or {}).values())
    elif task in DISTANCE_WITNESS_TASKS:
        edge_universe.update(spec.edges)
        node_universe.update(spec.nodes)
    elif task in SET_FEASIBILITY_TASKS:
        edge_universe.update(spec.edges)
        node_universe.update(spec.nodes)
    elif task in MST_TASKS:
        edge_universe.update(spec.edges)
        node_universe.update(spec.nodes)
    elif task in PATH_TASKS | TRAVERSAL_TASKS | {"hamiltonian_path", "topological_sort", "traveling_salesman_problem"}:
        pass
    else:
        edge_universe.update(spec.edges)
        if task in _generic_node_evidence_tasks() or not edge_universe:
            node_universe.update(spec.nodes)
        if task in BOOL_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
            edge_universe.update(spec.edges)
            node_universe.update(spec.nodes)
    return edge_universe, node_universe


def _potential_evidence_budget(spec: GraphLogicSpec) -> int:
    if spec.task in PATH_TASKS | TRAVERSAL_TASKS:
        return min(24, max(1, len(spec.edges) + len(spec.nodes)))
    if spec.task in MST_TASKS:
        return min(24, max(1, len(spec.edges) + len(spec.nodes)))
    if spec.task in {"topological_sort", "traveling_salesman_problem"}:
        return min(24, max(1, len(spec.edges) + len(spec.nodes)))
    if spec.task == "hamiltonian_path":
        return min(24, max(1, len(spec.nodes) + max(0, len(spec.nodes) - 1)))
    if spec.task == "isomophic_mapping" and spec.answer_mapping:
        return min(24, len(spec.answer_mapping))
    if spec.task == "maximum_flow":
        return min(24, max(1, len(spec.edges) + len(spec.nodes) + 1))
    if spec.task == "bridges":
        return min(24, max(1, len(spec.edges) + len(spec.nodes)))
    if spec.task in {"connected_component_number", "strongly_connected_number"}:
        return min(24, max(1, len(spec.edges) + len(spec.nodes)))
    if spec.task in DISTANCE_WITNESS_TASKS:
        if spec.task in {"closeness_centrality", "harmonic_centrality"} and spec.source is not None:
            return min(24, max(1, len(spec.nodes) - 1))
        if spec.task == "betweenness_centrality":
            return min(24, max(1, (len(spec.nodes) - 1) * max(0, len(spec.nodes) - 2) // 2))
        return min(24, max(1, len(spec.nodes) * max(0, len(spec.nodes) - 1) // 2))
    if spec.task in SET_FEASIBILITY_TASKS:
        if spec.task in {"bipartite_maximum_matching", "max_weight_matching"}:
            return min(24, max(1, len(spec.nodes) // 2))
        if spec.task == "min_vertex_cover":
            return min(24, max(1, len(spec.edges)))
        return min(24, max(1, len(spec.nodes)))
    if spec.task in COUNT_FORMULA_TASKS:
        return 4
    if spec.task in DEGREE_WITNESS_TASKS:
        return min(24, max(1, len(spec.nodes)))
    edge_universe, node_universe = _potential_evidence_universe(spec)
    return min(24, len(edge_universe) + len(node_universe))


def _potential_evidence_weight(
    spec: GraphLogicSpec,
    edge_orders: list[list[tuple[int, int]]],
    node_order: list[int],
) -> float:
    if _potential_evidence_budget(spec) <= 0:
        return 0.0
    task = spec.task
    if task == "hamiltonian_path":
        return 0.25
    if task == "isomophic_mapping":
        return 0.65
    if task == "maximum_flow":
        # Flow vocabulary is common in wrong greedy-path answers. Keep generic
        # evidence light; high-credit flow progress is handled by residual-aware
        # state units below.
        return 0.25
    if task == "bridges":
        return 0.50
    if task in {"connected_component_number", "strongly_connected_number"}:
        return 0.35
    if task == "betweenness_centrality":
        return 0.45
    if task in DISTANCE_WITNESS_TASKS:
        return 0.30
    if task == "triangles":
        return 0.55
    if task == "min_edge_covering":
        return 0.65
    if task in SET_FEASIBILITY_TASKS:
        return 0.50
    if task in COUNT_FORMULA_TASKS:
        return 0.45
    if task in DEGREE_WITNESS_TASKS:
        return 0.35
    if task in PATH_TASKS | TRAVERSAL_TASKS | {"hamiltonian_path", "topological_sort", "traveling_salesman_problem"}:
        return 0.16
    if task in MST_TASKS:
        return 0.20
    if task in BOOL_WITNESS_TASKS | GLOBAL_WITNESS_TASKS:
        return 0.20
    if task in ANSWER_EDGE_TASKS | ANSWER_NODE_TASKS:
        return 0.15
    has_core = bool(edge_orders or node_order or spec.target_edges or spec.target_nodes)
    return 0.25 if has_core else 0.35


def _has_unreachable_pair(spec: GraphLogicSpec) -> bool:
    nodes = sorted(spec.nodes)
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if not spec.directed and u > v:
                continue
            if not math.isfinite(_shortest_distance(spec, u, v)):
                return True
    return False


def _unit_names(units: Iterable[tuple]) -> set[str]:
    return {str(unit[0]) for unit in units if unit}


def _template_complete(spec: GraphLogicSpec, units: Iterable[tuple]) -> bool:
    names = _unit_names(units)

    def has(*items: str) -> bool:
        return any(item in names for item in items)

    direction_ok = True
    if spec.directed and spec.task in {
        "betweenness_centrality",
        "closeness_centrality",
        "global_efficiency",
        "harmonic_centrality",
        "wiener_index",
    }:
        direction_ok = has("directed_scope")

    if spec.task == "closeness_centrality":
        return direction_ok and has("closeness_formula") and has(
            "closeness_distance_sum", "distance_sum", "distance_value"
        ) and has("closeness_normalizer_value", "source_reachable_count")
    if spec.task == "harmonic_centrality":
        return direction_ok and has("harmonic_formula") and has("harmonic_sum", "distance_value")
    if spec.task == "betweenness_centrality":
        return (
            direction_ok
            and has("betweenness_formula")
            and has("betweenness_excludes_endpoints")
            and has("betweenness_normalizer_value")
            and has("betweenness_pair_via", "betweenness_path_via", "betweenness_pair_checked")
        )
    if spec.task == "global_efficiency":
        unreachable_ok = (not _has_unreachable_pair(spec)) or has("global_unreachable_zero_contribution")
        return (
            direction_ok
            and unreachable_ok
            and has("global_efficiency_formula")
            and has("global_efficiency_normalizer")
            and has("global_efficiency_denominator_value")
            and has("all_pairs_scope")
        )
    if spec.task == "wiener_index":
        return direction_ok and has("wiener_distance_sum_formula") and has("wiener_pair_scope") and has(
            "distance_value", "distance_sum", "metric_answer", "global_metric_answer"
        )
    if spec.task == "weighted_shortest_path":
        return has("weighted_path_uses_weights") and has(
            "source_distance",
            "dijkstra_relax_edge",
            "state_dijkstra_relax",
            "state_dijkstra_weighted_relax_cert",
            "state_dijkstra_decreased_dist",
        ) and has("path_reaches_target", "path_prefix", "state_path_reaches_target", "state_dijkstra_target_cert")
    if spec.task == "maximum_flow":
        return (
            has("flow_path", "state_flow_path")
            and has("flow_capacity", "state_flow_capacity_checked")
            and has("flow_bottleneck", "state_flow_bottleneck")
            and has(
                "state_flow_residual_bottleneck_cert",
                "state_flow_residual_update_after_augment",
                "state_flow_min_cut_cert",
            )
            and has("flow_value", "state_flow_value")
        )
    return False


def _legacy_core4_compat_enabled(cfg: GraphLogicConfig | None) -> bool:
    return bool(cfg is not None and getattr(cfg, "legacy_core4_compat", False))


def _legacy_core4_protected_task(spec: GraphLogicSpec, cfg: GraphLogicConfig | None) -> bool:
    return _legacy_core4_compat_enabled(cfg) and spec.task in LEGACY_CORE4_PROTECTED_TASKS


def _legacy_core4_potential_evidence_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    state: GraphLogicPotentialState,
    answer_hit: bool,
    check_final_answer: bool,
) -> set[tuple]:
    """Approximate the old core4 evidence surface by excluding later broad parsers."""
    units: set[tuple] = set()
    units.update(_distance_pair_units(spec, text, valid_edges, nodes))
    units.update(_distance_metric_units(spec, text))
    units.update(_distance_formula_units(spec, text))
    units.update(_path_process_units(spec, text, valid_edges, nodes))
    units.update(_mst_process_units(spec, text, valid_edges))
    if check_final_answer:
        mst_final_units, _ = _mst_final_units(spec, text)
        units.update(mst_final_units)
    units.update(_neighborhood_units(spec, text, nodes))
    units.update(_boolean_witness_units(spec, text, valid_edges, nodes, answer_hit))
    units.update(_degree_units(spec, text))
    units.update(_count_formula_units(spec, text, answer_hit))
    units.update(_triangle_units(spec, text))
    units.update(_tournament_units(spec, text))
    units.update(_hamiltonian_units(spec, text))
    units.update(_betweenness_units(spec, text))
    return units


def _template_completeness_units(
    spec: GraphLogicSpec,
    evidence_units: set[tuple],
    seen_evidence: set[tuple],
) -> set[tuple]:
    if spec.task not in STRICT_TEMPLATE_TASKS:
        return set()
    combined = set(seen_evidence) | set(evidence_units)
    if _template_complete(spec, combined):
        return {("template_complete", spec.task)}
    return set()


def _potential_evidence_units(
    spec: GraphLogicSpec,
    text: str,
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    family_candidate: bool,
    node_progress_cue: bool,
    state: GraphLogicPotentialState,
    answer_hit: bool,
    check_final_answer: bool = False,
    cfg: GraphLogicConfig | None = None,
) -> set[tuple]:
    if not (family_candidate or node_progress_cue):
        return set()
    edge_universe, node_universe = _potential_evidence_universe(spec)
    if spec.task == "weighted_shortest_path" and not _has_any(text, ACTION_CUES["weighted_path"]):
        edge_universe = set()
        node_universe = set()
    units: set[tuple] = set()
    for edge in valid_edges & edge_universe:
        units.add(("edge", edge))
    edge_nodes = {node for edge in valid_edges for node in edge}
    standalone_nodes = nodes - edge_nodes if valid_edges else nodes
    can_add_node_evidence = (
        spec.task not in PATH_TASKS
        and (node_progress_cue or spec.task in _generic_node_evidence_tasks() or not valid_edges)
    )
    if can_add_node_evidence:
        for node in standalone_nodes & node_universe:
            units.add(("node", int(node)))
    if _legacy_core4_compat_enabled(cfg):
        units.update(
            _legacy_core4_potential_evidence_units(
                spec,
                text,
                valid_edges,
                nodes,
                state,
                answer_hit,
                check_final_answer,
            )
        )
        return units
    units.update(_distance_pair_units(spec, text, valid_edges, nodes))
    units.update(_distance_metric_units(spec, text))
    units.update(_distance_formula_units(spec, text))
    units.update(_global_metric_formula_units(spec, text, answer_hit))
    units.update(_mapping_format_units(spec, text))
    units.update(_mapping_pair_units(spec, text))
    units.update(_set_feasibility_units(spec, text, valid_edges, nodes, state))
    units.update(_set_size_units(spec, text))
    units.update(_bridge_units(spec, text, valid_edges))
    units.update(_component_units(spec, text, nodes, answer_hit))
    units.update(_flow_units(spec, text, valid_edges, check_final_answer))
    units.update(_path_process_units(spec, text, valid_edges, nodes))
    units.update(_mst_process_units(spec, text, valid_edges))
    if check_final_answer:
        mst_final_units, _ = _mst_final_units(spec, text)
        units.update(mst_final_units)
    units.update(_neighborhood_units(spec, text, nodes))
    units.update(_boolean_witness_units(spec, text, valid_edges, nodes, answer_hit))
    units.update(_degree_units(spec, text))
    units.update(_count_formula_units(spec, text, answer_hit))
    units.update(_triangle_units(spec, text))
    units.update(_tournament_units(spec, text))
    units.update(_hamiltonian_units(spec, text))
    units.update(_betweenness_units(spec, text))
    units.update(_template_completeness_units(spec, units, state.seen_evidence))
    return units


def _potential_value(
    spec: GraphLogicSpec,
    state: GraphLogicPotentialState,
    edge_orders: list[list[tuple[int, int]]],
    node_order: list[int],
) -> float:
    target_edges = set(spec.target_edges or set())
    target_nodes = set(spec.target_nodes or set())
    w_order, w_edge, w_node, w_answer = _potential_weights(spec, edge_orders, node_order)
    order_value = _ordered_progress_value(state.edge_positions, edge_orders, state.node_pos, node_order)
    edge_value = (
        float(len(state.seen_edges & target_edges)) / max(1.0, float(len(target_edges))) if target_edges else 0.0
    )
    node_value = (
        float(len(state.seen_nodes & target_nodes)) / max(1.0, float(len(target_nodes))) if target_nodes else 0.0
    )
    answer_value = 1.0 if state.answer_seen else 0.0
    core_value = w_order * order_value + w_edge * edge_value + w_node * node_value + w_answer * answer_value
    evidence_weight = _potential_evidence_weight(spec, edge_orders, node_order)
    budget = _potential_evidence_budget(spec)
    if evidence_weight <= 0.0 or budget <= 0:
        return core_value
    # Keep the original linear shaping inside the evidence budget, then add a
    # small bounded tail so long correct traces do not hit a hard no-credit wall.
    evidence_count = float(len(state.seen_evidence))
    if evidence_count <= float(budget):
        evidence_value = evidence_count / float(budget)
    else:
        overflow = evidence_count - float(budget)
        evidence_value = 1.0 + 0.25 * (1.0 - math.exp(-overflow / max(1e-6, float(budget))))
    return core_value + evidence_weight * evidence_value


def _clone_potential_state(state: GraphLogicPotentialState) -> GraphLogicPotentialState:
    return GraphLogicPotentialState(
        seen_edges=set(state.seen_edges),
        seen_nodes=set(state.seen_nodes),
        edge_positions=list(state.edge_positions),
        node_pos=int(state.node_pos),
        answer_seen=bool(state.answer_seen),
        seen_evidence=set(state.seen_evidence),
    )


def _edge_signature(edges: Iterable[tuple[int, int]], limit: int = 1) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(u), int(v)) for u, v in edges)[:limit])


def _node_signature(nodes: Iterable[int], limit: int = 1) -> tuple[int, ...]:
    return tuple(sorted(int(node) for node in nodes)[:limit])


def _answer_signature(spec: GraphLogicSpec) -> tuple:
    if spec.answer_bool is not None:
        return ("answer_bool", bool(spec.answer_bool))
    if spec.answer_numbers:
        return ("answer_scalar",)
    return ("answer",)


def _step_anchor(
    spec: GraphLogicSpec,
    kind: str,
    matched_ordered_edges: list[tuple[int, int]],
    progressive_edges: set[tuple[int, int]],
    progressive_nodes: set[int],
    valid_edges: set[tuple[int, int]],
    nodes: set[int],
    invalid_edges: set[tuple[int, int]],
    answer_hit: bool,
) -> tuple:
    if kind == "invalid" and invalid_edges:
        return ("graph_logic", spec.task, "invalid_edge", _edge_signature(invalid_edges))
    if matched_ordered_edges:
        return ("graph_logic", spec.task, "edge", _edge_signature(matched_ordered_edges))
    if progressive_edges:
        return ("graph_logic", spec.task, "edge", _edge_signature(progressive_edges))
    if progressive_nodes:
        return ("graph_logic", spec.task, "node", _node_signature(progressive_nodes))
    if answer_hit:
        return ("graph_logic", spec.task, *_answer_signature(spec))
    if valid_edges:
        return ("graph_logic", spec.task, "legal_edge", _edge_signature(valid_edges))
    if nodes:
        return ("graph_logic", spec.task, "legal_node", _node_signature(nodes))
    return ("graph_logic", spec.task, kind)


def observe_graph_logic_response(
    text: str,
    spec: GraphLogicSpec,
    config: GraphLogicConfig | None = None,
    return_trace: bool = False,
    token_entropy_offsets: Sequence[Sequence[int] | tuple[int, int]] | None = None,
    token_entropies: Sequence[float] | None = None,
) -> (
    tuple[list[GraphLogicStep], dict[str, float]]
    | tuple[list[GraphLogicStep], dict[str, float], list[dict[str, Any]]]
    ):
    cfg = config or GraphLogicConfig()
    stats = defaultdict(float)
    legacy_core4_compat = _legacy_core4_compat_enabled(cfg)
    legacy_core4_protected = _legacy_core4_protected_task(spec, cfg)
    if legacy_core4_compat:
        stats["legacy_core4_compat"] = 1.0
        stats["legacy_core4_protected_task"] = float(legacy_core4_protected)
    reward_mode = str(getattr(cfg, "reward_mode", "event") or "event").lower()
    if reward_mode not in {"event", "potential"}:
        reward_mode = "event"
    target_edges = set(spec.target_edges or set())
    target_nodes = set(spec.target_nodes or set())
    edge_orders = [list(spec.target_edge_order or [])]
    edge_orders.extend(list(order) for order in (spec.alt_target_edge_orders or []))
    edge_orders = [order for order in edge_orders if order]
    node_order = list(spec.target_node_order or []) if spec.task in TRAVERSAL_TASKS | {"topological_sort"} else []
    edge_positions = [0 for _ in edge_orders]
    node_pos = 0
    seen_edges: set[tuple[int, int]] = set()
    seen_nodes: set[int] = set()
    observed_evidence: set[tuple] = set()
    potential_state = GraphLogicPotentialState(
        seen_edges=set(),
        seen_nodes=set(),
        edge_positions=[0 for _ in edge_orders],
        node_pos=0,
        answer_seen=False,
    )
    algorithm_state = _init_algorithm_state(spec, cfg)
    steps: list[GraphLogicStep] = []
    trace: list[dict[str, Any]] = []
    first_irrecoverable_error_seen = False
    weight = _task_weight(spec.task)
    check_final_answer = bool(getattr(cfg, "final_answer_checks", False))
    if algorithm_state is not None:
        stats["state_supported"] = 1.0
        stats["state_potential_initial"] = float(_algorithm_state_potential(spec, algorithm_state))
    if reward_mode == "potential":
        stats["potential_initial"] = float(_potential_value(spec, potential_state, edge_orders, node_order))

    router_budget_used = 0
    router_budget_limit = max(0, int(getattr(cfg, "action_router_max_segments_per_response", 0) or 0))
    if router_budget_limit > 0:
        stats["action_router_budget_enabled"] = 1.0
        stats["action_router_budget_limit"] = float(router_budget_limit)
    for start, end, seg_text in split_segments_with_spans(
        text,
        config=cfg,
        stats=stats,
        token_entropy_offsets=token_entropy_offsets,
        token_entropies=token_entropies,
    ):
        if len(steps) >= cfg.max_steps:
            break
        _advance_state_context_age(algorithm_state)
        answer_hit = _scalar_answer_hit(seg_text, spec) if check_final_answer else False
        fact_dump = _fact_dump(seg_text) and not _allow_solution_list_segment(spec.task, seg_text)
        family_candidate = _infer_family(spec.task, seg_text)
        state_mention = bool(algorithm_state is not None and _state_process_action_mentioned(spec, seg_text))
        candidate = family_candidate or _answer_like(seg_text) or answer_hit or state_mention
        segment_trace: dict[str, Any] | None = None
        if return_trace:
            segment_trace = {
                "segment_id": len(trace),
                "segment_text": seg_text,
                "char_span": [int(start), int(end)],
                "state_supported": bool(algorithm_state is not None),
                "candidate": bool(candidate),
                "family_candidate": bool(family_candidate),
                "answer_like": bool(_answer_like(seg_text)),
                "answer_hit": bool(answer_hit),
                "state_mention": bool(state_mention),
                "fact_dump": bool(fact_dump),
                "grounded": False,
                "executor_valid": False,
                "progressive": False,
                "ordered": False,
                "reward": 0.0,
                "kind": "none",
            }
        stats["segments"] += 1
        stats["candidate"] += float(candidate)
        stats["fact_dump"] += float(fact_dump)
        stats["answer_hit"] += float(answer_hit)
        if algorithm_state is not None:
            stats["state_raw_segments"] += 1.0
            stats["state_candidate_segments"] += float(candidate)
            stats["state_mentions"] += float(state_mention)
            if not candidate and not fact_dump:
                stats["state_not_candidate_segments"] += 1.0
        if fact_dump:
            if algorithm_state is not None:
                stats["state_fact_dump_segments"] += 1.0
            fact_dump_reward = 0.0
            if cfg.fact_dump_penalty > 0.0:
                fact_dump_reward = -float(cfg.fact_dump_penalty)
                steps.append(
                    GraphLogicStep(
                        fact_dump_reward,
                        ("graph_logic", spec.task, "fact_dump"),
                        (start, end),
                        label="NEG_REC",
                        unit_names=("fact_dump",),
                    )
                )
            if segment_trace is not None:
                segment_trace.update(
                    {
                        "skipped_reason": "fact_dump",
                        "reward": fact_dump_reward,
                        "kind": "fact_dump",
                    }
                )
                trace.append(segment_trace)
            continue
        router_routed: RoutedAction | None = None
        router_selected: ActionCandidate | None = None
        router_candidates: list[ActionCandidate] = []
        router_payload: RouterPayload | None = None
        router_active = bool(algorithm_state is not None and _action_router_active(spec, cfg))
        router_priority = 0
        router_budget_allows = True
        if router_active:
            router_priority = _action_router_segment_priority(
                spec,
                seg_text,
                candidate=bool(candidate),
                family_candidate=bool(family_candidate),
                answer_hit=bool(answer_hit),
                state_mention=bool(state_mention),
            )
            stats["action_router_high_value_score"] += float(router_priority)
            if router_priority > 0:
                stats["action_router_high_value_segments"] += 1.0
            router_budget_allows = router_budget_limit <= 0 or (
                router_budget_used < router_budget_limit and router_priority > 0
            )
            if router_budget_limit > 0:
                stats["action_router_budget_candidate_segments"] += float(router_priority > 0)
                if not router_budget_allows:
                    stats["action_router_budget_skipped_segments"] += 1.0
                    if segment_trace is not None:
                        segment_trace["action_router_budget_skipped"] = True
                        segment_trace["action_router_priority"] = int(router_priority)
        if router_active and router_budget_allows:
            if router_budget_limit > 0:
                router_budget_used += 1
                stats["action_router_budget_used_segments"] += 1.0
            router_routed, router_selected, router_candidates, router_payload = _route_action_segment(
                spec,
                seg_text,
                algorithm_state,
                cfg,
            )
            if router_budget_limit > 0 and not router_candidates:
                router_budget_used = max(0, router_budget_used - 1)
                stats["action_router_budget_used_segments"] = max(
                    0.0,
                    float(stats.get("action_router_budget_used_segments", 0.0)) - 1.0,
                )
            stats["action_router_segments"] += 1.0
            stats["action_router_candidates"] += float(len(router_candidates))
            stats["action_router_no_candidate_segments"] += float(not router_candidates)
            stats["action_router_selected_segments"] += float(
                bool(router_routed is not None and not router_routed.abstain)
            )
            stats["action_router_abstain_segments"] += float(
                bool(router_routed is None or router_routed.abstain)
            )
            if router_routed is not None:
                stats["action_router_confidence"] += float(router_routed.confidence)
                stats["action_router_ambiguous_segments"] += float(router_routed.ambiguous)
                stats[f"action_router_mode_{_action_router_mode(cfg)}"] += 1.0
                stats[f"action_router_source_{router_routed.source}"] += 1.0
                if str(router_routed.source) == "http_error":
                    stats["action_router_http_error_segments"] += 1.0
                if str(router_routed.source) == "http_missing_endpoint":
                    stats["action_router_http_missing_endpoint_segments"] += 1.0
                if router_selected is None and not router_routed.abstain:
                    stats["action_router_invalid_selection_segments"] += 1.0
            if _action_router_confident(router_routed, cfg):
                candidate = True
                stats["action_router_confident_segments"] += 1.0
                if segment_trace is not None:
                    segment_trace["candidate"] = True
                    segment_trace["action_router_candidate"] = True
        elif segment_trace is not None:
            segment_trace["action_router_candidate"] = False
        if not candidate:
            if segment_trace is not None:
                segment_trace["skipped_reason"] = "not_candidate"
                trace.append(segment_trace)
            continue

        explicit_edges = _explicit_edges(seg_text, spec)
        explicit_valid_edges = explicit_edges & spec.edges
        valid_edges = set(explicit_valid_edges)
        raw_invalid_edges = {
            e
            for e in explicit_edges
            if e not in spec.edges
            and e[0] in spec.nodes
            and e[1] in spec.nodes
            and _edge_has_hard_invalid_mention(seg_text, e, spec.directed)
        }
        suppressed_invalid_edges = {
            e
            for e in raw_invalid_edges
            if _edge_is_negated_or_rejected(seg_text, e, spec)
            or _edge_is_structural_pair_reference(seg_text, e, spec)
        }
        invalid_edges = raw_invalid_edges - suppressed_invalid_edges
        explicit_nodes = _numbers_in_text(seg_text, spec.nodes)
        nodes = set(explicit_nodes)
        context_edges, context_nodes = _state_contextual_grounding(
            spec,
            seg_text,
            algorithm_state,
            valid_edges,
            nodes,
            cfg,
        )
        if context_edges or context_nodes:
            valid_edges.update(context_edges)
            nodes.update(context_nodes)
            stats["state_context_grounded_segments"] += 1.0
            stats["state_context_edges"] += float(len(context_edges))
            stats["state_context_nodes"] += float(len(context_nodes))
        candidate_edges, candidate_nodes, candidate_grounding_stats = _state_candidate_action_grounding(
            spec,
            seg_text,
            algorithm_state,
            valid_edges,
            nodes,
            cfg,
        )
        if candidate_edges or candidate_nodes:
            valid_edges.update(candidate_edges)
            nodes.update(candidate_nodes)
            stats["state_candidate_grounded_segments"] += 1.0
            stats["state_candidate_grounded_edges"] += float(len(candidate_edges))
            stats["state_candidate_grounded_nodes"] += float(len(candidate_nodes))
            stats["state_candidate_grounding_confidence"] += float(
                candidate_grounding_stats.get("state_candidate_grounding_confidence", 0.0)
            )
            stats["state_candidate_grounding_commitment"] += float(
                candidate_grounding_stats.get("state_candidate_grounding_commitment", 0.0)
            )
        router_edges: set[tuple[int, int]] = set()
        router_nodes: set[int] = set()
        if router_active and _action_router_can_ground(cfg):
            router_edges, router_nodes = _action_router_edges_nodes(router_routed, router_selected, cfg)
            router_edges = {edge for edge in router_edges if edge in spec.edges}
            router_nodes = {node for node in router_nodes if node in spec.nodes}
            if router_edges or router_nodes:
                valid_edges.update(router_edges)
                nodes.update(router_nodes)
                stats["action_router_grounded_segments"] += 1.0
                stats["action_router_grounded_edges"] += float(len(router_edges))
                stats["action_router_grounded_nodes"] += float(len(router_nodes))
        formula_grounded = bool(
            algorithm_state is not None
            and state_mention
            and _state_formula_or_certificate_grounded(spec, seg_text)
        )
        grounded = bool(valid_edges or nodes or answer_hit or formula_grounded)
        stats["grounded"] += float(grounded)
        if formula_grounded:
            stats["state_formula_grounded_segments"] += 1.0
        stats["invalid_edges"] += float(len(invalid_edges))
        stats["suppressed_invalid_edges"] += float(len(suppressed_invalid_edges))
        if segment_trace is not None:
            segment_trace.update(
                {
                    "explicit_edges": _trace_value(explicit_edges),
                    "valid_edges": _trace_value(valid_edges),
                    "invalid_edges": _trace_value(invalid_edges),
                    "suppressed_invalid_edges": _trace_value(suppressed_invalid_edges),
                    "explicit_nodes": _trace_value(explicit_nodes),
                    "nodes": _trace_value(nodes),
                    "context_edges": _trace_value(context_edges),
                    "context_nodes": _trace_value(context_nodes),
                    "candidate_edges": _trace_value(candidate_edges),
                    "candidate_nodes": _trace_value(candidate_nodes),
                    "candidate_grounding": _trace_value(candidate_grounding_stats),
                    "router_edges": _trace_value(router_edges),
                    "router_nodes": _trace_value(router_nodes),
                    "action_router": _trace_action_router(
                        router_routed,
                        router_selected,
                        router_candidates,
                        payload=router_payload,
                        include_payload=bool(getattr(cfg, "action_router_payload_trace", False)),
                    ),
                    "grounded": bool(grounded),
                }
            )
        if not grounded:
            if algorithm_state is not None:
                stats["state_not_grounded_segments"] += 1.0
                if state_mention:
                    stats["state_mention_only_segments"] += 1.0
            if segment_trace is not None:
                segment_trace["skipped_reason"] = "not_grounded"
                trace.append(segment_trace)
            continue
        _remember_state_context(algorithm_state, explicit_valid_edges, explicit_nodes)

        segment_target_edges = target_edges
        segment_target_nodes = target_nodes
        if spec.task == "weighted_shortest_path":
            if not _has_any(seg_text, ACTION_CUES["weighted_path"]):
                segment_target_edges = set()
                segment_target_nodes = set()
        progressive_edges = (valid_edges & segment_target_edges) - seen_edges
        progressive_nodes = (nodes & segment_target_nodes) - seen_nodes
        new_edge_positions = list(edge_positions)
        new_node_pos = node_pos
        matched_ordered_edges = []
        for oi, order in enumerate(edge_orders):
            pos = new_edge_positions[oi]
            if pos < len(order) and order[pos] in valid_edges:
                matched_ordered_edges.append(order[pos])
                new_edge_positions[oi] += 1
        ordered = bool(matched_ordered_edges)
        if matched_ordered_edges:
            progressive_edges.update(matched_ordered_edges)
        elif not edge_orders and node_order and node_pos < len(node_order) and node_order[node_pos] in nodes:
            ordered = True
            progressive_nodes.add(node_order[node_pos])
            new_node_pos = node_pos + 1

        node_progress_cue = _has_node_progress_cue(seg_text) or _answer_like(seg_text) or answer_hit
        node_progress = bool(
            progressive_nodes
            and _allow_node_only_progress(spec.task)
            and node_progress_cue
        )
        node_only_progress = bool(node_progress and not valid_edges)
        format_invalid = _mapping_format_invalid(spec, seg_text) if check_final_answer else False
        numeric_final_invalid = (
            (
                _numeric_final_invalid(spec, seg_text)
                or _numeric_final_missing_invalid(spec, seg_text)
                or _distance_definition_invalid(spec, seg_text)
            )
            if check_final_answer
            else False
        )
        path_final_invalid = (
            (_path_final_invalid(spec, seg_text) or _weighted_path_cost_final_invalid(spec, seg_text))
            if check_final_answer
            else False
        )
        schema_conflict_reason = _schema_final_answer_conflict_reason(spec, seg_text)
        if legacy_core4_compat and not check_final_answer:
            # The old core4 path did not use the later schema-conflict parser as
            # an executor-valid/invalid source. Keep the parser visible in newer
            # modes, but do not let it alter the legacy compatibility reward.
            schema_conflict_reason = None
        schema_conflict_invalid = bool(schema_conflict_reason)
        template_seen_evidence = set(potential_state.seen_evidence) | observed_evidence
        evidence_units = _potential_evidence_units(
            spec,
            seg_text,
            valid_edges,
            nodes,
            family_candidate,
            node_progress_cue,
            potential_state,
            answer_hit,
            check_final_answer,
            cfg,
        )
        if not legacy_core4_compat:
            evidence_units.update(_template_completeness_units(spec, evidence_units, template_seen_evidence))
        observed_evidence.update(evidence_units)

        needs_action_events = str(getattr(cfg, "step_span_mode", "segment") or "segment").lower() in {
            "action",
            "actions",
            "action_based",
            "action-based",
        } or bool(router_active and _action_router_mode(cfg) != "disabled")
        if return_trace or needs_action_events:
            state_delta, state_step_stats, state_step_trace = _algorithm_state_step(
                spec,
                seg_text,
                valid_edges,
                nodes,
                algorithm_state,
                cfg,
                check_final_answer,
                return_trace=return_trace,
                return_action_events=needs_action_events,
            )
        else:
            state_delta, state_step_stats = _algorithm_state_step(
                spec,
                seg_text,
                valid_edges,
                nodes,
                algorithm_state,
                cfg,
                check_final_answer,
            )
            state_step_trace = {}
        router_accepted = bool(
            router_active
            and _action_router_confident(router_routed, cfg)
            and _action_router_acceptance(router_routed, router_selected, state_step_trace)
        )
        if router_active:
            stats["action_router_accepted_segments"] += float(router_accepted)
            if router_routed is not None and router_selected is not None:
                stats[f"action_router_type_{router_selected.action_type.lower()}"] += 1.0
            if segment_trace is not None:
                segment_trace["action_router"] = _trace_action_router(
                    router_routed,
                    router_selected,
                    router_candidates,
                    accepted=router_accepted,
                    payload=router_payload,
                    include_payload=bool(getattr(cfg, "action_router_payload_trace", False)),
                )
        if router_accepted and _action_router_can_ground(cfg):
            router_event = _action_router_event(router_routed, router_selected)  # type: ignore[arg-type]
            if router_event is not None:
                state_step_trace.setdefault("action_events", [])
                if isinstance(state_step_trace["action_events"], list):
                    state_step_trace["action_events"].append(router_event)
        action_event_spans = _global_action_event_spans(
            state_step_trace,
            start,
            end,
            seg_text,
            cfg,
        )
        if action_event_spans:
            stats["action_event_bound_segments"] += 1.0
            stats["action_event_bound_spans"] += float(len(action_event_spans))
        if algorithm_state is not None:
            stats["state_supported_segments"] += 1.0
            for key, value in state_step_stats.items():
                stats[key] += float(value)
        state_invalid_count = (
            float(state_step_stats.get("state_invalid_transitions", 0.0)) if algorithm_state is not None else 0.0
        )
        state_negative_delta = (
            float(state_step_stats.get("state_negative_delta", 0.0)) if algorithm_state is not None else 0.0
        )
        state_action_count = float(state_step_stats.get("state_actions", 0.0)) if algorithm_state is not None else 0.0
        state_rewardable_count = (
            float(state_step_stats.get("state_rewardable_new_units", 0.0)) if algorithm_state is not None else 0.0
        )
        state_consistent = bool(
            algorithm_state is not None
            and state_mention
            and state_invalid_count <= 0.0
            and (state_action_count > 0.0 or valid_edges or nodes or context_edges or context_nodes)
        )
        state_reward_consistent = bool(
            state_consistent
            and (
                state_delta > 0.0
                or state_rewardable_count > 0.0
                or float(state_step_stats.get("state_terminal_units", 0.0)) > 0.0
            )
        )
        if algorithm_state is not None:
            stats["state_mention_grounded_segments"] += float(state_mention)
            stats["state_consistent_segments"] += float(state_consistent)
            stats["state_reward_consistent_segments"] += float(state_reward_consistent)
        if segment_trace is not None:
            segment_trace.update(
                {
                    "state": _trace_value(state_step_trace),
                    "state_consistent": bool(state_consistent),
                    "state_reward_consistent": bool(state_reward_consistent),
                    "state_actions": float(state_step_stats.get("state_actions", 0.0)),
                    "state_invalid_transitions": float(
                        state_step_stats.get("state_invalid_transitions", 0.0)
                    ),
                    "state_rewardable_new_units": float(
                        state_step_stats.get("state_rewardable_new_units", 0.0)
                    ),
                    "state_negative_delta": float(state_negative_delta),
                    "state_weak_new_units": float(state_step_stats.get("state_weak_new_units", 0.0)),
                    "state_delta": float(state_delta),
                    "action_event_spans": [[int(left), int(right)] for left, right in action_event_spans],
                }
            )

        potential_delta = 0.0
        effective_potential_delta = 0.0
        if reward_mode == "potential":
            before = _potential_value(spec, potential_state, edge_orders, node_order)
            candidate_state = _clone_potential_state(potential_state)
            candidate_state.edge_positions = new_edge_positions
            candidate_state.node_pos = new_node_pos
            candidate_state.seen_edges.update(valid_edges & segment_target_edges)
            if node_progress:
                candidate_state.seen_nodes.update(nodes & segment_target_nodes)
            if answer_hit:
                candidate_state.answer_seen = True
            before_evidence_units = set(candidate_state.seen_evidence)
            candidate_state.seen_evidence.update(evidence_units)
            new_evidence_units = candidate_state.seen_evidence - before_evidence_units
            evidence_delta = len(new_evidence_units)
            task_evidence_delta = sum(
                1 for unit in new_evidence_units if unit and unit[0] not in {"edge", "node"}
            )
            after = _potential_value(spec, candidate_state, edge_orders, node_order)
            potential_delta = max(0.0, after - before)
            effective_potential_delta = (
                _calibrated_potential_delta(spec, potential_delta, state_delta, state_step_stats, cfg)
                if algorithm_state is not None
                else potential_delta
            )
            progressive = effective_potential_delta > 0.0 or state_delta > 0.0
            executor_valid = bool(
                valid_edges
                or node_progress
                or answer_hit
                or progressive
                or state_delta > 0.0
                or state_reward_consistent
                or format_invalid
                or numeric_final_invalid
                or path_final_invalid
                or schema_conflict_invalid
            )
        else:
            evidence_delta = 0
            task_evidence_delta = 0
            progressive = bool(progressive_edges or node_progress or answer_hit or state_delta > 0.0)
            executor_valid = bool(
                valid_edges
                or node_progress
                or (nodes and not target_edges)
                or answer_hit
                or state_delta > 0.0
                or state_reward_consistent
                or format_invalid
                or numeric_final_invalid
                or path_final_invalid
                or schema_conflict_invalid
            )

        stats["executor_valid"] += float(executor_valid)
        stats["progressive"] += float(progressive)
        stats["ordered"] += float(ordered)
        stats["node_only_progress"] += float(node_only_progress)
        stats["format_invalid"] += float(format_invalid)
        stats["numeric_final_invalid"] += float(numeric_final_invalid)
        stats["path_final_invalid"] += float(path_final_invalid)
        stats["schema_conflict_invalid"] += float(schema_conflict_invalid)
        if schema_conflict_reason:
            stats[f"schema_conflict_{schema_conflict_reason}"] += 1.0
        stats["template_complete"] += float(
            any(unit[:2] == ("template_complete", spec.task) for unit in evidence_units)
        )
        if segment_trace is not None:
            segment_trace.update(
                {
                    "executor_valid": bool(executor_valid),
                    "progressive": bool(progressive),
                    "ordered": bool(ordered),
                    "node_only_progress": bool(node_only_progress),
                    "progressive_edges": _trace_value(progressive_edges),
                    "progressive_nodes": _trace_value(progressive_nodes),
                    "matched_ordered_edges": _trace_value(matched_ordered_edges),
                    "format_invalid": bool(format_invalid),
                    "numeric_final_invalid": bool(numeric_final_invalid),
                    "path_final_invalid": bool(path_final_invalid),
                    "schema_conflict_invalid": bool(schema_conflict_invalid),
                    "schema_conflict_reason": schema_conflict_reason,
                    "evidence_units": _trace_units(evidence_units),
                    "evidence_delta": float(evidence_delta),
                    "task_evidence_delta": float(task_evidence_delta),
                    "potential_delta_raw": float(potential_delta),
                    "potential_delta": float(effective_potential_delta),
                }
            )
        if not executor_valid:
            if segment_trace is not None:
                segment_trace["skipped_reason"] = "not_executor_valid"
                trace.append(segment_trace)
            continue

        reward = 0.0
        kind = "legal"
        if reward_mode == "potential":
            reward += float(cfg.potential_scale) * effective_potential_delta
            if state_delta > 0.0:
                positive_scale = _state_positive_delta_scale(spec, state_step_stats)
                reward = max(reward, float(cfg.state_reward_scale) * state_delta * positive_scale)
            stats["potential_delta_raw"] += float(potential_delta)
            stats["potential_delta"] += float(effective_potential_delta)
            stats["potential_positive_steps"] += float(effective_potential_delta > 0.0)
            stats["potential_evidence_units"] += float(evidence_delta)
            stats["potential_evidence_steps"] += float(evidence_delta > 0)
            stats["potential_task_evidence_units"] += float(task_evidence_delta)
            stats["potential_task_evidence_steps"] += float(task_evidence_delta > 0)
            if ordered and effective_potential_delta > 0.0:
                kind = "ordered"
            elif effective_potential_delta > 0.0 or state_delta > 0.0:
                kind = "progress"
            else:
                kind = "legal"
        else:
            if ordered:
                reward += cfg.ordered_reward
                kind = "ordered"
            elif progressive:
                if answer_hit and not (progressive_edges or progressive_nodes):
                    reward += cfg.answer_reward
                else:
                    reward += cfg.progress_reward
                kind = "progress"
            else:
                reward += cfg.legal_reward
            if state_delta > 0.0:
                positive_scale = _state_positive_delta_scale(spec, state_step_stats)
                reward = max(reward, float(cfg.state_reward_scale) * state_delta * positive_scale)
        if algorithm_state is not None and state_invalid_count <= 0.0:
            state_floor_reward = 0.0
            if state_reward_consistent:
                floor_candidate = float(getattr(cfg, "state_consistent_reward", 0.0))
                if spec.task in ACCUMULATION_SENSITIVE_STATE_TASKS:
                    terminal_units = float(state_step_stats.get("state_terminal_units", 0.0))
                    if state_delta > 0.0:
                        # For hard numeric/optimization tasks, repeated formula or scope mentions should not
                        # accumulate a fixed positive floor unless the executable algorithm state actually moved.
                        floor_candidate = min(floor_candidate, float(cfg.state_reward_scale) * state_delta)
                    elif terminal_units > 0.0:
                        floor_candidate = min(floor_candidate, 0.25 * floor_candidate)
                    else:
                        floor_candidate = 0.0
                state_floor_reward = max(state_floor_reward, floor_candidate)
            if state_floor_reward > 0.0 and reward < state_floor_reward:
                reward = state_floor_reward
                kind = "state_consistent"
        if state_negative_delta > 0.0:
            reward -= float(cfg.state_reward_scale) * state_negative_delta
            kind = "invalid" if reward < 0 else kind
        duplicate_no_progress = bool(
            algorithm_state is not None
            and float(state_step_stats.get("state_duplicate_no_progress_segments", 0.0)) > 0.0
            and state_invalid_count <= 0.0
            and state_delta <= 0.0
        )
        if (
            duplicate_no_progress
            and float(getattr(cfg, "duplicate_no_progress_penalty", 0.0)) > 0.0
        ):
            duplicate_penalty = float(getattr(cfg, "duplicate_no_progress_penalty", 0.0)) * _state_invalid_penalty_scale(spec.task)
            reward -= duplicate_penalty
            stats["duplicate_no_progress_penalty"] += float(duplicate_penalty)
            kind = "duplicate" if reward < 0 else kind
            if segment_trace is not None:
                segment_trace["duplicate_no_progress"] = True
                segment_trace["duplicate_no_progress_penalty"] = float(duplicate_penalty)
        state_ambiguity_score = (
            _state_segment_ambiguity_score(
                grounded,
                state_mention,
                candidate_edges,
                candidate_nodes,
                state_step_stats,
                state_step_trace,
                duplicate_no_progress,
            )
            if algorithm_state is not None
            else 0.0
        )
        state_executable_commitment_score = (
            _state_executable_commitment_score(
                grounded,
                executor_valid,
                state_consistent,
                state_reward_consistent,
                state_step_stats,
                state_step_trace,
                state_ambiguity_score,
            )
            if algorithm_state is not None
            else 1.0
        )
        hard_cert_units = _state_segment_hard_cert_units(state_step_trace) if algorithm_state is not None else set()
        graph_sensitive_units = (
            _state_segment_graph_sensitive_units(state_step_trace) if algorithm_state is not None else set()
        )
        if algorithm_state is not None:
            stats["state_ambiguity_scored_segments"] += 1.0
            stats["state_ambiguity_score"] += float(state_ambiguity_score)
            stats["state_executable_commitment_score"] += float(state_executable_commitment_score)
            stats["state_hard_cert_segments"] += float(bool(hard_cert_units))
            stats["state_graph_sensitive_segments"] += float(bool(graph_sensitive_units))
        if state_invalid_count > 0.0 and float(getattr(cfg, "state_invalid_penalty", 0.0)) > 0.0:
            completion_invalid = float(state_step_stats.get("state_completion_invalid", 0.0)) > 0.0
            invalid_penalty_multiplier = 1.0 if completion_invalid else 0.35
            reward -= (
                float(getattr(cfg, "state_invalid_penalty", 0.0))
                * _state_invalid_penalty_scale(spec.task)
                * invalid_penalty_multiplier
                * min(3.0, state_invalid_count)
            )
            kind = "invalid" if reward < 0 else kind
        if invalid_edges:
            reward -= cfg.invalid_penalty * _invalid_penalty_scale(spec.task) * min(3, len(invalid_edges))
            kind = "invalid" if reward < 0 else kind
        if format_invalid:
            reward -= cfg.invalid_penalty * 2.0
            kind = "invalid" if reward < 0 else kind
        if numeric_final_invalid:
            reward -= cfg.invalid_penalty * 4.0
            kind = "invalid" if reward < 0 else kind
        if path_final_invalid:
            reward -= cfg.invalid_penalty * 4.0
            kind = "invalid" if reward < 0 else kind
        if schema_conflict_invalid:
            if reward > 0.0 and bool(getattr(cfg, "schema_conflict_gate_positive", True)):
                stats["schema_conflict_gate_blocked_segments"] += 1.0
                stats["schema_conflict_gate_blocked_reward"] += float(reward)
                reward = 0.0
                kind = "schema_conflict_gated"
            conflict_penalty_scale = max(0.0, float(getattr(cfg, "schema_conflict_penalty_scale", 0.50)))
            if conflict_penalty_scale > 0.0:
                conflict_penalty = (
                    float(cfg.invalid_penalty)
                    * _invalid_penalty_scale(spec.task)
                    * conflict_penalty_scale
                )
                reward -= conflict_penalty
                stats["schema_conflict_penalty"] += float(conflict_penalty)
                kind = "invalid" if reward < 0 else kind
                if segment_trace is not None:
                    segment_trace["schema_conflict_penalty"] = float(conflict_penalty)
        if fact_dump:
            reward -= cfg.fact_dump_penalty
        if (
            reward > 0.0
            and algorithm_state is not None
            and bool(getattr(cfg, "ambiguity_gate_positive", False))
            and state_ambiguity_score > float(getattr(cfg, "ambiguity_gate_threshold", 0.50))
        ):
            multiplier = max(0.0, min(1.0, float(getattr(cfg, "ambiguity_gate_multiplier", 0.0))))
            reward *= multiplier
            stats["state_ambiguity_gate_segments"] += 1.0
            stats["state_ambiguity_gate_multiplier"] += float(multiplier)
            if multiplier <= 0.0:
                kind = "ambiguity_gated"
        graph_sensitivity_gate_allowed = bool(
            algorithm_state is not None
            and bool(getattr(cfg, "graph_sensitivity_gate_positive", False))
            and _csv_allows_task(spec.task, getattr(cfg, "graph_sensitivity_gate_task_allowlist", ""))
        )
        if reward > 0.0 and graph_sensitivity_gate_allowed:
            graph_sensitivity_gate_pass = bool(graph_sensitive_units)
            stats["state_graph_sensitivity_gate_segments"] += 1.0
            stats["state_graph_sensitivity_gate_pass_segments"] += float(graph_sensitivity_gate_pass)
            if not graph_sensitivity_gate_pass:
                multiplier = max(0.0, min(1.0, float(getattr(cfg, "graph_sensitivity_gate_multiplier", 0.25))))
                reward *= multiplier
                stats["state_graph_sensitivity_gate_blocked_segments"] += 1.0
                stats["state_graph_sensitivity_gate_multiplier"] += float(multiplier)
                if multiplier <= 0.0:
                    kind = "graph_sensitivity_gated"
        certificate_gate_allowed = bool(
            algorithm_state is not None
            and bool(getattr(cfg, "certificate_only_hard_gate", False))
            and _csv_allows_task(spec.task, getattr(cfg, "certificate_gate_task_allowlist", ""))
        )
        if reward > 0.0 and certificate_gate_allowed:
            certificate_gate_pass = bool(hard_cert_units) or (
                state_rewardable_count > 0.0
                and state_executable_commitment_score >= float(getattr(cfg, "certificate_gate_min_exec_score", 0.60))
                and state_ambiguity_score <= float(getattr(cfg, "certificate_gate_max_ambiguity", 0.35))
            )
            stats["state_certificate_gate_segments"] += 1.0
            stats["state_certificate_gate_pass_segments"] += float(certificate_gate_pass)
            if not certificate_gate_pass:
                reward = 0.0
                kind = "certificate_gated"
                stats["state_certificate_gate_blocked_segments"] += 1.0
        component_precision_allowed = _component_precision_gate_allowed(spec, cfg)
        if reward > 0.0 and component_precision_allowed:
            unit_names = _unit_name_set_from_trace(state_step_trace)
            verified_component = _component_precision_verified_positive(spec, unit_names)
            stats["component_precision_gate_segments"] += 1.0
            stats["component_precision_gate_verified_segments"] += float(verified_component)
            if not verified_component:
                vague_potential = _component_precision_is_vague_potential(
                    potential_delta=float(effective_potential_delta),
                    state_delta=float(state_delta),
                    unit_names=unit_names,
                    hard_cert_units=hard_cert_units,
                )
                if vague_potential:
                    multiplier = max(
                        0.0,
                        min(1.0, float(getattr(cfg, "component_vague_potential_multiplier", 0.20))),
                    )
                    stats["component_precision_gate_vague_segments"] += 1.0
                    stats["component_precision_gate_vague_multiplier"] += float(multiplier)
                else:
                    multiplier = max(
                        0.0,
                        min(1.0, float(getattr(cfg, "component_noisy_positive_multiplier", 0.0))),
                    )
                    stats["component_precision_gate_noisy_segments"] += 1.0
                    stats["component_precision_gate_noisy_multiplier"] += float(multiplier)
                reward *= multiplier
                if multiplier <= 0.0:
                    kind = "component_precision_gated"
            if segment_trace is not None:
                segment_trace["component_precision_gate"] = True
                segment_trace["component_precision_verified"] = bool(verified_component)
                segment_trace["component_precision_unit_names"] = sorted(unit_names)
        lexical_commitment_score = _state_action_commitment_score(spec, seg_text) if algorithm_state is not None else 1.0
        state_commitment_score = (
            state_executable_commitment_score
            if bool(getattr(cfg, "executable_commitment_weight", False))
            else lexical_commitment_score
        )
        state_commitment_multiplier = 1.0
        if (
            reward > 0.0
            and algorithm_state is not None
            and bool(getattr(cfg, "state_commitment_weight", True))
            and (state_delta > 0.0 or state_reward_consistent)
        ):
            min_multiplier = max(0.0, float(getattr(cfg, "state_commitment_min_multiplier", 0.75)))
            max_multiplier = max(min_multiplier, float(getattr(cfg, "state_commitment_max_multiplier", 1.20)))
            state_commitment_multiplier = min_multiplier + (max_multiplier - min_multiplier) * state_commitment_score
            if state_invalid_count > 0.0:
                state_commitment_multiplier = min(1.0, state_commitment_multiplier)
            reward *= state_commitment_multiplier
            stats["state_commitment_weighted_segments"] += 1.0
            stats["state_commitment_score"] += float(state_commitment_score)
            stats["state_commitment_multiplier"] += float(state_commitment_multiplier)
        if segment_trace is not None:
            segment_trace["state_ambiguity_score"] = float(state_ambiguity_score)
            segment_trace["state_executable_commitment_score"] = float(state_executable_commitment_score)
            segment_trace["state_hard_cert_units"] = sorted(hard_cert_units)
            segment_trace["state_graph_sensitive_units"] = sorted(graph_sensitive_units)
            segment_trace["state_lexical_commitment_score"] = float(lexical_commitment_score)
            segment_trace["state_commitment_score"] = float(state_commitment_score)
            segment_trace["state_commitment_multiplier"] = float(state_commitment_multiplier)
        segment_unit_names = tuple(
            sorted(
                {
                    str(name)
                    for key in ("rewardable_unit_names", "new_unit_names", "unit_names")
                    for name in (state_step_trace.get(key, []) if isinstance(state_step_trace, dict) else [])
                    if str(name)
                }
            )
        )
        if schema_conflict_reason:
            segment_unit_names = tuple(sorted(set(segment_unit_names) | {f"schema_conflict:{schema_conflict_reason}"}))
        if (
            reward > 0.0
            and router_accepted
            and _action_router_can_reward(cfg)
            and router_routed is not None
        ):
            gate = max(0.0, min(1.0, float(router_routed.confidence)))
            router_bonus = min(
                max(0.0, float(getattr(cfg, "action_router_positive_cap", 0.05))),
                max(0.0, float(state_delta)) * gate,
            )
            if router_bonus > 0.0:
                reward += router_bonus
                stats["action_router_positive_bonus"] += float(router_bonus)
                stats["action_router_positive_bonus_segments"] += 1.0
                kind = "router_progress" if kind == "legal" else kind
                if segment_trace is not None:
                    segment_trace["action_router_positive_bonus"] = float(router_bonus)
        if (
            router_active
            and bool(getattr(cfg, "action_router_negative", False))
            and router_routed is not None
            and not router_routed.abstain
            and float(router_routed.confidence) >= float(getattr(cfg, "action_router_high_conf_threshold", 0.90))
            and (state_invalid_count > 0.0 or invalid_edges)
        ):
            router_penalty = 0.05 * max(0.0, min(1.0, float(router_routed.confidence)))
            reward -= router_penalty
            stats["action_router_high_conf_invalid_penalty"] += float(router_penalty)
            kind = "invalid" if reward < 0 else kind
            if segment_trace is not None:
                segment_trace["action_router_high_conf_invalid_penalty"] = float(router_penalty)
        completion_invalid = bool(
            algorithm_state is not None
            and float(state_step_stats.get("state_completion_invalid", 0.0)) > 0.0
        )
        hard_final_invalid = bool(
            format_invalid
            or numeric_final_invalid
            or path_final_invalid
            or schema_conflict_invalid
        )
        irrecoverable_error = bool(completion_invalid or hard_final_invalid)
        recoverable_error = bool(
            state_negative_delta > 0.0
            or state_invalid_count > 0.0
            or bool(invalid_edges)
        )
        step_label = "NEUTRAL"
        if reward > 0.0 and state_delta > 0.0 and state_invalid_count <= 0.0 and not invalid_edges:
            step_label = "POS_STRONG"
        elif reward > 0.0 and state_reward_consistent and state_invalid_count <= 0.0 and not invalid_edges:
            step_label = "POS_WEAK"
        elif irrecoverable_error:
            step_label = "NEG_IRREC"
        elif recoverable_error or reward < 0.0:
            step_label = "NEG_REC"
        elif reward > 0.0:
            step_label = "POS_WEAK"
        step_first_error = bool(step_label == "NEG_IRREC" and not first_irrecoverable_error_seen)
        if step_first_error:
            first_irrecoverable_error_seen = True
            stats["step_label_first_irrecoverable_error"] += 1.0
        stats[f"step_label_{step_label.lower()}"] += 1.0
        if segment_trace is not None:
            segment_trace["step_label"] = step_label
            segment_trace["first_error"] = bool(step_first_error)
        if reward == 0.0:
            if reward_mode == "potential" and effective_potential_delta > 0.0:
                potential_state = candidate_state
                seen_edges = potential_state.seen_edges
                seen_nodes = potential_state.seen_nodes
                edge_positions = potential_state.edge_positions
                node_pos = potential_state.node_pos
            if segment_trace is not None:
                segment_trace.update(
                    {
                        "reward": 0.0,
                        "kind": kind,
                        "skipped_reason": "zero_reward",
                    }
                )
                trace.append(segment_trace)
            continue
        if reward_mode == "potential":
            if effective_potential_delta > 0.0 or answer_hit:
                potential_state = candidate_state
                seen_edges = potential_state.seen_edges
                seen_nodes = potential_state.seen_nodes
                edge_positions = potential_state.edge_positions
                node_pos = potential_state.node_pos
        else:
            edge_positions = new_edge_positions
            node_pos = new_node_pos
            seen_edges.update(progressive_edges)
            seen_nodes.update(progressive_nodes)
        anchor = _step_anchor(
            spec,
            kind,
            matched_ordered_edges,
            progressive_edges,
            progressive_nodes,
            valid_edges,
            nodes,
            invalid_edges,
            answer_hit,
        )
        if segment_trace is not None:
            segment_trace.update(
                {
                    "reward": float(reward * weight),
                    "unweighted_reward": float(reward),
                    "kind": kind,
                    "anchor": _trace_value(anchor),
                }
            )
            trace.append(segment_trace)
        weighted_reward = float(reward * weight)
        remaining_steps = max(0, int(cfg.max_steps) - len(steps))
        reward_spans = _step_reward_char_spans(
            spec,
            cfg,
            seg_text,
            start,
            end,
            state_action_count,
            remaining_steps,
            action_event_spans=action_event_spans,
        )
        if not reward_spans:
            if segment_trace is not None and "skipped_reason" not in segment_trace:
                segment_trace["skipped_reason"] = "max_steps"
            continue
        if segment_trace is not None:
            segment_trace["reward_spans"] = [[int(left), int(right)] for left, right in reward_spans]
        per_span_reward = weighted_reward / float(len(reward_spans))
        for reward_span in reward_spans:
            steps.append(
                GraphLogicStep(
                    per_span_reward,
                    anchor,
                    reward_span,
                    label=step_label,
                    first_error=step_first_error,
                    unit_names=segment_unit_names,
                )
            )
        stats["action_span_rewarded_segments"] += 1.0
        stats["action_span_emitted"] += float(len(reward_spans))
        stats["action_span_split_segments"] += float(len(reward_spans) > 1)
        stats["action_event_rewarded_segments"] += float(bool(action_event_spans))
        stats["action_span_fallback_segments"] += float(
            (not action_event_spans) and len(reward_spans) == 1 and reward_spans[0] == (start, end)
        )

    schema_final_scalar_parseable = _schema_final_scalar_parseable(spec, text)
    schema_final_object_units, schema_final_object_missing = _schema_final_object_units(spec, text)
    schema_final_object_parseable = bool(schema_final_object_units)
    stats["schema_final_scalar_parseable"] = float(schema_final_scalar_parseable)
    stats["schema_final_object_parseable"] = float(schema_final_object_parseable)
    stats["schema_final_object_missing"] = float(schema_final_object_missing)
    stats["schema_final_object_units"] = float(len(schema_final_object_units))
    schema_final_object_rewardable = bool(schema_final_object_parseable)
    if spec.task == "isomophic_mapping":
        schema_final_object_rewardable = any(
            unit and str(unit[0]) in {"complete_mapping_dict_answer", "mapping_valid_dict"}
            for unit in schema_final_object_units
        )
    schema_tail_rewards_enabled = not legacy_core4_compat
    if text.strip() and schema_final_scalar_parseable and schema_tail_rewards_enabled:
        tail_len = min(512, len(text))
        scalar_schema_base = 0.010
        if spec.task in HARD_SCALAR_FINAL_SCHEMA_TASKS:
            scalar_schema_base = 0.030
        schema_reward = scalar_schema_base * max(0.65, weight)
        if _component_precision_unit_filter_allowed(spec, cfg):
            schema_reward = 0.0
            stats["component_precision_schema_scalar_blocked"] += 1.0
        elif _component_precision_gate_allowed(spec, cfg):
            schema_multiplier = max(
                0.0,
                min(1.0, float(getattr(cfg, "component_schema_parseable_multiplier", 0.05))),
            )
            schema_reward *= schema_multiplier
            stats["component_precision_schema_scalar_scaled"] += 1.0
            stats["component_precision_schema_multiplier_sum"] += float(schema_multiplier)
        if schema_reward > 0.0:
            steps.append(
                GraphLogicStep(
                    float(schema_reward),
                    ("graph_logic", spec.task, "schema_final_scalar_parseable"),
                    (max(0, len(text) - tail_len), len(text)),
                    label="POS_WEAK",
                    unit_names=("schema_final_scalar_parseable",),
                )
            )
            stats["schema_final_parseable_reward"] = float(
                stats.get("schema_final_parseable_reward", 0.0) + schema_reward
            )
            stats["step_label_pos_weak"] += 1.0
    if text.strip() and schema_final_object_rewardable and schema_tail_rewards_enabled:
        tail_len = min(512, len(text))
        strong_units = sum(
            1
            for unit in schema_final_object_units
            if unit and str(unit[0]) in {
                "mst_answer_spanning_tree",
                "state_mst_connected_acyclic",
                "state_mst_optimal_weight_cert",
                "mst_answer_valid",
                "complete_mapping_dict_answer",
                "edge_set_answer_feasible",
                "set_answer_feasible",
                "ordering_answer_valid",
                "node_answer_valid",
                "metric_target_set_complete",
                "state_path_cost_valid",
                "state_weighted_path_final_cert",
            }
        )
        object_schema_cap = 0.060
        object_schema_base = 0.018
        if spec.task in STRICT_OBJECT_FINAL_SCHEMA_TASKS:
            object_schema_cap = 0.120
            object_schema_base = 0.034
        if spec.task in {
            "weighted_minimum_spanning_tree",
            "isomophic_mapping",
            "min_vertex_cover",
            "min_edge_covering",
            "dominating_set",
            "maximal_independent_set",
            "bipartite_maximum_matching",
            "hamiltonian_path",
        } and spec.task not in STRICT_OBJECT_FINAL_SCHEMA_TASKS:
            object_schema_cap = 0.090
            object_schema_base = 0.026
        schema_reward = min(object_schema_cap, (object_schema_base + 0.008 * strong_units) * max(0.65, weight))
        schema_unit_names = {str(unit[0]) for unit in schema_final_object_units if unit}
        if _component_precision_unit_filter_allowed(spec, cfg):
            schema_verified = _component_precision_verified_positive(spec, schema_unit_names)
            if not schema_verified:
                schema_reward = 0.0
                stats["component_precision_schema_object_blocked"] += 1.0
        elif _component_precision_gate_allowed(spec, cfg):
            schema_verified = _component_precision_verified_positive(spec, schema_unit_names)
            if not schema_verified:
                schema_multiplier = max(
                    0.0,
                    min(1.0, float(getattr(cfg, "component_schema_parseable_multiplier", 0.05))),
                )
                schema_reward *= schema_multiplier
                stats["component_precision_schema_object_scaled"] += 1.0
                stats["component_precision_schema_multiplier_sum"] += float(schema_multiplier)
        if schema_reward > 0.0:
            steps.append(
                GraphLogicStep(
                    float(schema_reward),
                    ("graph_logic", spec.task, "schema_final_object_parseable"),
                    (max(0, len(text) - tail_len), len(text)),
                    label="POS_WEAK",
                    unit_names=tuple(sorted(schema_unit_names)),
                )
            )
            stats["schema_final_parseable_reward"] = float(
                stats.get("schema_final_parseable_reward", 0.0) + schema_reward
            )
            stats["step_label_pos_weak"] += 1.0
    schema_final_object_missing_penalized = (
        schema_final_object_missing
        and text.strip()
        and (
            len(text)
            >= (
                min(int(getattr(cfg, "schema_final_object_missing_min_chars", 8000)), 5000)
                if spec.task in STRICT_OBJECT_FINAL_SCHEMA_TASKS
                else int(getattr(cfg, "schema_final_object_missing_min_chars", 8000))
            )
            or _has_strong_final_answer_cue(text)
        )
    )
    stats["schema_final_object_missing_penalized"] = float(schema_final_object_missing_penalized)
    if schema_final_object_missing_penalized and schema_tail_rewards_enabled:
        tail_len = min(512, len(text))
        missing_scale = float(getattr(cfg, "schema_final_object_missing_penalty_scale", 0.12))
        if spec.task in STRICT_OBJECT_FINAL_SCHEMA_TASKS:
            missing_scale *= 2.0
        schema_penalty = (
            -float(cfg.invalid_penalty)
            * _invalid_penalty_scale(spec.task)
            * missing_scale
            * max(0.65, weight)
        )
        steps.append(
            GraphLogicStep(
                float(schema_penalty),
                ("graph_logic", spec.task, "schema_final_object_missing"),
                (max(0, len(text) - tail_len), len(text)),
                label="NEG_REC",
                first_error=False,
                unit_names=("schema_final_object_missing",),
            )
        )
        stats["step_label_neg_rec"] += 1.0
        stats["schema_final_object_missing_penalty"] = float(-schema_penalty)

    schema_final_numeric_missing = _schema_final_numeric_missing(spec, text)
    stats["schema_final_numeric_missing"] = float(schema_final_numeric_missing)
    if schema_final_numeric_missing and text.strip() and schema_tail_rewards_enabled:
        tail_len = min(512, len(text))
        numeric_missing_scale = 0.75
        if spec.task in HARD_SCALAR_FINAL_SCHEMA_TASKS:
            numeric_missing_scale = 1.10
        schema_penalty = (
            -float(cfg.invalid_penalty)
            * _invalid_penalty_scale(spec.task)
            * numeric_missing_scale
            * max(0.65, weight)
        )
        schema_first_error = not first_irrecoverable_error_seen
        if schema_first_error:
            first_irrecoverable_error_seen = True
            stats["step_label_first_irrecoverable_error"] += 1.0
        steps.append(
            GraphLogicStep(
                float(schema_penalty),
                ("graph_logic", spec.task, "schema_final_numeric_missing"),
                (max(0, len(text) - tail_len), len(text)),
                label="NEG_IRREC",
                first_error=schema_first_error,
                unit_names=("schema_final_numeric_missing",),
            )
        )
        stats["step_label_neg_irrec"] += 1.0
        stats["schema_final_numeric_missing_penalty"] = float(-schema_penalty)

    if algorithm_state is not None:
        state_potential_final = float(_algorithm_state_potential(spec, algorithm_state))
        stats["state_potential_final"] = state_potential_final
        stats["state_net_progress"] = float(
            max(0.0, state_potential_final - float(stats.get("state_potential_initial", 0.0)))
        )
    if reward_mode == "potential":
        potential_final = float(_potential_value(spec, potential_state, edge_orders, node_order))
        stats["potential_final"] = potential_final
        stats["potential_net_progress"] = float(
            max(0.0, potential_final - float(stats.get("potential_initial", 0.0)))
        )

    positive_reward_cap = float(getattr(cfg, "positive_reward_cap", 0.0) or 0.0)
    if reward_mode == "potential" and bool(getattr(cfg, "state_shaping", False)) and positive_reward_cap > 0.0:
        base_weighted_cap = max(0.0, positive_reward_cap * max(0.65, weight))
        weighted_cap, budget_stats = _response_positive_reward_budget(spec, cfg, stats, base_weighted_cap)
        stats.update(budget_stats)
        positive_total = sum(max(0.0, step.reward) for step in steps)
        stats["positive_reward_cap"] = float(weighted_cap)
        stats["positive_reward_base_cap"] = float(base_weighted_cap)
        stats["positive_reward_before_cap"] = float(positive_total)
        stats["positive_reward_upscale_applied"] = 0.0
        stats["positive_reward_upscale_scale"] = 1.0
        stats["positive_reward_target"] = float(weighted_cap)
        net_progress_budget = bool(float(budget_stats.get("positive_reward_budget_mode_net_progress", 0.0)) > 0.0)
        if positive_total > 0.0 and net_progress_budget:
            scale = weighted_cap / max(1e-12, positive_total)
            steps = [
                GraphLogicStep(
                    float(step.reward * scale) if step.reward > 0.0 else step.reward,
                    step.anchor,
                    step.char_span,
                    label=step.label,
                    first_error=step.first_error,
                    unit_names=step.unit_names,
                )
                for step in steps
            ]
            stats["positive_reward_cap_applied"] = float(scale < 1.0)
            stats["positive_reward_cap_scale"] = float(scale)
            stats["positive_reward_upscale_applied"] = float(scale > 1.0)
            stats["positive_reward_upscale_scale"] = float(scale if scale > 1.0 else 1.0)
            stats["positive_reward_budget_exact_applied"] = 1.0
        elif positive_total > 0.0 and positive_total > weighted_cap:
            scale = weighted_cap / max(1e-12, positive_total)
            steps = [
                GraphLogicStep(
                    float(step.reward * scale) if step.reward > 0.0 else step.reward,
                    step.anchor,
                    step.char_span,
                    label=step.label,
                    first_error=step.first_error,
                    unit_names=step.unit_names,
                )
                for step in steps
            ]
            stats["positive_reward_cap_applied"] = 1.0
            stats["positive_reward_cap_scale"] = float(scale)
        else:
            stats["positive_reward_cap_applied"] = 0.0
            stats["positive_reward_cap_scale"] = 1.0
            stats["positive_reward_budget_exact_applied"] = 0.0
            quality = float(budget_stats.get("positive_reward_budget_quality", 0.0))
            invalid_factor = float(budget_stats.get("positive_reward_budget_invalid_factor", 0.0))
            repetition_factor = float(budget_stats.get("positive_reward_budget_repetition_factor", 0.0))
            target_fraction = float(budget_stats.get("positive_reward_budget_target_fraction", 0.0))
            clean_enough = quality >= 0.55 and invalid_factor >= 0.65 and repetition_factor >= 0.70
            if positive_total > 0.0 and weighted_cap > positive_total and clean_enough:
                target_total = min(weighted_cap, max(positive_total, weighted_cap * target_fraction))
                max_upscale = 1.70 if spec.task in ACCUMULATION_SENSITIVE_STATE_TASKS else 1.50
                upscale = min(max_upscale, target_total / max(1e-12, positive_total))
                if upscale > 1.02:
                    steps = [
                        GraphLogicStep(
                            float(step.reward * upscale) if step.reward > 0.0 else step.reward,
                            step.anchor,
                            step.char_span,
                            label=step.label,
                            first_error=step.first_error,
                            unit_names=step.unit_names,
                        )
                        for step in steps
                    ]
                    stats["positive_reward_upscale_applied"] = 1.0
                    stats["positive_reward_upscale_scale"] = float(upscale)
                    stats["positive_reward_target"] = float(target_total)
        stats["positive_reward_after_scaling"] = float(sum(max(0.0, step.reward) for step in steps))

    if edge_orders:
        stats["ordered_total"] = float(max(len(order) for order in edge_orders))
        stats["ordered_recall"] = float(
            max(pos / max(1, len(order)) for pos, order in zip(edge_positions, edge_orders))
        )
    else:
        stats["ordered_total"] = float(len(node_order))
        stats["ordered_recall"] = float(node_pos / max(1, len(node_order)))
    if reward_mode == "potential":
        stats["potential_final"] = float(_potential_value(spec, potential_state, edge_orders, node_order))
    stats["reward_mode_potential"] = float(reward_mode == "potential")
    if return_trace:
        return steps, dict(stats), trace
    return steps, dict(stats)
