"""Router client implementations.

The production interface is intentionally tiny so a small 0.6B classifier can
replace the heuristic scorer later. The heuristic mode is useful for shadow
audits and CI because it has no model-serving dependency.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Iterable

from .action_schema import ActionCandidate, RoutedAction, RouterPayload


_ACTION_CUES = {
    "RELAX_EDGE": (
        "relax",
        "update",
        "distance",
        "dist",
        "through",
        "via",
        "parent",
        "cost",
        "tentative",
        "becomes",
        "set",
    ),
    "NO_RELAX_EDGE": ("no update", "not improve", "keep", "unchanged", "larger", "worse", "skip relax"),
    "SETTLE_NODE": (
        "settle",
        "settled",
        "finalize",
        "visited",
        "fixed",
        "permanent",
        "minimum",
        "minimum tentative",
        "unsettled",
        "next node",
        "smallest",
    ),
    "FINAL_PATH": ("final", "path", "route", "shortest", "answer"),
    "ACCEPT_EDGE": (
        "add",
        "accept",
        "choose",
        "include",
        "safe",
        "component",
        "mst",
        "spanning",
        "take",
        "select",
        "pick",
        "use edge",
    ),
    "REJECT_CYCLE_EDGE": ("reject", "skip", "cycle", "same component", "already connected"),
    "AUGMENT_PATH": (
        "augment",
        "bottleneck",
        "send",
        "push",
        "residual",
        "capacity",
        "flow",
        "remaining",
        "reverse",
        "increase",
    ),
    "PAIR_DISTANCE": ("distance", "dist", "pair", "shortest", "table", "sum", "d(", "d[", "between"),
    "NORMALIZER": ("denominator", "normalizer", "normalize", "divide", "n(n-1)", "n-1"),
    "MAPPING_PAIR": ("map", "mapping", "correspond", "graph1", "graph2"),
    "VISIT_NODE": ("visit", "visited", "discover", "enqueue", "dequeue", "push", "pop", "process", "mark"),
    "TRAVERSE_EDGE": ("visit", "discover", "tree", "parent", "neighbor", "explore", "edge"),
    "EMIT_NODE": ("emit", "output", "order", "topological", "next", "place", "append"),
    "FINAL_NODE_SEQUENCE": ("final", "answer", "path", "order", "sequence", "list"),
    "SELECT_SET_NODE": ("select", "choose", "include", "add", "take", "cover", "dominate", "independent"),
    "SELECT_SET_EDGE": ("select", "choose", "include", "add", "take", "matching", "cover", "edge"),
    "FINAL_NODE_SET": ("final", "answer", "set", "solution", "cover", "dominating", "independent"),
    "FINAL_EDGE_SET": ("final", "answer", "set", "solution", "matching", "cover"),
}


def _candidate_description(candidate: ActionCandidate) -> str:
    action = candidate.action_type
    args = candidate.args or {}
    if action == "RELAX_EDGE":
        return (
            f"RELAX_EDGE: update the best distance of node {args.get('v')} to {args.get('new_dist')} "
            f"through/predecessor/parent node {args.get('u')}."
        )
    if action == "NO_RELAX_EDGE":
        return (
            f"NO_RELAX_EDGE: check edge {args.get('u')}-{args.get('v')} but keep the old distance because "
            "there is no improvement."
        )
    if action == "SETTLE_NODE":
        return f"SETTLE_NODE: settle/finalize/mark permanent node {args.get('node', args.get('v'))}."
    if action == "FINAL_PATH":
        return "FINAL_PATH: declare the final path or route."
    if action == "ACCEPT_EDGE":
        return (
            f"ACCEPT_EDGE: accept/add/include edge {args.get('u')}-{args.get('v')} because it connects "
            "different DSU components."
        )
    if action == "REJECT_CYCLE_EDGE":
        return (
            f"REJECT_CYCLE_EDGE: reject/skip/do not add edge {args.get('u')}-{args.get('v')} because it "
            "forms a cycle or is inside the same component."
        )
    if action == "AUGMENT_PATH":
        return (
            f"AUGMENT_PATH: push/send/augment flow along path {args.get('path')} "
            f"with bottleneck {args.get('bottleneck')}."
        )
    if action == "PAIR_DISTANCE":
        return f"PAIR_DISTANCE: record shortest distance d({args.get('u')},{args.get('v')}) = {args.get('distance')}."
    if action == "NORMALIZER":
        return "NORMALIZER: state the denominator/normalization rule for the metric."
    if action == "MAPPING_PAIR":
        return f"MAPPING_PAIR: map/correspond node {args.get('u')} to node {args.get('v')}."
    if action == "VISIT_NODE":
        return f"VISIT_NODE: visit/discover/process node {args.get('node')} during BFS/DFS."
    if action == "TRAVERSE_EDGE":
        return f"TRAVERSE_EDGE: explore/tree/parent edge {args.get('u')}-{args.get('v')} during traversal."
    if action == "EMIT_NODE":
        return f"EMIT_NODE: output/append node {args.get('node')} in an ordered sequence such as topological sort."
    if action == "FINAL_NODE_SEQUENCE":
        return "FINAL_NODE_SEQUENCE: declare a final ordered node sequence/path/order."
    if action == "SELECT_SET_NODE":
        return f"SELECT_SET_NODE: select/include node {args.get('node')} for {args.get('task')}."
    if action == "SELECT_SET_EDGE":
        return f"SELECT_SET_EDGE: select/include edge {args.get('u')}-{args.get('v')} for {args.get('task')}."
    if action == "FINAL_NODE_SET":
        return f"FINAL_NODE_SET: declare the final node set for {args.get('task')}."
    if action == "FINAL_EDGE_SET":
        return f"FINAL_EDGE_SET: declare the final edge set for {args.get('task')}."
    if action == "ABSTAIN":
        return "ABSTAIN: no candidate action is clearly expressed by the segment."
    return str(candidate.reason or action)


def _value_label(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(_jsonable(value))


def _edge_label(args: dict) -> str:
    u = args.get("u")
    v = args.get("v")
    return f"{_value_label(u)}-{_value_label(v)}"


def _path_label(path) -> str:
    if isinstance(path, (list, tuple)) and path:
        return "-".join(_value_label(node) for node in path)
    return "the listed path"


def _required_evidence(candidate: ActionCandidate) -> list[str]:
    """Human-readable evidence requirements shown to the semantic router.

    These strings are not used by the executor. They tighten the small model's
    semantic routing objective so it chooses a candidate only when the segment
    contains enough local evidence for that specific action.
    """

    action = candidate.action_type
    args = candidate.args or {}
    if action == "RELAX_EDGE":
        return [
            f"mentions the updated node {args.get('v')}",
            (
                f"mentions either new distance {_value_label(args.get('new_dist'))} "
                f"or predecessor/through node {args.get('u')}"
            ),
            "uses an update/relax/shorter-distance/parent/predecessor meaning",
        ]
    if action == "NO_RELAX_EDGE":
        return [
            f"mentions edge or endpoints {_edge_label(args)}",
            "states no update/no improvement/keep the old distance",
        ]
    if action == "SETTLE_NODE":
        node = args.get("node", args.get("v"))
        return [
            f"mentions node {node}",
            "uses settle/finalize/permanent/minimum-tentative/next-minimum meaning",
        ]
    if action == "FINAL_PATH":
        return [
            "declares a final path/route answer",
            "mentions the path nodes or final distance, not just a plan to find a path",
        ]
    if action == "ACCEPT_EDGE":
        return [
            f"mentions edge {_edge_label(args)} or both endpoints",
            "uses add/accept/include/take/select edge meaning",
            "mentions safe/different components/no cycle, or clearly commits the edge to the tree",
        ]
    if action == "REJECT_CYCLE_EDGE":
        return [
            f"mentions edge {_edge_label(args)} or both endpoints",
            "uses reject/skip/do-not-add meaning",
            "mentions cycle/same component/already connected or another explicit rejection reason",
        ]
    if action == "AUGMENT_PATH":
        return [
            f"mentions path {_path_label(args.get('path'))} or all path nodes in order",
            f"mentions bottleneck/send/push amount {_value_label(args.get('bottleneck'))}",
            "uses augment/send/push flow meaning",
        ]
    if action == "PAIR_DISTANCE":
        return [
            f"mentions pair ({args.get('u')},{args.get('v')})",
            f"mentions distance or contribution value {_value_label(args.get('distance'))}",
        ]
    if action == "NORMALIZER":
        return [
            "mentions denominator/normalization/division rule",
            "states a concrete denominator or formula, not only the metric name",
        ]
    if action == "MAPPING_PAIR":
        return [
            f"mentions source node {args.get('u')}",
            f"mentions target node {args.get('v')}",
            "uses map/corresponds-to/matched-with meaning",
        ]
    if action == "VISIT_NODE":
        return [f"mentions node {args.get('node')}", "uses visit/discover/process/mark/enqueue/dequeue meaning"]
    if action == "TRAVERSE_EDGE":
        return [
            f"mentions edge {_edge_label(args)} or both endpoints",
            "uses explore/traverse/tree-edge/parent-edge meaning",
        ]
    if action == "EMIT_NODE":
        return [f"mentions node {args.get('node')}", "uses output/append/emit/place-in-order meaning"]
    if action == "FINAL_NODE_SEQUENCE":
        return ["declares a final ordered sequence/path/order", "lists multiple nodes in order"]
    if action == "SELECT_SET_NODE":
        return [
            f"mentions node {args.get('node')}",
            "uses select/include/add/take meaning for the set solution",
        ]
    if action == "SELECT_SET_EDGE":
        return [
            f"mentions edge {_edge_label(args)} or both endpoints",
            "uses select/include/add/take meaning for the edge set",
        ]
    if action == "FINAL_NODE_SET":
        return ["declares a final node set answer", "lists the selected nodes explicitly"]
    if action == "FINAL_EDGE_SET":
        return ["declares a final edge set answer", "lists the selected edges explicitly"]
    if action == "ABSTAIN":
        return ["use when no non-ABSTAIN candidate has explicit local evidence"]
    return ["segment explicitly states this action type and its concrete arguments"]


def _abstain_rules(candidate: ActionCandidate) -> list[str]:
    action = candidate.action_type
    rules = [
        "abstain if the segment is only a plan, formula reminder, restatement of the problem, or generic reasoning",
        "abstain if the concrete arguments in the segment do not match this candidate",
    ]
    if action == "RELAX_EDGE":
        rules.append("abstain if it says shortest path/Dijkstra but does not name the updated node and distance/parent")
    elif action in {"ACCEPT_EDGE", "REJECT_CYCLE_EDGE"}:
        rules.append("abstain if it mentions MST/Kruskal/safe edge but not this concrete edge")
    elif action == "AUGMENT_PATH":
        rules.append("abstain if it only says a residual path exists without bottleneck/send amount")
    elif action == "PAIR_DISTANCE":
        rules.append("abstain if it mentions all-pairs distances or a table without this exact pair and value")
    elif action == "MAPPING_PAIR":
        rules.append("abstain if it discusses isomorphism/refinement without this explicit source-target pair")
    elif action in {"FINAL_NODE_SET", "FINAL_EDGE_SET", "FINAL_NODE_SEQUENCE", "FINAL_PATH"}:
        rules.append("abstain if it is not the final object or does not list the object explicitly")
    return rules


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", text))


def _jsonable(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return round(float(value), 6)
        return "inf" if value > 0 else "-inf"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _limited_sorted(values, limit: int = 24):
    try:
        out = sorted(values)
    except Exception:
        out = list(values)
    return [_jsonable(item) for item in out[: max(0, limit)]]


def candidate_to_payload(candidate: ActionCandidate) -> dict:
    """Serialize one deterministic candidate for a small model router."""

    return {
        "id": candidate.candidate_id,
        "type": candidate.action_type,
        "args": _jsonable(candidate.args),
        "description": _candidate_description(candidate),
        "required_evidence": _required_evidence(candidate),
        "abstain_if": _abstain_rules(candidate),
        "weak": bool(candidate.weak),
        "terminal": bool(candidate.terminal),
    }


def compact_state_summary(task: str, spec, state, *, max_items: int = 24) -> dict:
    """Build the compact state passed to a future 0.6B action router.

    This intentionally avoids dumping the full graph or full observer object.
    The router sees only the state fields needed to match text to candidate
    actions; the executor still validates correctness.
    """

    summary = {
        "task": str(task),
        "directed": bool(getattr(spec, "directed", False)),
        "num_nodes": len(getattr(spec, "nodes", set()) or set()),
        "num_edges": len(getattr(spec, "edges", set()) or set()),
        "source": _jsonable(getattr(spec, "source", None)),
        "target": _jsonable(getattr(spec, "target", None)),
        "recent_nodes": _jsonable(getattr(state, "recent_nodes", [])[-max_items:]),
        "recent_edges": _jsonable(getattr(state, "recent_edges", [])[-max_items:]),
    }

    def seen_units_sample() -> list:
        return _jsonable(sorted(getattr(state, "seen_units", set()) or set(), key=str)[:max_items])

    if task == "weighted_shortest_path":
        dist = getattr(state, "dist", {}) or {}
        parents = getattr(state, "parent_node", {}) or {}
        summary.update(
            {
                "settled": _limited_sorted(getattr(state, "settled_nodes", set()) or set(), max_items),
                "dist": {
                    str(k): _jsonable(v)
                    for k, v in sorted(dist.items(), key=lambda item: str(item[0]))[:max_items]
                },
                "parents": {
                    str(k): _jsonable(v)
                    for k, v in sorted(parents.items(), key=lambda item: str(item[0]))[:max_items]
                },
            }
        )
    elif task in {"bfs", "dfs"}:
        summary.update(
            {
                "visited": _limited_sorted(getattr(state, "visited_nodes", set()) or set(), max_items),
                "order_pos": _jsonable(getattr(state, "order_pos", 0)),
                "node_order_pos": _jsonable(getattr(state, "node_order_pos", 0)),
                "target_edge_order_sample": _jsonable((getattr(spec, "target_edge_order", []) or [])[:max_items]),
                "target_node_order_sample": _jsonable((getattr(spec, "target_node_order", []) or [])[:max_items]),
            }
        )
    elif task in {"minimum_spanning_tree", "weighted_minimum_spanning_tree"}:
        parent = getattr(state, "dsu_parent", {}) or {}
        component_map = {}
        for node in sorted(getattr(spec, "nodes", set()) or set())[:max_items]:
            root = parent.get(node, node)
            component_map.setdefault(str(root), []).append(int(node))
        summary.update(
            {
                "accepted_edges": _jsonable(sorted(getattr(state, "accepted_edges", set()) or set())[:max_items]),
                "rejected_edges": _jsonable(sorted(getattr(state, "rejected_edges", set()) or set())[:max_items]),
                "total_weight": _jsonable(getattr(state, "total_weight", 0.0)),
                "components_sample": component_map,
            }
        )
    elif task == "maximum_flow":
        residual = getattr(state, "residual", {}) or {}
        positive_residual = [
            (edge, cap)
            for edge, cap in sorted(residual.items(), key=lambda item: str(item[0]))
            if float(cap) > 1e-12
        ][:max_items]
        summary.update(
            {
                "flow_value": _jsonable(getattr(state, "flow_value", 0.0)),
                "max_flow_value": _jsonable(getattr(state, "max_flow_value", 0.0)),
                "positive_residual_sample": _jsonable(positive_residual),
            }
        )
    elif task in {"global_efficiency", "wiener_index", "betweenness_centrality", "closeness_centrality"}:
        summary.update(
            {
                "seen_units_sample": seen_units_sample(),
                "answer_numbers": _jsonable(getattr(spec, "answer_numbers", []) or []),
            }
        )
    elif task in {"topological_sort", "hamiltonian_path", "traveling_salesman_problem"}:
        summary.update(
            {
                "seen_units_sample": seen_units_sample(),
                "target_node_order_sample": _jsonable((getattr(spec, "target_node_order", []) or [])[:max_items]),
                "target_edge_order_sample": _jsonable((getattr(spec, "target_edge_order", []) or [])[:max_items]),
            }
        )
    elif task in {
        "bipartite_maximum_matching",
        "max_weight_matching",
        "min_edge_covering",
        "min_vertex_cover",
        "dominating_set",
        "maximal_independent_set",
    }:
        summary.update(
            {
                "seen_units_sample": seen_units_sample(),
                "target_node_count": len(getattr(spec, "target_nodes", set()) or set()),
                "target_edge_count": len(getattr(spec, "target_edges", set()) or set()),
            }
        )
    elif task == "isomophic_mapping":
        summary.update(
            {
                "partial_mapping": _jsonable(getattr(state, "partial_mapping", {}) or {}),
                "answer_mapping_keys": _jsonable(
                    sorted((getattr(spec, "answer_mapping", {}) or {}).keys())[:max_items]
                ),
            }
        )
    else:
        summary["seen_units_sample"] = seen_units_sample()
    return summary


def build_router_payload(
    *,
    task: str,
    spec,
    state,
    segment_text: str,
    candidates: Iterable[ActionCandidate],
    max_state_items: int = 24,
) -> RouterPayload:
    """Build the exact JSON-ready input intended for a model-backed router."""

    candidate_list = list(candidates)
    return RouterPayload(
        task=str(task),
        segment=str(segment_text),
        state_summary=compact_state_summary(str(task), spec, state, max_items=max_state_items),
        candidate_actions=[candidate_to_payload(candidate) for candidate in candidate_list],
    )


_ROUTER_MAX_SEGMENT_CHARS = max(128, int(os.environ.get("GRAPH_LOGIC_ACTION_ROUTER_MAX_SEGMENT_CHARS", "1200")))
_ROUTER_MAX_STRING_CHARS = max(32, int(os.environ.get("GRAPH_LOGIC_ACTION_ROUTER_MAX_STRING_CHARS", "160")))
_ROUTER_MAX_LIST_ITEMS = max(2, int(os.environ.get("GRAPH_LOGIC_ACTION_ROUTER_MAX_LIST_ITEMS", "16")))
_ROUTER_ENDPOINT_LOCK = threading.Lock()
_ROUTER_ENDPOINT_COUNTER = 0


def _router_endpoints(base_url: str) -> list[str]:
    return [item.strip().rstrip("/") for item in str(base_url or "").split(",") if item.strip()]


def _ordered_router_endpoints(base_url: str) -> list[str]:
    endpoints = _router_endpoints(base_url)
    if len(endpoints) <= 1:
        return endpoints
    global _ROUTER_ENDPOINT_COUNTER
    with _ROUTER_ENDPOINT_LOCK:
        start = _ROUTER_ENDPOINT_COUNTER % len(endpoints)
        _ROUTER_ENDPOINT_COUNTER += 1
    return endpoints[start:] + endpoints[:start]


def _truncate_router_text(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    # Keep the beginning and the end: model segments often start with the
    # action phrase and end with the object/certificate.
    head = max_chars // 2
    tail = max_chars - head - 24
    return raw[:head] + " ...[truncated]... " + raw[-max(0, tail) :]


def _compact_router_value(value, *, depth: int = 0):
    if depth > 4:
        return str(type(value).__name__)
    if isinstance(value, str):
        return _truncate_router_text(value, _ROUTER_MAX_STRING_CHARS)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        out = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _ROUTER_MAX_LIST_ITEMS:
                out["..."] = f"{len(value) - _ROUTER_MAX_LIST_ITEMS} more"
                break
            out[str(key)] = _compact_router_value(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [_compact_router_value(item, depth=depth + 1) for item in seq[:_ROUTER_MAX_LIST_ITEMS]]
        if len(seq) > _ROUTER_MAX_LIST_ITEMS:
            out.append(f"... {len(seq) - _ROUTER_MAX_LIST_ITEMS} more")
        return out
    return _truncate_router_text(str(value), _ROUTER_MAX_STRING_CHARS)


def _compact_router_candidate(candidate: dict) -> dict:
    # The model needs the candidate identity and concrete arguments. Long prose
    # descriptions are useful for synthetic tests but can push real training
    # requests beyond the router context window.
    return {
        "id": candidate.get("id"),
        "type": candidate.get("type"),
        "args": _compact_router_value(candidate.get("args", {})),
        "required_evidence": _compact_router_value(candidate.get("required_evidence", [])),
        "weak": bool(candidate.get("weak", False)),
        "terminal": bool(candidate.get("terminal", False)),
    }


def payload_to_json(payload: RouterPayload) -> str:
    return json.dumps(
        {
            "instruction": payload.instruction,
            "task": payload.task,
            "segment": _truncate_router_text(payload.segment, _ROUTER_MAX_SEGMENT_CHARS),
            "state_summary": _compact_router_value(payload.state_summary),
            "candidate_actions": [_compact_router_candidate(candidate) for candidate in payload.candidate_actions],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def parse_router_output(
    raw_output,
    candidates: Iterable[ActionCandidate],
    *,
    segment_text: str = "",
    source: str = "model",
) -> RoutedAction:
    """Parse the 0.6B router output into a validated ``RoutedAction``.

    Expected model output:

    ``{"selected_candidate_id": "...", "confidence": 0.91,
       "evidence_span": [l, r], "abstain": false}``
    """

    if isinstance(raw_output, str):
        try:
            data = json.loads(raw_output)
        except Exception:
            data = _extract_json_object(raw_output)
            if data is None:
                return RoutedAction(selected_candidate_id="abstain", source=source)
    elif isinstance(raw_output, dict):
        data = dict(raw_output)
    else:
        return RoutedAction(selected_candidate_id="abstain", source=source)

    # Accept a full OpenAI-compatible response as well as the raw assistant JSON.
    if "choices" in data and isinstance(data.get("choices"), list) and data["choices"]:
        message = data["choices"][0].get("message", {}) if isinstance(data["choices"][0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        nested = _extract_json_object(str(content))
        if nested is None:
            return RoutedAction(selected_candidate_id="abstain", source=source)
        data = nested

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_id = str(
        data.get("selected_candidate_id")
        or data.get("candidate_id")
        or data.get("id")
        or ("abstain" if data.get("abstain") else "")
    )
    if not selected_id or selected_id not in by_id or selected_id == "abstain" or bool(data.get("abstain", False)):
        return RoutedAction(selected_candidate_id="abstain", source=source)
    candidate = by_id[selected_id]
    raw_confidence = data.get("confidence", 0.0)
    malformed_confidence = False
    if isinstance(raw_confidence, bool):
        confidence = 0.0
        malformed_confidence = True
    elif isinstance(raw_confidence, str) and raw_confidence.strip().lower() in {"true", "false", "yes", "no"}:
        confidence = 0.0
        malformed_confidence = True
    else:
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except Exception:
            confidence = 0.0
            malformed_confidence = True
    span = data.get("evidence_span")
    evidence_span = None
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        try:
            left, right = int(span[0]), int(span[1])
            if 0 <= left < right <= max(len(segment_text), right):
                evidence_span = (left, right)
        except Exception:
            evidence_span = None
    if evidence_span is None and segment_text:
        evidence_span = _evidence_span(segment_text, candidate)
    scores = data.get("candidate_scores", ())
    parsed_scores = []
    if isinstance(scores, list):
        for item in scores[:8]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    parsed_scores.append((str(item[0]), float(item[1])))
                except Exception:
                    continue
    return RoutedAction(
        selected_candidate_id=selected_id,
        action_type=candidate.action_type,
        action_args=dict(candidate.args),
        confidence=float(confidence),
        evidence_span=evidence_span,
        abstain=False,
        ambiguous=bool(data.get("ambiguous", False)) or malformed_confidence,
        candidate_scores=tuple(parsed_scores),
        source=source,
    )


def _extract_json_object(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def call_openai_router(
    payload: RouterPayload,
    candidates: Iterable[ActionCandidate],
    *,
    base_url: str,
    model: str,
    timeout: float = 2.0,
    max_tokens: int = 64,
) -> RoutedAction:
    """Call an OpenAI-compatible router service and parse its JSON output."""

    candidate_list = list(candidates)
    endpoints = _ordered_router_endpoints(base_url)
    if not endpoints or not str(model or "").strip():
        return RoutedAction(selected_candidate_id="abstain", source="http_missing_endpoint")
    request_body = {
        "model": str(model),
        "temperature": 0,
        "top_p": 1,
        "max_tokens": int(max(8, max_tokens)),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict semantic action router, not a reward model. "
                    "Return exactly one compact JSON object with keys: selected_candidate_id, confidence, "
                    "evidence_span, abstain, ambiguous. Do not explain and do not use markdown. "
                    "confidence must be a numeric float in [0,1], never true/false. "
                    "evidence_span must be a [start,end] character-offset array inside the segment, never text. "
                    "Default to ABSTAIN. Select a non-ABSTAIN candidate only when the segment explicitly satisfies "
                    "that candidate's required_evidence. Do not infer an action from the current state alone. "
                    "Do not select generic mentions of shortest path, MST, flow, all-pairs distance, centrality, "
                    "mapping, formula, answer, or verification unless the concrete candidate arguments are stated. "
                    "If two candidates are plausible, if the concrete arguments are missing, or if the segment is "
                    "only a plan/restatement/formula reminder, return ABSTAIN with ambiguous=true. "
                    "Reject/skip/do-not-add statements are valid only when the concrete rejected edge is stated. "
                    "Parent/predecessor updates in shortest path are RELAX_EDGE only when the updated node and "
                    "new distance or predecessor are stated."
                ),
            },
            {"role": "user", "content": payload_to_json(payload)},
        ],
    }

    def post_json(endpoint: str, body: dict) -> str:
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        # Ray workers may inherit proxy variables from the launcher. The router
        # is usually an internal IP service, so force direct connections here.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=float(timeout)) as resp:
            return resp.read().decode("utf-8", errors="replace")

    for endpoint in endpoints:
        try:
            body = post_json(endpoint, request_body)
            return parse_router_output(body, candidate_list, segment_text=payload.segment, source="http")
        except urllib.error.HTTPError as exc:
            # Some OpenAI-compatible servers do not implement response_format. Retry
            # once without it while keeping thinking disabled.
            if exc.code not in {400, 404, 422}:
                continue
            fallback_body = dict(request_body)
            fallback_body.pop("response_format", None)
            try:
                body = post_json(endpoint, fallback_body)
                return parse_router_output(body, candidate_list, segment_text=payload.segment, source="http")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                continue
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue
    return RoutedAction(selected_candidate_id="abstain", source="http_error")


def _candidate_numbers(candidate: ActionCandidate) -> set[str]:
    nums: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            nums.add(str(value))
        elif isinstance(value, float) and math.isfinite(value):
            nums.add(str(int(value)) if float(value).is_integer() else str(round(float(value), 6)))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(candidate.args)
    visit(candidate.edges)
    visit(candidate.nodes)
    return nums


def _evidence_span(text: str, candidate: ActionCandidate) -> tuple[int, int] | None:
    if not text:
        return None
    hits: list[tuple[int, int]] = []
    for raw in sorted(_candidate_numbers(candidate), key=len, reverse=True):
        if not raw:
            continue
        m = re.search(rf"\b{re.escape(raw)}\b", text)
        if m:
            hits.append((m.start(), m.end()))
    cues = _ACTION_CUES.get(candidate.action_type, ())
    for cue in cues:
        m = re.search(re.escape(cue), text, flags=re.I)
        if m:
            hits.append((m.start(), m.end()))
            break
    if not hits:
        return None
    left = max(0, min(h[0] for h in hits) - 24)
    right = min(len(text), max(h[1] for h in hits) + 48)
    return (left, right)


def _score_candidate(text: str, candidate: ActionCandidate) -> float:
    if candidate.action_type == "ABSTAIN":
        return 0.0
    low = text.lower()
    score = 0.0
    cues = _ACTION_CUES.get(candidate.action_type, ())
    cue_hits = sum(1 for cue in cues if cue in low)
    score += min(0.45, 0.12 * cue_hits)
    text_nums = _numbers(text)
    cand_nums = _candidate_numbers(candidate)
    if cand_nums:
        overlap = len(text_nums & cand_nums)
        score += min(0.40, 0.10 * overlap)
        # A numeric value plus its endpoint/context nodes is often how graph
        # actions are expressed without canonical verbs, e.g. "7 via 3 is 12".
        if overlap >= 3:
            score += 0.12
    if candidate.edges:
        for u, v in candidate.edges:
            if re.search(rf"\b{u}\b.{0,20}\b{v}\b|\b{v}\b.{0,20}\b{u}\b", text):
                score += 0.20
                break
    if candidate.nodes and any(str(node) in text_nums for node in candidate.nodes):
        score += 0.10
    if candidate.action_type == "RELAX_EDGE":
        new_dist = candidate.args.get("new_dist")
        u = candidate.args.get("u")
        v = candidate.args.get("v")
        has_new_dist = new_dist is not None and str(int(float(new_dist))) in text_nums
        has_endpoint_context = {str(u), str(v)} & text_nums
        if has_new_dist and has_endpoint_context:
            score += 0.18
    if candidate.action_type == "SETTLE_NODE":
        dist = candidate.args.get("dist")
        node = candidate.args.get("node", candidate.args.get("v"))
        if node is not None and str(node) in text_nums:
            score += 0.20
        if dist is not None and str(int(float(dist))) in text_nums:
            score += 0.12
        if any(cue in low for cue in ("permanent", "minimum tentative", "unsettled", "settled vertex")):
            score += 0.18
    if candidate.action_type == "NO_RELAX_EDGE" and any(
        cue in low for cue in ("does not improve", "no improvement", "not improve", "unchanged")
    ):
        score += 0.35
    if candidate.action_type == "AUGMENT_PATH":
        bottleneck = candidate.args.get("bottleneck")
        path = candidate.args.get("path") or []
        has_bottleneck = bottleneck is not None and str(int(float(bottleneck))) in text_nums
        has_path_context = sum(str(x) in text_nums for x in path) >= 2
        if has_bottleneck and has_path_context:
            score += 0.18
    if candidate.action_type == "PAIR_DISTANCE":
        dist = candidate.args.get("distance")
        if dist is not None and str(int(float(dist))) in text_nums:
            score += 0.12
    if candidate.weak:
        score *= 0.85
    return max(0.0, min(0.99, score))


def route_action_heuristic(
    segment_text: str,
    candidates: Iterable[ActionCandidate],
    *,
    source: str = "heuristic",
) -> RoutedAction:
    """Pick the most text-compatible candidate with a calibrated heuristic score."""

    candidate_list = list(candidates)
    if not candidate_list:
        return RoutedAction(selected_candidate_id="abstain", source=source)
    scored = [(candidate, _score_candidate(segment_text, candidate)) for candidate in candidate_list]
    scored.sort(key=lambda item: item[1], reverse=True)
    best, best_score = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    ambiguous = bool(best_score > 0.0 and second > 0.0 and (best_score - second) < 0.08)
    if best.action_type == "ABSTAIN" or best_score <= 0.0:
        return RoutedAction(
            selected_candidate_id="abstain",
            source=source,
            candidate_scores=tuple((c.candidate_id, float(s)) for c, s in scored[:5]),
        )
    return RoutedAction(
        selected_candidate_id=best.candidate_id,
        action_type=best.action_type,
        action_args=dict(best.args),
        confidence=float(best_score),
        evidence_span=_evidence_span(segment_text, best),
        abstain=False,
        ambiguous=ambiguous,
        candidate_scores=tuple((c.candidate_id, float(s)) for c, s in scored[:5]),
        source=source,
    )


def route_action_from_payload(
    payload: RouterPayload,
    candidates: Iterable[ActionCandidate],
    *,
    backend: str = "heuristic",
    raw_model_output=None,
    base_url: str = "",
    model: str = "",
    timeout: float = 2.0,
    max_tokens: int = 64,
    heuristic_fallback: bool = False,
    heuristic_fallback_threshold: float = 0.75,
) -> RoutedAction:
    """Route one segment through the configured backend.

    Backends:
    - ``heuristic``: dependency-free local scorer.
    - ``model_stub`` / ``model``: parse ``raw_model_output`` if supplied,
      otherwise abstain. This is the stable I/O seam for a future 0.6B service.
    """

    backend = str(backend or "heuristic").strip().lower()
    candidate_list = list(candidates)
    if backend in {"heuristic", "local"}:
        return route_action_heuristic(payload.segment, candidate_list, source="heuristic")
    if backend in {"http", "openai", "vllm"}:
        if raw_model_output is not None:
            return parse_router_output(raw_model_output, candidate_list, segment_text=payload.segment, source=backend)
        routed = call_openai_router(
            payload,
            candidate_list,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if heuristic_fallback and (routed.abstain or routed.ambiguous):
            fallback = route_action_heuristic(payload.segment, candidate_list, source=f"{backend}_heuristic_fallback")
            if (
                not fallback.abstain
                and not fallback.ambiguous
                and float(fallback.confidence) >= float(heuristic_fallback_threshold)
            ):
                return fallback
        return routed
    if backend in {"model", "model_stub", "external"}:
        if raw_model_output is None:
            return RoutedAction(selected_candidate_id="abstain", source=backend)
        return parse_router_output(raw_model_output, candidate_list, segment_text=payload.segment, source=backend)
    return route_action_heuristic(payload.segment, candidate_list, source=f"fallback:{backend}")
