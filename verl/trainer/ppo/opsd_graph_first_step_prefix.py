# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instance-specific graph first-step prefixes for OPSD teacher scoring.

This builder is intentionally deterministic. It does not solve the full task or
insert the final answer. It parses the current graph problem and emits only the
first verifier-compatible algorithmic move, so OPSD can compare the sampled
trajectory under a more informative teacher context.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from verl.trainer.ppo.opsd_teacher_prefix import OPSDTeacherPrefixBuilder, _coerce_mapping, _string_or_empty


@dataclass(frozen=True)
class ParsedGraphProblem:
    task: str
    directed: bool
    nodes: list[Any]
    edges: list[tuple[Any, Any, float | None]]
    source: Any | None = None
    target: Any | None = None
    query_node: Any | None = None
    prompt: str = ""

    @property
    def weighted(self) -> bool:
        return any(w is not None for _, _, w in self.edges)


class GraphFirstStepOPSDPrefixBuilder(OPSDTeacherPrefixBuilder):
    """Build an instance-specific first-step algorithm prefix for graph tasks.

    Config keys under ``actor_rollout_ref.actor.policy_loss``:

    - ``opsd_graph_prefix_task_allowlist``: comma-separated task allowlist.
    - ``opsd_graph_prefix_max_chars``: final prefix character cap.
    - ``opsd_graph_prefix_fallback_algorithm_text``: fallback to task text.
    - ``opsd_graph_prefix_include_negative_warning``: add "do not reward" hint.
    """

    DEFAULT_ALLOWLIST = {"all"}

    def __init__(self, config: Any = None):
        super().__init__(config)
        policy_cfg = getattr(self.config, "policy_loss", None)
        self.static_prefix = ""
        self.max_chars = 1000
        self.fallback_algorithm_text = True
        self.include_negative_warning = True
        allowlist_raw = ""
        if policy_cfg is not None:
            self.static_prefix = str(policy_cfg.get("opsd_teacher_prefix", "") or "").strip()
            self.max_chars = int(
                policy_cfg.get(
                    "opsd_graph_prefix_max_chars",
                    policy_cfg.get("opsd_teacher_prefix_max_chars", 1000),
                )
                or 1000
            )
            self.fallback_algorithm_text = _as_bool(
                policy_cfg.get("opsd_graph_prefix_fallback_algorithm_text", True), default=True
            )
            self.include_negative_warning = _as_bool(
                policy_cfg.get("opsd_graph_prefix_include_negative_warning", True), default=True
            )
            allowlist_raw = str(policy_cfg.get("opsd_graph_prefix_task_allowlist", "") or "")
        self.task_allowlist = _parse_task_set(allowlist_raw) or set(self.DEFAULT_ALLOWLIST)

        try:
            from scripts.graph_task_algorithm_texts import GraphTaskAlgorithmTexts
        except Exception as exc:  # pragma: no cover
            self._algorithm_texts = None
            self._import_error = exc
        else:
            self._algorithm_texts = GraphTaskAlgorithmTexts
            self._import_error = None

    def build_one(self, sample: dict[str, Any]) -> str:
        task = _extract_task_name(sample)
        if task == "isomorphic_mapping":
            task = "isomophic_mapping"
        blocks: list[str] = []
        if self.static_prefix:
            blocks.append(self.static_prefix)

        if task not in self.task_allowlist and "all" not in self.task_allowlist and "*" not in self.task_allowlist:
            return self._join_and_trim(blocks)

        problem = parse_graph_problem(sample, task)
        first_step = build_first_step_hint(problem)
        if first_step:
            blocks.append(_format_first_step_prefix(problem, first_step, self.include_negative_warning))
        elif self.fallback_algorithm_text:
            fallback = self._task_algorithm_prefix(task)
            if fallback:
                blocks.append(fallback)

        return self._join_and_trim(blocks)

    def _task_algorithm_prefix(self, task: str) -> str:
        if self._algorithm_texts is None:
            raise RuntimeError(
                "GraphFirstStepOPSDPrefixBuilder could not import scripts.graph_task_algorithm_texts"
            ) from self._import_error
        if task and self._algorithm_texts.has_task(task):
            return (
                "Teacher algorithm hint for OPSD scoring. Use this fallback procedure "
                "when an instance-specific first step cannot be parsed.\n"
                + self._algorithm_texts.prompt_block(task)
            )
        return ""

    def _join_and_trim(self, blocks: list[str]) -> str:
        prefix = "\n\n".join(block for block in blocks if str(block or "").strip()).strip()
        if self.max_chars > 0 and len(prefix) > self.max_chars:
            prefix = prefix[: self.max_chars].rstrip()
        return prefix


def parse_graph_problem(sample: dict[str, Any], task: str = "") -> ParsedGraphProblem:
    extra = _coerce_mapping(sample.get("extra_info"))
    reward_model = _coerce_mapping(sample.get("reward_model"))
    prompt = _extract_prompt(sample)
    task = task or _extract_task_name(sample)
    if task == "isomorphic_mapping":
        task = "isomophic_mapping"

    direction = str(extra.get("direction", "") or "").lower()
    if not direction:
        low = prompt.lower()
        if "directed graph" in low or "directed acyclic graph" in low:
            direction = "directed"
        elif "undirected graph" in low:
            direction = "undirected"
    directed = direction == "directed"

    nodes = _parse_nodes(extra.get("nodes"))
    edges = _parse_edges(extra.get("edges"))
    if not edges:
        edges = _parse_edges_from_prompt(prompt)
    if not nodes:
        nodes = _nodes_from_edges(edges)

    source, target, query_node = _parse_sample(extra.get("sample"), task)
    if source is None and task in {"bfs", "dfs"}:
        source = query_node
    if query_node is None and task in {"betweenness_centrality", "closeness_centrality"}:
        query_node = source
    if task == "maximum_flow":
        if source is None or target is None:
            source, target = _parse_source_target_from_prompt(prompt, source, target)

    # Reward model is intentionally not used to compute the first step; it is
    # only kept available for future diagnostics and to avoid final-answer leak.
    del reward_model

    return ParsedGraphProblem(
        task=task,
        directed=directed,
        nodes=nodes,
        edges=edges,
        source=source,
        target=target,
        query_node=query_node,
        prompt=prompt,
    )


def build_first_step_hint(problem: ParsedGraphProblem) -> str:
    task = problem.task
    if not problem.nodes and not problem.edges:
        return ""
    if task in {"weighted_shortest_path", "shortest_path"}:
        return _first_step_shortest_path(problem, weighted=(task == "weighted_shortest_path" or problem.weighted))
    if task in {"weighted_minimum_spanning_tree", "minimum_spanning_tree"}:
        return _first_step_mst(problem)
    if task == "maximum_flow":
        return _first_step_flow(problem)
    if task in {"bfs", "dfs"}:
        return _first_step_traversal(problem, dfs=(task == "dfs"))
    if task in {"closeness_centrality", "betweenness_centrality"}:
        return _first_step_node_metric(problem)
    if task in {"global_efficiency", "wiener_index"}:
        return _first_step_all_pairs_metric(problem)
    if task == "isomophic_mapping":
        return _first_step_isomorphism(problem)
    return ""


def _first_step_shortest_path(problem: ParsedGraphProblem, *, weighted: bool) -> str:
    source = problem.source if problem.source is not None else _first_node(problem.nodes)
    if source is None:
        return ""
    adj = _adjacency(problem)
    neigh = sorted(adj.get(source, []), key=lambda x: str(x[0]))
    lines = [
        f"Initialize distance[{source}]=0 and every other distance to infinity; set all parents to None.",
        f"The first settled node is {source}, because it is the only node with finite tentative distance.",
    ]
    if neigh:
        relax = []
        for v, w in neigh[:8]:
            if weighted:
                relax.append(f"relax edge ({source},{v}) with weight {w:g}: tentative distance[{v}]={w:g}")
            else:
                relax.append(f"discover {v} from {source}: distance[{v}]=1 and parent[{v}]={source}")
        lines.append("Then process its outgoing/incident edges: " + "; ".join(relax) + ".")
    if problem.target is not None:
        lines.append(f"Do not report the final path to target {problem.target} before these initialization and relaxation steps.")
    return " ".join(lines)


def _first_step_mst(problem: ParsedGraphProblem) -> str:
    if not problem.edges:
        return ""
    weighted_edges = [(u, v, 1.0 if w is None else float(w)) for u, v, w in problem.edges]
    weighted_edges.sort(key=lambda e: (e[2], str(e[0]), str(e[1])))
    u, v, w = weighted_edges[0]
    return (
        "Initialize a DSU with every node in its own component. Sort edges by nondecreasing weight "
        f"using the task tie-breaking convention. The first candidate edge is ({u},{v}) with weight {w:g}. "
        f"Since {u} and {v} start in different components, the first valid MST action is to accept/add this edge and union the two components. "
        "Do not replay a final edge list as a Kruskal trace unless each edge is checked for components/cycles."
    )


def _first_step_flow(problem: ParsedGraphProblem) -> str:
    source = problem.source
    target = problem.target
    if source is None or target is None:
        return ""
    adj = _adjacency(problem)
    parent: dict[Any, Any | None] = {source: None}
    q: deque[Any] = deque([source])
    while q and target not in parent:
        u = q.popleft()
        for v, _w in sorted(adj.get(u, []), key=lambda x: str(x[0])):
            if v not in parent:
                parent[v] = u
                q.append(v)
                if v == target:
                    break
    path: list[Any] = []
    if target in parent:
        cur = target
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    if not path:
        return (
            f"Initialize all flows to 0 and residual capacities to the original capacities. "
            f"Start an Edmonds-Karp BFS from source {source}; if no residual path reaches sink {target}, the current flow is already maximal."
        )
    bottleneck = _path_bottleneck(path, problem.edges)
    amount = f" with bottleneck {bottleneck:g}" if bottleneck is not None else ""
    return (
        "Initialize all flows to 0 and construct the residual graph from the original capacities. "
        f"Run BFS from source {source} to sink {target}; a first residual path is {_format_path(path)}{amount}. "
        "The first valid flow action is to augment along that full path, then decrease forward residual capacities and increase reverse residual capacities by the bottleneck. "
        "A generic statement that a path exists is not enough without path and bottleneck/update evidence."
    )


def _first_step_traversal(problem: ParsedGraphProblem, *, dfs: bool) -> str:
    source = problem.source if problem.source is not None else _first_node(problem.nodes)
    if source is None:
        return ""
    adj = _adjacency(problem)
    neigh = [v for v, _ in sorted(adj.get(source, []), key=lambda x: str(x[0]))]
    if dfs:
        first = neigh[0] if neigh else None
        tail = f" The next DFS edge is ({source},{first}) if {first} is unvisited." if first is not None else ""
        return (
            f"Initialize visited as empty and start DFS at source {source}. "
            f"The first action is to visit {source}, then examine neighbors in the required order: {neigh[:10]}.{tail}"
        )
    return (
        f"Initialize visited={{ {source} }} and queue=[{source}]. "
        f"The first action is to dequeue/visit {source}; then enqueue its unvisited neighbors in required order: {neigh[:10]}."
    )


def _first_step_node_metric(problem: ParsedGraphProblem) -> str:
    q = problem.query_node if problem.query_node is not None else problem.source
    first_source = _first_node(problem.nodes)
    if problem.task == "closeness_centrality":
        if q is None:
            return ""
        return (
            f"For closeness of node {q}, first run shortest-path search from {q} "
            "under the prompt convention. Initialize dist[{q}]=0, queue/frontier=[{q}], and all other distances as unknown. "
            "The first certificate should be concrete pair distances from this node before computing the reciprocal/normalization."
        ).format(q=q)
    if first_source is None:
        return ""
    target = f" for target node {q}" if q is not None else ""
    return (
        f"For betweenness{target}, start Brandes-style shortest-path counting from source {first_source}. "
        f"Initialize dist[{first_source}]=0, sigma[{first_source}]=1, predecessor lists empty, and queue=[{first_source}]. "
        "The first valid evidence is a distance/sigma update from this source, not a final centrality value."
    )


def _first_step_all_pairs_metric(problem: ParsedGraphProblem) -> str:
    first = _first_node(problem.nodes)
    if first is None:
        return ""
    if problem.task == "wiener_index":
        return (
            f"Begin by computing shortest-path distances from node {first}. "
            "Record concrete pair distances d(first,v). For the Wiener index, add each unordered pair distance exactly once; do not use only the final scalar without pair-distance evidence."
        )
    return (
        f"Begin by computing shortest-path distances from node {first}. "
        "For each reachable ordered pair (first,v), record contribution 1/d(first,v). Global efficiency is the average over the prompt's ordered-pair denominator, so first show pair-distance/contribution evidence."
    )


def _first_step_isomorphism(problem: ParsedGraphProblem) -> str:
    graph1, graph2 = _split_isomorphism_edges(problem)
    if not graph1 or not graph2:
        return (
            "Parse the two graphs separately. First compare node counts and degree signatures. "
            "Start with candidate mappings only between nodes with the same degree/local signature, and verify bijection plus edge preservation before committing a pair."
        )
    deg1 = _degrees(graph1, directed=False)
    deg2 = _degrees(graph2, directed=False)
    node = sorted(deg1, key=lambda x: (-deg1[x], str(x)))[0]
    candidates = [v for v, d in sorted(deg2.items(), key=lambda x: str(x[0])) if d == deg1[node]][:8]
    return (
        f"Parse Graph1 and Graph2 separately. First compare node counts and degree signatures. "
        f"Choose Graph1 node {node} with degree {deg1[node]}; its first candidate images in Graph2 must have the same degree: {candidates}. "
        "Do not reward a mapping pair unless it preserves one-to-one consistency and all checked edges/non-edges remain possible."
    )


def _format_first_step_prefix(problem: ParsedGraphProblem, first_step: str, include_negative_warning: bool) -> str:
    node_count = len(problem.nodes)
    edge_count = len(problem.edges)
    query = []
    if problem.source is not None:
        query.append(f"source={problem.source}")
    if problem.target is not None:
        query.append(f"target={problem.target}")
    if problem.query_node is not None and problem.query_node != problem.source:
        query.append(f"query_node={problem.query_node}")
    lines = [
        "Teacher instance-specific graph hint for OPSD scoring.",
        "Use this hint only to score whether the sampled reasoning follows the correct first algorithmic move; do not treat it as the final answer.",
        f"Task: {problem.task}",
        f"Graph summary: directed={str(problem.directed).lower()}, weighted={str(problem.weighted).lower()}, nodes={node_count}, edges={edge_count}",
    ]
    if query:
        lines.append("Query: " + ", ".join(query))
    lines.append("Correct first step: " + first_step)
    if include_negative_warning:
        lines.append(
            "Do not assign higher OPSD preference to reasoning that jumps to a final answer or uses generic graph terminology without this concrete first-step evidence."
        )
    return "\n".join(lines)


def _extract_task_name(sample: dict[str, Any]) -> str:
    for key in ("task", "problem_type"):
        task = _string_or_empty(sample.get(key))
        if task:
            return task
    extra = _coerce_mapping(sample.get("extra_info"))
    for key in ("task", "problem_type", "task_class"):
        task = _string_or_empty(extra.get(key))
        if task:
            return task
    reward_model = _coerce_mapping(sample.get("reward_model"))
    for key in ("task", "problem_type", "task_class"):
        task = _string_or_empty(reward_model.get(key))
        if task:
            return task
    return ""


def _extract_prompt(sample: dict[str, Any]) -> str:
    for key in ("prompt", "query", "raw_prompt"):
        text = _prompt_to_text(sample.get(key))
        if text:
            return text
    extra = _coerce_mapping(sample.get("extra_info"))
    for key in ("prompt", "query", "raw_prompt"):
        text = _prompt_to_text(extra.get(key))
        if text:
            return text
    return ""


def _prompt_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _prompt_to_text(value.item())
        if value.size == 1:
            return _prompt_to_text(value.reshape(-1)[0])
        return "\n".join(_prompt_to_text(x) for x in value.reshape(-1) if _prompt_to_text(x))
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "") or ""))
            else:
                parts.append(_prompt_to_text(item))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        return str(value.get("content", "") or value.get("text", "") or "")
    return str(value)


def _parse_nodes(value: Any) -> list[Any]:
    parsed = _literal(value)
    if parsed is None:
        return []
    if isinstance(parsed, tuple) and len(parsed) == 2 and all(isinstance(x, int) for x in parsed):
        a, b = parsed
        if a <= b and b - a <= 10000:
            return list(range(a, b + 1))
    if isinstance(parsed, (list, tuple, set)):
        if len(parsed) == 2 and all(isinstance(x, list) for x in parsed):
            return list(parsed[0]) + list(parsed[1])
        return list(parsed)
    return []


def _parse_edges(value: Any) -> list[tuple[Any, Any, float | None]]:
    parsed = _literal(value)
    return _normalize_edges(parsed)


def _parse_edges_from_prompt(prompt: str) -> list[tuple[Any, Any, float | None]]:
    if not prompt:
        return []
    match = re.search(r"edges are:\s*(.+?)(?:\.\n|Question:|\n\nQuestion:)", prompt, flags=re.I | re.S)
    if not match:
        return []
    text = match.group(1).strip().rstrip(".")
    return _normalize_edges(_literal("[" + text + "]") if not text.startswith("[") else _literal(text))


def _normalize_edges(parsed: Any) -> list[tuple[Any, Any, float | None]]:
    if parsed is None:
        return []
    if isinstance(parsed, tuple) and len(parsed) == 2 and all(isinstance(x, list) for x in parsed):
        # Isomorphism stores (graph1_edges, graph2_edges). Keep both in the
        # flat view for summary; _split_isomorphism_edges can recover them.
        return _normalize_edges(parsed[0]) + _normalize_edges(parsed[1])
    out = []
    if not isinstance(parsed, (list, tuple, set)):
        return out
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        u, v = item[0], item[1]
        w = None
        if len(item) >= 3:
            try:
                w = float(item[2])
            except Exception:
                w = None
        out.append((u, v, w))
    return out


def _parse_sample(value: Any, task: str) -> tuple[Any | None, Any | None, Any | None]:
    parsed = _literal(value)
    if parsed in (None, "none", "None"):
        return None, None, None
    if isinstance(parsed, tuple) and len(parsed) >= 2:
        return parsed[0], parsed[1], None
    if isinstance(parsed, list) and len(parsed) >= 2 and task in {"weighted_shortest_path", "shortest_path", "maximum_flow"}:
        return parsed[0], parsed[1], None
    return parsed, None, parsed


def _parse_source_target_from_prompt(prompt: str, source: Any | None, target: Any | None) -> tuple[Any | None, Any | None]:
    if source is not None and target is not None:
        return source, target
    patterns = [
        r"source node\s+([A-Za-z0-9#_-]+).*?(?:sink|target) node\s+([A-Za-z0-9#_-]+)",
        r"from node\s+([A-Za-z0-9#_-]+).*?to node\s+([A-Za-z0-9#_-]+)",
        r"between nodes?\s+([A-Za-z0-9#_-]+)\s+and\s+([A-Za-z0-9#_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, prompt, flags=re.I | re.S)
        if m:
            return _coerce_node(m.group(1)), _coerce_node(m.group(2))
    return source, target


def _literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _literal(value.item())
        if value.size == 1:
            return _literal(value.reshape(-1)[0])
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() == "none":
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _nodes_from_edges(edges: Iterable[tuple[Any, Any, float | None]]) -> list[Any]:
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    return sorted(nodes, key=lambda x: str(x))


def _adjacency(problem: ParsedGraphProblem) -> dict[Any, list[tuple[Any, float]]]:
    adj: dict[Any, list[tuple[Any, float]]] = {}
    for u, v, w in problem.edges:
        weight = 1.0 if w is None else float(w)
        adj.setdefault(u, []).append((v, weight))
        if not problem.directed:
            adj.setdefault(v, []).append((u, weight))
    return adj


def _first_node(nodes: list[Any]) -> Any | None:
    if not nodes:
        return None
    return sorted(nodes, key=lambda x: str(x))[0]


def _format_path(path: list[Any]) -> str:
    return "->".join(str(x) for x in path)


def _path_bottleneck(path: list[Any], edges: list[tuple[Any, Any, float | None]]) -> float | None:
    weights = {}
    for u, v, w in edges:
        weights[(u, v)] = 1.0 if w is None else float(w)
    vals = []
    for u, v in zip(path, path[1:]):
        if (u, v) not in weights:
            return None
        vals.append(weights[(u, v)])
    return min(vals) if vals else None


def _split_isomorphism_edges(problem: ParsedGraphProblem) -> tuple[list[tuple[Any, Any, float | None]], list[tuple[Any, Any, float | None]]]:
    extra_edges = None
    # Re-parse from prompt is not reliable for isomorphism; use edge id ranges
    # as a fallback when the flattened list is all we have.
    if problem.edges:
        left = []
        right = []
        for edge in problem.edges:
            u, v, _ = edge
            if isinstance(u, int) and isinstance(v, int) and u < 100 and v < 100:
                left.append(edge)
            else:
                right.append(edge)
        if left and right:
            return left, right
    return [], []


def _degrees(edges: list[tuple[Any, Any, float | None]], *, directed: bool) -> dict[Any, int]:
    deg: dict[Any, int] = {}
    for u, v, _ in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + (0 if directed else 1)
    return deg


def _coerce_node(text: str) -> Any:
    try:
        return int(text)
    except Exception:
        return text


def _parse_task_set(raw: str) -> set[str]:
    out = {item.strip().lower() for item in re.split(r"[,;|\s]+", str(raw or "")) if item.strip()}
    if "isomorphic_mapping" in out:
        out.add("isomophic_mapping")
    return out


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
