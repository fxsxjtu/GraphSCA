"""Small data contracts for routing natural-language segments to graph actions.

The router is deliberately not a reward model. It selects one candidate action
from a deterministic candidate list; the graph executor remains responsible for
checking legality and computing process reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionCandidate:
    """A deterministic graph action candidate exposed to the semantic router."""

    candidate_id: str
    action_type: str
    args: dict[str, Any] = field(default_factory=dict)
    canonical_key: tuple[Any, ...] = field(default_factory=tuple)
    unit_names: tuple[str, ...] = field(default_factory=tuple)
    edges: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    nodes: tuple[int, ...] = field(default_factory=tuple)
    weak: bool = False
    terminal: bool = False
    reason: str = ""


@dataclass(frozen=True)
class RoutedAction:
    """Router output for one segment."""

    selected_candidate_id: str
    action_type: str = "ABSTAIN"
    action_args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_span: tuple[int, int] | None = None
    abstain: bool = True
    ambiguous: bool = False
    candidate_scores: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    source: str = "heuristic"


@dataclass(frozen=True)
class RouterPayload:
    """JSON-ready input contract for a model-backed action router."""

    task: str
    segment: str
    state_summary: dict[str, Any]
    candidate_actions: list[dict[str, Any]]
    instruction: str = (
        "Select the single candidate graph action explicitly expressed by the segment. "
        "Default to ABSTAIN unless the segment satisfies the selected candidate's required_evidence. "
        "Do not infer an action merely because it is plausible in the current state or because the segment mentions "
        "generic graph terms such as shortest path, MST, flow, distance table, mapping, or answer. "
        "For RELAX_EDGE the segment must name the updated node and either the new distance or "
        "predecessor/through node. "
        "For SETTLE_NODE it must name the node and a settle/finalize/minimum-tentative meaning. "
        "For ACCEPT_EDGE/REJECT_CYCLE_EDGE it must name the edge and the accept/reject/cycle/component meaning. "
        "For AUGMENT_PATH it must name the path and bottleneck/send amount; a generic residual path is not enough. "
        "For PAIR_DISTANCE it must name the exact pair and distance/contribution. "
        "For MAPPING_PAIR it must name both mapped nodes. "
        "If two candidates are plausible, or if evidence is only a formula/plan/restatement, return ABSTAIN with "
        "ambiguous=true. "
        "Do not judge reward or correctness. "
        "Return exactly one compact JSON object only. confidence must be a numeric float in [0,1], never true/false. "
        "evidence_span must be a [start,end] character-offset array inside segment, never text."
    )
