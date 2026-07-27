from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
from difflib import SequenceMatcher
import re
import numpy as np
import warnings

from .registry import register

# Suppress misleading fast tokenizer warning
# This warning is a false positive when using return_offsets_mapping with __call__
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')


def _parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off", ""):
            return False
        raise ValueError(f"Cannot parse boolean flag from string: {value!r}")
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return bool(value)

@register("graph")
class GraphRewardManager:
    """The reward manager with activation-based step reward support.

    This reward manager computes two types of rewards:
    1. Step rewards: Assigned to individual entity tokens based on their activations
    2. Final reward: Assigned to the last token based on correctness

    Args:
        tokenizer: Tokenizer for decoding responses
        num_examine: Number of samples to print for debugging
        compute_score: Function to compute correctness score
        reward_fn_key: Key to retrieve reward function from data
        activation_reward_fn: Function to compute step rewards from embeddings
                             Should accept (hidden_size,) tensor and return scalar
        activation_weight: (Deprecated) Not used in step reward mode
        entity_patterns: Regex patterns for extracting entities
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source',
                 # Activation-based reward parameters
                 activation_reward_fn=None,
                 activation_weight=0.0,
                 entity_patterns=None,
                 step_reward_occurrence: str = "last",
                 entity_reward_mode: str = "span",
                 use_sequence_context: bool = True,
                 broadcast_final_reward_to_all_tokens: bool = False,
                 occurrence_decay=None,
                 truncation_penalty=None,
                 # Direction 3 skip switch: when true, graph manager bypasses the
                 # activation_reward_fn forward pass (Direction 3 replaces the step-reward
                 # tensors downstream in ray_trainer).
                 skip_activation_reward: bool = False,
                 **_direction3_extra) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.skip_activation_reward = _parse_bool_flag(skip_activation_reward)
        if self.skip_activation_reward:
            print(
                "[GraphRewardManager] skip_activation_reward=True; activation_reward_fn forward "
                "passes will be bypassed (Direction 3 will produce step rewards downstream)."
            )
            # Force activation_reward_fn to None so every downstream branch becomes a no-op
            # that still returns the expected non_tensor_batch keys.
            activation_reward_fn = None

        # Activation-based reward config
        # Support both direct function and config dict
        if activation_reward_fn is not None:
            # Check if it's a dict or DictConfig
            from omegaconf import DictConfig
            is_config = isinstance(activation_reward_fn, (dict, DictConfig))

            if is_config:
                # Load from config dict
                print("Loading activation reward function from config...")

                hazard_model_path = activation_reward_fn.get('hazard_model_path', None)
                model_dir = activation_reward_fn.get('model_dir', None)

                if hazard_model_path:
                    # Hazard-rate prediction model (train_hazard_prediction.py)
                    from .hazard_reward_fn import create_hazard_reward_fn

                    self.activation_reward_fn = create_hazard_reward_fn(
                        model_path=hazard_model_path,
                        device=activation_reward_fn.get('device', 'cuda'),
                        normalize_reward=activation_reward_fn.get('normalize_reward', True),
                        reward_scale=activation_reward_fn.get('reward_scale', 1.0),
                        pair_mode=activation_reward_fn.get('pair_mode', 'prev_concat'),
                    )
                    print(f"Loaded hazard reward model: {hazard_model_path}")
                elif model_dir:
                    from .pair_stage2_reward_fn import create_pair_stage2_activation_reward_fn

                    self.activation_reward_fn = create_pair_stage2_activation_reward_fn(
                        model_dir=model_dir,
                        device=activation_reward_fn.get('device', 'cuda'),
                        normalize_reward=activation_reward_fn.get('normalize_reward', True),
                        reward_scale=activation_reward_fn.get('reward_scale', 1.0),
                        pair_mode=activation_reward_fn.get('pair_mode', 'prev_concat'),
                        stats_path=activation_reward_fn.get('stats_path', None),
                        auto_load_stats=activation_reward_fn.get('auto_load_stats', True),
                    )
                    print(f"Loaded pair-stage2 activation reward model: {model_dir}")
                else:
                    from .activation_reward_fn import create_activation_reward_fn

                    self.activation_reward_fn = create_activation_reward_fn(
                        model_type=activation_reward_fn.get('model_type', 'mlp'),
                        label_type=activation_reward_fn.get('label_type', 'mi'),
                        base_dir=activation_reward_fn.get('base_dir', ''),
                        device=activation_reward_fn.get('device', 'cuda'),
                        normalize_reward=activation_reward_fn.get('normalize_reward', True),
                        reward_scale=activation_reward_fn.get('reward_scale', 1.0),
                        stats_path=activation_reward_fn.get('stats_path', None),
                        auto_load_stats=activation_reward_fn.get('auto_load_stats', True),
                    )
                    print(f"Loaded activation reward model: {activation_reward_fn.get('model_type')}_{activation_reward_fn.get('label_type')}")
            else:
                # Direct function provided
                self.activation_reward_fn = activation_reward_fn
        else:
            self.activation_reward_fn = None

        self.activation_weight = activation_weight
        self.step_reward_occurrence = step_reward_occurrence
        if self.step_reward_occurrence not in ("all", "last"):
            raise ValueError(
                f"step_reward_occurrence must be 'all' or 'last', got: {self.step_reward_occurrence}"
            )
        self.entity_reward_mode = str(entity_reward_mode).lower()
        if self.entity_reward_mode not in ("span", "first", "last"):
            raise ValueError(
                f"entity_reward_mode must be 'span', 'first', or 'last', got: {self.entity_reward_mode}"
            )
        self.use_sequence_context = _parse_bool_flag(use_sequence_context)
        self.broadcast_final_reward_to_all_tokens = broadcast_final_reward_to_all_tokens
        self.occurrence_decay_schedule = self._parse_occurrence_decay(occurrence_decay)
        print(
            f"[GraphRewardManager] occurrence_decay_schedule = {self.occurrence_decay_schedule} "
            f"(raw input: {occurrence_decay!r})"
        )
        # Three-valued outcome reward: override correctness with a soft penalty for samples
        # whose response hit max_response_length AND failed to produce a correct answer.
        # None / "none" / False → disabled (legacy 0/1 behaviour).
        self.truncation_penalty = self._parse_truncation_penalty(truncation_penalty)
        print(
            f"[GraphRewardManager] truncation_penalty = {self.truncation_penalty} "
            f"(raw input: {truncation_penalty!r})"
        )

        # Entity recognition patterns (can be customized)
        self.entity_patterns = entity_patterns or {
            'node': [
                # Standard formats with various separators (ordered by specificity)
                r'node[s]?\s*[:\-_#]?\s*(\d+)',           # node 1, node:1, node-1, node_1, node#1
                r'node[s]?\s*[\(\[\{]\s*(\d+)\s*[\)\]\}]',  # node(1), node[1], node{1}
                r'vertex[s]?\s*[:\-_#]?\s*(\d+)',         # vertex 2, vertex-2, vertex_2
                r'vertex[s]?\s*[\(\[\{]\s*(\d+)\s*[\)\]\}]',  # vertex(2), vertex[2], vertex{2}
                r'[Nn]ode[s]?\s*[:\-_#]?\s*(\d+)',        # Node: 3, nodes-4
                r'[Vv]ertex[s]?\s*[:\-_#]?\s*(\d+)',      # Vertex 5, vertices_6
                r'\b[Vv][\-_#: ]?(\d+)\b',                # V1, v-2, V_3, V 4
                r'\b[Nn][\-_#: ]?(\d+)\b',                # N1, n-2, N_3, N 4
                # Single letter nodes - more conservative matching
                r'(?:node|vertex|visit|at|from|to|start|check)\s+([A-Z])\b(?!\s+(?:is|are|and|or|the|because|think|option))',  # "node A", "visit B" but not "option B"
            ],
            'edge': [
                # --- Bracketed notation (numeric) — tolerate whitespace inside the bracket ---
                r'edge[s]?\s*[\(\[\{]\s*(\d+)\s*,?\s*(\d+)\s*[\)\]\}]',  # edge(1,2), edge[ 1, 2 ], edge{1 2}
                r'[\(\[\{]\s*(\d+)\s*,?\s*(\d+)\s*[\)\]\}]',             # (1,2), [ 1 , 2 ], { 1, 2 }
                # --- Arrow notations (bidirectional before uni-directional to avoid partial match) ---
                r'(\d+)\s*<[-=]+>\s*(\d+)',                              # 1<->2, 1<=>2
                r'(\d+)\s*[-=]+>\s*(\d+)',                               # 1->2, 1-->2, 1=>2
                r'(\d+)\s*[→➜➔⇒]\s*(\d+)',                              # 1→2, 1➜2, 1⇒2
                r'(\d+)\s*<[-=]+\s*(\d+)',                               # 1<-2, 1<--2 (reverse)
                # --- Edge keyword + flexible separator (full arrow charset, mirrors _build_edge_search_patterns) ---
                r'edge[s]?\s*[:\-_#]?\s*(\d+)\s*[-,→➜➔⇒]\s*(\d+)',      # edge:1-2, edge_1→2, edge 1➜2
                # --- Letter-node edges ---
                r'[\(\[\{]\s*([A-Z])\s*,?\s*([A-Z])\s*[\)\]\}]',         # (A,B), [ A, B ], { A B }
                r'\b([A-Z])\s*<[-=]+>\s*([A-Z])\b',                      # A<->B
                r'\b([A-Z])\s*[-–—→➜➔⇒]\s*([A-Z])\b',                   # A-B, A→B, A➜B
            ]
        }

        # OPTIMIZATION: Pre-compile regex patterns for faster matching
        self._compiled_node_patterns = [re.compile(p, re.IGNORECASE) for p in self.entity_patterns['node']]
        self._compiled_edge_patterns = [re.compile(p, re.IGNORECASE) for p in self.entity_patterns['edge']]

    def requires_full_response_embeddings(self) -> bool:
        if not self.use_sequence_context:
            return False
        if self.activation_reward_fn is None or not hasattr(self.activation_reward_fn, "compute_sequence"):
            return False

        # Try lightweight metadata load first (no CUDA needed)
        if hasattr(self.activation_reward_fn, "_load_checkpoint_metadata"):
            try:
                self.activation_reward_fn._load_checkpoint_metadata()
            except Exception as exc:
                print(f"[Warning] Failed to load reward model metadata: {exc}")

        backbone = getattr(self.activation_reward_fn, "backbone", None)
        if backbone is None and hasattr(self.activation_reward_fn, "get_backbone_name"):
            try:
                backbone = self.activation_reward_fn.get_backbone_name()
            except Exception as exc:
                print(f"[Warning] Failed to infer reward backbone via get_backbone_name: {exc}")
                backbone = None
        if backbone is None and hasattr(self.activation_reward_fn, "get_model_info"):
            try:
                backbone = self.activation_reward_fn.get_model_info().get("backbone")
            except Exception as exc:
                print(f"[Warning] Failed to infer reward backbone via get_model_info: {exc}")
                backbone = None

        result = backbone not in (None, "mlp")
        print(f"[GraphRewardManager] requires_full_response_embeddings: backbone={backbone}, result={result}")
        return result

    def _default_reward_extra_value(self, key: str):
        if key == "step_reward/occurrence":
            return self.step_reward_occurrence
        if key.startswith("step_reward/") or key.startswith("entity/") or key == "correctness_reward":
            return 0.0
        return None

    @staticmethod
    def _parse_occurrence_decay(value):
        """Parse the ``occurrence_decay`` config into a numeric schedule.

        Accepts any of the following:
            * ``None`` / ``"none"`` / ``False``       → disabled (equivalent to ``[1.0]``)
            * ``"default"`` / ``True``                → legacy schedule ``[1.0, 0.7, 0.5, 0.3]``
            * ``list[float]`` / ``tuple[float]``      → custom schedule (length ≥ 1)

        The returned schedule is always a non-empty list of floats; lookups at
        ``index >= len(schedule)`` should clamp to ``schedule[-1]``.
        """
        # Handle falsy / None / explicit disable first
        if value is None or value is False:
            return [1.0]
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("", "none", "off", "false", "disable", "disabled", "no"):
                return [1.0]
            if normalized in ("default", "legacy", "true", "on", "yes"):
                return [1.0, 0.7, 0.5, 0.3]
            # Accept comma-separated strings like "1.0,0.7,0.5"
            if "," in normalized:
                try:
                    return [float(x) for x in normalized.split(",") if x.strip()]
                except ValueError:
                    raise ValueError(f"Failed to parse occurrence_decay string: {value!r}")
            raise ValueError(
                f"Unsupported occurrence_decay string: {value!r}. "
                "Use 'none', 'default', or a comma-separated list of floats."
            )
        if value is True:
            return [1.0, 0.7, 0.5, 0.3]
        # Sequence of floats
        try:
            schedule = [float(x) for x in value]
        except TypeError:
            raise ValueError(f"Unsupported occurrence_decay type: {type(value).__name__}")
        if not schedule:
            return [1.0]
        return schedule

    @staticmethod
    def _parse_truncation_penalty(value):
        """Parse the ``truncation_penalty`` config into either None (disabled) or a float.

        Accepts ``None`` / ``"none"`` / ``False`` / ``""`` to disable, or any numeric /
        numeric-looking string to set the penalty. Typical values: ``-0.3`` or ``-0.5``.
        """
        if value is None or value is False:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("", "none", "off", "false", "disable", "disabled", "no"):
                return None
            try:
                return float(normalized)
            except ValueError:
                raise ValueError(f"Unsupported truncation_penalty string: {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported truncation_penalty type: {type(value).__name__}")

    def _select_entity_reward_positions(self, token_span):
        token_span = [int(pos) for pos in token_span]
        if not token_span:
            return []
        if self.entity_reward_mode == "first":
            return [token_span[0]]
        if self.entity_reward_mode == "last":
            return [token_span[-1]]
        return token_span

    def _clip_token_span(self, token_span, response_length: int):
        return [
            int(pos) for pos in token_span
            if 0 <= int(pos) < int(response_length)
        ]

    def _assign_entity_reward(self, target_tensor, sample_idx: int, token_span, reward_value: float):
        reward_positions = self._select_entity_reward_positions(token_span)
        for token_pos in reward_positions:
            if token_pos >= target_tensor.shape[1]:
                raise AssertionError(
                    f"token_pos {token_pos} >= response_length {target_tensor.shape[1]}"
                )
            target_tensor[sample_idx, token_pos] = reward_value
        return reward_positions

    def _read_entity_reward(self, reward_tensor, sample_idx: int, token_span):
        reward_positions = self._select_entity_reward_positions(token_span)
        if not reward_positions:
            return None, ()
        return reward_tensor[sample_idx, reward_positions[0]].item(), tuple(reward_positions)

    def _standard_reward_extra_keys(self):
        return [
            "step_reward/occurrence",
            "step_reward/count",
            "step_reward/mean",
            "step_reward/std",
            "step_reward/min",
            "step_reward/max",
            "step_reward/sum",
            "step_reward/node_count",
            "step_reward/node_mean",
            "step_reward/node_std",
            "step_reward/node_min",
            "step_reward/node_max",
            "step_reward/edge_count",
            "step_reward/edge_mean",
            "step_reward/edge_std",
            "step_reward/edge_min",
            "step_reward/edge_max",
            "entity/nodes_detected",
            "entity/edges_detected",
            "entity/nodes_with_position",
            "entity/edges_with_position",
            "entity/total_detected",
            "entity/total_with_position",
            "entity/node_position_rate",
            "entity/edge_position_rate",
            "correctness_reward",
            "outcome/is_truncated",
            "outcome/truncation_penalty_applied",
        ]

    def get_reward_extra_keys(self):
        return list(self._standard_reward_extra_keys())

    def _finalize_reward_extra_info(self, reward_extra_info, expected_len: int):
        for key in self._standard_reward_extra_keys():
            reward_extra_info.setdefault(key, [])
        for key, values in list(reward_extra_info.items()):
            cur_len = len(values)
            if cur_len < expected_len:
                fill_value = self._default_reward_extra_value(key)
                values.extend([fill_value] * (expected_len - cur_len))
            elif cur_len > expected_len:
                del values[expected_len:]
        return reward_extra_info

    def extract_entities(self, response_str):
        """
        Extract node and edge entities from response string
        Returns: {'nodes': [node_ids], 'edges': [(src, dst)]}
        """
        entities = {'nodes': set(), 'edges': set()}

        # Extract nodes using pre-compiled patterns
        for pattern in self._compiled_node_patterns:
            matches = pattern.finditer(response_str)
            for match in matches:
                node_id = match.group(1)
                entities['nodes'].add(node_id)

        # Extract edges using pre-compiled patterns
        for pattern in self._compiled_edge_patterns:
            matches = pattern.finditer(response_str)
            for match in matches:
                if len(match.groups()) >= 2:
                    src, dst = match.group(1), match.group(2)
                    entities['edges'].add((src, dst))

        return {
            'nodes': sorted(list(entities['nodes'])),
            'edges': sorted(list(entities['edges']))
        }

    def find_entity_token_positions(self, response_str, response_ids, entities):
        """
        Find entity positions in token sequence
        Returns: {
            'node_positions': {node_id: [[token_idx1, token_idx2, ...], ...]},
            'edge_positions': {edge_tuple: [[token_idx1, token_idx2, ...], ...]}
        }
        """
        # Tokenize with offset mapping to get token-to-character alignment
        encoding = self.tokenizer(
            response_str,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        token_ids = [int(token_id) for token_id in encoding['input_ids']]
        offset_mapping = encoding['offset_mapping']
        original_token_ids = [int(token_id) for token_id in response_ids]
        token_position_map = self._build_token_position_map(
            retokenized_ids=token_ids,
            original_ids=original_token_ids,
        )

        entity_positions = {
            'node_positions': {},
            'edge_positions': {}
        }

        # Find node positions - keep all span occurrences
        for node_id in entities['nodes']:
            spans = self._find_entity_in_text(
                response_str,
                node_id,
                offset_mapping,
                token_position_map,
                entity_type='node'
            )
            if spans:
                entity_positions['node_positions'][node_id] = spans

        # Find edge positions - keep all span occurrences
        for edge in entities['edges']:
            spans = self._find_entity_in_text(
                response_str,
                edge,
                offset_mapping,
                token_position_map,
                entity_type='edge'
            )
            if spans:
                entity_positions['edge_positions'][edge] = spans

        return entity_positions

    def _build_token_position_map(self, retokenized_ids, original_ids):
        if not retokenized_ids:
            return []
        if not original_ids:
            return [[] for _ in retokenized_ids]
        if retokenized_ids == original_ids:
            return [[idx] for idx in range(len(original_ids))]

        mapping = [[] for _ in retokenized_ids]
        matcher = SequenceMatcher(a=retokenized_ids, b=original_ids, autojunk=False)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    mapping[i1 + offset] = [j1 + offset]
                continue

            if tag == "replace":
                retok_span = i2 - i1
                orig_span = j2 - j1
                if retok_span <= 0 or orig_span <= 0:
                    continue

                for offset in range(retok_span):
                    start = j1 + (offset * orig_span) // retok_span
                    end = j1 + ((offset + 1) * orig_span + retok_span - 1) // retok_span
                    end = min(end, j2)
                    if start >= end:
                        start = min(start, j2 - 1)
                        end = start + 1
                    mapping[i1 + offset] = list(range(start, end))
                continue

            if tag == "insert":
                if len(original_ids) == 0:
                    continue
                anchor = j1 if j1 < len(original_ids) else len(original_ids) - 1
                if anchor < 0:
                    continue
                for idx in range(i1, i2):
                    mapping[idx] = [anchor]

        return mapping

    def _dedupe_patterns(self, patterns):
        unique_patterns = []
        seen = set()
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        return unique_patterns

    def _build_node_search_patterns(self, entity_str):
        escaped = re.escape(entity_str)
        open_bracket = r"[\(\[\{]"
        close_bracket = r"[\)\]\}]"
        patterns = [
            rf'\bnode[s]?\s*[:\-_#]?\s*{escaped}\b',
            rf'\bnode[s]?\s*{open_bracket}\s*{escaped}\s*{close_bracket}',
            rf'\bvertex[s]?\s*[:\-_#]?\s*{escaped}\b',
            rf'\bvertex[s]?\s*{open_bracket}\s*{escaped}\s*{close_bracket}',
            rf'\b{escaped}\b',
        ]

        if entity_str.isdigit():
            patterns.extend([
                rf'\b[Vv][\-_#: ]?{escaped}\b',
                rf'\b[Nn][\-_#: ]?{escaped}\b',
            ])
        else:
            patterns.append(
                rf'(?:node|vertex|visit|at|from|to|start|check)\s+{escaped}\b'
            )

        return self._dedupe_patterns(patterns)

    def _build_edge_search_patterns(self, edge):
        src, dst = edge
        src_escaped = re.escape(src)
        dst_escaped = re.escape(dst)
        open_bracket = r"[\(\[\{]"
        close_bracket = r"[\)\]\}]"
        # Must stay symmetric with ``entity_patterns['edge']`` used by ``extract_entities``;
        # any format added there must be mirrored here, otherwise the edge will be
        # extractable but not mappable back to a token span.
        patterns = [
            rf'edge[s]?\s*{open_bracket}\s*{src_escaped}\s*,?\s*{dst_escaped}\s*{close_bracket}',
            rf'{open_bracket}\s*{src_escaped}\s*,?\s*{dst_escaped}\s*{close_bracket}',
            # Bidirectional must come before uni-directional to avoid partial match on `<-`
            rf'{src_escaped}\s*<[-=]+>\s*{dst_escaped}',
            rf'{src_escaped}\s*[-=]+>\s*{dst_escaped}',
            rf'{src_escaped}\s*[→➜➔⇒]\s*{dst_escaped}',
            rf'{src_escaped}\s*<[-=]+\s*{dst_escaped}',
            rf'edge[s]?\s*[:\-_#]?\s*{src_escaped}\s*[-,→➜➔⇒]\s*{dst_escaped}',
            # Plain dash / arrow between tokens (covers letter edges A-B / A→B and numeric 1-2)
            rf'{src_escaped}\s*[-–—→➜➔⇒]\s*{dst_escaped}',
        ]
        return self._dedupe_patterns(patterns)

    def _find_entity_in_text(self, text, entity_value, offset_mapping, token_position_map, entity_type='node'):
        """
        Find entity string in text and return corresponding token positions
        """
        if entity_type == 'node':
            patterns = self._build_node_search_patterns(entity_value)
        else:
            patterns = self._build_edge_search_patterns(entity_value)

        token_spans = []

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char = match.start()
                end_char = match.end()

                # Find tokens covering this character range
                covering_tokens = []
                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start >= start_char and token_end <= end_char:
                        covering_tokens.append(token_idx)
                    elif token_start < end_char and token_end > start_char:
                        # Partial overlap
                        covering_tokens.append(token_idx)

                if covering_tokens:
                    original_positions = []
                    for token_idx in covering_tokens:
                        if 0 <= token_idx < len(token_position_map):
                            original_positions.extend(token_position_map[token_idx])
                    normalized_span = tuple(dict.fromkeys(int(pos) for pos in original_positions))
                    if normalized_span and normalized_span not in token_spans:
                        token_spans.append(normalized_span)

        return [list(span) for span in token_spans]

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards with optional activation-based component"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        pure_step_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Check if embeddings are available and determine format
        embeddings = None
        if "response_embeddings" in data.batch:
            embeddings = data.batch["response_embeddings"]
        elif "response_embeddings" in data.meta_info:
            embeddings = data.meta_info["response_embeddings"]

        has_embeddings = embeddings is not None
        use_precomputed_entity_embeddings = False
        entity_positions_batch = None
        entity_span_positions_batch = None
        entity_metadata_batch = None

        flat_entity_keys = None
        if "flat_entity_keys" in data.meta_info:
            flat_entity_keys = data.meta_info["flat_entity_keys"]
        elif "flat_entity_keys" in data.non_tensor_batch:
            flat_entity_keys = data.non_tensor_batch["flat_entity_keys"]

        if has_embeddings:
            # Check if we have precomputed entity embeddings (memory-optimized format)
            # Entity embeddings: (total_entities, hidden_size) - 2D tensor
            # Full embeddings: (batch, response_len, hidden_size) - 3D tensor
            if embeddings.dim() == 2:
                use_precomputed_entity_embeddings = True
                if self.requires_full_response_embeddings():
                    print("[Warning] Sequence reward model received 2D precomputed entity embeddings; context will be disabled.")
                # Get entity metadata if available
                entity_positions_batch = data.non_tensor_batch.get("entity_positions", None)
                entity_span_positions_batch = data.non_tensor_batch.get("entity_span_positions", None)
                entity_metadata_batch = data.non_tensor_batch.get("entity_metadata", None)
                if flat_entity_keys is None:
                    raise RuntimeError("[graph] flat_entity_keys missing for 2D (entity) embeddings")

                if entity_positions_batch is not None:
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        total_entities = embeddings.shape[0]
                        memory_mb = embeddings.numel() * 4 / (1024**2)
                        print(f"[Memory Optimization] Using precomputed entity embeddings: {total_entities} entities, {memory_mb:.2f} MB")
                else:
                    # Fallback: entity embeddings without metadata
                    print("[Warning] Entity embeddings detected but no metadata found. Falling back to full extraction.")
                    use_precomputed_entity_embeddings = False

        # Initialize step rewards and anchor observations for GiGPO
        step_rewards_list = []  # Will store (batch_size, max_entities) with NaN padding
        anchor_obs_list = []    # Will store (batch_size, max_entities) with entity positions
        step_token_positions_list = []  # Will store (batch_size, max_entities) with token positions
        max_entities = 0  # Track maximum number of entities across batch

        # Track entity embedding index for precomputed format
        entity_embedding_idx = 0

        # Precompute rewards and per-sample key mapping for flat entity embeddings
        flat_step_rewards_raw = None
        flat_keys_by_sample = None
        flat_span_keys = None
        if has_embeddings and self.activation_reward_fn is not None and use_precomputed_entity_embeddings and flat_entity_keys is not None:
            if hasattr(self.activation_reward_fn, 'compute_batch'):
                span_indices = []
                flat_span_keys = []
                cursor = 0
                while cursor < len(flat_entity_keys):
                    sample_idx, entity_type, entity_id, span_positions, _token_pos = flat_entity_keys[cursor]
                    span_positions = tuple(int(pos) for pos in span_positions)
                    span_len = max(len(span_positions), 1)
                    end = min(cursor + span_len, len(flat_entity_keys))
                    span_indices.append(list(range(cursor, end)))
                    flat_span_keys.append((sample_idx, entity_type, entity_id, span_positions))
                    cursor = end
                try:
                    step_rewards_batch = self.activation_reward_fn.compute_batch(
                        embeddings, span_indices=span_indices
                    )
                except TypeError:
                    step_rewards_batch = self.activation_reward_fn.compute_batch(embeddings)
                if isinstance(step_rewards_batch, torch.Tensor):
                    if step_rewards_batch.dim() == 0:
                        raise TypeError(
                            "[graph] activation_reward_fn.compute_batch returned a scalar; "
                            "expected a 1D tensor/list with one reward per entity."
                        )
                    flat_step_rewards_raw = step_rewards_batch.detach().cpu().tolist()
                elif isinstance(step_rewards_batch, (list, tuple, np.ndarray)):
                    flat_step_rewards_raw = list(step_rewards_batch)
                else:
                    raise TypeError(
                        "[graph] activation_reward_fn.compute_batch returned unsupported type; "
                        "expected tensor/list/tuple/ndarray."
                    )
                if len(flat_step_rewards_raw) != len(flat_span_keys):
                    raise ValueError(
                        "[graph] activation_reward_fn.compute_batch length mismatch: "
                        f"got {len(flat_step_rewards_raw)}, expected {len(flat_span_keys)}."
                    )

            # Build per-sample list of (idx, entity_type, entity_id, token_pos)
            flat_keys_by_sample = [[] for _ in range(len(data))]
            limit = min(len(flat_span_keys or []), len(flat_step_rewards_raw or []))
            for idx in range(limit):
                sample_idx, entity_type, entity_id, span_positions = flat_span_keys[idx]
                flat_keys_by_sample[sample_idx].append((idx, entity_type, entity_id, span_positions))

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            extra_info['prompt'] = prompt_str
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # ===== Truncation-aware outcome reward (方案 3) =====
            # When a response hits max_response_length AND its parsed correctness is non-positive,
            # we cannot tell whether the model is "wrong" or merely "cut off mid-reasoning".
            # Overriding such samples' outcome with a soft penalty keeps GRPO group baselines clean:
            # the group mean is no longer dragged to ~0 by cut-off noise, and truly-wrong samples
            # retain distinguishable punishment. Only touches ``reward`` (the outcome scalar), not
            # ``pure_step_reward_tensor``, so the Y1 decoupling invariant is preserved.
            max_response_length_int = int(response_ids.shape[0])
            is_truncated_flag = int(valid_response_length >= max_response_length_int)
            penalty_applied = 0
            if (
                self.truncation_penalty is not None
                and is_truncated_flag == 1
                and float(reward) <= 0.0
            ):
                reward = self.truncation_penalty
                penalty_applied = 1
            reward_extra_info['outcome/is_truncated'].append(is_truncated_flag)
            reward_extra_info['outcome/truncation_penalty_applied'].append(penalty_applied)

            # ===== Activation-based Step Rewards =====
            # Assign step rewards to individual entity tokens
            if has_embeddings and self.activation_reward_fn is not None:
                # Ensure entity_positions is defined for downstream GiGPO collection
                entity_positions = {"node_positions": defaultdict(list), "edge_positions": defaultdict(list)}
                if use_precomputed_entity_embeddings:
                    # ===== MEMORY-OPTIMIZED PATH: Use precomputed entity embeddings =====
                    # Embeddings are already extracted at entity positions
                    # Format: (total_entities, hidden_size) with metadata

                    # Get entity positions and metadata for this sample
                    sample_entity_positions = (
                        entity_positions_batch[i] if entity_positions_batch is not None and len(entity_positions_batch) > 0 else []
                    )
                    sample_entity_span_positions = (
                        entity_span_positions_batch[i]
                        if entity_span_positions_batch is not None and len(entity_span_positions_batch) > 0 else []
                    )
                    sample_entity_metadata = (
                        entity_metadata_batch[i] if entity_metadata_batch is not None and len(entity_metadata_batch) > 0 else []
                    )
                    # Reconstruct entity_positions from metadata for downstream processing
                    for token_span, meta in zip(sample_entity_span_positions, sample_entity_metadata):
                        entity_type, entity_id = meta
                        token_span = [int(pos) for pos in token_span]
                        if len(token_span) == 0:
                            continue
                        if entity_type == "node":
                            entity_positions["node_positions"][entity_id].append(token_span)
                        else:
                            entity_positions["edge_positions"][entity_id].append(token_span)

                    # Ensure positions and metadata are aligned
                    # Use explicit flat mapping if available
                    if flat_entity_keys is not None:
                        # Track per-sample stats using precomputed flat rewards
                        sample_step_rewards_for_stats = []
                        node_step_rewards = []
                        edge_step_rewards = []
                        decay_schedule = self.occurrence_decay_schedule
                        occurrence_counters = defaultdict(int)

                        if flat_keys_by_sample is not None and flat_step_rewards_raw is not None:
                            for idx, entity_type, entity_id, token_span in flat_keys_by_sample[i]:
                                if idx >= len(flat_step_rewards_raw):
                                    break
                                raw_reward = flat_step_rewards_raw[idx]
                                entity_key = (entity_type, entity_id)
                                occurrence_idx = occurrence_counters[entity_key]
                                occurrence_counters[entity_key] += 1

                                if self.step_reward_occurrence == "all":
                                    decay_idx = min(occurrence_idx, len(decay_schedule) - 1)
                                    decay_factor = decay_schedule[decay_idx]
                                else:
                                    decay_factor = decay_schedule[0]
                                decayed_reward = raw_reward * decay_factor

                                try:
                                    self._assign_entity_reward(pure_step_reward_tensor, i, token_span, decayed_reward)
                                except AssertionError:
                                    print(f"\n[ERROR] token_pos out of bounds at sample {i}:")
                                    print(f"  token_span: {token_span}")
                                    print(f"  selected_positions: {self._select_entity_reward_positions(token_span)}")
                                    print(f"  reward_tensor.shape: {reward_tensor.shape}")
                                    print(f"  response_length (reward_tensor.shape[1]): {reward_tensor.shape[1]}")
                                    print(f"  entity_type: {entity_type}, entity_id: {entity_id}")
                                    print(f"  prompt_length: {prompt_length}")
                                    print(f"  valid_response_length: {valid_response_length}")
                                    print(f"  response_ids.shape: {response_ids.shape}")
                                    print(f"  data.batch['responses'].shape: {data.batch['responses'].shape}")
                                    print(f"  flat_entity_keys entry: (sample_idx={i}, entity_type={entity_type}, entity_id={entity_id}, token_span={token_span})")
                                    raise
                                sample_step_rewards_for_stats.append(decayed_reward)
                                if entity_type == 'node':
                                    node_step_rewards.append(decayed_reward)
                                else:
                                    edge_step_rewards.append(decayed_reward)

                        # Extract entities for statistics (we still need this for counting)
                        entities = self.extract_entities(response_str)
                        num_nodes_detected = len(entities['nodes'])
                        num_edges_detected = len(entities['edges'])
                        num_nodes_with_position = len(entity_positions['node_positions'])
                        num_edges_with_position = len(entity_positions['edge_positions'])
                    else:
                        num_entities = min(len(sample_entity_span_positions), len(sample_entity_metadata))
                        sample_entity_span_positions = sample_entity_span_positions[:num_entities]
                        sample_entity_metadata = sample_entity_metadata[:num_entities]

                        # Guard against running past available embeddings
                        available = embeddings.shape[0] - entity_embedding_idx
                        if available <= 0:
                            num_entities = 0
                        else:
                            num_entities = min(num_entities, available)
                            sample_entity_span_positions = sample_entity_span_positions[:num_entities]
                            sample_entity_metadata = sample_entity_metadata[:num_entities]

                        if num_entities > 0:
                            # Extract entity embeddings for this sample
                            entity_embeddings_batch = embeddings[entity_embedding_idx:entity_embedding_idx + num_entities]
                            entity_embedding_idx += num_entities

                            # Compute rewards for all entities at once
                            if hasattr(self.activation_reward_fn, 'compute_batch'):
                                step_rewards_batch = self.activation_reward_fn.compute_batch(entity_embeddings_batch)
                                step_rewards_raw = step_rewards_batch.cpu().tolist()
                            else:
                                # Fallback: loop through
                                step_rewards_raw = []
                                for emb in entity_embeddings_batch:
                                    reward = self.activation_reward_fn(emb)
                                    if isinstance(reward, torch.Tensor):
                                        reward = reward.item()
                                    step_rewards_raw.append(reward)

                            # Assign rewards to token positions
                            sample_step_rewards_for_stats = []
                            node_step_rewards = []
                            edge_step_rewards = []

                            decay_schedule = self.occurrence_decay_schedule
                            occurrence_counters = defaultdict(int)

                            for idx, (token_span, (entity_type, entity_id)) in enumerate(zip(sample_entity_span_positions, sample_entity_metadata)):
                                if idx >= len(step_rewards_raw):
                                    break
                                raw_reward = step_rewards_raw[idx]

                                entity_key = (entity_type, entity_id)
                                occurrence_idx = occurrence_counters[entity_key]
                                occurrence_counters[entity_key] += 1

                                if self.step_reward_occurrence == "all":
                                    decay_idx = min(occurrence_idx, len(decay_schedule) - 1)
                                    decay_factor = decay_schedule[decay_idx]
                                else:
                                    decay_factor = decay_schedule[0]
                                decayed_reward = raw_reward * decay_factor

                                try:
                                    self._assign_entity_reward(pure_step_reward_tensor, i, token_span, decayed_reward)
                                except AssertionError:
                                    print(f"\n[ERROR] token_pos out of bounds at sample {i} (non-flat path):")
                                    print(f"  token_span: {token_span}")
                                    print(f"  selected_positions: {self._select_entity_reward_positions(token_span)}")
                                    print(f"  reward_tensor.shape: {reward_tensor.shape}")
                                    print(f"  response_length: {reward_tensor.shape[1]}")
                                    print(f"  entity_type: {entity_type}, entity_id: {entity_id}")
                                    print(f"  prompt_length: {prompt_length}")
                                    print(f"  valid_response_length: {valid_response_length}")
                                    print(f"  response_ids.shape: {response_ids.shape}")
                                    print(f"  sample_entity_positions length: {len(sample_entity_span_positions)}")
                                    print(f"  sample_entity_metadata length: {len(sample_entity_metadata)}")
                                    raise

                                # Track for statistics
                                sample_step_rewards_for_stats.append(decayed_reward)
                                if entity_type == 'node':
                                    node_step_rewards.append(decayed_reward)
                                else:  # edge
                                    edge_step_rewards.append(decayed_reward)

                            # Extract entities for statistics (we still need this for counting)
                            entities = self.extract_entities(response_str)
                            num_nodes_detected = len(entities['nodes'])
                            num_edges_detected = len(entities['edges'])
                            num_nodes_with_position = len([m for m in sample_entity_metadata if m[0] == 'node'])
                            num_edges_with_position = len([m for m in sample_entity_metadata if m[0] == 'edge'])
                        else:
                            # No entities for this sample
                            sample_step_rewards_for_stats = []
                            node_step_rewards = []
                            edge_step_rewards = []
                            num_nodes_detected = 0
                            num_edges_detected = 0
                            num_nodes_with_position = 0
                            num_edges_with_position = 0

                else:
                    # ===== ORIGINAL PATH: Extract from full embeddings =====
                    # Extract embeddings for this sample: (response_len, hidden_size)
                    # response_embeddings may come from batch or meta_info depending on rollout path
                    sample_embeddings = embeddings[i]

                    # Extract entities from response
                    entities = self.extract_entities(response_str)
                    entity_positions = self.find_entity_token_positions(
                        response_str, valid_response_ids.tolist(), entities
                    )

                    # Track entity detection statistics
                    num_nodes_detected = len(entities['nodes'])
                    num_edges_detected = len(entities['edges'])
                    num_nodes_with_position = len(entity_positions['node_positions'])
                    num_edges_with_position = len(entity_positions['edge_positions'])

                    # ===== OPTIMIZED: Batch reward computation =====
                    # Collect entity metadata: (token_span, decay_factor, entity_type)
                    entity_metadata_list = []
                    decay_schedule = self.occurrence_decay_schedule

                    # Collect node entities
                    for node_id, token_spans in entity_positions['node_positions'].items():
                        if len(token_spans) == 0:
                            continue
                        if self.step_reward_occurrence == "all":
                            positions_iter = enumerate(token_spans)
                        else:
                            positions_iter = [(len(token_spans) - 1, token_spans[-1])]
                        for occurrence_idx, token_span in positions_iter:
                            if occurrence_idx < len(decay_schedule):
                                decay_factor = decay_schedule[occurrence_idx]
                            else:
                                decay_factor = decay_schedule[-1]
                            clipped_span = self._clip_token_span(token_span, valid_response_length)
                            if clipped_span:
                                entity_metadata_list.append((tuple(clipped_span), decay_factor, 'node'))

                    # Collect edge entities
                    for edge, token_spans in entity_positions['edge_positions'].items():
                        if len(token_spans) == 0:
                            continue
                        if self.step_reward_occurrence == "all":
                            positions_iter = enumerate(token_spans)
                        else:
                            positions_iter = [(len(token_spans) - 1, token_spans[-1])]
                        for occurrence_idx, token_span in positions_iter:
                            if occurrence_idx < len(decay_schedule):
                                decay_factor = decay_schedule[occurrence_idx]
                            else:
                                decay_factor = decay_schedule[-1]
                            clipped_span = self._clip_token_span(token_span, valid_response_length)
                            if clipped_span:
                                entity_metadata_list.append((tuple(clipped_span), decay_factor, 'edge'))

                    # Batch compute rewards if we have entities
                    sample_step_rewards_for_stats = []
                    node_step_rewards = []
                    edge_step_rewards = []

                    if len(entity_metadata_list) > 0:
                        # --- New path: full-sequence inference then pick entity positions ---
                        if hasattr(self.activation_reward_fn, 'compute_sequence'):
                            span_indices = [
                                self._select_entity_reward_positions(token_span)
                                for token_span, _, _ in entity_metadata_list
                            ]
                            try:
                                span_rewards = self.activation_reward_fn.compute_sequence(
                                    sample_embeddings[:valid_response_length],
                                    valid_length=valid_response_length,
                                    span_indices=span_indices,
                                )
                            except TypeError:
                                all_position_rewards = self.activation_reward_fn.compute_sequence(
                                    sample_embeddings[:valid_response_length],
                                    valid_length=valid_response_length,
                                )
                                span_rewards = torch.stack(
                                    [
                                        all_position_rewards[
                                            torch.as_tensor(
                                                self._select_entity_reward_positions(token_span),
                                                dtype=torch.long,
                                                device=all_position_rewards.device,
                                            )
                                        ].mean()
                                        for token_span, _, _ in entity_metadata_list
                                    ]
                                ) if entity_metadata_list else torch.empty(0, device=sample_embeddings.device)

                            for span_reward, (token_span, decay_factor, entity_type) in zip(span_rewards, entity_metadata_list):
                                raw_reward = span_reward.item() if isinstance(span_reward, torch.Tensor) else float(span_reward)

                                decayed_reward = raw_reward * decay_factor

                                try:
                                    self._assign_entity_reward(pure_step_reward_tensor, i, token_span, decayed_reward)
                                except AssertionError:
                                    print(f"\n[ERROR] token_pos out of bounds at sample {i} (full embeddings path):")
                                    print(f"  token_span: {token_span}")
                                    print(f"  selected_positions: {self._select_entity_reward_positions(token_span)}")
                                    print(f"  reward_tensor.shape: {reward_tensor.shape}")
                                    raise
                                sample_step_rewards_for_stats.append(decayed_reward)
                                if entity_type == 'node':
                                    node_step_rewards.append(decayed_reward)
                                else:
                                    edge_step_rewards.append(decayed_reward)
                        else:
                            # --- Legacy path: per-span batch inference ---
                            entity_embeddings_list = []
                            span_indices = []
                            for token_span, _, _ in entity_metadata_list:
                                span_start = len(entity_embeddings_list)
                                for token_pos in self._select_entity_reward_positions(token_span):
                                    if token_pos < sample_embeddings.shape[0]:
                                        entity_embeddings_list.append(sample_embeddings[token_pos])
                                span_indices.append(list(range(span_start, len(entity_embeddings_list))))

                            entity_embeddings_batch = torch.stack(entity_embeddings_list)

                            if hasattr(self.activation_reward_fn, 'compute_batch'):
                                try:
                                    step_rewards_batch = self.activation_reward_fn.compute_batch(
                                        entity_embeddings_batch, span_indices=span_indices
                                    )
                                except TypeError:
                                    step_rewards_batch = self.activation_reward_fn.compute_batch(entity_embeddings_batch)
                                step_rewards_raw = step_rewards_batch.cpu().tolist()
                            else:
                                step_rewards_raw = []
                                for span in span_indices:
                                    span_rewards = []
                                    for emb_idx in span:
                                        reward = self.activation_reward_fn(entity_embeddings_list[emb_idx])
                                        if isinstance(reward, torch.Tensor):
                                            reward = reward.item()
                                        span_rewards.append(reward)
                                    step_rewards_raw.append(float(np.mean(span_rewards)) if span_rewards else 0.0)

                            for idx, (token_span, decay_factor, entity_type) in enumerate(entity_metadata_list):
                                raw_reward = step_rewards_raw[idx]
                                decayed_reward = raw_reward * decay_factor

                                try:
                                    self._assign_entity_reward(pure_step_reward_tensor, i, token_span, decayed_reward)
                                except AssertionError:
                                    print(f"\n[ERROR] token_pos out of bounds at sample {i} (full embeddings path):")
                                    print(f"  token_span: {token_span}")
                                    print(f"  selected_positions: {self._select_entity_reward_positions(token_span)}")
                                    print(f"  reward_tensor.shape: {reward_tensor.shape}")
                                    raise
                                sample_step_rewards_for_stats.append(decayed_reward)
                                if entity_type == 'node':
                                    node_step_rewards.append(decayed_reward)
                                else:
                                    edge_step_rewards.append(decayed_reward)
                    # ===== END OPTIMIZED =====

                # Compute and store comprehensive statistics
                num_entities_with_reward = len(sample_step_rewards_for_stats)

                if num_entities_with_reward > 0:
                    # Overall step reward statistics
                    step_rewards_array = np.array(sample_step_rewards_for_stats)
                    reward_extra_info['step_reward/occurrence'].append(self.step_reward_occurrence)
                    reward_extra_info['step_reward/count'].append(num_entities_with_reward)
                    reward_extra_info['step_reward/mean'].append(float(np.mean(step_rewards_array)))
                    reward_extra_info['step_reward/std'].append(float(np.std(step_rewards_array)))
                    reward_extra_info['step_reward/min'].append(float(np.min(step_rewards_array)))
                    reward_extra_info['step_reward/max'].append(float(np.max(step_rewards_array)))
                    reward_extra_info['step_reward/sum'].append(float(np.sum(step_rewards_array)))

                    # Node-specific statistics
                    if len(node_step_rewards) > 0:
                        node_rewards_array = np.array(node_step_rewards)
                        reward_extra_info['step_reward/node_count'].append(len(node_step_rewards))
                        reward_extra_info['step_reward/node_mean'].append(float(np.mean(node_rewards_array)))
                        reward_extra_info['step_reward/node_std'].append(float(np.std(node_rewards_array)))
                        reward_extra_info['step_reward/node_min'].append(float(np.min(node_rewards_array)))
                        reward_extra_info['step_reward/node_max'].append(float(np.max(node_rewards_array)))
                    else:
                        reward_extra_info['step_reward/node_count'].append(0)
                        reward_extra_info['step_reward/node_mean'].append(0.0)
                        reward_extra_info['step_reward/node_std'].append(0.0)
                        reward_extra_info['step_reward/node_min'].append(0.0)
                        reward_extra_info['step_reward/node_max'].append(0.0)

                    # Edge-specific statistics
                    if len(edge_step_rewards) > 0:
                        edge_rewards_array = np.array(edge_step_rewards)
                        reward_extra_info['step_reward/edge_count'].append(len(edge_step_rewards))
                        reward_extra_info['step_reward/edge_mean'].append(float(np.mean(edge_rewards_array)))
                        reward_extra_info['step_reward/edge_std'].append(float(np.std(edge_rewards_array)))
                        reward_extra_info['step_reward/edge_min'].append(float(np.min(edge_rewards_array)))
                        reward_extra_info['step_reward/edge_max'].append(float(np.max(edge_rewards_array)))
                    else:
                        reward_extra_info['step_reward/edge_count'].append(0)
                        reward_extra_info['step_reward/edge_mean'].append(0.0)
                        reward_extra_info['step_reward/edge_std'].append(0.0)
                        reward_extra_info['step_reward/edge_min'].append(0.0)
                        reward_extra_info['step_reward/edge_max'].append(0.0)
                else:
                    # No entities with rewards - record zeros
                    reward_extra_info['step_reward/occurrence'].append(self.step_reward_occurrence)
                    reward_extra_info['step_reward/count'].append(0)
                    reward_extra_info['step_reward/mean'].append(0.0)
                    reward_extra_info['step_reward/std'].append(0.0)
                    reward_extra_info['step_reward/min'].append(0.0)
                    reward_extra_info['step_reward/max'].append(0.0)
                    reward_extra_info['step_reward/sum'].append(0.0)
                    reward_extra_info['step_reward/node_count'].append(0)
                    reward_extra_info['step_reward/node_mean'].append(0.0)
                    reward_extra_info['step_reward/node_std'].append(0.0)
                    reward_extra_info['step_reward/node_min'].append(0.0)
                    reward_extra_info['step_reward/node_max'].append(0.0)
                    reward_extra_info['step_reward/edge_count'].append(0)
                    reward_extra_info['step_reward/edge_mean'].append(0.0)
                    reward_extra_info['step_reward/edge_std'].append(0.0)
                    reward_extra_info['step_reward/edge_min'].append(0.0)
                    reward_extra_info['step_reward/edge_max'].append(0.0)

                # Entity detection statistics
                reward_extra_info['entity/nodes_detected'].append(num_nodes_detected)
                reward_extra_info['entity/edges_detected'].append(num_edges_detected)
                reward_extra_info['entity/nodes_with_position'].append(num_nodes_with_position)
                reward_extra_info['entity/edges_with_position'].append(num_edges_with_position)
                reward_extra_info['entity/total_detected'].append(num_nodes_detected + num_edges_detected)
                reward_extra_info['entity/total_with_position'].append(num_nodes_with_position + num_edges_with_position)

                # Position matching rate (how many detected entities got valid positions)
                if num_nodes_detected > 0:
                    node_position_rate = num_nodes_with_position / num_nodes_detected
                else:
                    node_position_rate = 0.0
                if num_edges_detected > 0:
                    edge_position_rate = num_edges_with_position / num_edges_detected
                else:
                    edge_position_rate = 0.0

                reward_extra_info['entity/node_position_rate'].append(node_position_rate)
                reward_extra_info['entity/edge_position_rate'].append(edge_position_rate)

                # Add final correctness reward after step rewards so the final reward is
                # additive to existing step-level rewards rather than replacing them.
                if valid_response_length > 0:
                    if self.broadcast_final_reward_to_all_tokens:
                        reward_tensor[i, :valid_response_length] += reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = reward

                # ===== Collect step rewards and anchor observations for GiGPO =====
                sample_step_rewards = []
                sample_anchor_obs = []
                sample_step_positions = []
                response_len = valid_response_length

                if flat_entity_keys is not None and use_precomputed_entity_embeddings:
                    # Use flat mapping order to collect GiGPO items
                    for idx, (sample_idx, entity_type, entity_id, span_positions) in enumerate(flat_span_keys or []):
                        if sample_idx != i:
                            continue
                        span_positions = [int(pos) for pos in span_positions if int(pos) < response_len]
                        if span_positions:
                            entity_reward, selected_positions = self._read_entity_reward(
                                pure_step_reward_tensor, i, span_positions
                            )
                            if entity_reward is not None:
                                sample_step_rewards.append(entity_reward)
                                sample_anchor_obs.append(entity_id)
                                sample_step_positions.append(selected_positions)
                else:
                    # Collect node entities
                    for node_id, token_spans in entity_positions['node_positions'].items():
                        if len(token_spans) == 0:
                            continue
                        if self.step_reward_occurrence == "all":
                            positions = token_spans
                        else:
                            positions = [token_spans[-1]]
                        for token_span in positions:
                            # Bounds check using response length (works for both precomputed and full embeddings)
                            token_span = [int(pos) for pos in token_span if int(pos) < response_len]
                            if token_span:
                                entity_reward, selected_positions = self._read_entity_reward(
                                    pure_step_reward_tensor, i, token_span
                                )
                                if entity_reward is not None:
                                    sample_step_rewards.append(entity_reward)
                                    sample_anchor_obs.append(node_id)  # Store entity ID instead of token position
                                    sample_step_positions.append(selected_positions)

                    # Collect edge entities
                    for edge, token_spans in entity_positions['edge_positions'].items():
                        if len(token_spans) == 0:
                            continue
                        if self.step_reward_occurrence == "all":
                            positions = token_spans
                        else:
                            positions = [token_spans[-1]]
                        for token_span in positions:
                            token_span = [int(pos) for pos in token_span if int(pos) < response_len]
                            if token_span:
                                entity_reward, selected_positions = self._read_entity_reward(
                                    pure_step_reward_tensor, i, token_span
                                )
                                if entity_reward is not None:
                                    sample_step_rewards.append(entity_reward)
                                    sample_anchor_obs.append(edge)  # Store entity ID instead of token position
                                    sample_step_positions.append(selected_positions)

                step_rewards_list.append(sample_step_rewards)
                anchor_obs_list.append(sample_anchor_obs)
                step_token_positions_list.append(sample_step_positions)
                max_entities = max(max_entities, len(sample_step_rewards))
            else:
                # No embeddings or activation reward function
                step_rewards_list.append([])
                anchor_obs_list.append([])
                step_token_positions_list.append([])
                if valid_response_length > 0:
                    if self.broadcast_final_reward_to_all_tokens:
                        reward_tensor[i, :valid_response_length] += reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = reward
            # ======================================
            reward_extra_info['correctness_reward'].append(reward)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("=" * 80)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

                # Print detailed step reward and entity statistics if available
                if has_embeddings and self.activation_reward_fn is not None:
                    if reward_extra_info.get('step_reward/count') and len(reward_extra_info['step_reward/count']) > 0:
                        idx = -1  # Last added item
                        print("\n--- Entity Detection Statistics ---")
                        print(f"[entities/nodes_detected]      {reward_extra_info['entity/nodes_detected'][idx]}")
                        print(f"[entities/edges_detected]      {reward_extra_info['entity/edges_detected'][idx]}")
                        print(f"[entities/total_detected]      {reward_extra_info['entity/total_detected'][idx]}")
                        print(f"[entities/nodes_with_position] {reward_extra_info['entity/nodes_with_position'][idx]}")
                        print(f"[entities/edges_with_position] {reward_extra_info['entity/edges_with_position'][idx]}")
                        print(f"[entities/total_with_position] {reward_extra_info['entity/total_with_position'][idx]}")
                        print(f"[entities/node_position_rate]  {reward_extra_info['entity/node_position_rate'][idx]:.2%}")
                        print(f"[entities/edge_position_rate]  {reward_extra_info['entity/edge_position_rate'][idx]:.2%}")

                        print("\n--- Step Reward Statistics (Overall) ---")
                        print(f"[step_reward/count]  {reward_extra_info['step_reward/count'][idx]}")
                        print(f"[step_reward/mean]   {reward_extra_info['step_reward/mean'][idx]:.6f}")
                        print(f"[step_reward/std]    {reward_extra_info['step_reward/std'][idx]:.6f}")
                        print(f"[step_reward/min]    {reward_extra_info['step_reward/min'][idx]:.6f}")
                        print(f"[step_reward/max]    {reward_extra_info['step_reward/max'][idx]:.6f}")
                        print(f"[step_reward/sum]    {reward_extra_info['step_reward/sum'][idx]:.6f}")

                        print("\n--- Step Reward Statistics (Nodes) ---")
                        print(f"[step_reward/node_count] {reward_extra_info['step_reward/node_count'][idx]}")
                        print(f"[step_reward/node_mean]  {reward_extra_info['step_reward/node_mean'][idx]:.6f}")
                        print(f"[step_reward/node_std]   {reward_extra_info['step_reward/node_std'][idx]:.6f}")
                        print(f"[step_reward/node_min]   {reward_extra_info['step_reward/node_min'][idx]:.6f}")
                        print(f"[step_reward/node_max]   {reward_extra_info['step_reward/node_max'][idx]:.6f}")

                        print("\n--- Step Reward Statistics (Edges) ---")
                        print(f"[step_reward/edge_count] {reward_extra_info['step_reward/edge_count'][idx]}")
                        print(f"[step_reward/edge_mean]  {reward_extra_info['step_reward/edge_mean'][idx]:.6f}")
                        print(f"[step_reward/edge_std]   {reward_extra_info['step_reward/edge_std'][idx]:.6f}")
                        print(f"[step_reward/edge_min]   {reward_extra_info['step_reward/edge_min'][idx]:.6f}")
                        print(f"[step_reward/edge_max]   {reward_extra_info['step_reward/edge_max'][idx]:.6f}")

                        print("\n--- Final Reward ---")
                        print(f"[correctness_reward]     {reward_extra_info['correctness_reward'][idx]:.6f}")
                        print("=" * 80)

        # ===== Construct step_rewards, anchor_obs and step_token_positions arrays for GiGPO =====
        if max_entities > 0:
            # Convert to padded arrays
            step_rewards_tensor = torch.full(
                (len(data), max_entities),
                float('nan'),
                dtype=torch.float32,
                device=reward_tensor.device,
            )
            anchor_obs_array = np.full((len(data), max_entities), None, dtype=object)  # Use object dtype for entity IDs
            step_token_positions_array = np.full((len(data), max_entities), None, dtype=object)

            for i, (rewards, entity_ids, token_positions) in enumerate(
                zip(step_rewards_list, anchor_obs_list, step_token_positions_list)
            ):
                if len(rewards) > 0:
                    step_rewards_tensor[i, :len(rewards)] = torch.tensor(rewards, dtype=torch.float32)
                    # Ensure edge tuples are kept as single objects (avoid 2D broadcast)
                    entity_ids_arr = np.empty(len(entity_ids), dtype=object)
                    entity_ids_arr[:] = entity_ids
                    anchor_obs_array[i, :len(entity_ids)] = entity_ids_arr
                    token_positions_arr = np.empty(len(token_positions), dtype=object)
                    token_positions_arr[:] = token_positions
                    step_token_positions_array[i, :len(token_positions)] = token_positions_arr
        else:
            # No entities found, return empty arrays
            step_rewards_tensor = torch.full(
                (len(data), 1),
                float('nan'),
                dtype=torch.float32,
                device=reward_tensor.device,
            )
            anchor_obs_array = np.full((len(data), 1), None, dtype=object)
            step_token_positions_array = np.full((len(data), 1), None, dtype=object)

        if return_dict:
            reward_extra_info = self._finalize_reward_extra_info(reward_extra_info, len(data))
            return {
                "reward_tensor": reward_tensor,
                "pure_step_reward_tensor": pure_step_reward_tensor,
                "reward_extra_info": reward_extra_info,
                "step_rewards": step_rewards_tensor,
                "anchor_obs": anchor_obs_array,
                "step_token_positions": step_token_positions_array,
            }
        else:
            return reward_tensor
