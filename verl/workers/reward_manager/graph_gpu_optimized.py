from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
import re
import numpy as np
import warnings
import gc

from .registry import register

# Suppress misleading fast tokenizer warning
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

@register("graph_gpu_optimized")
class GraphRewardManagerGPUOptimized:
    """GPU-optimized reward manager that avoids CPU transfers.

    Key optimizations:
    1. Immediate detach() and clone() to break reference chains
    2. Batch processing of entities with size limits
    3. Explicit cleanup after each batch
    4. NO CPU transfers (stays on GPU)
    5. Forced garbage collection

    Args:
        tokenizer: Tokenizer for decoding responses
        num_examine: Number of samples to print for debugging
        compute_score: Function to compute correctness score
        reward_fn_key: Key to retrieve reward function from data
        activation_reward_fn: Function to compute step rewards from embeddings
        entity_patterns: Regex patterns for extracting entities
        step_reward_occurrence: Use 'all' or 'last' entity occurrences
        max_entity_batch_size: Maximum entities to process at once (default: 256)
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source',
                 activation_reward_fn=None,
                 activation_weight=0.0,
                 entity_patterns=None,
                 step_reward_occurrence: str = "last",
                 max_entity_batch_size: int = 256) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_entity_batch_size = max_entity_batch_size

        # Activation-based reward config
        if activation_reward_fn is not None:
            from omegaconf import DictConfig
            is_config = isinstance(activation_reward_fn, (dict, DictConfig))

            if is_config:
                print("Loading activation reward function from config...")

                model_dir = activation_reward_fn.get('model_dir', None)
                if model_dir:
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
                    print(f"✓ Loaded pair-stage2 activation reward model: {model_dir}")
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
                    print(f"✓ Loaded activation reward model: {activation_reward_fn.get('model_type')}_{activation_reward_fn.get('label_type')}")
            else:
                self.activation_reward_fn = activation_reward_fn
        else:
            self.activation_reward_fn = None

        self.activation_weight = activation_weight
        self.step_reward_occurrence = step_reward_occurrence
        if self.step_reward_occurrence not in ("all", "last"):
            raise ValueError(
                f"step_reward_occurrence must be 'all' or 'last', got: {self.step_reward_occurrence}"
            )

        # Entity recognition patterns
        self.entity_patterns = entity_patterns or {
            'node': [
                r'node[s]?\s*(\d+)',
                r'vertex[s]?\s*(\d+)',
                r'[Nn]ode[s]?\s*[:\-]?\s*(\d+)',
                r'\b([A-Z])\b',
            ],
            'edge': [
                r'edge[s]?\s*\((\d+),\s*(\d+)\)',
                r'\((\d+),\s*(\d+)\)',
                r'(\d+)\s*-+\s*(\d+)',
                r'(\d+)\s*→\s*(\d+)',
            ]
        }

        # Pre-compile regex patterns
        self._compiled_node_patterns = [re.compile(p, re.IGNORECASE) for p in self.entity_patterns['node']]
        self._compiled_edge_patterns = [re.compile(p, re.IGNORECASE) for p in self.entity_patterns['edge']]

    def extract_entities(self, response_str):
        """Extract node and edge entities from response string"""
        entities = {'nodes': set(), 'edges': set()}

        for pattern in self._compiled_node_patterns:
            matches = pattern.finditer(response_str)
            for match in matches:
                node_id = match.group(1)
                entities['nodes'].add(node_id)

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
        """Find entity positions in token sequence"""
        encoding = self.tokenizer(
            response_str,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        token_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']

        entity_positions = {
            'node_positions': {},
            'edge_positions': {}
        }

        for node_id in entities['nodes']:
            positions = self._find_entity_in_text(
                response_str,
                node_id,
                offset_mapping,
                entity_type='node'
            )
            if positions:
                entity_positions['node_positions'][node_id] = positions

        for edge in entities['edges']:
            src, dst = edge
            edge_str = f"({src}, {dst})"
            positions = self._find_entity_in_text(
                response_str,
                edge_str,
                offset_mapping,
                entity_type='edge'
            )
            if positions:
                entity_positions['edge_positions'][edge] = positions

        return entity_positions

    def _find_entity_in_text(self, text, entity_str, offset_mapping, entity_type='node'):
        """Find entity string in text and return corresponding token positions"""
        if entity_type == 'node':
            patterns = [
                rf'\bnode[s]?\s*[:\-]?\s*{re.escape(entity_str)}\b',
                rf'\b{re.escape(entity_str)}\b'
            ]
        else:
            patterns = [rf'{re.escape(entity_str)}']

        token_positions = []

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_char = match.start()
                end_char = match.end()

                covering_tokens = []
                for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start >= start_char and token_end <= end_char:
                        covering_tokens.append(token_idx)
                    elif token_start < end_char and token_end > start_char:
                        covering_tokens.append(token_idx)

                if covering_tokens:
                    token_positions.append(covering_tokens[-1])

        return token_positions

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards with GPU-optimized memory management"""

        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Check if embeddings are available
        has_embeddings = "response_embeddings" in data.batch

        # Initialize step rewards and anchor observations
        step_rewards_list = []
        anchor_obs_list = []
        max_entities = 0

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
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
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # ===== GPU-Optimized Activation-based Step Rewards =====
            if has_embeddings and self.activation_reward_fn is not None:
                # CRITICAL: Detach immediately to break reference chain
                # This prevents the large response_embeddings tensor from being held in memory
                sample_embeddings = data.batch["response_embeddings"][i].detach()

                # Extract entities
                entities = self.extract_entities(response_str)
                entity_positions = self.find_entity_token_positions(
                    response_str, valid_response_ids.tolist(), entities
                )

                # Track statistics
                num_nodes_detected = len(entities['nodes'])
                num_edges_detected = len(entities['edges'])
                num_nodes_with_position = len(entity_positions['node_positions'])
                num_edges_with_position = len(entity_positions['edge_positions'])

                # Collect entity positions and metadata
                entity_token_positions = []
                entity_metadata = []
                decay_schedule = [1.0, 0.7, 0.5, 0.3]

                # Collect node entities
                for node_id, token_positions in entity_positions['node_positions'].items():
                    if len(token_positions) == 0:
                        continue
                    if self.step_reward_occurrence == "all":
                        positions_iter = enumerate(token_positions)
                    else:
                        positions_iter = [(len(token_positions) - 1, token_positions[-1])]

                    for occurrence_idx, token_pos in positions_iter:
                        if token_pos < sample_embeddings.shape[0]:
                            entity_token_positions.append(token_pos)
                            decay_factor = decay_schedule[occurrence_idx] if occurrence_idx < len(decay_schedule) else decay_schedule[-1]
                            entity_metadata.append((token_pos, decay_factor, 'node', node_id))

                # Collect edge entities
                for edge, token_positions in entity_positions['edge_positions'].items():
                    if len(token_positions) == 0:
                        continue
                    if self.step_reward_occurrence == "all":
                        positions_iter = enumerate(token_positions)
                    else:
                        positions_iter = [(len(token_positions) - 1, token_positions[-1])]

                    for occurrence_idx, token_pos in positions_iter:
                        if token_pos < sample_embeddings.shape[0]:
                            entity_token_positions.append(token_pos)
                            decay_factor = decay_schedule[occurrence_idx] if occurrence_idx < len(decay_schedule) else decay_schedule[-1]
                            entity_metadata.append((token_pos, decay_factor, 'edge', edge))

                # Process entities in batches
                sample_step_rewards_for_stats = []
                node_step_rewards = []
                edge_step_rewards = []
                sample_step_rewards = []
                sample_anchor_obs = []

                if len(entity_token_positions) > 0:
                    # Process in batches to limit memory usage
                    for start_idx in range(0, len(entity_token_positions), self.max_entity_batch_size):
                        end_idx = min(start_idx + self.max_entity_batch_size, len(entity_token_positions))

                        # CRITICAL: Clone and detach to create independent copies
                        # This breaks the reference chain to sample_embeddings
                        batch_embeddings = torch.stack([
                            sample_embeddings[pos].clone().detach()
                            for pos in entity_token_positions[start_idx:end_idx]
                        ])

                        # Compute rewards for this batch
                        if hasattr(self.activation_reward_fn, 'compute_batch'):
                            # Use batch computation (stays on GPU)
                            batch_rewards = self.activation_reward_fn.compute_batch(
                                batch_embeddings,
                                batch_size=64  # Sub-batch size within activation_reward_fn
                            )
                            # Convert to list (still on GPU, just Python list of scalars)
                            step_rewards_raw = [r.item() for r in batch_rewards]
                        else:
                            # Fallback: loop through
                            step_rewards_raw = []
                            for emb in batch_embeddings:
                                reward = self.activation_reward_fn(emb)
                                if isinstance(reward, torch.Tensor):
                                    reward = reward.item()
                                step_rewards_raw.append(reward)

                        # Apply decay and assign to reward tensor
                        for idx_in_batch, global_idx in enumerate(range(start_idx, end_idx)):
                            token_pos, decay_factor, entity_type, entity_id = entity_metadata[global_idx]
                            raw_reward = step_rewards_raw[idx_in_batch]
                            decayed_reward = raw_reward * decay_factor

                            # Assign to reward tensor
                            reward_tensor[i, token_pos] = decayed_reward

                            # Track for statistics
                            sample_step_rewards_for_stats.append(decayed_reward)
                            if entity_type == 'node':
                                node_step_rewards.append(decayed_reward)
                            else:
                                edge_step_rewards.append(decayed_reward)

                            # Collect for GiGPO
                            sample_step_rewards.append(decayed_reward)
                            sample_anchor_obs.append(entity_id)

                        # CRITICAL: Immediately free batch memory
                        del batch_embeddings
                        if 'batch_rewards' in locals():
                            del batch_rewards

                        # Force garbage collection every few batches
                        if (start_idx // self.max_entity_batch_size) % 4 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()

                # CRITICAL: Free sample_embeddings immediately
                del sample_embeddings

                # Compute statistics (same as before)
                num_entities_with_reward = len(sample_step_rewards_for_stats)

                if num_entities_with_reward > 0:
                    step_rewards_array = np.array(sample_step_rewards_for_stats)
                    reward_extra_info['step_reward/occurrence'].append(self.step_reward_occurrence)
                    reward_extra_info['step_reward/count'].append(num_entities_with_reward)
                    reward_extra_info['step_reward/mean'].append(float(np.mean(step_rewards_array)))
                    reward_extra_info['step_reward/std'].append(float(np.std(step_rewards_array)))
                    reward_extra_info['step_reward/min'].append(float(np.min(step_rewards_array)))
                    reward_extra_info['step_reward/max'].append(float(np.max(step_rewards_array)))
                    reward_extra_info['step_reward/sum'].append(float(np.sum(step_rewards_array)))

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
                    # No entities - record zeros
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

                node_position_rate = num_nodes_with_position / num_nodes_detected if num_nodes_detected > 0 else 0.0
                edge_position_rate = num_edges_with_position / num_edges_detected if num_edges_detected > 0 else 0.0
                reward_extra_info['entity/node_position_rate'].append(node_position_rate)
                reward_extra_info['entity/edge_position_rate'].append(edge_position_rate)

                step_rewards_list.append(sample_step_rewards)
                anchor_obs_list.append(sample_anchor_obs)
                max_entities = max(max_entities, len(sample_step_rewards))
            else:
                step_rewards_list.append([])
                anchor_obs_list.append([])

            # Assign final correctness reward
            reward_tensor[i, valid_response_length - 1] = reward
            reward_extra_info['correctness_reward'].append(reward)

            # Print debug info (abbreviated)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("=" * 80)
                print("[prompt]", prompt_str[:200] + "..." if len(prompt_str) > 200 else prompt_str)
                print("[response]", response_str[:200] + "..." if len(response_str) > 200 else response_str)
                print("[ground_truth]", ground_truth)
                print(f"[score]", score if not isinstance(score, dict) else score.get('score'))

                if has_embeddings and self.activation_reward_fn is not None:
                    if reward_extra_info.get('step_reward/count') and len(reward_extra_info['step_reward/count']) > 0:
                        idx = -1
                        print(f"[entities] detected: {reward_extra_info['entity/total_detected'][idx]}, "
                              f"with_position: {reward_extra_info['entity/total_with_position'][idx]}")
                        print(f"[step_reward] count: {reward_extra_info['step_reward/count'][idx]}, "
                              f"mean: {reward_extra_info['step_reward/mean'][idx]:.6f}, "
                              f"sum: {reward_extra_info['step_reward/sum'][idx]:.6f}")
                        print(f"[correctness_reward] {reward_extra_info['correctness_reward'][idx]:.6f}")
                print("=" * 80)

        # CRITICAL: Delete response_embeddings from data.batch
        # This is the main source of memory accumulation
        if has_embeddings and "response_embeddings" in data.batch:
            del data.batch["response_embeddings"]
            torch.cuda.empty_cache()
            gc.collect()

        # Construct step_rewards and anchor_obs arrays
        if max_entities > 0:
            step_rewards_tensor = torch.full(
                (len(data), max_entities),
                float('nan'),
                dtype=torch.float32,
                device=reward_tensor.device,
            )
            anchor_obs_array = np.full((len(data), max_entities), None, dtype=object)

            for i, (rewards, entity_ids) in enumerate(zip(step_rewards_list, anchor_obs_list)):
                if len(rewards) > 0:
                    step_rewards_tensor[i, :len(rewards)] = torch.tensor(rewards, dtype=torch.float32)
                    anchor_obs_array[i, :len(entity_ids)] = np.array(entity_ids, dtype=object)
        else:
            step_rewards_tensor = torch.full(
                (len(data), 1),
                float('nan'),
                dtype=torch.float32,
                device=reward_tensor.device,
            )
            anchor_obs_array = np.full((len(data), 1), None, dtype=object)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "step_rewards": step_rewards_tensor,
                "anchor_obs": anchor_obs_array,
            }
        else:
            return reward_tensor
