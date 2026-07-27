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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available
from verl.utils.import_utils import load_extern_object


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # Add rnc_classifier to PYTHONPATH for activation reward models
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
        rnc_classifier_path = os.environ.get("RNC_CLASSIFIER_PATH", "")
        if rnc_classifier_path and os.path.exists(rnc_classifier_path):
            current_pythonpath = runtime_env_vars.get("PYTHONPATH", "")
            if rnc_classifier_path not in current_pythonpath:
                if current_pythonpath:
                    runtime_env_vars["PYTHONPATH"] = f"{rnc_classifier_path}:{current_pythonpath}"
                else:
                    runtime_env_vars["PYTHONPATH"] = rnc_classifier_path
                runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _need_reference_policy_for_config(config) -> bool:
    loss_mode = str(config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")).lower()
    direction3_enabled = _direction3_enabled(config)
    method = _resolve_step_reward_method(config)
    # KL step reward also needs ref forward (for ref_log_prob in batch).
    return bool(
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
        or loss_mode in ("mgpo", "opsd", "opsd_future_kl")
        or direction3_enabled
        or method in ("kl", "direction3")
    )


def _resolve_step_reward_method(config) -> str:
    """Return one of {direction3, kl, consistency, graph_progress, none}.

    Honours the new ``step_reward_method`` key first; falls back to the legacy
    ``direction3_enabled`` flag for backward compatibility.
    """
    rk = config.reward_model.get("reward_kwargs", None)
    if rk is None:
        return "none"
    method = rk.get("step_reward_method", None)
    if method is not None:
        m = str(method).strip().lower()
        if m in ("direction3", "kl", "consistency", "graph_progress", "entity_info_only", "none"):
            return m
    # Legacy fallback.
    if _direction3_enabled(config):
        return "direction3"
    return "none"


def _direction3_enabled(config) -> bool:
    rk = config.reward_model.get("reward_kwargs", None)
    if rk is None:
        return False
    flag = rk.get("direction3_enabled", False)
    if isinstance(flag, str):
        return flag.strip().lower() in ("true", "1", "yes", "on")
    return bool(flag)


def _build_progress_reward_computer(config, tokenizer, graph_reward_manager):
    """Return a ProgressRewardComputer or None depending on config.reward_model.reward_kwargs."""
    if not _direction3_enabled(config):
        return None
    rk = config.reward_model.get("reward_kwargs", {}) or {}
    from verl.workers.reward_manager.progress_reward_fn import ProgressRewardComputer

    mode = str(rk.get("direction3_mode", "n")).lower()
    n = int(rk.get("direction3_n", 10))
    expansion = str(rk.get("direction3_expansion", "raw")).lower()
    logp_clip = float(rk.get("direction3_logp_clip", 20.0))
    scale = float(rk.get("direction3_scale", 0.3))
    pos_scale = float(rk.get("direction3_pos_scale", 1.0))
    neg_scale = float(rk.get("direction3_neg_scale", 1.0))
    length_normalize_raw = rk.get("direction3_length_normalize", True)
    if isinstance(length_normalize_raw, str):
        length_normalize = length_normalize_raw.strip().lower() in ("true", "1", "yes", "on")
    else:
        length_normalize = bool(length_normalize_raw)
    max_virtual_batch_per_call = int(rk.get("direction3_max_virtual_batch_per_call", 4096))
    shared_pool_size = int(rk.get("direction3_shared_pool_size", 30))
    min_rollout_freq = int(rk.get("direction3_min_rollout_freq", 2))
    anchor_mode = str(rk.get("direction3_anchor_mode", "span")).lower()
    # Share the entity_reward_mode with the graph reward manager so Direction 3
    # scatters at the same token positions the existing pipeline expects.
    entity_reward_mode = getattr(graph_reward_manager, "entity_reward_mode", "span")

    print(
        f"[Direction3] enabled. mode={mode} n={n} expansion={expansion} "
        f"anchor_mode={anchor_mode} "
        f"logp_clip={logp_clip} scale={scale} pos/neg={pos_scale}/{neg_scale} "
        f"length_normalize={length_normalize} entity_reward_mode={entity_reward_mode} "
        f"max_virtual_batch_per_call={max_virtual_batch_per_call} "
        f"shared_pool_size={shared_pool_size} min_rollout_freq={min_rollout_freq}"
    )
    return ProgressRewardComputer(
        tokenizer=tokenizer,
        graph_reward_manager=graph_reward_manager,
        mode=mode,
        n=n,
        expansion=expansion,
        logp_clip=logp_clip,
        scale=scale,
        pos_scale=pos_scale,
        neg_scale=neg_scale,
        length_normalize=length_normalize,
        entity_reward_mode=entity_reward_mode,
        anchor_mode=anchor_mode,
        max_virtual_batch_per_call=max_virtual_batch_per_call,
        shared_pool_size=shared_pool_size,
        min_rollout_freq=min_rollout_freq,
    )


def _build_kl_step_reward_computer(config, tokenizer, graph_reward_manager):
    rk = config.reward_model.get("reward_kwargs", {}) or {}
    from verl.workers.reward_manager.kl_step_reward import KLStepRewardComputer

    mode = str(rk.get("kl_entity_mode", "shared")).lower()
    n = int(rk.get("kl_n", 10))
    aggregation = str(rk.get("kl_aggregation", "mean")).lower()
    sign_mode = str(rk.get("kl_sign_mode", "with_outcome")).lower()
    scale = float(rk.get("kl_scale", 1.0))
    per_token_clip = float(rk.get("kl_per_token_clip", 5.0))
    agg_clip = float(rk.get("kl_agg_clip", 50.0))
    shared_pool_size = int(rk.get("kl_shared_pool_size", 30))
    min_rollout_freq = int(rk.get("kl_min_rollout_freq", 2))
    entity_reward_mode = getattr(graph_reward_manager, "entity_reward_mode", "span")

    print(
        f"[KLStepReward] mode={mode} n={n} aggregation={aggregation} "
        f"sign_mode={sign_mode} scale={scale} per_token_clip={per_token_clip} "
        f"agg_clip={agg_clip} shared_pool_size={shared_pool_size} "
        f"min_rollout_freq={min_rollout_freq} entity_reward_mode={entity_reward_mode}"
    )
    return KLStepRewardComputer(
        tokenizer=tokenizer,
        graph_reward_manager=graph_reward_manager,
        mode=mode,
        n=n,
        aggregation=aggregation,
        sign_mode=sign_mode,
        scale=scale,
        per_token_clip=per_token_clip,
        agg_clip=agg_clip,
        entity_reward_mode=entity_reward_mode,
        shared_pool_size=shared_pool_size,
        min_rollout_freq=min_rollout_freq,
    )


def _build_consistency_step_reward_computer(config, tokenizer, graph_reward_manager):
    rk = config.reward_model.get("reward_kwargs", {}) or {}
    from verl.workers.reward_manager.consistency_step_reward import (
        ConsistencyStepRewardComputer,
    )

    mode = str(rk.get("consistency_entity_mode", "shared")).lower()
    n = int(rk.get("consistency_n", 10))
    metric = str(rk.get("consistency_metric", "majority_share")).lower()
    sign_mode = str(rk.get("consistency_sign_mode", "with_outcome")).lower()
    normalize_raw = rk.get("consistency_normalize", True)
    if isinstance(normalize_raw, str):
        normalize = normalize_raw.strip().lower() in ("true", "1", "yes", "on")
    else:
        normalize = bool(normalize_raw)
    min_group_size = int(rk.get("consistency_min_group_size", 2))
    shared_pool_size = int(rk.get("consistency_shared_pool_size", 30))
    min_rollout_freq = int(rk.get("consistency_min_rollout_freq", 2))
    entity_reward_mode = getattr(graph_reward_manager, "entity_reward_mode", "span")

    print(
        f"[ConsistencyStepReward] mode={mode} n={n} metric={metric} "
        f"sign_mode={sign_mode} normalize={normalize} min_group_size={min_group_size} "
        f"shared_pool_size={shared_pool_size} min_rollout_freq={min_rollout_freq} "
        f"entity_reward_mode={entity_reward_mode}"
    )
    return ConsistencyStepRewardComputer(
        tokenizer=tokenizer,
        graph_reward_manager=graph_reward_manager,
        mode=mode,
        n=n,
        metric=metric,
        sign_mode=sign_mode,
        normalize=normalize,
        min_group_size=min_group_size,
        entity_reward_mode=entity_reward_mode,
        shared_pool_size=shared_pool_size,
        min_rollout_freq=min_rollout_freq,
    )


def _build_entity_info_only_computer(config, tokenizer, graph_reward_manager):
    rk = config.reward_model.get("reward_kwargs", {}) or {}
    from verl.workers.reward_manager.entity_info_computer import EntityInfoOnlyComputer

    mode = str(rk.get("entity_info_mode", "all")).lower()
    n = int(rk.get("entity_info_n", 10))
    shared_pool_size = int(rk.get("entity_info_shared_pool_size", 30))
    min_rollout_freq = int(rk.get("entity_info_min_rollout_freq", 2))
    max_anchors_per_rollout = int(rk.get("entity_info_max_anchors", 64))

    print(
        f"[EntityInfoOnly] mode={mode} n={n} max_anchors={max_anchors_per_rollout} "
        f"shared_pool_size={shared_pool_size} min_rollout_freq={min_rollout_freq}"
    )
    return EntityInfoOnlyComputer(
        tokenizer=tokenizer,
        graph_reward_manager=graph_reward_manager,
        mode=mode,
        n=n,
        shared_pool_size=shared_pool_size,
        min_rollout_freq=min_rollout_freq,
        max_anchors_per_rollout=max_anchors_per_rollout,
    )


def _build_graph_progress_reward_computer(config, tokenizer, graph_reward_manager):
    rk = config.reward_model.get("reward_kwargs", {}) or {}
    from verl.workers.reward_manager.graph_progress_reward import GraphProgressRewardComputer

    def _as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    max_entities = int(rk.get("graph_progress_max_entities", 64))
    reward_clip = float(rk.get("graph_progress_reward_clip", 1.0))
    positive_reward = float(rk.get("graph_progress_positive_reward", 1.0))
    focus_reward = float(rk.get("graph_progress_focus_reward", 0.2))
    irrelevant_node_reward = float(rk.get("graph_progress_irrelevant_node_reward", 0.0))
    irrelevant_edge_reward = float(rk.get("graph_progress_irrelevant_edge_reward", -0.1))
    invalid_edge_reward = float(rk.get("graph_progress_invalid_edge_reward", -1.0))
    order_hit_reward = float(rk.get("graph_progress_order_hit_reward", 1.0))
    order_seen_reward = float(rk.get("graph_progress_order_seen_reward", -0.2))
    order_skip_reward = float(rk.get("graph_progress_order_skip_reward", 0.2))
    coverage_mode = str(rk.get("graph_progress_coverage_mode", "prefix")).lower()
    prefix_decay = float(rk.get("graph_progress_prefix_decay", 0.995))
    scorer = str(rk.get("graph_progress_scorer", "answer")).lower()
    if scorer == "graph_logic":
        coverage_mode = "segment"
    relative_alpha = float(rk.get("graph_progress_relative_alpha", 2.0))
    relative_min_position_weight = float(rk.get("graph_progress_relative_min_position_weight", 0.2))
    relative_final_gate_eta = float(rk.get("graph_progress_relative_final_gate_eta", 0.2))
    graph_logic_ordered_reward = float(rk.get("graph_logic_ordered_reward", 1.0))
    graph_logic_progress_reward = float(rk.get("graph_logic_progress_reward", 0.25))
    graph_logic_answer_reward = float(rk.get("graph_logic_answer_reward", 0.5))
    graph_logic_legal_reward = float(rk.get("graph_logic_legal_reward", 0.0))
    graph_logic_invalid_penalty = float(rk.get("graph_logic_invalid_penalty", 0.2))
    graph_logic_fact_dump_penalty = float(rk.get("graph_logic_fact_dump_penalty", 0.0))
    graph_logic_final_gate_eta = float(rk.get("graph_logic_final_gate_eta", 0.2))
    graph_logic_correct_process_weight = float(rk.get("graph_logic_correct_process_weight", 1.0))
    graph_logic_wrong_process_weight = float(rk.get("graph_logic_wrong_process_weight", graph_logic_final_gate_eta))
    graph_logic_correct_only_positive = _as_bool(
        rk.get("graph_logic_correct_only_positive", False), default=False
    )
    graph_logic_keep_wrong_negative = _as_bool(
        rk.get("graph_logic_keep_wrong_negative", True), default=True
    )
    graph_logic_unit_trust_filter = _as_bool(
        rk.get("graph_logic_unit_trust_filter", False), default=False
    )
    graph_logic_unit_trust_margin = float(rk.get("graph_logic_unit_trust_margin", 0.05))
    graph_logic_unit_trust_min_task_samples = int(
        rk.get("graph_logic_unit_trust_min_task_samples", 8)
    )
    graph_logic_unit_trust_min_correct_hits = int(
        rk.get("graph_logic_unit_trust_min_correct_hits", 1)
    )
    graph_logic_unit_trust_low_sample_keep = _as_bool(
        rk.get("graph_logic_unit_trust_low_sample_keep", True), default=True
    )
    graph_logic_unit_trust_keep_unknown_positive = _as_bool(
        rk.get("graph_logic_unit_trust_keep_unknown_positive", False), default=False
    )
    graph_logic_process_task_blocklist = str(rk.get("graph_logic_process_task_blocklist", ""))
    graph_logic_good_prefix_preserve = _as_bool(
        rk.get("graph_logic_good_prefix_preserve", False), default=False
    )
    graph_logic_good_prefix_wrong_weight = float(rk.get("graph_logic_good_prefix_wrong_weight", 0.25))
    graph_logic_good_prefix_wrong_cap = float(rk.get("graph_logic_good_prefix_wrong_cap", 0.0))
    graph_logic_token_mapping = str(rk.get("graph_logic_token_mapping", "exact")).lower()
    graph_logic_segment_merge_strategy = str(
        rk.get("graph_logic_segment_merge_strategy", "none")
    ).lower()
    graph_logic_segment_merge_target_count = int(rk.get("graph_logic_segment_merge_target_count", 0))
    graph_logic_segment_merge_max_chars = int(rk.get("graph_logic_segment_merge_max_chars", 1200))
    graph_logic_segment_merge_min_info_score = float(rk.get("graph_logic_segment_merge_min_info_score", 0.0))
    graph_logic_reward_mode = str(rk.get("graph_logic_reward_mode", "event")).lower()
    graph_logic_potential_scale = float(rk.get("graph_logic_potential_scale", 1.0))
    graph_logic_state_shaping = _as_bool(rk.get("graph_logic_state_shaping", False), default=False)
    graph_logic_state_reward_scale = float(rk.get("graph_logic_state_reward_scale", 1.0))
    graph_logic_state_tasks = str(rk.get("graph_logic_state_tasks", ""))
    graph_logic_state_calibrate_potential = _as_bool(
        rk.get("graph_logic_state_calibrate_potential", True), default=True
    )
    graph_logic_state_invalid_penalty = float(rk.get("graph_logic_state_invalid_penalty", 0.03))
    graph_logic_state_mention_reward = float(rk.get("graph_logic_state_mention_reward", 0.005))
    graph_logic_state_consistent_reward = float(rk.get("graph_logic_state_consistent_reward", 0.02))
    graph_logic_duplicate_no_progress_penalty = float(rk.get("graph_logic_duplicate_no_progress_penalty", 0.002))
    graph_logic_state_context_max_age = int(rk.get("graph_logic_state_context_max_age", 2))
    graph_logic_positive_reward_cap = float(rk.get("graph_logic_positive_reward_cap", 0.25))
    graph_logic_positive_reward_budget_mode = str(
        rk.get("graph_logic_positive_reward_budget_mode", "quality")
    ).lower()
    graph_logic_step_span_mode = str(rk.get("graph_logic_step_span_mode", "segment")).lower()
    graph_logic_action_span_max_per_segment = int(rk.get("graph_logic_action_span_max_per_segment", 6))
    graph_logic_action_span_context_chars = int(rk.get("graph_logic_action_span_context_chars", 0))
    graph_logic_state_candidate_grounding = _as_bool(
        rk.get("graph_logic_state_candidate_grounding", True), default=True
    )
    graph_logic_state_candidate_max_edges = int(rk.get("graph_logic_state_candidate_max_edges", 3))
    graph_logic_state_commitment_weight = _as_bool(
        rk.get("graph_logic_state_commitment_weight", True), default=True
    )
    graph_logic_state_commitment_min_multiplier = float(
        rk.get("graph_logic_state_commitment_min_multiplier", 0.75)
    )
    graph_logic_state_commitment_max_multiplier = float(
        rk.get("graph_logic_state_commitment_max_multiplier", 1.20)
    )
    graph_logic_executable_commitment_weight = _as_bool(
        rk.get("graph_logic_executable_commitment_weight", False), default=False
    )
    graph_logic_ambiguity_gate_positive = _as_bool(
        rk.get("graph_logic_ambiguity_gate_positive", False), default=False
    )
    graph_logic_ambiguity_gate_threshold = float(rk.get("graph_logic_ambiguity_gate_threshold", 0.50))
    graph_logic_ambiguity_gate_multiplier = float(rk.get("graph_logic_ambiguity_gate_multiplier", 0.0))
    graph_logic_certificate_only_hard_gate = _as_bool(
        rk.get("graph_logic_certificate_only_hard_gate", False), default=False
    )
    graph_logic_certificate_gate_task_allowlist = str(
        rk.get("graph_logic_certificate_gate_task_allowlist", "")
    )
    graph_logic_certificate_gate_min_exec_score = float(
        rk.get("graph_logic_certificate_gate_min_exec_score", 0.60)
    )
    graph_logic_certificate_gate_max_ambiguity = float(
        rk.get("graph_logic_certificate_gate_max_ambiguity", 0.35)
    )
    graph_logic_graph_sensitivity_gate_positive = _as_bool(
        rk.get("graph_logic_graph_sensitivity_gate_positive", False), default=False
    )
    graph_logic_graph_sensitivity_gate_task_allowlist = str(
        rk.get("graph_logic_graph_sensitivity_gate_task_allowlist", "")
    )
    graph_logic_graph_sensitivity_gate_multiplier = float(
        rk.get("graph_logic_graph_sensitivity_gate_multiplier", 0.25)
    )
    graph_logic_component_precision_gate = _as_bool(
        rk.get("graph_logic_component_precision_gate", False), default=False
    )
    graph_logic_component_precision_filter_units = _as_bool(
        rk.get("graph_logic_component_precision_filter_units", False), default=False
    )
    graph_logic_component_precision_task_allowlist = str(
        rk.get("graph_logic_component_precision_task_allowlist", "")
    )
    graph_logic_component_vague_potential_multiplier = float(
        rk.get("graph_logic_component_vague_potential_multiplier", 0.20)
    )
    graph_logic_component_noisy_positive_multiplier = float(
        rk.get("graph_logic_component_noisy_positive_multiplier", 0.0)
    )
    graph_logic_component_schema_parseable_multiplier = float(
        rk.get("graph_logic_component_schema_parseable_multiplier", 0.05)
    )
    graph_logic_oracle_milestones = _as_bool(rk.get("graph_logic_oracle_milestones", False), default=False)
    graph_logic_oracle_milestone_max_units = int(rk.get("graph_logic_oracle_milestone_max_units", 96))
    graph_logic_oracle_milestone_positive_only = _as_bool(
        rk.get("graph_logic_oracle_milestone_positive_only", False), default=False
    )
    graph_logic_oracle_milestone_count_actions = _as_bool(
        rk.get("graph_logic_oracle_milestone_count_actions", True), default=True
    )
    graph_logic_oracle_milestone_task_allowlist = str(
        rk.get("graph_logic_oracle_milestone_task_allowlist", "")
    )
    graph_logic_action_router_enable = _as_bool(rk.get("graph_logic_action_router_enable", False), default=False)
    graph_logic_action_router_mode = str(rk.get("graph_logic_action_router_mode", "shadow"))
    graph_logic_action_router_task_allowlist = str(rk.get("graph_logic_action_router_task_allowlist", ""))
    graph_logic_action_router_conf_threshold = float(rk.get("graph_logic_action_router_conf_threshold", 0.75))
    graph_logic_action_router_high_conf_threshold = float(
        rk.get("graph_logic_action_router_high_conf_threshold", 0.90)
    )
    graph_logic_action_router_positive_cap = float(rk.get("graph_logic_action_router_positive_cap", 0.05))
    graph_logic_action_router_negative = _as_bool(rk.get("graph_logic_action_router_negative", False), default=False)
    graph_logic_action_router_max_candidates = int(rk.get("graph_logic_action_router_max_candidates", 16))
    graph_logic_action_router_backend = str(rk.get("graph_logic_action_router_backend", "heuristic"))
    graph_logic_action_router_base_url = str(rk.get("graph_logic_action_router_base_url", ""))
    graph_logic_action_router_model = str(rk.get("graph_logic_action_router_model", "qwen3-0.6b-action-router"))
    graph_logic_action_router_timeout = float(rk.get("graph_logic_action_router_timeout", 2.0))
    graph_logic_action_router_max_tokens = int(rk.get("graph_logic_action_router_max_tokens", 96))
    graph_logic_action_router_heuristic_fallback = _as_bool(
        rk.get("graph_logic_action_router_heuristic_fallback", False), default=False
    )
    graph_logic_action_router_heuristic_fallback_threshold = float(
        rk.get("graph_logic_action_router_heuristic_fallback_threshold", 0.75)
    )
    graph_logic_action_router_trace = _as_bool(rk.get("graph_logic_action_router_trace", True), default=True)
    graph_logic_action_router_payload_trace = _as_bool(
        rk.get("graph_logic_action_router_payload_trace", False), default=False
    )
    graph_logic_prefix_credit = _as_bool(rk.get("graph_logic_prefix_credit", False), default=False)
    graph_logic_prefix_credit_window = int(rk.get("graph_logic_prefix_credit_window", 256))
    graph_logic_prefix_credit_decay = float(rk.get("graph_logic_prefix_credit_decay", 0.99))
    graph_logic_prefix_credit_local_alpha = float(rk.get("graph_logic_prefix_credit_local_alpha", 0.6))
    graph_logic_prefix_credit_negative = _as_bool(
        rk.get("graph_logic_prefix_credit_negative", False), default=False
    )
    graph_logic_family_balance = _as_bool(rk.get("graph_logic_family_balance", False), default=False)
    graph_logic_family_scale_csv = str(rk.get("graph_logic_family_scale_csv", ""))
    graph_logic_family_scale_default = float(rk.get("graph_logic_family_scale_default", 1.0))
    graph_logic_final_answer_checks = _as_bool(rk.get("graph_logic_final_answer_checks", False), default=False)
    graph_logic_schema_conflict_gate_positive = _as_bool(
        rk.get("graph_logic_schema_conflict_gate_positive", True), default=True
    )
    graph_logic_schema_conflict_penalty_scale = float(
        rk.get("graph_logic_schema_conflict_penalty_scale", 0.50)
    )
    graph_logic_legacy_core4_compat = _as_bool(
        rk.get("graph_logic_legacy_core4_compat", False), default=False
    )
    first_occurrence_only = _as_bool(rk.get("graph_progress_first_occurrence_only", True), default=True)
    extract_backend = str(rk.get("graph_progress_extract_backend", "process")).lower()
    extract_workers = int(rk.get("graph_progress_extract_workers", 0))
    print_timing = _as_bool(rk.get("graph_progress_print_timing", True), default=True)

    print(
        f"[GraphProgress] max_entities={max_entities} clip={reward_clip} "
        f"pos={positive_reward} focus={focus_reward} invalid_edge={invalid_edge_reward} "
        f"coverage={coverage_mode} prefix_decay={prefix_decay} scorer={scorer} "
        f"rel_alpha={relative_alpha} rel_min_w={relative_min_position_weight} rel_eta={relative_final_gate_eta} "
        f"logic_ordered={graph_logic_ordered_reward} logic_progress={graph_logic_progress_reward} "
        f"logic_answer={graph_logic_answer_reward} logic_invalid={graph_logic_invalid_penalty} "
        f"logic_eta={graph_logic_final_gate_eta} logic_token_mapping={graph_logic_token_mapping} "
        f"logic_process_weight={graph_logic_correct_process_weight}/{graph_logic_wrong_process_weight} "
        f"logic_correct_only_positive={graph_logic_correct_only_positive} "
        f"logic_keep_wrong_negative={graph_logic_keep_wrong_negative} "
        f"logic_unit_trust={graph_logic_unit_trust_filter}@{graph_logic_unit_trust_margin},"
        f"min_task={graph_logic_unit_trust_min_task_samples},"
        f"min_hits={graph_logic_unit_trust_min_correct_hits},"
        f"low_sample_keep={graph_logic_unit_trust_low_sample_keep},"
        f"keep_unknown={graph_logic_unit_trust_keep_unknown_positive} "
        f"logic_process_task_blocklist={graph_logic_process_task_blocklist or 'none'} "
        f"logic_good_prefix={graph_logic_good_prefix_preserve}/{graph_logic_good_prefix_wrong_weight} "
        f"logic_good_prefix_cap={graph_logic_good_prefix_wrong_cap} "
        f"logic_segment_merge={graph_logic_segment_merge_strategy}:target={graph_logic_segment_merge_target_count},"
        f"max_chars={graph_logic_segment_merge_max_chars},min_info={graph_logic_segment_merge_min_info_score} "
        f"logic_reward_mode={graph_logic_reward_mode} logic_potential_scale={graph_logic_potential_scale} "
        f"logic_state={graph_logic_state_shaping} logic_state_scale={graph_logic_state_reward_scale} "
        f"logic_state_tasks={graph_logic_state_tasks or 'all'} "
        f"logic_state_calibrate={graph_logic_state_calibrate_potential} "
        f"logic_state_invalid_penalty={graph_logic_state_invalid_penalty} "
        f"logic_state_mention_reward={graph_logic_state_mention_reward} "
        f"logic_state_consistent_reward={graph_logic_state_consistent_reward} "
        f"logic_duplicate_no_progress_penalty={graph_logic_duplicate_no_progress_penalty} "
        f"logic_state_context_max_age={graph_logic_state_context_max_age} "
        f"logic_positive_reward_cap={graph_logic_positive_reward_cap} "
        f"logic_positive_reward_budget_mode={graph_logic_positive_reward_budget_mode} "
        f"logic_step_span_mode={graph_logic_step_span_mode} "
        f"logic_action_span_max={graph_logic_action_span_max_per_segment} "
        f"logic_action_span_context={graph_logic_action_span_context_chars} "
        f"logic_candidate_grounding={graph_logic_state_candidate_grounding} "
        f"logic_commitment_weight={graph_logic_state_commitment_weight} "
        f"logic_executable_commitment={graph_logic_executable_commitment_weight} "
        f"logic_ambiguity_gate={graph_logic_ambiguity_gate_positive}@{graph_logic_ambiguity_gate_threshold} "
        f"logic_certificate_gate={graph_logic_certificate_only_hard_gate}:{graph_logic_certificate_gate_task_allowlist or 'all'} "
        f"logic_graph_sensitivity_gate={graph_logic_graph_sensitivity_gate_positive}:{graph_logic_graph_sensitivity_gate_task_allowlist or 'all'} "
        f"logic_component_precision_gate={graph_logic_component_precision_gate}:{graph_logic_component_precision_task_allowlist or 'default'} "
        f"logic_component_precision_filter_units={graph_logic_component_precision_filter_units} "
        f"logic_component_precision_mult={graph_logic_component_vague_potential_multiplier}/"
        f"{graph_logic_component_noisy_positive_multiplier}/{graph_logic_component_schema_parseable_multiplier} "
        f"logic_oracle={graph_logic_oracle_milestones}:{graph_logic_oracle_milestone_task_allowlist or 'all'} "
        f"logic_action_router={graph_logic_action_router_enable}:{graph_logic_action_router_mode}, "
        f"backend={graph_logic_action_router_backend}, base_url={graph_logic_action_router_base_url}, "
        f"model={graph_logic_action_router_model}, allowlist={graph_logic_action_router_task_allowlist or 'all'}, "
        f"conf={graph_logic_action_router_conf_threshold}, pos_cap={graph_logic_action_router_positive_cap}, "
        f"negative={graph_logic_action_router_negative}, fallback={graph_logic_action_router_heuristic_fallback}@{graph_logic_action_router_heuristic_fallback_threshold}, "
        f"payload_trace={graph_logic_action_router_payload_trace} "
        f"logic_prefix_credit={graph_logic_prefix_credit} "
        f"logic_prefix_credit_window={graph_logic_prefix_credit_window} "
        f"logic_prefix_credit_decay={graph_logic_prefix_credit_decay} "
        f"logic_prefix_credit_local_alpha={graph_logic_prefix_credit_local_alpha} "
        f"logic_prefix_credit_negative={graph_logic_prefix_credit_negative} "
        f"logic_family_balance={graph_logic_family_balance} "
        f"logic_family_scales={graph_logic_family_scale_csv or 'default'} "
        f"logic_family_default={graph_logic_family_scale_default} "
        f"logic_final_answer_checks={graph_logic_final_answer_checks} "
        f"logic_schema_conflict_gate={graph_logic_schema_conflict_gate_positive},"
        f"penalty_scale={graph_logic_schema_conflict_penalty_scale} "
        f"logic_legacy_core4_compat={graph_logic_legacy_core4_compat} "
        f"first_only={first_occurrence_only} "
        f"extract_backend={extract_backend} extract_workers={extract_workers or 'auto'}"
    )
    return GraphProgressRewardComputer(
        tokenizer=tokenizer,
        graph_reward_manager=graph_reward_manager,
        max_entities=max_entities,
        reward_clip=reward_clip,
        positive_reward=positive_reward,
        focus_reward=focus_reward,
        irrelevant_node_reward=irrelevant_node_reward,
        irrelevant_edge_reward=irrelevant_edge_reward,
        invalid_edge_reward=invalid_edge_reward,
        order_hit_reward=order_hit_reward,
        order_seen_reward=order_seen_reward,
        order_skip_reward=order_skip_reward,
        coverage_mode=coverage_mode,
        prefix_decay=prefix_decay,
        scorer=scorer,
        relative_alpha=relative_alpha,
        relative_min_position_weight=relative_min_position_weight,
        relative_final_gate_eta=relative_final_gate_eta,
        graph_logic_ordered_reward=graph_logic_ordered_reward,
        graph_logic_progress_reward=graph_logic_progress_reward,
        graph_logic_answer_reward=graph_logic_answer_reward,
        graph_logic_legal_reward=graph_logic_legal_reward,
        graph_logic_invalid_penalty=graph_logic_invalid_penalty,
        graph_logic_fact_dump_penalty=graph_logic_fact_dump_penalty,
        graph_logic_final_gate_eta=graph_logic_final_gate_eta,
        graph_logic_correct_process_weight=graph_logic_correct_process_weight,
        graph_logic_wrong_process_weight=graph_logic_wrong_process_weight,
        graph_logic_correct_only_positive=graph_logic_correct_only_positive,
        graph_logic_keep_wrong_negative=graph_logic_keep_wrong_negative,
        graph_logic_unit_trust_filter=graph_logic_unit_trust_filter,
        graph_logic_unit_trust_margin=graph_logic_unit_trust_margin,
        graph_logic_unit_trust_min_task_samples=graph_logic_unit_trust_min_task_samples,
        graph_logic_unit_trust_min_correct_hits=graph_logic_unit_trust_min_correct_hits,
        graph_logic_unit_trust_low_sample_keep=graph_logic_unit_trust_low_sample_keep,
        graph_logic_unit_trust_keep_unknown_positive=graph_logic_unit_trust_keep_unknown_positive,
        graph_logic_process_task_blocklist=graph_logic_process_task_blocklist,
        graph_logic_good_prefix_preserve=graph_logic_good_prefix_preserve,
        graph_logic_good_prefix_wrong_weight=graph_logic_good_prefix_wrong_weight,
        graph_logic_good_prefix_wrong_cap=graph_logic_good_prefix_wrong_cap,
        graph_logic_token_mapping=graph_logic_token_mapping,
        graph_logic_segment_merge_strategy=graph_logic_segment_merge_strategy,
        graph_logic_segment_merge_target_count=graph_logic_segment_merge_target_count,
        graph_logic_segment_merge_max_chars=graph_logic_segment_merge_max_chars,
        graph_logic_segment_merge_min_info_score=graph_logic_segment_merge_min_info_score,
        graph_logic_reward_mode=graph_logic_reward_mode,
        graph_logic_potential_scale=graph_logic_potential_scale,
        graph_logic_state_shaping=graph_logic_state_shaping,
        graph_logic_state_reward_scale=graph_logic_state_reward_scale,
        graph_logic_state_tasks=graph_logic_state_tasks,
        graph_logic_state_calibrate_potential=graph_logic_state_calibrate_potential,
        graph_logic_state_invalid_penalty=graph_logic_state_invalid_penalty,
        graph_logic_state_mention_reward=graph_logic_state_mention_reward,
        graph_logic_state_consistent_reward=graph_logic_state_consistent_reward,
        graph_logic_duplicate_no_progress_penalty=graph_logic_duplicate_no_progress_penalty,
        graph_logic_state_context_max_age=graph_logic_state_context_max_age,
        graph_logic_positive_reward_cap=graph_logic_positive_reward_cap,
        graph_logic_positive_reward_budget_mode=graph_logic_positive_reward_budget_mode,
        graph_logic_step_span_mode=graph_logic_step_span_mode,
        graph_logic_action_span_max_per_segment=graph_logic_action_span_max_per_segment,
        graph_logic_action_span_context_chars=graph_logic_action_span_context_chars,
        graph_logic_state_candidate_grounding=graph_logic_state_candidate_grounding,
        graph_logic_state_candidate_max_edges=graph_logic_state_candidate_max_edges,
        graph_logic_state_commitment_weight=graph_logic_state_commitment_weight,
        graph_logic_state_commitment_min_multiplier=graph_logic_state_commitment_min_multiplier,
        graph_logic_state_commitment_max_multiplier=graph_logic_state_commitment_max_multiplier,
        graph_logic_executable_commitment_weight=graph_logic_executable_commitment_weight,
        graph_logic_ambiguity_gate_positive=graph_logic_ambiguity_gate_positive,
        graph_logic_ambiguity_gate_threshold=graph_logic_ambiguity_gate_threshold,
        graph_logic_ambiguity_gate_multiplier=graph_logic_ambiguity_gate_multiplier,
        graph_logic_certificate_only_hard_gate=graph_logic_certificate_only_hard_gate,
        graph_logic_certificate_gate_task_allowlist=graph_logic_certificate_gate_task_allowlist,
        graph_logic_certificate_gate_min_exec_score=graph_logic_certificate_gate_min_exec_score,
        graph_logic_certificate_gate_max_ambiguity=graph_logic_certificate_gate_max_ambiguity,
        graph_logic_graph_sensitivity_gate_positive=graph_logic_graph_sensitivity_gate_positive,
        graph_logic_graph_sensitivity_gate_task_allowlist=graph_logic_graph_sensitivity_gate_task_allowlist,
        graph_logic_graph_sensitivity_gate_multiplier=graph_logic_graph_sensitivity_gate_multiplier,
        graph_logic_component_precision_gate=graph_logic_component_precision_gate,
        graph_logic_component_precision_filter_units=graph_logic_component_precision_filter_units,
        graph_logic_component_precision_task_allowlist=graph_logic_component_precision_task_allowlist,
        graph_logic_component_vague_potential_multiplier=graph_logic_component_vague_potential_multiplier,
        graph_logic_component_noisy_positive_multiplier=graph_logic_component_noisy_positive_multiplier,
        graph_logic_component_schema_parseable_multiplier=graph_logic_component_schema_parseable_multiplier,
        graph_logic_oracle_milestones=graph_logic_oracle_milestones,
        graph_logic_oracle_milestone_max_units=graph_logic_oracle_milestone_max_units,
        graph_logic_oracle_milestone_positive_only=graph_logic_oracle_milestone_positive_only,
        graph_logic_oracle_milestone_count_actions=graph_logic_oracle_milestone_count_actions,
        graph_logic_oracle_milestone_task_allowlist=graph_logic_oracle_milestone_task_allowlist,
        graph_logic_action_router_enable=graph_logic_action_router_enable,
        graph_logic_action_router_mode=graph_logic_action_router_mode,
        graph_logic_action_router_task_allowlist=graph_logic_action_router_task_allowlist,
        graph_logic_action_router_conf_threshold=graph_logic_action_router_conf_threshold,
        graph_logic_action_router_high_conf_threshold=graph_logic_action_router_high_conf_threshold,
        graph_logic_action_router_positive_cap=graph_logic_action_router_positive_cap,
        graph_logic_action_router_negative=graph_logic_action_router_negative,
        graph_logic_action_router_max_candidates=graph_logic_action_router_max_candidates,
        graph_logic_action_router_backend=graph_logic_action_router_backend,
        graph_logic_action_router_base_url=graph_logic_action_router_base_url,
        graph_logic_action_router_model=graph_logic_action_router_model,
        graph_logic_action_router_timeout=graph_logic_action_router_timeout,
        graph_logic_action_router_max_tokens=graph_logic_action_router_max_tokens,
        graph_logic_action_router_heuristic_fallback=graph_logic_action_router_heuristic_fallback,
        graph_logic_action_router_heuristic_fallback_threshold=graph_logic_action_router_heuristic_fallback_threshold,
        graph_logic_action_router_trace=graph_logic_action_router_trace,
        graph_logic_action_router_payload_trace=graph_logic_action_router_payload_trace,
        graph_logic_prefix_credit=graph_logic_prefix_credit,
        graph_logic_prefix_credit_window=graph_logic_prefix_credit_window,
        graph_logic_prefix_credit_decay=graph_logic_prefix_credit_decay,
        graph_logic_prefix_credit_local_alpha=graph_logic_prefix_credit_local_alpha,
        graph_logic_prefix_credit_negative=graph_logic_prefix_credit_negative,
        graph_logic_family_balance=graph_logic_family_balance,
        graph_logic_family_scale_csv=graph_logic_family_scale_csv,
        graph_logic_family_scale_default=graph_logic_family_scale_default,
        graph_logic_final_answer_checks=graph_logic_final_answer_checks,
        graph_logic_schema_conflict_gate_positive=graph_logic_schema_conflict_gate_positive,
        graph_logic_schema_conflict_penalty_scale=graph_logic_schema_conflict_penalty_scale,
        graph_logic_legacy_core4_compat=graph_logic_legacy_core4_compat,
        first_occurrence_only=first_occurrence_only,
        extract_backend=extract_backend,
        extract_workers=extract_workers if extract_workers > 0 else None,
        print_timing=print_timing,
    )


def _build_step_reward_computer(config, tokenizer, graph_reward_manager):
    """Dispatch on `reward_kwargs.step_reward_method`. Returns a computer
    object exposing `.compute(data, ref_policy_wg=None) -> dict` and
    `.last_stats: dict`, or None if no step reward is requested.
    """
    method = _resolve_step_reward_method(config)
    if method == "none":
        return None, "none"
    if method == "direction3":
        return _build_progress_reward_computer(config, tokenizer, graph_reward_manager), "direction3"
    if method == "kl":
        return _build_kl_step_reward_computer(config, tokenizer, graph_reward_manager), "kl"
    if method == "consistency":
        return _build_consistency_step_reward_computer(config, tokenizer, graph_reward_manager), "consistency"
    if method == "graph_progress":
        return _build_graph_progress_reward_computer(config, tokenizer, graph_reward_manager), "graph_progress"
    if method == "entity_info_only":
        return _build_entity_info_only_computer(config, tokenizer, graph_reward_manager), "entity_info_only"
    raise ValueError(f"Unknown step_reward_method: {method}")


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if _need_reference_policy_for_config(config):
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            # elif use_legacy_worker_impl == "disable":
            #     from verl.workers.engine_workers import RewardModelWorker
            #
            #     print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker group.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if _need_reference_policy_for_config(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Direction 3: optional Progress-reward computer (see
        # verl/workers/reward_manager/progress_reward_fn.py). When enabled via
        # +reward_model.reward_kwargs.direction3_enabled=true it replaces the
        # hazard_rnn / activation step-reward path and instead derives
        # pure_step_reward_tensor from the chosen step-reward method
        # (direction3 / kl / consistency / none — see _resolve_step_reward_method).
        progress_reward_computer, _step_reward_method = _build_step_reward_computer(
            config=config,
            tokenizer=tokenizer,
            graph_reward_manager=reward_fn,
        )
        print(f"[main_ppo] step_reward_method = {_step_reward_method}")

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            progress_reward_computer=progress_reward_computer,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
