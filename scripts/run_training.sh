#!/usr/bin/env bash
set -euo pipefail

# Anonymous training entry for the paper's final GraphLogic configuration.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3-4B
#   TRAIN_FILE=/path/to/train.parquet
#   VAL_FILE=/path/to/validation.parquet
#
# Optional:
#   OUTPUT_DIR=./outputs/grasca_core4
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   PYTHON_BIN=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

: "${MODEL_PATH:?Set MODEL_PATH to the base model or Hugging Face model ID.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file.}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet file.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/grasca_core4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grasca_core4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RAY_INCLUDE_DASHBOARD=0
export RAY_DISABLE_DASHBOARD_LOG_INFO=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/grasca_ray}"

mkdir -p "${OUTPUT_DIR}" "${RAY_TMPDIR}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-43}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"

TRAIN_FILES="['${TRAIN_FILE}']"
VAL_FILES="['${VAL_FILE}']"
CORE4_TASKS="weighted_shortest_path,weighted_minimum_spanning_tree,betweenness_centrality,wiener_index"

echo "Launching anonymous GraphLogic training"
echo "  model=${MODEL_PATH}"
echo "  train=${TRAIN_FILE}"
echo "  validation=${VAL_FILE}"
echo "  output=${OUTPUT_DIR}"
echo "  gpus=${CUDA_VISIBLE_DEVICES}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo_no_step_norm \
    +algorithm.gigpo_step_weight=0.2 \
    +algorithm.gigpo_mode=mean_norm \
    +algorithm.gigpo_epsilon=1e-6 \
    +algorithm.gigpo_gamma=0.0 \
    +algorithm.gigpo_advantage_scope=entity \
    +algorithm.gigpo_episode_advantage_scope=token \
    +algorithm.gigpo_step_reward_gate=none \
    +algorithm.gigpo_step_length_norm=none \
    +algorithm.gigpo_step_length_ref=batch_median \
    +algorithm.gigpo_single_step_fallback_weight=0.5 \
    +algorithm.gigpo_step_scatter_pool=max_abs \
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.dataloader_num_workers=0 \
    data.prompt_key=prompt \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=observer_future_kl \
    actor_rollout_ref.actor.policy_loss.action_future_kl_decay_rate=32.0 \
    actor_rollout_ref.actor.policy_loss.action_future_kl_beta=1.0 \
    actor_rollout_ref.actor.policy_loss.action_future_kl_clip_low=0.88 \
    actor_rollout_ref.actor.policy_loss.action_future_kl_clip_high=1.18 \
    actor_rollout_ref.actor.policy_loss.action_future_kl_start=after_action \
    actor_rollout_ref.actor.policy_loss.action_future_kl_normalize=sum \
    actor_rollout_ref.actor.policy_loss.action_future_kl_direction=episode \
    actor_rollout_ref.actor.policy_loss.action_future_kl_lambda=0.20 \
    actor_rollout_ref.actor.policy_loss.action_future_kl_renormalize_per_response=true \
    actor_rollout_ref.actor.policy_loss.action_future_kl_reweight_episode=false \
    actor_rollout_ref.actor.policy_loss.action_future_kl_prefix_credit=false \
    actor_rollout_ref.actor.policy_loss.action_future_kl_credit_assignment=off \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=graph \
    +reward_model.reward_kwargs.step_reward_method=graph_progress \
    +reward_model.reward_kwargs.graph_progress_max_entities=64 \
    +reward_model.reward_kwargs.graph_progress_reward_clip=1.0 \
    +reward_model.reward_kwargs.graph_progress_positive_reward=1.0 \
    +reward_model.reward_kwargs.graph_progress_focus_reward=0.2 \
    +reward_model.reward_kwargs.graph_progress_irrelevant_node_reward=0.0 \
    +reward_model.reward_kwargs.graph_progress_irrelevant_edge_reward=-0.1 \
    +reward_model.reward_kwargs.graph_progress_invalid_edge_reward=-1.0 \
    +reward_model.reward_kwargs.graph_progress_order_hit_reward=1.0 \
    +reward_model.reward_kwargs.graph_progress_order_seen_reward=-0.2 \
    +reward_model.reward_kwargs.graph_progress_order_skip_reward=0.2 \
    +reward_model.reward_kwargs.graph_progress_coverage_mode=segment \
    +reward_model.reward_kwargs.graph_progress_prefix_decay=0.995 \
    +reward_model.reward_kwargs.graph_progress_scorer=graph_logic \
    +reward_model.reward_kwargs.graph_progress_relative_alpha=2.0 \
    +reward_model.reward_kwargs.graph_progress_relative_min_position_weight=0.2 \
    +reward_model.reward_kwargs.graph_progress_relative_final_gate_eta=0.2 \
    +reward_model.reward_kwargs.graph_progress_first_occurrence_only=true \
    +reward_model.reward_kwargs.graph_progress_extract_backend=process \
    +reward_model.reward_kwargs.graph_progress_extract_workers=64 \
    +reward_model.reward_kwargs.graph_progress_print_timing=true \
    +reward_model.reward_kwargs.graph_logic_ordered_reward=1.0 \
    +reward_model.reward_kwargs.graph_logic_progress_reward=0.25 \
    +reward_model.reward_kwargs.graph_logic_answer_reward=0.5 \
    +reward_model.reward_kwargs.graph_logic_legal_reward=0.0 \
    +reward_model.reward_kwargs.graph_logic_invalid_penalty=0.05 \
    +reward_model.reward_kwargs.graph_logic_fact_dump_penalty=0.005 \
    +reward_model.reward_kwargs.graph_logic_final_gate_eta=0.0 \
    +reward_model.reward_kwargs.graph_logic_correct_process_weight=1.0 \
    +reward_model.reward_kwargs.graph_logic_wrong_process_weight=0.0 \
    +reward_model.reward_kwargs.graph_logic_keep_wrong_negative=true \
    +reward_model.reward_kwargs.graph_logic_correct_only_positive=false \
    +reward_model.reward_kwargs.graph_logic_unit_trust_filter=false \
    +reward_model.reward_kwargs.graph_logic_process_task_blocklist="''" \
    +reward_model.reward_kwargs.graph_logic_good_prefix_preserve=false \
    +reward_model.reward_kwargs.graph_logic_token_mapping=exact \
    +reward_model.reward_kwargs.graph_logic_segment_merge_strategy=none \
    +reward_model.reward_kwargs.graph_logic_reward_mode=potential \
    +reward_model.reward_kwargs.graph_logic_potential_scale=1.15 \
    +reward_model.reward_kwargs.graph_logic_state_shaping=true \
    +reward_model.reward_kwargs.graph_logic_state_reward_scale=1.05 \
    +reward_model.reward_kwargs.graph_logic_state_tasks="'all'" \
    +reward_model.reward_kwargs.graph_logic_state_calibrate_potential=true \
    +reward_model.reward_kwargs.graph_logic_state_invalid_penalty=0.0005 \
    +reward_model.reward_kwargs.graph_logic_state_mention_reward=0.005 \
    +reward_model.reward_kwargs.graph_logic_state_consistent_reward=0.0 \
    +reward_model.reward_kwargs.graph_logic_duplicate_no_progress_penalty=0.002 \
    +reward_model.reward_kwargs.graph_logic_state_context_max_age=2 \
    +reward_model.reward_kwargs.graph_logic_state_candidate_grounding=true \
    +reward_model.reward_kwargs.graph_logic_state_candidate_max_edges=3 \
    +reward_model.reward_kwargs.graph_logic_state_commitment_weight=true \
    +reward_model.reward_kwargs.graph_logic_state_commitment_min_multiplier=0.75 \
    +reward_model.reward_kwargs.graph_logic_state_commitment_max_multiplier=1.20 \
    +reward_model.reward_kwargs.graph_logic_positive_reward_cap=0.30 \
    +reward_model.reward_kwargs.graph_logic_positive_reward_budget_mode=net_progress \
    +reward_model.reward_kwargs.graph_logic_step_span_mode=action \
    +reward_model.reward_kwargs.graph_logic_action_span_max_per_segment=8 \
    +reward_model.reward_kwargs.graph_logic_action_span_context_chars=96 \
    +reward_model.reward_kwargs.graph_logic_prefix_credit=true \
    +reward_model.reward_kwargs.graph_logic_prefix_credit_window=64 \
    +reward_model.reward_kwargs.graph_logic_prefix_credit_decay=0.96 \
    +reward_model.reward_kwargs.graph_logic_prefix_credit_local_alpha=0.85 \
    +reward_model.reward_kwargs.graph_logic_prefix_credit_negative=false \
    +reward_model.reward_kwargs.graph_logic_oracle_milestones=true \
    +reward_model.reward_kwargs.graph_logic_oracle_milestone_max_units=48 \
    +reward_model.reward_kwargs.graph_logic_oracle_milestone_positive_only=true \
    +reward_model.reward_kwargs.graph_logic_oracle_milestone_count_actions=false \
    +reward_model.reward_kwargs.graph_logic_oracle_milestone_task_allowlist="'${CORE4_TASKS}'" \
    +reward_model.reward_kwargs.graph_logic_executable_commitment_weight=false \
    +reward_model.reward_kwargs.graph_logic_ambiguity_gate_positive=false \
    +reward_model.reward_kwargs.graph_logic_certificate_only_hard_gate=false \
    +reward_model.reward_kwargs.graph_logic_graph_sensitivity_gate_positive=false \
    +reward_model.reward_kwargs.graph_logic_component_precision_gate=false \
    +reward_model.reward_kwargs.graph_logic_schema_conflict_gate_positive=false \
    +reward_model.reward_kwargs.graph_logic_legacy_core4_compat=true \
    +reward_model.reward_kwargs.graph_logic_action_router_enable=false \
    +reward_model.reward_kwargs.graph_logic_family_balance=false \
    +reward_model.reward_kwargs.graph_logic_final_answer_checks=false \
    +reward_model.reward_kwargs.entity_reward_mode=span \
    +reward_model.reward_kwargs.broadcast_final_reward_to_all_tokens=true \
    +reward_model.reward_kwargs.skip_activation_reward=true \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=anonymous_graph_reasoning \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.val_before_train=False \
    trainer.test_freq="${TEST_FREQ}" \
    +trainer.val_max_batches=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs=1 2>&1 | tee "${OUTPUT_DIR}/training.log"
