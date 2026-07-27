# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_reward_extra_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.opsd_teacher_prefix import build_opsd_teacher_batch, get_opsd_teacher_prefix_builder
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in (AdvantageEstimator.GiGPO, AdvantageEstimator.GiGPO_NO_STEP_NORM):
        # GiGPO-family estimators require step_rewards, anchor_obs, index, and traj_index
        print(data.batch.keys())
        if "step_rewards" not in data.batch:
            raise ValueError("GiGPO requires step_rewards from reward manager. Make sure reward manager returns step_rewards in dict format.")
        if "anchor_obs" not in data.non_tensor_batch:
            raise ValueError("GiGPO requires anchor_obs from reward manager. Make sure reward manager returns anchor_obs in dict format.")
        if "traj_index" not in data.non_tensor_batch:
            raise ValueError("GiGPO requires traj_index to be generated. This should be done automatically before compute_advantage.")

        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "step_rewards": data.batch["step_rewards"],
            "response_mask": data.batch["response_mask"],
            "anchor_obs": data.non_tensor_batch["anchor_obs"],
            "index": data.non_tensor_batch["uid"],
            "traj_index": data.non_tensor_batch["traj_index"],
            "config": config,
        }
        if "step_token_positions" in data.non_tensor_batch:
            adv_kwargs["step_token_positions"] = data.non_tensor_batch["step_token_positions"]
        # Pass pure step reward tensor if available so the estimator can compute
        # outcome_only_rewards = token_level_rewards - pure_step_reward_tensor and
        # use it for the episode path (Y1 design: episode and step fully decoupled).
        if "pure_step_reward_tensor" in data.batch:
            adv_kwargs["pure_step_reward_tensor"] = data.batch["pure_step_reward_tensor"]

        # Optional gating for step rewards. By default this is disabled to avoid hard masking.
        gate_mode = "none"
        if config is not None:
            gate_mode = str(config.get("gigpo_step_reward_gate", "none")).lower()

        if gate_mode in ("none", "off", "false", "0"):
            pass
        elif gate_mode in ("positive_final", "final_positive", "gt0"):
            token_level_scores = data.batch.get("token_level_scores", data.batch["token_level_rewards"])
            response_mask = data.batch["response_mask"]
            valid_resp_len = response_mask.sum(dim=-1).long().clamp(min=1)
            row_idx = torch.arange(token_level_scores.shape[0], device=token_level_scores.device)
            last_token_idx = valid_resp_len - 1
            final_scores = token_level_scores[row_idx, last_token_idx]
            adv_kwargs["step_reward_valid_mask"] = final_scores > 0
        elif gate_mode in ("nonnegative_final", "final_nonnegative", "ge0"):
            token_level_scores = data.batch.get("token_level_scores", data.batch["token_level_rewards"])
            response_mask = data.batch["response_mask"]
            valid_resp_len = response_mask.sum(dim=-1).long().clamp(min=1)
            row_idx = torch.arange(token_level_scores.shape[0], device=token_level_scores.device)
            last_token_idx = valid_resp_len - 1
            final_scores = token_level_scores[row_idx, last_token_idx]
            adv_kwargs["step_reward_valid_mask"] = final_scores >= 0
        else:
            raise ValueError(
                f"Unsupported gigpo_step_reward_gate={gate_mode}. "
                "Expected one of: none, positive_final, nonnegative_final."
            )

        # Get GiGPO-specific parameters from config
        adv_kwargs["return_components"] = True
        if config is not None:
            adv_kwargs["epsilon"] = config.get("gigpo_epsilon", 1e-6)
            adv_kwargs["step_advantage_w"] = config.get("gigpo_step_weight", 1.0)
            adv_kwargs["mode"] = config.get("gigpo_mode", "mean_norm")
            if str(config.get("gigpo_advantage_scope", "token")).lower() == "entity" and (
                "step_token_positions" not in data.non_tensor_batch
            ):
                raise ValueError(
                    "GiGPO with gigpo_advantage_scope=entity requires step_token_positions in non_tensor_batch."
                )

        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_result = adv_estimator_fn(**adv_kwargs)
        if len(adv_result) == 4:
            advantages, returns, advantages_episode, advantages_step = adv_result
            data.batch["advantages_episode"] = advantages_episode
            data.batch["advantages_step"] = advantages_step
        else:
            advantages, returns = adv_result
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        progress_reward_computer=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        # Optional step-reward computer (direction3 / kl / consistency).
        # Each computer derives pure_step_reward_tensor from a different signal but
        # all sign their rewards, so gigpo_step_reward_gate must be 'none' (positive_final
        # would zero out negatives). Auto-override here.
        self.progress_reward_computer = progress_reward_computer
        if self.progress_reward_computer is not None:
            try:
                from omegaconf import OmegaConf

                OmegaConf.set_struct(config.algorithm, False)
            except Exception:
                pass
            prev_gate = None
            try:
                prev_gate = config.algorithm.get("gigpo_step_reward_gate", "none")
            except Exception:
                prev_gate = "none"
            try:
                config.algorithm.gigpo_step_reward_gate = "none"
                print(
                    f"[StepReward] overriding gigpo_step_reward_gate: {prev_gate} -> none "
                    "(signed step rewards conflict with positive_final gate)."
                )
            except Exception as cfg_err:
                print(
                    f"[StepReward] failed to override gigpo_step_reward_gate ({cfg_err}); "
                    "make sure the shell script sets gigpo_step_reward_gate=none."
                )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )
        self.policy_loss_mode = str(
            config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        ).lower()
        self.opsd_like_loss = self.policy_loss_mode in {"opsd", "opsd_future_kl"}
        self.opsd_skip_reward = (
            self.opsd_like_loss
            and bool(config.actor_rollout_ref.actor.policy_loss.get("opsd_skip_reward", False))
        )
        self.opsd_teacher_prefix_builder = None
        self._opsd_prefix_log_printed = False
        self._opsd_teacher_batch_log_printed = False
        if self.opsd_like_loss:
            if not bool(config.actor_rollout_ref.actor.policy_loss.get("opsd_use_ref_policy", True)):
                raise NotImplementedError(
                    f"loss_mode='{self.policy_loss_mode}' currently supports only opsd_use_ref_policy=True. "
                    "Custom teacher workers can be added after the prefix-batch rewrite is implemented."
                )
            if not self.use_reference_policy:
                raise ValueError(f"loss_mode='{self.policy_loss_mode}' requires a reference policy worker for teacher log-probs.")
            if self.policy_loss_mode == "opsd_future_kl" and self.opsd_skip_reward:
                raise ValueError(
                    "loss_mode='opsd_future_kl' requires reward-manager step_token_positions; "
                    "set opsd_skip_reward=false."
                )
            self.opsd_teacher_prefix_builder = get_opsd_teacher_prefix_builder(config.actor_rollout_ref.actor)
            print(
                "[OPSD] enabled: "
                f"loss_mode={self.policy_loss_mode}, "
                f"builder={self.opsd_teacher_prefix_builder.__class__.__module__}."
                f"{self.opsd_teacher_prefix_builder.__class__.__name__}, "
                f"use_ref_policy={bool(config.actor_rollout_ref.actor.policy_loss.get('opsd_use_ref_policy', True))}, "
                f"skip_reward={self.opsd_skip_reward}"
            )
            if self.opsd_skip_reward:
                print("[OPSD] pure mode enabled: reward manager and advantage computation will be skipped.")

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _extract_entity_positions_for_embeddings(self, batch: DataProto):
        """
        Extract entity positions from responses for memory-efficient embedding extraction.

        This method decodes responses and identifies entity token positions before
        embedding extraction, allowing us to only extract entity embeddings instead
        of all token embeddings (saving ~98% memory: 16GB -> 200MB).

        Args:
            batch: DataProto containing responses and prompts

        Returns:
            entity_positions_batch: List[List[int]], flattened token positions for hook-based embedding extraction
            entity_metadata_batch: List[List[Tuple]], metadata for each entity/span (type, id)
            entity_span_positions_batch: List[List[List[int]]], full token spans for each entity occurrence
            flat_entity_keys: flattened token-level mapping
        """
        # Check if reward_fn has entity extraction capability
        if self.reward_fn is None:
            return None, None, None, None

        if not hasattr(self.reward_fn, 'extract_entities') or not hasattr(self.reward_fn, 'find_entity_token_positions'):
            # Reward manager doesn't support entity extraction
            return None, None, None, None

        entity_positions_batch = []
        entity_metadata_batch = []
        entity_span_positions_batch = []
        flat_entity_keys = []
        occurrence_mode = getattr(self.reward_fn, "step_reward_occurrence", "all")
        if occurrence_mode not in ("all", "last"):
            occurrence_mode = "all"

        for i in range(len(batch)):
            data_item = batch[i]

            # Get prompt and response
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']

            # Get valid lengths
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # Decode response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Extract entities
            entities = self.reward_fn.extract_entities(response_str)

            # Find token positions
            entity_positions = self.reward_fn.find_entity_token_positions(
                response_str, valid_response_ids.tolist(), entities
            )

            # Collect all entity positions and metadata
            sample_positions = []
            sample_metadata = []
            sample_span_positions = []

            # Collect node entities
            for node_id, spans in entity_positions['node_positions'].items():
                normalized_spans = []
                for span in spans:
                    normalized_span = tuple(
                        sorted({int(pos) for pos in span if int(pos) < valid_response_length})
                    )
                    if normalized_span:
                        normalized_spans.append(normalized_span)
                if occurrence_mode == "last" and normalized_spans:
                    normalized_spans = [normalized_spans[-1]]
                for span in normalized_spans:
                    sample_metadata.append(('node', node_id))
                    sample_span_positions.append(list(span))
                    sample_positions.extend(span)
                    for pos in span:
                        flat_entity_keys.append((i, 'node', node_id, span, pos))

            # Collect edge entities
            for edge, spans in entity_positions['edge_positions'].items():
                normalized_spans = []
                for span in spans:
                    normalized_span = tuple(
                        sorted({int(pos) for pos in span if int(pos) < valid_response_length})
                    )
                    if normalized_span:
                        normalized_spans.append(normalized_span)
                if occurrence_mode == "last" and normalized_spans:
                    normalized_spans = [normalized_spans[-1]]
                for span in normalized_spans:
                    sample_metadata.append(('edge', edge))
                    sample_span_positions.append(list(span))
                    sample_positions.extend(span)
                    for pos in span:
                        flat_entity_keys.append((i, 'edge', edge, span, pos))

            entity_positions_batch.append(sample_positions)
            entity_metadata_batch.append(sample_metadata)
            entity_span_positions_batch.append(sample_span_positions)

        return entity_positions_batch, entity_metadata_batch, entity_span_positions_batch, flat_entity_keys

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return_dict_result = {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
                # Also check for step_rewards and anchor_obs in batch for GiGPO
                if "step_rewards" in batch.batch:
                    return_dict_result["step_rewards"] = batch.batch["step_rewards"]
                if "anchor_obs" in batch.non_tensor_batch:
                    return_dict_result["anchor_obs"] = batch.non_tensor_batch["anchor_obs"]
                if "step_token_positions" in batch.non_tensor_batch:
                    return_dict_result["step_token_positions"] = batch.non_tensor_batch["step_token_positions"]
                return return_dict_result
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            # Preserve step_rewards and anchor_obs for GiGPO
            return_dict_result = {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            if "step_rewards" in result:
                return_dict_result["step_rewards"] = result["step_rewards"]
            if "anchor_obs" in result:
                return_dict_result["anchor_obs"] = result["anchor_obs"]
            if "step_token_positions" in result:
                return_dict_result["step_token_positions"] = result["step_token_positions"]
            return return_dict_result
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_train_accuracy_metrics(self, batch: DataProto) -> dict[str, float]:
        """Compute train-time terminal correctness metrics from token-level scores.

        For graph-style rewards, the final correctness reward is assigned to the last
        valid response token. This helper extracts that token score per sample and
        reports mean accuracy both globally and (when available) per data source.
        """
        metric_dict: dict[str, float] = {}

        token_level_scores = batch.batch.get("token_level_scores", None)
        response_mask = batch.batch.get("response_mask", None)
        if token_level_scores is None or response_mask is None:
            return metric_dict

        try:
            valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
            valid_row_mask = valid_lengths > 0
            if not torch.any(valid_row_mask):
                return metric_dict

            row_indices = torch.nonzero(valid_row_mask, as_tuple=False).squeeze(-1)
            col_indices = valid_lengths[valid_row_mask] - 1
            terminal_scores = token_level_scores[row_indices, col_indices].detach().float().cpu().numpy()
            metric_dict["train-core/acc/mean@1"] = float(np.mean(terminal_scores))

            data_sources = batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                return metric_dict

            if isinstance(data_sources, str):
                metric_dict[f"train-core/{data_sources}/acc/mean@1"] = float(np.mean(terminal_scores))
                return metric_dict

            data_sources_arr = np.asarray(data_sources, dtype=object)
            valid_row_mask_np = valid_row_mask.detach().cpu().numpy()
            if data_sources_arr.ndim == 0:
                metric_dict[f"train-core/{str(data_sources_arr.item())}/acc/mean@1"] = float(np.mean(terminal_scores))
                return metric_dict
            if data_sources_arr.shape[0] != valid_row_mask_np.shape[0]:
                return metric_dict

            valid_data_sources = data_sources_arr[valid_row_mask_np]
            for data_source in np.unique(valid_data_sources):
                source_mask = valid_data_sources == data_source
                if np.any(source_mask):
                    metric_dict[f"train-core/{data_source}/acc/mean@1"] = float(np.mean(terminal_scores[source_mask]))
        except Exception as e:
            # Metrics should never break training.
            print(f"[TRAIN_ACC] failed to compute training accuracy metrics: {type(e).__name__}: {e}")

        return metric_dict

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        val_max_batches = int(self.config.trainer.get("val_max_batches", 0) or 0)
        for val_batch_idx, test_data in enumerate(self.val_dataloader):
            if val_max_batches > 0 and val_batch_idx >= val_max_batches:
                break
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if not data_source_lst:
            return {}

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _prepare_opsd_teacher_prefixes(self, batch: DataProto) -> dict[str, float]:
        """Build OPSD teacher prefixes and attach them to the batch metadata."""

        if self.opsd_teacher_prefix_builder is None:
            return {}
        prefixes = self.opsd_teacher_prefix_builder.build_batch(batch)
        batch_size = int(batch.batch["responses"].shape[0])
        if len(prefixes) != batch_size:
            raise ValueError(
                f"OPSD teacher prefix builder returned {len(prefixes)} prefixes for batch size {batch_size}."
            )
        prefixes = [str(prefix or "") for prefix in prefixes]
        batch.non_tensor_batch["opsd_teacher_prefixes"] = np.array(prefixes, dtype=object)
        non_empty = sum(1 for prefix in prefixes if prefix)
        if not self._opsd_prefix_log_printed:
            preview = next((prefix[:180].replace("\n", " ") for prefix in prefixes if prefix), "")
            print(
                "[OPSD] teacher prefixes built: "
                f"batch_size={batch_size}, non_empty={non_empty}, "
                f"non_empty_frac={non_empty / max(1, batch_size):.4f}, "
                f"preview={preview!r}"
            )
            self._opsd_prefix_log_printed = True
        return {
            "opsd/teacher_prefix_nonempty_frac": float(non_empty / max(1, batch_size)),
            "opsd/teacher_prefix_builder_enabled": 1.0,
        }

    def _build_opsd_teacher_ref_batch(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Build the prefix-aware reference batch used by OPSD teacher scoring."""

        if not self.opsd_like_loss:
            return batch, {}
        prefixes = batch.non_tensor_batch.get("opsd_teacher_prefixes", None)
        if prefixes is None:
            return batch, {}
        prefix_list = [str(prefix or "") for prefix in prefixes.tolist()]
        if not any(prefix_list):
            return batch, {
                "opsd/teacher_prefix_nonempty_frac": 0.0,
                "opsd/teacher_prefix_builder_enabled": 1.0,
            }
        teacher_batch, metrics = build_opsd_teacher_batch(batch=batch, tokenizer=self.tokenizer, prefixes=prefix_list)
        if not self._opsd_teacher_batch_log_printed:
            print(
                "[OPSD] prefix-aware teacher batch ready: "
                f"input_shape={tuple(batch.batch['input_ids'].shape)}, "
                f"teacher_input_shape={tuple(teacher_batch.batch['input_ids'].shape)}, "
                f"response_shape={tuple(batch.batch['responses'].shape)}, "
                f"prefix_token_mean={metrics.get('opsd/teacher_prefix_token_mean', 0.0):.2f}, "
                f"truncated_frac={metrics.get('opsd/teacher_prefix_truncated_frac', 0.0):.4f}"
            )
            self._opsd_teacher_batch_log_printed = True
        return teacher_batch, metrics

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            # Disable embedding extraction for old_log_prob computation to avoid batch mismatch
            batch.meta_info["extract_embeddings"] = False
            batch.meta_info.pop("embedding_layer", None)

            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        val_core_acc_key = "val-core/PKU-ML/Erdos/acc/mean@1"
        train_core_acc_key = "train-core/PKU-ML/Erdos/acc/mean@1"
        train_core_acc_fallback_key = "train-core/acc/mean@1"
        latest_val_core_acc = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        # Extract embeddings and compute rewards
                        # Two modes: streaming (memory-efficient) or batch (original)
                        use_streaming_reward = getattr(self.config.actor_rollout_ref.rollout, 'streaming_reward_computation', False)

                        if hasattr(self.config.actor_rollout_ref.rollout, 'extract_embeddings') and \
                           self.config.actor_rollout_ref.rollout.extract_embeddings:

                            if use_streaming_reward and self.reward_fn is not None:
                                # Streaming mode: compute rewards immediately without storing embeddings
                                with marked_timer("extract_embeddings_and_rewards", timing_raw, color="cyan"):
                                    use_full_response_embeddings = False
                                    if hasattr(self.reward_fn, "requires_full_response_embeddings"):
                                        use_full_response_embeddings = self.reward_fn.requires_full_response_embeddings()

                                    if use_full_response_embeddings:
                                        entity_positions_batch = None
                                        entity_metadata_batch = None
                                        entity_span_positions_batch = None
                                        flat_entity_keys = None
                                    else:
                                        # Pre-extract entity positions to reduce embedding memory in streaming path
                                        entity_positions_batch, entity_metadata_batch, entity_span_positions_batch, flat_entity_keys = (
                                            self._extract_entity_positions_for_embeddings(gen_batch_output)
                                        )

                                    # Prepare batch for streaming computation
                                    extract_batch = gen_batch_output.select(
                                        batch_keys=["responses", "input_ids", "attention_mask", "position_ids"]
                                    )

                                    # Get reward_model keys from the original batch and repeat them
                                    # The original batch contains reward_model keys that were kept during _get_gen_batch
                                    reward_model_keys = {"data_source", "reward_model", "extra_info", "uid"}

                                    # Repeat the original batch's non_tensor_batch to match gen_batch_output size
                                    repeated_batch_non_tensor = {}
                                    for key in reward_model_keys:
                                        if key in batch.non_tensor_batch:
                                            # Repeat each value n times (interleaved)
                                            original_values = batch.non_tensor_batch[key]
                                            repeated_values = []
                                            n = self.config.actor_rollout_ref.rollout.n
                                            for val in original_values:
                                                repeated_values.extend([val] * n)
                                            repeated_batch_non_tensor[key] = np.array(repeated_values, dtype=object)

                                    # Attach entity metadata/positions for memory-optimized embedding extraction
                                    if entity_positions_batch is not None:
                                        repeated_batch_non_tensor["entity_positions"] = np.array(
                                            entity_positions_batch, dtype=object
                                        )
                                        if entity_metadata_batch is not None:
                                            repeated_batch_non_tensor["entity_metadata"] = np.array(
                                                entity_metadata_batch, dtype=object
                                            )
                                        if entity_span_positions_batch is not None:
                                            repeated_batch_non_tensor["entity_span_positions"] = np.array(
                                                entity_span_positions_batch, dtype=object
                                            )

                                    extract_batch.non_tensor_batch = repeated_batch_non_tensor

                                    if use_full_response_embeddings:
                                        extract_batch.non_tensor_batch.pop("entity_positions", None)
                                        extract_batch.non_tensor_batch.pop("entity_metadata", None)
                                        extract_batch.non_tensor_batch.pop("entity_span_positions", None)
                                        extract_batch.meta_info.pop("entity_positions", None)
                                        extract_batch.meta_info.pop("flat_entity_keys", None)
                                        extract_batch.meta_info["force_full_response_embeddings"] = True
                                    else:
                                        extract_batch.meta_info["force_full_response_embeddings"] = False

                                    extract_batch.meta_info["extract_embeddings"] = True
                                    extract_batch.meta_info["embedding_layer"] = getattr(self.config.actor_rollout_ref.rollout, 'embedding_layer', 30)
                                    extract_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                    extract_batch.meta_info["micro_batch_size"] = gen_batch_output.meta_info.get("micro_batch_size", 8)
                                    extract_batch.meta_info["use_dynamic_bsz"] = False
                                    extract_batch.meta_info["force_full_response_embeddings"] = use_full_response_embeddings
                                    if entity_positions_batch is not None and not use_full_response_embeddings:
                                        extract_batch.meta_info["entity_positions"] = entity_positions_batch
                                        if flat_entity_keys is not None:
                                            extract_batch.meta_info["flat_entity_keys"] = flat_entity_keys

                                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                        if use_full_response_embeddings:
                                            print("[Streaming Reward] Sequence reward model detected: forcing full response embeddings")

                                    # Streaming computation: embeddings are computed and immediately used for rewards
                                    result = self.actor_rollout_wg.compute_log_prob_with_rewards(
                                        self.reward_fn, extract_batch, self.tokenizer
                                    )

                                    # Extract reward_tensor from DataProto result
                                    reward_tensor = result.batch["streaming_reward_tensor"]

                                    # Store rewards directly (no embeddings stored)
                                    gen_batch_output.batch["streaming_reward_tensor"] = reward_tensor

                                    # Extract and store step_rewards and anchor_obs for GiGPO if available
                                    if "step_rewards" in result.batch:
                                        gen_batch_output.batch["step_rewards"] = result.batch["step_rewards"]
                                    if "anchor_obs" in result.non_tensor_batch:
                                        gen_batch_output.non_tensor_batch["anchor_obs"] = result.non_tensor_batch["anchor_obs"]
                                    if "step_token_positions" in result.non_tensor_batch:
                                        gen_batch_output.non_tensor_batch["step_token_positions"] = result.non_tensor_batch[
                                            "step_token_positions"
                                        ]
                                    # Store pure step reward tensor for outcome-only episode advantage
                                    if "pure_step_reward_tensor" in result.batch:
                                        gen_batch_output.batch["pure_step_reward_tensor"] = result.batch[
                                            "pure_step_reward_tensor"
                                        ]
                                    reward_extra_keys = result.meta_info.get("reward_extra_keys", [])
                                    if reward_extra_keys:
                                        for key in reward_extra_keys:
                                            if key in result.non_tensor_batch:
                                                gen_batch_output.non_tensor_batch[key] = result.non_tensor_batch[key]
                                        gen_batch_output.meta_info["reward_extra_keys"] = reward_extra_keys

                                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                        log_interval = getattr(self.config, "streaming_reward_log_interval", 50)
                                        if log_interval and self.global_steps % log_interval == 0:
                                            reward_memory_mb = reward_tensor.numel() * 4 / (1024**2)
                                            print(f"[Streaming Reward] Computed rewards with shape: {reward_tensor.shape}, memory: {reward_memory_mb:.2f} MB")
                                            if "step_rewards" in result.batch:
                                                print(f"[Streaming Reward] Extracted step_rewards with shape: {result.batch['step_rewards'].shape}")
                                            if "anchor_obs" in result.non_tensor_batch:
                                                print(f"[Streaming Reward] Extracted anchor_obs with shape: {result.non_tensor_batch['anchor_obs'].shape}")
                                            if "step_token_positions" in result.non_tensor_batch:
                                                print(
                                                    "[Streaming Reward] Extracted step_token_positions with "
                                                    f"shape: {result.non_tensor_batch['step_token_positions'].shape}"
                                                )
                            else:
                                # Original batch mode: extract and store embeddings
                                with marked_timer("extract_embeddings", timing_raw, color="cyan"):
                                    # OPTIMIZATION: Pre-extract entity positions to reduce memory usage
                                    # This allows us to only extract entity embeddings (200MB) instead of all tokens (16GB)
                                    entity_positions_batch, entity_metadata_batch, entity_span_positions_batch, flat_entity_keys = self._extract_entity_positions_for_embeddings(gen_batch_output)

                                    # Prepare batch for embedding extraction
                                    extract_batch = gen_batch_output.select(
                                        batch_keys=["responses", "input_ids", "attention_mask", "position_ids"]
                                    )
                                    extract_batch.meta_info["extract_embeddings"] = True
                                    extract_batch.meta_info["embedding_layer"] = getattr(self.config.actor_rollout_ref.rollout, 'embedding_layer', 30)
                                    extract_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                    extract_batch.meta_info["micro_batch_size"] = gen_batch_output.meta_info.get("micro_batch_size", 8)
                                    extract_batch.meta_info["use_dynamic_bsz"] = False

                                    # Pass entity positions if available
                                    if entity_positions_batch is not None:
                                        extract_batch.meta_info["entity_positions"] = entity_positions_batch
                                        if flat_entity_keys is not None:
                                            extract_batch.meta_info["flat_entity_keys"] = flat_entity_keys
                                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                            total_entities = sum(len(positions) for positions in entity_positions_batch)
                                            print(f"[Memory Optimization] Pre-extracted {total_entities} entity positions across {len(entity_positions_batch)} samples")

                                    # Extract embeddings using compute_log_prob
                                    _, _, response_embeddings = self.actor_rollout_wg.compute_log_prob(extract_batch)

                                    # Add embeddings to gen_batch_output
                                    if response_embeddings is not None:
                                        gen_batch_output.batch["response_embeddings"] = response_embeddings
                                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                            if entity_positions_batch is not None:
                                                # Entity embeddings: (total_entities, hidden_size)
                                                memory_mb = response_embeddings.numel() * 4 / (1024**2)
                                                print(f"[Embedding Extraction] Extracted entity embeddings with shape: {response_embeddings.shape}, memory: {memory_mb:.2f} MB")
                                            else:
                                                # Full embeddings: (batch, response_len, hidden_size)
                                                memory_gb = response_embeddings.numel() * 4 / (1024**3)
                                                print(f"[Embedding Extraction] Extracted full embeddings with shape: {response_embeddings.shape}, memory: {memory_gb:.2f} GB")

                                        # Store entity metadata if available
                                        if entity_metadata_batch is not None:
                                            gen_batch_output.non_tensor_batch["entity_metadata"] = entity_metadata_batch
                                            gen_batch_output.non_tensor_batch["entity_positions"] = entity_positions_batch
                                            gen_batch_output.non_tensor_batch["entity_span_positions"] = entity_span_positions_batch

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            # Extract embeddings for REMAX baseline if configured
                            if hasattr(self.config.actor_rollout_ref.rollout, 'extract_embeddings') and \
                               self.config.actor_rollout_ref.rollout.extract_embeddings:
                                # OPTIMIZATION: Pre-extract entity positions for baseline
                                entity_positions_baseline, entity_metadata_baseline, entity_span_positions_baseline, flat_entity_keys_baseline = self._extract_entity_positions_for_embeddings(gen_baseline_output)

                                # Prepare batch for embedding extraction
                                extract_baseline_batch = gen_baseline_output.select(
                                    batch_keys=["responses", "input_ids", "attention_mask", "position_ids"]
                                )
                                extract_baseline_batch.meta_info["extract_embeddings"] = True
                                extract_baseline_batch.meta_info["embedding_layer"] = getattr(self.config.actor_rollout_ref.rollout, 'embedding_layer', 30)
                                extract_baseline_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                extract_baseline_batch.meta_info["micro_batch_size"] = gen_baseline_output.meta_info.get("micro_batch_size", 8)
                                extract_baseline_batch.meta_info["use_dynamic_bsz"] = False

                                # Pass entity positions if available
                                if entity_positions_baseline is not None:
                                    extract_baseline_batch.meta_info["entity_positions"] = entity_positions_baseline
                                    if flat_entity_keys_baseline is not None:
                                        extract_baseline_batch.meta_info["flat_entity_keys"] = flat_entity_keys_baseline

                                # Extract embeddings using compute_log_prob
                                _, _, baseline_embeddings = self.actor_rollout_wg.compute_log_prob(extract_baseline_batch)

                                # Add embeddings to gen_baseline_output
                                if baseline_embeddings is not None:
                                    gen_baseline_output.batch["response_embeddings"] = baseline_embeddings
                                    if entity_metadata_baseline is not None:
                                        gen_baseline_output.non_tensor_batch["entity_metadata"] = entity_metadata_baseline
                                        gen_baseline_output.non_tensor_batch["entity_positions"] = entity_positions_baseline
                                        gen_baseline_output.non_tensor_batch["entity_span_positions"] = entity_span_positions_baseline
                                        if flat_entity_keys_baseline is not None:
                                            # Keep flat_entity_keys in meta_info only to avoid non_tensor_batch size mismatch.
                                            gen_baseline_output.meta_info["flat_entity_keys"] = flat_entity_keys_baseline

                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    reward_kwargs_cfg = self.config.reward_model.get("reward_kwargs", {})
                    segment_merge_strategy = str(
                        reward_kwargs_cfg.get("graph_logic_segment_merge_strategy", "none") or "none"
                    ).lower()
                    needs_model_entropy_merge = segment_merge_strategy in {
                        "model_entropy",
                        "token_entropy",
                        "logit_entropy",
                        "logits_entropy",
                    }
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = (
                        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    )
                    precomputed_old_log_prob = None
                    precomputed_old_log_prob_mfu = 0
                    if needs_model_entropy_merge:
                        if bypass_recomputing_logprobs:
                            raise RuntimeError(
                                "graph_logic_segment_merge_strategy=model_entropy requires recomputed actor "
                                "entropys, but rollout_correction.bypass_mode is enabled."
                            )
                        with marked_timer("old_log_prob_pre_reward", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics.update(
                                {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "actor/entropy_for_segment_merge": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                            )
                            batch.batch["response_token_entropys"] = entropys.float()
                            old_log_prob.batch.pop("entropys")
                            precomputed_old_log_prob = old_log_prob
                            precomputed_old_log_prob_mfu = old_log_prob_mfu

                    reward_extra_infos_dict = {}
                    future_reward = None
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.opsd_skip_reward:
                            zero = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            batch.batch["token_level_scores"] = zero
                            batch.batch["token_level_rewards"] = zero
                            batch.batch["advantages"] = zero
                            batch.batch["returns"] = zero
                            metrics["opsd/skip_reward"] = 1.0
                            metrics["opsd/pure_mode"] = 1.0
                        else:
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # Compute or extract reward for training
                            # Check if streaming rewards were already computed
                            use_streaming_reward = getattr(self.config.actor_rollout_ref.rollout, 'streaming_reward_computation', False)
                            if use_streaming_reward and "streaming_reward_tensor" in batch.batch:
                                # Use pre-computed streaming rewards
                                reward_tensor = batch.batch["streaming_reward_tensor"]
                                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                                reward_extra_infos_dict = (
                                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys if key in batch.non_tensor_batch}
                                    if reward_extra_keys
                                    else {}
                                )
                                step_rewards = batch.batch.get("step_rewards", None)
                                anchor_obs = batch.non_tensor_batch.get("anchor_obs", None)
                                step_token_positions = batch.non_tensor_batch.get("step_token_positions", None)
                                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                    print(f"[Streaming Reward] Using pre-computed rewards from streaming computation")
                                    if step_rewards is not None:
                                        print(f"[Streaming Reward] Found step_rewards with shape: {step_rewards.shape}")
                                    if anchor_obs is not None:
                                        print(f"[Streaming Reward] Found anchor_obs with shape: {anchor_obs.shape}")
                                    if step_token_positions is not None:
                                        print(
                                            "[Streaming Reward] Found step_token_positions with "
                                            f"shape: {step_token_positions.shape}"
                                        )

                                # Store reward tensor and extra info immediately for streaming path
                                batch.batch["token_level_scores"] = reward_tensor

                                # Store step_rewards and anchor_obs if available (already in batch from rollout)
                                # No need to store again, they're already there
                            elif self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_result = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, return_dict=True
                                )
                                # Handle both old format (tuple) and new format (dict)
                                if isinstance(reward_result, tuple):
                                    reward_tensor, reward_extra_infos_dict = reward_result
                                    step_rewards = None
                                    anchor_obs = None
                                    step_token_positions = None
                                    pure_step_reward_tensor = None
                                elif isinstance(reward_result, dict):
                                    reward_tensor = reward_result.get("reward_tensor")
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                    step_rewards = reward_result.get("step_rewards", None)
                                    anchor_obs = reward_result.get("anchor_obs", None)
                                    step_token_positions = reward_result.get("step_token_positions", None)
                                    pure_step_reward_tensor = reward_result.get("pure_step_reward_tensor", None)
                                else:
                                    # Fallback for old format
                                    reward_tensor = reward_result
                                    reward_extra_infos_dict = {}
                                    step_rewards = None
                                    anchor_obs = None
                                    step_token_positions = None
                                    pure_step_reward_tensor = None

                                # Store reward tensor and extra info immediately for sync path
                                batch.batch["token_level_scores"] = reward_tensor

                                # Store step rewards and anchor observations if available
                                if step_rewards is not None:
                                    batch.batch["step_rewards"] = step_rewards
                                if anchor_obs is not None:
                                    batch.non_tensor_batch["anchor_obs"] = anchor_obs
                                if step_token_positions is not None:
                                    batch.non_tensor_batch["step_token_positions"] = step_token_positions
                                if pure_step_reward_tensor is not None:
                                    batch.batch["pure_step_reward_tensor"] = pure_step_reward_tensor

                                if reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            if precomputed_old_log_prob is not None:
                                old_log_prob = precomputed_old_log_prob
                                old_log_prob_mfu = precomputed_old_log_prob_mfu
                            else:
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if self.opsd_like_loss:
                                metrics.update(self._prepare_opsd_teacher_prefixes(batch))
                                ref_batch, ref_prefix_metrics = self._build_opsd_teacher_ref_batch(batch)
                                metrics.update(ref_prefix_metrics)
                            else:
                                ref_batch = batch
                            ref_log_prob = self._compute_ref_log_prob(ref_batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.opsd_skip_reward:
                            metrics["opsd/skip_advantage"] = 1.0
                            metrics["opsd/dummy_advantages"] = 1.0
                        else:
                            # we combine with rule-based rm
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_result = ray.get(future_reward)

                                # Handle both old format (tuple) and new format (dict)
                                if isinstance(reward_result, tuple):
                                    reward_tensor, reward_extra_infos_dict = reward_result
                                    step_rewards = None
                                    anchor_obs = None
                                    step_token_positions = None
                                    pure_step_reward_tensor = None
                                elif isinstance(reward_result, dict):
                                    reward_tensor = reward_result.get("reward_tensor")
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                    step_rewards = reward_result.get("step_rewards", None)
                                    anchor_obs = reward_result.get("anchor_obs", None)
                                    step_token_positions = reward_result.get("step_token_positions", None)
                                    pure_step_reward_tensor = reward_result.get("pure_step_reward_tensor", None)
                                else:
                                    # Fallback for old format
                                    reward_tensor = reward_result
                                    reward_extra_infos_dict = {}
                                    step_rewards = None
                                    anchor_obs = None
                                    step_token_positions = None
                                    pure_step_reward_tensor = None

                                # Store reward tensor and extra info immediately for async path
                                batch.batch["token_level_scores"] = reward_tensor

                                # Store step rewards and anchor observations if available
                                if step_rewards is not None:
                                    batch.batch["step_rewards"] = step_rewards
                                if anchor_obs is not None:
                                    batch.non_tensor_batch["anchor_obs"] = anchor_obs
                                if step_token_positions is not None:
                                    batch.non_tensor_batch["step_token_positions"] = step_token_positions
                                if pure_step_reward_tensor is not None:
                                    batch.batch["pure_step_reward_tensor"] = pure_step_reward_tensor

                                if reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if reward_extra_infos_dict:
                                metrics.update(compute_reward_extra_metrics(reward_extra_infos_dict))

                            # Free embeddings memory after reward computation
                            # Embeddings are only needed for reward calculation and can be safely removed now
                            if "response_embeddings" in batch.batch:
                                embeddings_shape = batch.batch["response_embeddings"].shape
                                embeddings_memory_mb = batch.batch["response_embeddings"].numel() * 4 / (1024**2)
                                batch.batch.pop("response_embeddings")
                                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                                    print(f"[Memory Optimization] Freed response_embeddings: shape={embeddings_shape}, memory={embeddings_memory_mb:.2f} MB")
                                del embeddings_shape, embeddings_memory_mb

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            # Generate traj_index for GiGPO
                            # traj_index distinguishes different rollouts of the same prompt
                            # For n=2: [0, 1, 0, 1, ...], for n=3: [0, 1, 2, 0, 1, 2, ...]
                            if "uid" in batch.non_tensor_batch:
                                n = self.config.actor_rollout_ref.rollout.n
                                batch_size = len(batch.non_tensor_batch["uid"])
                                num_prompts = batch_size // n
                                traj_index = np.tile(np.arange(n), num_prompts)
                                # Handle remainder if batch size is not divisible by n
                                remainder = batch_size % n
                                if remainder > 0:
                                    traj_index = np.concatenate([traj_index, np.arange(remainder)])
                                batch.non_tensor_batch["traj_index"] = traj_index

                            # Step reward (direction3 / kl / consistency): override the four
                            # GiGPO-consumed fields with method-specific signals. Each computer
                            # validates its own input requirements internally; ref_policy_wg is
                            # passed for direction3 (KL/consistency ignore it).
                            if self.progress_reward_computer is not None:
                                with marked_timer("step_reward", timing_raw, color="magenta"):
                                    ref_wg = (
                                        self.ref_policy_wg
                                        if self.use_reference_policy else None
                                    )
                                    step_out = self.progress_reward_computer.compute(
                                        batch,
                                        ref_policy_wg=ref_wg,
                                        outcome_scores=batch.batch.get("token_level_scores", None),
                                    )
                                    resp_device = batch.batch["responses"].device
                                    batch.batch["pure_step_reward_tensor"] = step_out[
                                        "pure_step_reward_tensor"
                                    ].to(resp_device)
                                    batch.batch["step_rewards"] = step_out["step_rewards"].to(resp_device)
                                    batch.non_tensor_batch["anchor_obs"] = step_out["anchor_obs"]
                                    batch.non_tensor_batch["step_token_positions"] = step_out[
                                        "step_token_positions"
                                    ]
                                    # Optional anchor tensors (used by entity_anchored_grouped_kl loss).
                                    # We store BOTH the tensor (in batch.batch, in case dispatch preserves it)
                                    # AND a numpy mirror (in non_tensor_batch) which reliably survives the
                                    # DataProto chunk + Ray dispatch into FSDP workers.
                                    for anchor_key in (
                                        "anchor_positions",
                                        "anchor_group_ids",
                                        "anchor_prompt_ids",
                                        "segment_id_per_token",
                                        "num_anchors",
                                    ):
                                        if anchor_key in step_out:
                                            batch.batch[anchor_key] = step_out[anchor_key].to(resp_device)
                                        np_key = f"{anchor_key}_np"
                                        if np_key in step_out:
                                            batch.non_tensor_batch[np_key] = step_out[np_key]
                                    step_stats = self.progress_reward_computer.last_stats or {}
                                    for k, v in step_stats.items():
                                        try:
                                            metrics[k] = float(v)
                                        except (TypeError, ValueError):
                                            pass

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(self._compute_train_accuracy_metrics(batch=batch))
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                if val_core_acc_key in metrics:
                    latest_val_core_acc = metrics[val_core_acc_key]
                if latest_val_core_acc is None:
                    print(f"[VAL_ACC] step={self.global_steps} {val_core_acc_key}=N/A")
                else:
                    try:
                        val_core_acc_to_print = float(latest_val_core_acc)
                    except (TypeError, ValueError):
                        val_core_acc_to_print = latest_val_core_acc
                    print(f"[VAL_ACC] step={self.global_steps} {val_core_acc_key}={val_core_acc_to_print}")

                train_core_acc_to_print = None
                train_core_acc_key_to_print = None
                if train_core_acc_key in metrics:
                    train_core_acc_to_print = metrics[train_core_acc_key]
                    train_core_acc_key_to_print = train_core_acc_key
                elif train_core_acc_fallback_key in metrics:
                    train_core_acc_to_print = metrics[train_core_acc_fallback_key]
                    train_core_acc_key_to_print = train_core_acc_fallback_key

                if train_core_acc_to_print is None:
                    print(f"[TRAIN_ACC] step={self.global_steps} {train_core_acc_key}=N/A")
                else:
                    try:
                        train_core_acc_to_print = float(train_core_acc_to_print)
                    except (TypeError, ValueError):
                        pass
                    print(
                        f"[TRAIN_ACC] step={self.global_steps} {train_core_acc_key_to_print}={train_core_acc_to_print}"
                    )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
