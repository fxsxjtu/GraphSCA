# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    GiGPO = "gigpo"
    GiGPO_NO_STEP_NORM = "gigpo_no_step_norm"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: ``dp_size``, ``batch_num_tokens``, and ``global_batch_size`` are only compatible with the new model engine
        for now, while the legacy model engines conduct the aggregation outside ``agg_loss``.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    # TODO: Consider the numerical stability?

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size. When appling manual aggregation,
            scaling up the ``loss`` by ``dp_size`` can cancel out FSDP averaging.
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    # NOTE: `masked_sum` is more robust than multiplying the `mask`.
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode.startswith("seq-mean"):
        # TODO: Correct and unify the denominator logic.
        if global_batch_size is not None:
            seq_denominator = global_batch_size * dp_size
        else:  # The default logic which is only correct when the batch sizes are even.
            local_bsz = loss_mat.shape[0]
            seq_denominator = local_bsz

        if loss_agg_mode.startswith("seq-mean-token-sum"):
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1)  # token-sum per sequence

            if loss_agg_mode == "seq-mean-token-sum":
                pass  # TODO: Add assertation.
            elif loss_agg_mode == "seq-mean-token-sum-norm":
                if loss_scale_factor is None:
                    loss_scale_factor = loss_mask.shape[-1]
                seq_losses = seq_losses / loss_scale_factor
            else:
                raise ValueError(f"Invalid {loss_agg_mode=}")
        elif loss_agg_mode == "seq-mean-token-mean":
            token_counts = torch.sum(loss_mask, dim=-1)  # per-sequence token count
            # token-mean per sequence
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1) / (token_counts + 1e-8)
        else:
            raise ValueError(f"Invalid {loss_agg_mode=}")
        loss = torch.sum(seq_losses) / seq_denominator  # seq-mean
    else:
        raise ValueError(f"Invalid {loss_agg_mode=}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("opsd")  # type: ignore[arg-type]
def compute_policy_loss_opsd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sampled-token OPSD loss on student rollout trajectories.

    OPSD scores the student-generated response tokens with a teacher forward and
    uses the sampled-token log-prob gap as a detached policy-gradient advantage:

        A_t = log pi_teacher(y_t | teacher_context) - log pi_student(y_t | student_context)
        L_t = -A_t * log pi_student(y_t | student_context)

    The current framework uses ``ref_log_prob`` as the teacher-token log-prob.
    A teacher-prefix builder is wired in the trainer as an extension point; the
    first migrated mode keeps that prefix empty.
    """

    del uid
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    if ref_log_prob is None:
        raise ValueError(
            "loss_mode='opsd' requires ref_log_prob. Enable the reference policy "
            "or keep the default OPSD reference-policy setup."
        )

    policy_cfg = config.policy_loss
    loss_type = str(policy_cfg.get("opsd_loss_type", "sampled_pg") or "sampled_pg").lower()
    if loss_type not in {"sampled_pg", "sampled-token-pg", "sampled_token_pg"}:
        raise ValueError(f"Unsupported opsd_loss_type={loss_type}. Expected sampled_pg.")

    teacher_log_prob = ref_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
    valid_mask = response_mask.to(dtype=log_prob.dtype)
    opsd_advantage = (teacher_log_prob - log_prob).detach()

    if bool(policy_cfg.get("opsd_center_advantage", False)):
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (opsd_advantage * valid_mask).sum(dim=-1, keepdim=True) / denom
        opsd_advantage = (opsd_advantage - mean) * valid_mask

    adv_clip = policy_cfg.get("opsd_adv_clip", 5.0)
    if adv_clip is not None and float(adv_clip) > 0:
        opsd_advantage = opsd_advantage.clamp(min=-float(adv_clip), max=float(adv_clip))

    opsd_coef = float(policy_cfg.get("opsd_coef", 1.0))
    loss_mat = -opsd_coef * opsd_advantage * log_prob
    if rollout_is_weights is not None:
        loss_mat = loss_mat * rollout_is_weights

    opsd_loss = agg_loss(
        loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    current_vs_old = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    teacher_gap = teacher_log_prob - log_prob
    metrics = {
        "actor/opsd_active": 1.0,
        "actor/opsd_loss": opsd_loss.detach().item(),
        "actor/opsd_adv_mean": verl_F.masked_mean(opsd_advantage, response_mask).detach().item(),
        "actor/opsd_adv_abs_mean": verl_F.masked_mean(opsd_advantage.abs(), response_mask).detach().item(),
        "actor/opsd_teacher_gap_mean": verl_F.masked_mean(teacher_gap, response_mask).detach().item(),
        "actor/opsd_teacher_gap_abs_mean": verl_F.masked_mean(teacher_gap.abs(), response_mask).detach().item(),
        "actor/opsd_student_old_kl": verl_F.masked_mean(-current_vs_old, response_mask).detach().item(),
        "actor/opsd_coef": opsd_coef,
    }
    return opsd_loss, metrics


def _mask_advantage_to_observer_action_spans(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    step_token_positions: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep token advantages only on observer action spans.

    This is used by ``loss_mode=opsd_future_kl``: OPSD produces a dense
    sampled-token teacher gap, but action Future-KL needs an action-local
    source signal to redistribute. Non-action tokens are zeroed before the
    existing observer Future-KL credit-assignment code is applied.
    """

    default_metrics = {
        "actor/opsd_fkl_action_source_enabled": 0.0,
        "actor/opsd_fkl_action_source_action_count": 0.0,
        "actor/opsd_fkl_action_source_token_coverage": 0.0,
        "actor/opsd_fkl_action_source_no_valid_span_frac": 0.0,
        "actor/opsd_fkl_action_source_mass_ratio": 0.0,
    }
    if step_token_positions is None:
        return torch.zeros_like(advantages), default_metrics

    batch_size, response_len = response_mask.shape
    try:
        stp_np = np.asarray(step_token_positions, dtype=object)
    except Exception:
        return torch.zeros_like(advantages), default_metrics
    if stp_np.ndim == 1:
        stp_np = stp_np.reshape(batch_size, -1) if stp_np.shape[0] == batch_size else stp_np.reshape(1, -1)
    if stp_np.shape[0] != batch_size:
        return torch.zeros_like(advantages), default_metrics

    mask = response_mask.to(dtype=torch.float32)
    span_mask = torch.zeros_like(mask)
    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).detach().cpu().numpy()
    max_steps = stp_np.shape[1] if stp_np.ndim >= 2 else 0
    action_count = 0
    no_valid_span = 0

    for b in range(batch_size):
        valid_len = int(valid_lengths[b])
        if valid_len <= 0:
            continue
        for s_idx in range(max_steps):
            pos_spec = stp_np[b, s_idx]
            if pos_spec is None:
                continue
            if isinstance(pos_spec, float) and np.isnan(pos_spec):
                continue
            span = _action_only_range_for_observer_future_kl(pos_spec, response_len, valid_len)
            if span is None:
                no_valid_span += 1
                continue
            start, end = span
            if start > end:
                no_valid_span += 1
                continue
            token_count = mask[b, start : end + 1].sum()
            if bool(token_count <= 0):
                no_valid_span += 1
                continue
            span_mask[b, start : end + 1] = torch.maximum(
                span_mask[b, start : end + 1],
                mask[b, start : end + 1],
            )
            action_count += 1

    action_advantages = advantages * span_mask.to(dtype=advantages.dtype)
    valid_token_count = mask.sum().clamp(min=1.0)
    raw_abs = (advantages.detach().to(torch.float32).abs() * mask).sum().clamp(min=1e-8)
    action_abs = (action_advantages.detach().to(torch.float32).abs() * mask).sum()
    metrics = dict(default_metrics)
    metrics["actor/opsd_fkl_action_source_enabled"] = 1.0
    metrics["actor/opsd_fkl_action_source_action_count"] = float(action_count)
    metrics["actor/opsd_fkl_action_source_token_coverage"] = float(
        ((span_mask * mask).sum() / valid_token_count).detach().item()
    )
    metrics["actor/opsd_fkl_action_source_no_valid_span_frac"] = float(
        no_valid_span / max(1, no_valid_span + action_count)
    )
    metrics["actor/opsd_fkl_action_source_mass_ratio"] = float((action_abs / raw_abs).detach().item())
    return action_advantages, metrics


@register_policy_loss("opsd_future_kl")  # type: ignore[arg-type]
def compute_policy_loss_opsd_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
    advantages_episode: torch.Tensor | None = None,
    advantages_step: torch.Tensor | None = None,
    step_token_positions: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """OPSD sampled-token loss with observer action Future-KL redistribution.

    OPSD first builds a detached sampled-token advantage:

        A_opsd = log pi_teacher(y_t) - log pi_student(y_t)

    Then, instead of applying that dense token advantage directly, this mode
    keeps the OPSD signal on observer action spans and reuses the existing
    action Future-KL credit assignment to move that signal to prefix/action
    tokens whose current-vs-rollout policy shift supports the action.
    """

    del uid, advantages, advantages_episode, advantages_step
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    if ref_log_prob is None:
        raise ValueError(
            "loss_mode='opsd_future_kl' requires ref_log_prob. Enable the reference policy "
            "and keep opsd_use_ref_policy=true."
        )

    policy_cfg = config.policy_loss
    loss_type = str(policy_cfg.get("opsd_loss_type", "sampled_pg") or "sampled_pg").lower()
    if loss_type not in {"sampled_pg", "sampled-token-pg", "sampled_token_pg"}:
        raise ValueError(f"Unsupported opsd_loss_type={loss_type}. Expected sampled_pg.")

    teacher_log_prob = ref_log_prob.to(device=log_prob.device, dtype=log_prob.dtype)
    valid_mask = response_mask.to(dtype=log_prob.dtype)
    opsd_advantage = (teacher_log_prob - log_prob).detach()

    if bool(policy_cfg.get("opsd_center_advantage", False)):
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (opsd_advantage * valid_mask).sum(dim=-1, keepdim=True) / denom
        opsd_advantage = (opsd_advantage - mean) * valid_mask

    adv_clip = policy_cfg.get("opsd_adv_clip", 5.0)
    if adv_clip is not None and float(adv_clip) > 0:
        opsd_advantage = opsd_advantage.clamp(min=-float(adv_clip), max=float(adv_clip))

    source_mode = str(policy_cfg.get("opsd_future_kl_source", "action_spans") or "action_spans").lower()
    source_metrics: dict[str, float]
    if source_mode in {"action_spans", "action", "span", "spans"}:
        source_advantage, source_metrics = _mask_advantage_to_observer_action_spans(
            advantages=opsd_advantage,
            response_mask=response_mask,
            step_token_positions=step_token_positions,
        )
    elif source_mode in {"all", "all_tokens", "dense"}:
        source_advantage = opsd_advantage
        source_metrics = {
            "actor/opsd_fkl_action_source_enabled": 0.0,
            "actor/opsd_fkl_action_source_action_count": 0.0,
            "actor/opsd_fkl_action_source_token_coverage": 1.0,
            "actor/opsd_fkl_action_source_no_valid_span_frac": 0.0,
            "actor/opsd_fkl_action_source_mass_ratio": 1.0,
        }
    else:
        raise ValueError(f"Unsupported opsd_future_kl_source={source_mode}. Expected action_spans or all_tokens.")

    credit_assignment_mode = str(policy_cfg.get("action_future_kl_credit_assignment", "transfer") or "transfer").lower()
    if credit_assignment_mode == "transfer":
        weighted_advantage, credit_metrics = _compute_observer_future_kl_credit_assignment(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            step_token_positions=step_token_positions,
            advantages_step=source_advantage,
            config=config,
        )
    elif credit_assignment_mode in {"off", "none", "false", "0"}:
        weighted_advantage = source_advantage
        credit_metrics = {
            "observer_fipo/credit_assignment_enabled": 0.0,
            "observer_fipo/credit_assignment_action_count": 0.0,
            "observer_fipo/credit_assignment_transferred_action_count": 0.0,
            "observer_fipo/credit_assignment_token_coverage": 0.0,
        }
    else:
        raise ValueError(
            f"Unsupported action_future_kl_credit_assignment={credit_assignment_mode}. "
            "Expected one of: off, transfer."
        )

    credit_scale = float(policy_cfg.get("opsd_future_kl_credit_scale", 1.0))
    if credit_scale != 1.0:
        weighted_advantage = weighted_advantage * credit_scale

    cap_ratio = float(policy_cfg.get("opsd_future_kl_credit_abs_cap_ratio", 1.0))
    cap_applied_frac = 0.0
    cap_scale_mean = 1.0
    if cap_ratio > 0:
        mask_f = response_mask.to(dtype=torch.float32)
        source_abs = (source_advantage.detach().to(torch.float32).abs() * mask_f).sum(dim=-1, keepdim=True)
        weighted_abs = (weighted_advantage.detach().to(torch.float32).abs() * mask_f).sum(dim=-1, keepdim=True)
        max_abs = source_abs * cap_ratio
        cap_scale = torch.where(
            (weighted_abs > max_abs) & (weighted_abs > 1e-8),
            max_abs / weighted_abs.clamp(min=1e-8),
            torch.ones_like(weighted_abs),
        )
        weighted_advantage = weighted_advantage * cap_scale.to(dtype=weighted_advantage.dtype)
        valid_response = (response_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=torch.float32)
        denom = valid_response.sum().clamp(min=1.0)
        cap_applied_frac = float((((cap_scale < 0.999999).to(torch.float32) * valid_response).sum() / denom).detach().item())
        cap_scale_mean = float(((cap_scale.to(torch.float32) * valid_response).sum() / denom).detach().item())

    opsd_coef = float(policy_cfg.get("opsd_coef", 1.0))
    loss_mat = -opsd_coef * weighted_advantage.detach().to(dtype=log_prob.dtype) * log_prob
    if rollout_is_weights is not None:
        loss_mat = loss_mat * rollout_is_weights

    opsd_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    current_vs_old = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    teacher_gap = teacher_log_prob - log_prob
    metrics: dict[str, Any] = {
        "actor/opsd_future_kl_active": 1.0,
        "actor/opsd_active": 1.0,
        "actor/opsd_loss": opsd_loss.detach().item(),
        "actor/opsd_adv_mean": verl_F.masked_mean(opsd_advantage, response_mask).detach().item(),
        "actor/opsd_adv_abs_mean": verl_F.masked_mean(opsd_advantage.abs(), response_mask).detach().item(),
        "actor/opsd_fkl_source_adv_mean": verl_F.masked_mean(source_advantage, response_mask).detach().item(),
        "actor/opsd_fkl_source_adv_abs_mean": verl_F.masked_mean(source_advantage.abs(), response_mask).detach().item(),
        "actor/opsd_fkl_weighted_adv_mean": verl_F.masked_mean(weighted_advantage, response_mask).detach().item(),
        "actor/opsd_fkl_weighted_adv_abs_mean": verl_F.masked_mean(
            weighted_advantage.abs(), response_mask
        ).detach().item(),
        "actor/opsd_fkl_credit_scale": credit_scale,
        "actor/opsd_fkl_credit_abs_cap_ratio": cap_ratio,
        "actor/opsd_fkl_credit_abs_cap_applied_frac": cap_applied_frac,
        "actor/opsd_fkl_credit_abs_cap_scale_mean": cap_scale_mean,
        "actor/opsd_teacher_gap_mean": verl_F.masked_mean(teacher_gap, response_mask).detach().item(),
        "actor/opsd_teacher_gap_abs_mean": verl_F.masked_mean(teacher_gap.abs(), response_mask).detach().item(),
        "actor/opsd_student_old_kl": verl_F.masked_mean(-current_vs_old, response_mask).detach().item(),
        "actor/opsd_coef": opsd_coef,
    }
    metrics.update(source_metrics)
    metrics.update({f"actor/opsd_fkl_{k}": v for k, v in credit_metrics.items()})
    return opsd_loss, metrics


@register_policy_loss("future_kl")  # type: ignore[arg-type]
def compute_policy_loss_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """FIPO-style Future-KL modulated PPO loss.

    Adapted from FIPO/recipe/fipo/run_fipo_qwen2.5_32b.sh + verl core_algos.py
    (see Qwen Pilot, FIPO 2026).

    Per-token signal is the same `negative_approx_kl = log_prob - old_log_prob`
    that vanilla PPO already computes (so this is the *current vs rollout*
    log-prob shift, not an external reference KL). For each token t we
    accumulate the discounted future shift over the rest of the response:

        FutureKL_t = sum_{k>=t} gamma^(k-t) * mask_k * (log_prob_k - old_log_prob_k)

    Then convert to a multiplicative influence weight on the advantage:

        f_t = clip(exp(FutureKL_t), 1 - eps_low, 1 + eps_high)
        weighted_adv_t = adv_t * f_t

    Standard PPO clipping is applied with the modulated advantage. A safety
    threshold caps influence at 1.0 for negative-advantage tokens whose
    importance ratio is too high.

    Config keys (all on `config.policy_loss`):
        decay_rate          (float, default 32.0) — half-life in tokens; gamma = 2^(-1/decay_rate)
        chunk_size          (int, default 128)    — chunk size for the matmul accumulator
        future_kl_clip_ratio (float, default 0.2) — eps for the asymmetric clip
        future_kl_clip_high_only (bool, default True) — clip only above 1
        safety_thresh       (float, default 10.0) — ratio threshold above which neg-adv influence is capped
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    cliprange_low = clip_ratio_low if clip_ratio_low is not None else clip_ratio
    cliprange_high = clip_ratio_high if clip_ratio_high is not None else clip_ratio
    assert clip_ratio_c > 1.0

    negative_approx_kl = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ---------- Future-KL accumulation (chunked matmul) ----------
    batch_size, response_len = log_prob.shape
    device, dtype = log_prob.device, log_prob.dtype
    decay_rate = float(config.policy_loss.get("decay_rate", 32.0))
    chunk_size = int(config.policy_loss.get("chunk_size", 128))
    gamma = 2.0 ** (-1.0 / max(decay_rate, 1e-6))

    # Filter out tokens whose ratio already exceeds clip_ratio_c — they have no
    # gradient anyway and would distort the future signal.
    filter_threshold = torch.log(torch.tensor(clip_ratio_c, device=device, dtype=dtype))
    participation_mask = (negative_approx_kl <= filter_threshold).to(dtype)
    masked_kl = negative_approx_kl * response_mask.to(dtype) * participation_mask

    future_kl = torch.zeros((batch_size, response_len), device=device, dtype=dtype)
    pos_i = torch.arange(response_len, device=device).unsqueeze(1)  # (L, 1)
    gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)
        distance = j_idx - pos_i                              # (L, Kb)
        valid = distance >= 0
        decay_block = torch.pow(gamma_t, distance.clamp(min=0)) * valid.to(dtype)
        kl_block = masked_kl[:, j_start:j_end]                # (B, Kb)
        future_kl += torch.matmul(kl_block, decay_block.t())  # (B, L)

    # ---------- Influence weight ----------
    fkl_clip_ratio = float(config.policy_loss.get("future_kl_clip_ratio", 0.2))
    high_only = bool(config.policy_loss.get("future_kl_clip_high_only", True))
    if fkl_clip_ratio != 0.0:
        if high_only:
            upper_bound, lower_bound = 1.0 + fkl_clip_ratio, 1.0
        else:
            upper_bound, lower_bound = 1.0 + fkl_clip_ratio, 1.0 - fkl_clip_ratio
        influence_weights = torch.exp(future_kl).clamp(min=lower_bound, max=upper_bound).detach()
    else:
        upper_bound, lower_bound = 10.0, 0.0
        influence_weights = torch.exp(future_kl).clamp(max=10.0).detach()

    # Safety: cap negative-advantage tokens with high importance ratio so they
    # cannot have their suppression amplified by future_kl.
    safety_thresh = float(config.policy_loss.get("safety_thresh", 10.0))
    mask_neg_high_is = (advantages < 0) & (ratio > safety_thresh)
    influence_weights = torch.where(
        mask_neg_high_is, influence_weights.clamp(min=0.8, max=1.0), influence_weights
    )

    weighted_advantages = advantages * influence_weights

    # ---------- Standard PPO surrogate, with modulated advantage ----------
    pg_losses1 = -weighted_advantages * ratio
    pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    # Drop sequences with too many lower-clip activations (DAPO/FIPO trick).
    lower_clip_mask = (
        (advantages < 0) & (clip_pg_losses1 > pg_losses3) & response_mask.bool()
    )
    seq_has_low_clip = lower_clip_mask.sum(dim=1) > 1
    seq_valid_mask = (~seq_has_low_clip).unsqueeze(1)
    final_mask = (response_mask.bool() & seq_valid_mask).to(dtype)

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=final_mask, loss_agg_mode=loss_agg_mode,
        **config.global_batch_info
    )

    # ---------- Stats ----------
    influence_weights_mean = verl_F.masked_mean(influence_weights, response_mask)
    raw_iw = torch.exp(future_kl)
    raw_iw_mean = verl_F.masked_mean(raw_iw, response_mask)
    valid_iw = influence_weights[response_mask.bool()]
    valid_raw = raw_iw[response_mask.bool()]
    iw_min = valid_iw.min() if valid_iw.numel() else torch.tensor(1.0, device=device)
    iw_max = valid_iw.max() if valid_iw.numel() else torch.tensor(1.0, device=device)
    raw_iw_min = valid_raw.min() if valid_raw.numel() else torch.tensor(1.0, device=device)
    raw_iw_max = valid_raw.max() if valid_raw.numel() else torch.tensor(1.0, device=device)
    clip_high_frac = verl_F.masked_mean((influence_weights >= upper_bound - 1e-7).float(), response_mask)
    clip_low_frac = verl_F.masked_mean((influence_weights <= lower_bound + 1e-7).float(), response_mask)
    seq_dropped_frac = float(seq_has_low_clip.float().mean().detach().item())

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "future_kl/influence_weight_mean": influence_weights_mean.detach().item(),
        "future_kl/influence_weight_min": iw_min.detach().item(),
        "future_kl/influence_weight_max": iw_max.detach().item(),
        "future_kl/raw_iw_mean": raw_iw_mean.detach().item(),
        "future_kl/raw_iw_min": raw_iw_min.detach().item(),
        "future_kl/raw_iw_max": raw_iw_max.detach().item(),
        "future_kl/clip_high_frac": clip_high_frac.detach().item(),
        "future_kl/clip_low_frac": clip_low_frac.detach().item(),
        "future_kl/seq_dropped_frac": seq_dropped_frac,
        "future_kl/decay_rate": decay_rate,
        "future_kl/chunk_size": chunk_size,
    }
    return pg_loss, pg_metrics


def _action_range_for_observer_future_kl(pos_spec, response_len: int, valid_len: int) -> tuple[int, int] | None:
    """Resolve a reward-manager token-position object to a compact action span."""
    if pos_spec is None or valid_len <= 0 or response_len <= 0:
        return None

    backward_decay = _backward_decay_range_for_step_adv(pos_spec)
    if backward_decay is not None:
        prefix_start, _action_start, action_end, _decay, _local_alpha = backward_decay
        start, end = prefix_start, action_end
    else:
        decay_prefix = _decay_prefix_for_step_adv(pos_spec)
        if decay_prefix is not None:
            end, _ = decay_prefix
            start = end
        else:
            compact_range = _compact_range_for_step_adv(pos_spec)
            if compact_range is not None:
                start, end = compact_range
            else:
                positions = list(_normalize_token_positions_for_step_adv(pos_spec))
                if not positions:
                    return None
                start, end = min(positions), max(positions)

    start = max(0, min(int(start), response_len - 1, valid_len - 1))
    end = max(0, min(int(end), response_len - 1, valid_len - 1))
    if start > end:
        return None
    return start, end


def _action_only_range_for_observer_future_kl(pos_spec, response_len: int, valid_len: int) -> tuple[int, int] | None:
    """Resolve the local action span without reward-manager prefix expansion.

    Reward-level prefix credit may encode positions as backward_decay_range:
        (prefix_start, action_start, action_end, decay, local_alpha)
    For loss-level attribution we need the causal target action span, not the
    whole prefix window, otherwise prefix credit is recursively treated as an
    action and becomes too broad.
    """
    if pos_spec is None or valid_len <= 0 or response_len <= 0:
        return None

    backward_decay = _backward_decay_range_for_step_adv(pos_spec)
    if backward_decay is not None:
        _prefix_start, start, end, _decay, _local_alpha = backward_decay
    else:
        decay_prefix = _decay_prefix_for_step_adv(pos_spec)
        if decay_prefix is not None:
            end, _ = decay_prefix
            start = end
        else:
            compact_range = _compact_range_for_step_adv(pos_spec)
            if compact_range is not None:
                start, end = compact_range
            else:
                positions = list(_normalize_token_positions_for_step_adv(pos_spec))
                if not positions:
                    return None
                start, end = min(positions), max(positions)

    start = max(0, min(int(start), response_len - 1, valid_len - 1))
    end = max(0, min(int(end), response_len - 1, valid_len - 1))
    if start > end:
        return None
    return start, end


def _policy_loss_bool(policy_cfg: Any, key: str, default: bool = False) -> bool:
    value = policy_cfg.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _renormalize_step_advantages_per_response(
    before: torch.Tensor,
    after: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Preserve each response's positive and negative step-advantage budgets.

    Future-KL/RLSD should redistribute observer step credit across action spans,
    not increase the total amount of process credit available to a response.
    We therefore conserve positive mass and negative mass separately.
    """

    mask = response_mask.to(dtype=after.dtype)
    before = before * mask
    after = after * mask
    eps = torch.tensor(1e-8, device=after.device, dtype=after.dtype)

    before_pos = before.clamp(min=0.0).sum(dim=-1, keepdim=True)
    after_pos = after.clamp(min=0.0).sum(dim=-1, keepdim=True)
    before_neg = (-before.clamp(max=0.0)).sum(dim=-1, keepdim=True)
    after_neg = (-after.clamp(max=0.0)).sum(dim=-1, keepdim=True)

    pos_scale = torch.where(
        (before_pos > eps) & (after_pos > eps),
        before_pos / after_pos.clamp(min=eps.item()),
        torch.ones_like(before_pos),
    )
    neg_scale = torch.where(
        (before_neg > eps) & (after_neg > eps),
        before_neg / after_neg.clamp(min=eps.item()),
        torch.ones_like(before_neg),
    )

    renormed = torch.where(
        after > 0.0,
        after * pos_scale,
        torch.where(after < 0.0, after * neg_scale, after),
    ) * mask

    valid_response = (mask.sum(dim=-1, keepdim=True) > 0).to(dtype=after.dtype)
    response_count = valid_response.sum().clamp(min=1.0)
    pos_valid = ((before_pos > eps) & (after_pos > eps)).to(dtype=after.dtype)
    neg_valid = ((before_neg > eps) & (after_neg > eps)).to(dtype=after.dtype)
    pos_scale_mean = (pos_scale * pos_valid).sum() / pos_valid.sum().clamp(min=1.0)
    neg_scale_mean = (neg_scale * neg_valid).sum() / neg_valid.sum().clamp(min=1.0)
    metrics = {
        "observer_fipo/renormalize_per_response": 1.0,
        "observer_fipo/renorm_positive_scale_mean": float(pos_scale_mean.detach().item()),
        "observer_fipo/renorm_negative_scale_mean": float(neg_scale_mean.detach().item()),
        "observer_fipo/renorm_positive_mass_before_mean": float(
            ((before_pos * valid_response).sum() / response_count).detach().item()
        ),
        "observer_fipo/renorm_positive_mass_after_raw_mean": float(
            ((after_pos * valid_response).sum() / response_count).detach().item()
        ),
        "observer_fipo/renorm_negative_mass_before_mean": float(
            ((before_neg * valid_response).sum() / response_count).detach().item()
        ),
        "observer_fipo/renorm_negative_mass_after_raw_mean": float(
            ((after_neg * valid_response).sum() / response_count).detach().item()
        ),
    }
    return renormed, metrics


def _compute_observer_future_kl_prefix_credit(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    step_token_positions: Any,
    advantages_step: torch.Tensor,
    config: ActorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assign signed, action-conditioned Future-KL credit to pre-action tokens.

    This is a conservative proxy for "which prefix tokens made the following
    observer action more/less likely" using the same policy shift available to
    PPO:

        delta_t = log pi_theta(t) - log pi_old(t)
        impact(prefix_t -> action) = tanh(delta_t * mean(delta_action) / temp)

    The action's step advantage is then scattered to the prefix with a distance
    decay and a per-response absolute-budget cap. Positive and negative action
    rewards naturally produce the four desired cases:
      good action + prefix supports action   -> positive credit
      good action + prefix suppresses action -> negative credit
      bad action  + prefix supports action   -> negative credit
      bad action  + prefix suppresses action -> positive credit
    """
    policy_cfg = config.policy_loss
    default_metrics = {
        "observer_fipo/prefix_fkl_enabled": 0.0,
        "observer_fipo/prefix_fkl_action_count": 0.0,
        "observer_fipo/prefix_fkl_token_coverage": 0.0,
        "observer_fipo/prefix_fkl_abs_mean": 0.0,
        "observer_fipo/prefix_fkl_pos_mean": 0.0,
        "observer_fipo/prefix_fkl_neg_mean": 0.0,
        "observer_fipo/prefix_fkl_budget_ratio_mean": 0.0,
        "observer_fipo/prefix_fkl_action_adv_abs_sum_mean": 0.0,
        "observer_fipo/prefix_fkl_prefix_adv_abs_sum_mean": 0.0,
        "observer_fipo/prefix_fkl_support_action_frac": 0.0,
        "observer_fipo/prefix_fkl_suppress_action_frac": 0.0,
        "observer_fipo/prefix_fkl_skipped_no_prefix_frac": 0.0,
        "observer_fipo/prefix_fkl_skipped_low_action_delta_frac": 0.0,
        "observer_fipo/prefix_fkl_no_valid_span_frac": 0.0,
    }
    if not _policy_loss_bool(policy_cfg, "action_future_kl_prefix_credit", False):
        return torch.zeros_like(advantages_step), default_metrics
    if step_token_positions is None:
        return torch.zeros_like(advantages_step), default_metrics

    batch_size, response_len = response_mask.shape
    device = log_prob.device
    dtype = log_prob.dtype
    try:
        stp_np = np.asarray(step_token_positions, dtype=object)
    except Exception:
        return torch.zeros_like(advantages_step), default_metrics
    if stp_np.ndim == 1:
        stp_np = stp_np.reshape(batch_size, -1) if stp_np.shape[0] == batch_size else stp_np.reshape(1, -1)
    if stp_np.shape[0] != batch_size:
        return torch.zeros_like(advantages_step), default_metrics

    prefix_window = max(0, int(policy_cfg.get("action_future_kl_prefix_window", 64)))
    prefix_decay = max(0.0, min(1.0, float(policy_cfg.get("action_future_kl_prefix_decay", 0.96))))
    prefix_budget_ratio = max(0.0, float(policy_cfg.get("action_future_kl_prefix_budget_ratio", 0.15)))
    min_action_delta = max(0.0, float(policy_cfg.get("action_future_kl_prefix_min_action_delta", 1e-4)))
    temperature = max(1e-8, float(policy_cfg.get("action_future_kl_prefix_temperature", 0.003)))
    if prefix_window <= 0 or prefix_budget_ratio <= 0:
        metrics = dict(default_metrics)
        metrics["observer_fipo/prefix_fkl_enabled"] = 1.0
        return torch.zeros_like(advantages_step), metrics

    delta = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0).detach().to(dtype=torch.float32)
    step_adv = advantages_step.detach().to(dtype=torch.float32)
    mask = response_mask.to(dtype=torch.float32)
    prefix_adv = torch.zeros_like(step_adv, dtype=torch.float32)
    prefix_covered = torch.zeros_like(mask, dtype=torch.float32)
    action_abs_mass = torch.zeros(batch_size, device=device, dtype=torch.float32)

    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).detach().cpu().numpy()
    max_steps = stp_np.shape[1] if stp_np.ndim >= 2 else 0
    action_count = 0
    no_valid_span = 0
    skipped_no_prefix = 0
    skipped_low_action_delta = 0
    support_count = 0
    suppress_count = 0
    impact_count = 0
    gamma_t = torch.tensor(prefix_decay, device=device, dtype=torch.float32)

    for b in range(batch_size):
        valid_len = int(valid_lengths[b])
        if valid_len <= 0:
            continue
        for s_idx in range(max_steps):
            pos_spec = stp_np[b, s_idx]
            if pos_spec is None:
                continue
            if isinstance(pos_spec, float) and np.isnan(pos_spec):
                continue
            span = _action_only_range_for_observer_future_kl(pos_spec, response_len, valid_len)
            if span is None:
                no_valid_span += 1
                continue
            start, end = span
            span_mask = mask[b, start : end + 1]
            span_tokens = span_mask.sum()
            if bool(span_tokens <= 0):
                no_valid_span += 1
                continue
            action_count += 1
            action_adv = (step_adv[b, start : end + 1] * span_mask).sum() / span_tokens.clamp(min=1.0)
            action_delta = (delta[b, start : end + 1] * span_mask).sum() / span_tokens.clamp(min=1.0)
            action_abs_mass[b] += (step_adv[b, start : end + 1].abs() * span_mask).sum()
            if bool(action_adv.abs() <= 1e-12):
                continue
            if bool(action_delta.abs() < min_action_delta):
                skipped_low_action_delta += 1
                continue

            prefix_start = max(0, start - prefix_window)
            prefix_end = start
            if prefix_start >= prefix_end:
                skipped_no_prefix += 1
                continue
            prefix_mask = mask[b, prefix_start:prefix_end]
            if bool(prefix_mask.sum() <= 0):
                skipped_no_prefix += 1
                continue

            prefix_delta = delta[b, prefix_start:prefix_end]
            distance = torch.arange(prefix_end - prefix_start, 0, -1, device=device, dtype=torch.float32)
            distance_decay = torch.pow(gamma_t, distance)
            signed_impact = torch.tanh((prefix_delta * action_delta) / temperature) * distance_decay * prefix_mask
            prefix_adv[b, prefix_start:prefix_end] += action_adv.to(torch.float32) * signed_impact
            prefix_covered[b, prefix_start:prefix_end] = torch.maximum(
                prefix_covered[b, prefix_start:prefix_end],
                (signed_impact.abs() > 1e-8).to(torch.float32) * prefix_mask,
            )
            support_count += int(((signed_impact > 1e-8) * prefix_mask.bool()).sum().detach().item())
            suppress_count += int(((signed_impact < -1e-8) * prefix_mask.bool()).sum().detach().item())
            impact_count += int(((signed_impact.abs() > 1e-8) * prefix_mask.bool()).sum().detach().item())

    prefix_adv = prefix_adv * mask
    raw_prefix_abs = prefix_adv.abs().sum(dim=-1, keepdim=True)
    cap = action_abs_mass.unsqueeze(-1) * prefix_budget_ratio
    scale = torch.where(
        (raw_prefix_abs > cap) & (raw_prefix_abs > 1e-8),
        cap / raw_prefix_abs.clamp(min=1e-8),
        torch.ones_like(raw_prefix_abs),
    )
    prefix_adv = (prefix_adv * scale * mask).to(dtype=advantages_step.dtype).detach()

    valid_response = (mask.sum(dim=-1) > 0).to(torch.float32)
    response_count = valid_response.sum().clamp(min=1.0)
    prefix_abs_sum = prefix_adv.detach().to(torch.float32).abs().sum(dim=-1)
    action_abs_sum = action_abs_mass
    budget_ratio = torch.where(
        action_abs_sum > 1e-8,
        prefix_abs_sum / action_abs_sum.clamp(min=1e-8),
        torch.zeros_like(prefix_abs_sum),
    )
    valid_token_count = mask.sum().clamp(min=1.0)
    covered_count = (prefix_covered * mask).sum()
    pos_mass = prefix_adv.detach().to(torch.float32).clamp(min=0.0)
    neg_mass = (-prefix_adv.detach().to(torch.float32).clamp(max=0.0))
    metrics = dict(default_metrics)
    metrics["observer_fipo/prefix_fkl_enabled"] = 1.0
    metrics["observer_fipo/prefix_fkl_action_count"] = float(action_count)
    metrics["observer_fipo/prefix_fkl_token_coverage"] = float(((prefix_covered * mask).sum() / valid_token_count).detach().item())
    metrics["observer_fipo/prefix_fkl_abs_mean"] = float((prefix_adv.detach().to(torch.float32).abs() * mask).sum().div(valid_token_count).detach().item())
    metrics["observer_fipo/prefix_fkl_pos_mean"] = float((pos_mass * mask).sum().div(valid_token_count).detach().item())
    metrics["observer_fipo/prefix_fkl_neg_mean"] = float((neg_mass * mask).sum().div(valid_token_count).detach().item())
    metrics["observer_fipo/prefix_fkl_budget_ratio_mean"] = float(
        ((budget_ratio * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/prefix_fkl_action_adv_abs_sum_mean"] = float(
        ((action_abs_sum * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/prefix_fkl_prefix_adv_abs_sum_mean"] = float(
        ((prefix_abs_sum * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/prefix_fkl_support_action_frac"] = float(support_count / max(1, impact_count))
    metrics["observer_fipo/prefix_fkl_suppress_action_frac"] = float(suppress_count / max(1, impact_count))
    metrics["observer_fipo/prefix_fkl_skipped_no_prefix_frac"] = float(skipped_no_prefix / max(1, action_count))
    metrics["observer_fipo/prefix_fkl_skipped_low_action_delta_frac"] = float(
        skipped_low_action_delta / max(1, action_count)
    )
    metrics["observer_fipo/prefix_fkl_no_valid_span_frac"] = float(no_valid_span / max(1, no_valid_span + action_count))
    if bool(covered_count <= 0):
        metrics["observer_fipo/prefix_fkl_support_action_frac"] = 0.0
        metrics["observer_fipo/prefix_fkl_suppress_action_frac"] = 0.0
    return prefix_adv, metrics


def _compute_observer_future_kl_credit_assignment(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    step_token_positions: Any,
    advantages_step: torch.Tensor,
    config: ActorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Transfer verified action step advantage to likely causal prefix/action tokens.

    This is a mass-preserving alternative to weighting the whole observer action
    span. For each action span, policy-shift agreement between candidate tokens
    and the action span is used as a local attribution proxy. When no supporting
    token is found, the original step advantage is left unchanged.
    """

    policy_cfg = config.policy_loss
    default_metrics = {
        "observer_fipo/credit_assignment_enabled": 0.0,
        "observer_fipo/credit_assignment_action_count": 0.0,
        "observer_fipo/credit_assignment_transferred_action_count": 0.0,
        "observer_fipo/credit_assignment_token_coverage": 0.0,
        "observer_fipo/credit_assignment_support_empty_frac": 0.0,
        "observer_fipo/credit_assignment_no_valid_span_frac": 0.0,
        "observer_fipo/credit_assignment_mass_before_mean": 0.0,
        "observer_fipo/credit_assignment_mass_after_mean": 0.0,
        "observer_fipo/credit_assignment_abs_mass_before_mean": 0.0,
        "observer_fipo/credit_assignment_abs_mass_after_mean": 0.0,
        "observer_fipo/credit_assignment_transfer_abs_mass_mean": 0.0,
        "observer_fipo/credit_assignment_local_alpha": 1.0,
        "observer_fipo/credit_assignment_window": 0.0,
        "observer_fipo/credit_assignment_prefix_and_action": 0.0,
        "observer_fipo/credit_assignment_action_only": 0.0,
        "observer_fipo/credit_assignment_preserve_mass": 1.0,
    }
    if step_token_positions is None:
        return advantages_step, default_metrics

    batch_size, response_len = response_mask.shape
    device = log_prob.device
    try:
        stp_np = np.asarray(step_token_positions, dtype=object)
    except Exception:
        return advantages_step, default_metrics
    if stp_np.ndim == 1:
        stp_np = stp_np.reshape(batch_size, -1) if stp_np.shape[0] == batch_size else stp_np.reshape(1, -1)
    if stp_np.shape[0] != batch_size:
        return advantages_step, default_metrics

    window = max(0, int(policy_cfg.get("action_future_kl_credit_window", 64)))
    target_mode = str(policy_cfg.get("action_future_kl_credit_target", "prefix") or "prefix").lower()
    local_alpha = max(0.0, min(1.0, float(policy_cfg.get("action_future_kl_credit_local_alpha", 0.2))))
    decay = max(0.0, min(1.0, float(policy_cfg.get("action_future_kl_credit_decay", 0.96))))
    temperature = max(1e-8, float(policy_cfg.get("action_future_kl_credit_temperature", 0.003)))
    min_support = max(0.0, float(policy_cfg.get("action_future_kl_credit_min_support", 1e-6)))
    preserve_mass = _policy_loss_bool(policy_cfg, "action_future_kl_credit_preserve_mass", True)
    action_only = target_mode in {"action_only", "action-only", "action"}
    include_action = target_mode in {"prefix_and_action", "prefix+action", "both"} or action_only

    step_adv_before = advantages_step.detach().to(dtype=torch.float32)
    step_adv_after = step_adv_before.clone()
    delta = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0).detach().to(dtype=torch.float32)
    mask = response_mask.to(dtype=torch.float32)
    covered = torch.zeros_like(mask)
    transfer_abs_mass = torch.zeros(batch_size, device=device, dtype=torch.float32)

    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).detach().cpu().numpy()
    max_steps = stp_np.shape[1] if stp_np.ndim >= 2 else 0
    action_count = 0
    transferred_count = 0
    support_empty = 0
    no_valid_span = 0

    for b in range(batch_size):
        valid_len = int(valid_lengths[b])
        if valid_len <= 0:
            continue
        for s_idx in range(max_steps):
            pos_spec = stp_np[b, s_idx]
            if pos_spec is None:
                continue
            if isinstance(pos_spec, float) and np.isnan(pos_spec):
                continue

            span = _action_only_range_for_observer_future_kl(pos_spec, response_len, valid_len)
            if span is None:
                no_valid_span += 1
                continue
            start, end = span
            span_mask = mask[b, start : end + 1]
            span_tokens = span_mask.sum()
            if bool(span_tokens <= 0):
                no_valid_span += 1
                continue

            action_count += 1
            action_values = step_adv_before[b, start : end + 1] * span_mask
            action_mass = action_values.sum()
            if bool(action_mass.abs() <= 1e-12):
                support_empty += 1
                continue

            action_delta = (delta[b, start : end + 1] * span_mask).sum() / span_tokens.clamp(min=1.0)
            if bool(action_delta.abs() <= 1e-12):
                support_empty += 1
                continue

            if action_only:
                cand_start = start
                cand_end = end + 1
            else:
                cand_start = max(0, start - window)
                cand_end = end + 1 if include_action else start
            cand_end = max(cand_start, min(cand_end, valid_len, response_len))
            if cand_start >= cand_end:
                support_empty += 1
                continue

            cand_mask = mask[b, cand_start:cand_end]
            if bool(cand_mask.sum() <= 0):
                support_empty += 1
                continue

            cand_delta = delta[b, cand_start:cand_end]
            cols = torch.arange(cand_start, cand_end, device=device, dtype=torch.float32)
            if action_only:
                distance = (cols - float(start)).clamp(min=0.0)
            else:
                distance = torch.where(
                    cols < float(start),
                    float(start) - cols,
                    cols - float(start),
                )
            distance_decay = torch.pow(
                torch.tensor(decay, device=device, dtype=torch.float32),
                distance,
            )
            impact = torch.tanh((cand_delta * action_delta.to(torch.float32)) / temperature)
            support = torch.relu(impact) * distance_decay * cand_mask
            support_sum = support.sum()
            if bool(support_sum <= min_support):
                support_empty += 1
                continue

            transfer_ratio = 1.0 - local_alpha
            transfer_mass = action_mass.to(torch.float32) * float(transfer_ratio)
            if bool(transfer_mass.abs() <= 1e-12):
                support_empty += 1
                continue

            if preserve_mass:
                step_adv_after[b, start : end + 1] -= action_values * float(transfer_ratio)
            credit = transfer_mass * support / support_sum.clamp(min=1e-8)
            step_adv_after[b, cand_start:cand_end] += credit
            covered[b, cand_start:cand_end] = torch.maximum(
                covered[b, cand_start:cand_end],
                (support > min_support).to(dtype=torch.float32) * cand_mask,
            )
            transfer_abs_mass[b] += transfer_mass.abs()
            transferred_count += 1

    step_adv_after = (step_adv_after * mask).to(dtype=advantages_step.dtype).detach()
    valid_response = (mask.sum(dim=-1, keepdim=True) > 0).to(dtype=torch.float32)
    response_count = valid_response.sum().clamp(min=1.0)
    valid_token_count = mask.sum().clamp(min=1.0)
    before_masked = step_adv_before * mask
    after_masked = step_adv_after.detach().to(dtype=torch.float32) * mask
    metrics = dict(default_metrics)
    metrics["observer_fipo/credit_assignment_enabled"] = 1.0
    metrics["observer_fipo/credit_assignment_action_count"] = float(action_count)
    metrics["observer_fipo/credit_assignment_transferred_action_count"] = float(transferred_count)
    metrics["observer_fipo/credit_assignment_token_coverage"] = float((covered * mask).sum().div(valid_token_count).detach().item())
    metrics["observer_fipo/credit_assignment_support_empty_frac"] = float(support_empty / max(1, action_count))
    metrics["observer_fipo/credit_assignment_no_valid_span_frac"] = float(no_valid_span / max(1, no_valid_span + action_count))
    metrics["observer_fipo/credit_assignment_mass_before_mean"] = float(
        ((before_masked.sum(dim=-1, keepdim=True) * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/credit_assignment_mass_after_mean"] = float(
        ((after_masked.sum(dim=-1, keepdim=True) * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/credit_assignment_abs_mass_before_mean"] = float(
        ((before_masked.abs().sum(dim=-1, keepdim=True) * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/credit_assignment_abs_mass_after_mean"] = float(
        ((after_masked.abs().sum(dim=-1, keepdim=True) * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/credit_assignment_transfer_abs_mass_mean"] = float(
        ((transfer_abs_mass.view(-1, 1) * valid_response).sum() / response_count).detach().item()
    )
    metrics["observer_fipo/credit_assignment_local_alpha"] = float(local_alpha)
    metrics["observer_fipo/credit_assignment_window"] = float(window)
    metrics["observer_fipo/credit_assignment_prefix_and_action"] = 1.0 if include_action else 0.0
    metrics["observer_fipo/credit_assignment_action_only"] = 1.0 if action_only else 0.0
    metrics["observer_fipo/credit_assignment_preserve_mass"] = 1.0 if preserve_mass else 0.0
    return step_adv_after, metrics


def _compute_observer_future_kl_action_weight_map(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    step_token_positions: Any,
    advantages_step: torch.Tensor,
    advantages_episode: torch.Tensor | None,
    config: ActorConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute FIPO-style influence weights at observer-action granularity.

    The observer supplies action/segment spans. For each action span [l, r],
    we measure the decayed policy shift strictly after the action:

        F_a = sum_{k > r} gamma^(k-r-1) (log pi_theta_k - log pi_old_k)

    `action_future_kl_normalize=mean` divides that discounted sum by the
    discounted suffix mass. `sum` keeps the original FIPO-style scale.

    When `action_future_kl_direction=episode`, the weight follows the RLSD
    direction-aware form:

        w_a = (1 - lambda) + lambda * clip(exp(beta * sign(A_episode) * F_a))

    Positive-outcome responses therefore amplify actions whose future suffix is
    reinforced by the policy update, while negative-outcome responses amplify
    blame for actions whose future suffix is suppressed. The sign is detached
    and only gates credit redistribution; it does not create another gradient
    path.

    The resulting action weight is applied only to tokens inside [l, r].
    """
    batch_size, response_len = response_mask.shape
    device = log_prob.device
    dtype = log_prob.dtype

    default_metrics = {
        "observer_fipo/enabled": 0.0,
        "observer_fipo/action_count": 0.0,
        "observer_fipo/action_token_coverage": 0.0,
        "observer_fipo/future_kl_mean": 0.0,
        "observer_fipo/future_kl_std": 0.0,
        "observer_fipo/weight_mean": 1.0,
        "observer_fipo/weight_std": 0.0,
        "observer_fipo/weight_min": 1.0,
        "observer_fipo/weight_max": 1.0,
        "observer_fipo/amplify_frac": 0.0,
        "observer_fipo/dampen_frac": 0.0,
        "observer_fipo/nontrivial_weight_frac": 0.0,
        "observer_fipo/positive_action_weight_mean": 1.0,
        "observer_fipo/negative_action_weight_mean": 1.0,
        "observer_fipo/clip_low_frac": 0.0,
        "observer_fipo/clip_high_frac": 0.0,
        "observer_fipo/no_future_suffix_frac": 0.0,
        "observer_fipo/no_valid_span_frac": 0.0,
        "observer_fipo/normalize_sum": 0.0,
        "observer_fipo/direction_episode": 0.0,
        "observer_fipo/direction_step": 0.0,
        "observer_fipo/mix_lambda": 1.0,
        "observer_fipo/directional_future_kl_mean": 0.0,
        "observer_fipo/directional_future_kl_abs_mean": 0.0,
        "observer_fipo/directional_future_kl_positive_frac": 0.0,
        "observer_fipo/directional_future_kl_negative_frac": 0.0,
        "observer_fipo/pre_mix_clip_low_frac": 0.0,
        "observer_fipo/pre_mix_clip_high_frac": 0.0,
        "observer_fipo/positive_episode_weight_mean": 1.0,
        "observer_fipo/negative_episode_weight_mean": 1.0,
        "observer_fipo/zero_episode_weight_mean": 1.0,
        "observer_fipo/future_kl_abs_mean": 0.0,
        "observer_fipo/weight_abs_deviation_mean": 0.0,
        "observer_fipo/suffix_mass_mean": 0.0,
        "observer_fipo/suffix_mass_std": 0.0,
        "observer_fipo/action_span_token_mean": 0.0,
        "observer_fipo/action_span_adv_abs_mean": 0.0,
        "observer_fipo/covered_step_adv_abs_mean": 0.0,
        "observer_fipo/renormalize_per_response": 0.0,
        "observer_fipo/renorm_positive_scale_mean": 1.0,
        "observer_fipo/renorm_negative_scale_mean": 1.0,
        "observer_fipo/renorm_positive_mass_before_mean": 0.0,
        "observer_fipo/renorm_positive_mass_after_raw_mean": 0.0,
        "observer_fipo/renorm_negative_mass_before_mean": 0.0,
        "observer_fipo/renorm_negative_mass_after_raw_mean": 0.0,
    }
    if step_token_positions is None:
        return torch.ones_like(log_prob), default_metrics

    try:
        stp_np = np.asarray(step_token_positions, dtype=object)
    except Exception:
        return torch.ones_like(log_prob), default_metrics
    if stp_np.ndim == 1:
        stp_np = stp_np.reshape(batch_size, -1) if stp_np.shape[0] == batch_size else stp_np.reshape(1, -1)
    if stp_np.shape[0] != batch_size:
        return torch.ones_like(log_prob), default_metrics

    policy_cfg = config.policy_loss
    decay_rate = float(policy_cfg.get("action_future_kl_decay_rate", policy_cfg.get("decay_rate", 32.0)))
    gamma = 2.0 ** (-1.0 / max(decay_rate, 1e-6))
    beta = float(policy_cfg.get("action_future_kl_beta", 1.0))
    clip_low = float(policy_cfg.get("action_future_kl_clip_low", 0.85))
    clip_high = float(policy_cfg.get("action_future_kl_clip_high", 1.20))
    start_mode = str(policy_cfg.get("action_future_kl_start", "after_action")).lower()
    normalize_mode = str(policy_cfg.get("action_future_kl_normalize", "sum")).lower()
    direction_mode = str(policy_cfg.get("action_future_kl_direction", "none")).lower()
    mix_lambda = float(policy_cfg.get("action_future_kl_lambda", 1.0))
    mix_lambda = max(0.0, min(1.0, mix_lambda))
    if start_mode != "after_action":
        raise ValueError("observer_future_kl currently supports action_future_kl_start=after_action only.")
    if normalize_mode not in ("mean", "sum"):
        raise ValueError("observer_future_kl action_future_kl_normalize must be one of: mean, sum.")
    if direction_mode not in ("none", "episode", "step"):
        raise ValueError("observer_future_kl action_future_kl_direction must be one of: none, episode, step.")
    if clip_low > clip_high:
        clip_low, clip_high = clip_high, clip_low
    if direction_mode == "episode" and advantages_episode is None:
        direction_mode = "none"

    delta = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0).detach()
    mask = response_mask.to(dtype=dtype)
    suffix_sum_after = torch.zeros_like(delta)
    suffix_mass_after = torch.zeros_like(delta)
    running_sum = torch.zeros(batch_size, device=device, dtype=dtype)
    running_mass = torch.zeros(batch_size, device=device, dtype=dtype)
    gamma_t = torch.tensor(gamma, device=device, dtype=dtype)

    for t in range(response_len - 1, -1, -1):
        suffix_sum_after[:, t] = running_sum
        suffix_mass_after[:, t] = running_mass
        token_mask = mask[:, t]
        running_sum = delta[:, t] * token_mask + gamma_t * running_sum
        running_mass = token_mask + gamma_t * running_mass

    weight_accum = torch.zeros_like(log_prob, dtype=torch.float32)
    weight_denom = torch.zeros_like(log_prob, dtype=torch.float32)
    token_covered = torch.zeros_like(response_mask, dtype=torch.float32)

    future_values: list[torch.Tensor] = []
    directional_future_values: list[torch.Tensor] = []
    clipped_action_weights: list[torch.Tensor] = []
    action_weights: list[torch.Tensor] = []
    positive_weights: list[torch.Tensor] = []
    negative_weights: list[torch.Tensor] = []
    positive_episode_weights: list[torch.Tensor] = []
    negative_episode_weights: list[torch.Tensor] = []
    zero_episode_weights: list[torch.Tensor] = []
    suffix_masses: list[torch.Tensor] = []
    span_token_counts: list[torch.Tensor] = []
    span_adv_abs_values: list[torch.Tensor] = []
    no_future = 0
    no_valid_span = 0
    action_count = 0

    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).detach().cpu().numpy()
    max_steps = stp_np.shape[1] if stp_np.ndim >= 2 else 0
    for b in range(batch_size):
        valid_len = int(valid_lengths[b])
        if valid_len <= 0:
            continue
        for s_idx in range(max_steps):
            pos_spec = stp_np[b, s_idx]
            if pos_spec is None:
                continue
            if isinstance(pos_spec, float) and np.isnan(pos_spec):
                continue
            span = _action_range_for_observer_future_kl(pos_spec, response_len, valid_len)
            if span is None:
                no_valid_span += 1
                continue
            start, end = span
            action_count += 1
            mass = suffix_mass_after[b, end]
            if bool(mass <= 1e-8):
                future_val = torch.zeros((), device=device, dtype=dtype)
                no_future += 1
            else:
                future_sum = suffix_sum_after[b, end]
                if normalize_mode == "mean":
                    future_val = future_sum / mass.clamp(min=1e-8)
                else:
                    future_val = future_sum

            span_mask = response_mask[b, start : end + 1].to(dtype=torch.float32)
            span_tokens = span_mask.sum()
            if bool(span_tokens <= 0):
                no_valid_span += 1
                continue
            span_adv = (
                advantages_step[b, start : end + 1].detach().to(dtype=torch.float32) * span_mask
            ).sum() / span_tokens.clamp(min=1.0)
            strength = span_adv.abs().clamp(min=1e-6)
            direction_sign = torch.ones((), device=device, dtype=torch.float32)
            episode_span_adv = torch.zeros((), device=device, dtype=torch.float32)
            if direction_mode == "episode" and advantages_episode is not None:
                episode_span_adv = (
                    advantages_episode[b, start : end + 1].detach().to(dtype=torch.float32) * span_mask
                ).sum() / span_tokens.clamp(min=1.0)
                direction_sign = torch.sign(episode_span_adv)
            elif direction_mode == "step":
                direction_sign = torch.sign(span_adv)

            directional_future_val = future_val.to(torch.float32) * direction_sign
            clipped_weight = torch.exp(beta * directional_future_val).clamp(min=clip_low, max=clip_high)
            raw_weight = ((1.0 - mix_lambda) + mix_lambda * clipped_weight).detach()

            weight_accum[b, start : end + 1] += raw_weight.to(torch.float32) * strength * span_mask
            weight_denom[b, start : end + 1] += strength * span_mask
            token_covered[b, start : end + 1] = torch.maximum(token_covered[b, start : end + 1], span_mask)
            future_values.append(future_val.to(torch.float32))
            directional_future_values.append(directional_future_val.to(torch.float32))
            clipped_action_weights.append(clipped_weight.to(torch.float32))
            action_weights.append(raw_weight.to(torch.float32))
            suffix_masses.append(mass.to(torch.float32))
            span_token_counts.append(span_tokens.to(torch.float32))
            span_adv_abs_values.append(span_adv.abs().to(torch.float32))
            if bool(span_adv > 0):
                positive_weights.append(raw_weight.to(torch.float32))
            elif bool(span_adv < 0):
                negative_weights.append(raw_weight.to(torch.float32))
            if bool(episode_span_adv > 0):
                positive_episode_weights.append(raw_weight.to(torch.float32))
            elif bool(episode_span_adv < 0):
                negative_episode_weights.append(raw_weight.to(torch.float32))
            else:
                zero_episode_weights.append(raw_weight.to(torch.float32))

    weight_map = torch.where(
        weight_denom > 0,
        weight_accum / weight_denom.clamp(min=1e-8),
        torch.ones_like(weight_accum),
    ).to(dtype=dtype)
    weight_map = weight_map.detach()

    valid_mask = response_mask.to(dtype=torch.float32)
    coverage = (token_covered * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    metrics = dict(default_metrics)
    metrics["observer_fipo/enabled"] = 1.0
    metrics["observer_fipo/action_count"] = float(action_count)
    metrics["observer_fipo/action_token_coverage"] = float(coverage.detach().item())
    metrics["observer_fipo/no_future_suffix_frac"] = float(no_future / max(1, action_count))
    metrics["observer_fipo/no_valid_span_frac"] = float(no_valid_span / max(1, no_valid_span + action_count))
    metrics["observer_fipo/normalize_sum"] = 1.0 if normalize_mode == "sum" else 0.0
    metrics["observer_fipo/direction_episode"] = 1.0 if direction_mode == "episode" else 0.0
    metrics["observer_fipo/direction_step"] = 1.0 if direction_mode == "step" else 0.0
    metrics["observer_fipo/mix_lambda"] = mix_lambda
    covered_token_count = (token_covered * valid_mask).sum()
    if bool(covered_token_count > 0):
        covered_step_abs = (advantages_step.detach().to(torch.float32).abs() * token_covered * valid_mask).sum()
        metrics["observer_fipo/covered_step_adv_abs_mean"] = float(
            (covered_step_abs / covered_token_count.clamp(min=1.0)).detach().item()
        )

    if action_weights:
        weights_t = torch.stack(action_weights)
        future_t = torch.stack(future_values)
        directional_future_t = torch.stack(directional_future_values)
        clipped_weights_t = torch.stack(clipped_action_weights)
        metrics["observer_fipo/future_kl_mean"] = float(future_t.mean().detach().item())
        metrics["observer_fipo/future_kl_std"] = float(future_t.std(unbiased=False).detach().item())
        metrics["observer_fipo/future_kl_abs_mean"] = float(future_t.abs().mean().detach().item())
        metrics["observer_fipo/directional_future_kl_mean"] = float(
            directional_future_t.mean().detach().item()
        )
        metrics["observer_fipo/directional_future_kl_abs_mean"] = float(
            directional_future_t.abs().mean().detach().item()
        )
        metrics["observer_fipo/weight_mean"] = float(weights_t.mean().detach().item())
        metrics["observer_fipo/weight_std"] = float(weights_t.std(unbiased=False).detach().item())
        metrics["observer_fipo/weight_min"] = float(weights_t.min().detach().item())
        metrics["observer_fipo/weight_max"] = float(weights_t.max().detach().item())
        metrics["observer_fipo/weight_abs_deviation_mean"] = float((weights_t - 1.0).abs().mean().detach().item())
        metrics["observer_fipo/amplify_frac"] = float((weights_t > 1.0 + 1e-7).float().mean().detach().item())
        metrics["observer_fipo/dampen_frac"] = float((weights_t < 1.0 - 1e-7).float().mean().detach().item())
        metrics["observer_fipo/nontrivial_weight_frac"] = float(
            ((weights_t - 1.0).abs() >= 0.005).float().mean().detach().item()
        )
        metrics["observer_fipo/directional_future_kl_positive_frac"] = float(
            (directional_future_t > 1e-8).float().mean().detach().item()
        )
        metrics["observer_fipo/directional_future_kl_negative_frac"] = float(
            (directional_future_t < -1e-8).float().mean().detach().item()
        )
        metrics["observer_fipo/clip_low_frac"] = float(
            (weights_t <= ((1.0 - mix_lambda) + mix_lambda * clip_low) + 1e-7).float().mean().detach().item()
        )
        metrics["observer_fipo/clip_high_frac"] = float(
            (weights_t >= ((1.0 - mix_lambda) + mix_lambda * clip_high) - 1e-7).float().mean().detach().item()
        )
        metrics["observer_fipo/pre_mix_clip_low_frac"] = float(
            (clipped_weights_t <= clip_low + 1e-7).float().mean().detach().item()
        )
        metrics["observer_fipo/pre_mix_clip_high_frac"] = float(
            (clipped_weights_t >= clip_high - 1e-7).float().mean().detach().item()
        )
    if suffix_masses:
        suffix_mass_t = torch.stack(suffix_masses)
        metrics["observer_fipo/suffix_mass_mean"] = float(suffix_mass_t.mean().detach().item())
        metrics["observer_fipo/suffix_mass_std"] = float(suffix_mass_t.std(unbiased=False).detach().item())
    if span_token_counts:
        metrics["observer_fipo/action_span_token_mean"] = float(torch.stack(span_token_counts).mean().detach().item())
    if span_adv_abs_values:
        metrics["observer_fipo/action_span_adv_abs_mean"] = float(
            torch.stack(span_adv_abs_values).mean().detach().item()
        )
    if positive_weights:
        metrics["observer_fipo/positive_action_weight_mean"] = float(
            torch.stack(positive_weights).mean().detach().item()
        )
    if negative_weights:
        metrics["observer_fipo/negative_action_weight_mean"] = float(
            torch.stack(negative_weights).mean().detach().item()
        )
    if positive_episode_weights:
        metrics["observer_fipo/positive_episode_weight_mean"] = float(
            torch.stack(positive_episode_weights).mean().detach().item()
        )
    if negative_episode_weights:
        metrics["observer_fipo/negative_episode_weight_mean"] = float(
            torch.stack(negative_episode_weights).mean().detach().item()
        )
    if zero_episode_weights:
        metrics["observer_fipo/zero_episode_weight_mean"] = float(
            torch.stack(zero_episode_weights).mean().detach().item()
        )

    return weight_map, metrics


@register_policy_loss("observer_future_kl")  # type: ignore[arg-type]
def compute_policy_loss_observer_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
    advantages_episode: torch.Tensor | None = None,
    advantages_step: torch.Tensor | None = None,
    step_token_positions: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO loss that applies Future-KL only to observer/GiGPO step advantages."""
    del ref_log_prob, uid
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    assert clip_ratio_c > 1.0

    negative_approx_kl = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    observer_metrics: dict[str, Any]
    if advantages_episode is None or advantages_step is None:
        weighted_advantages = advantages
        observer_metrics = {
            "observer_fipo/enabled": 0.0,
            "observer_fipo/missing_components": 1.0,
            "observer_fipo/step_adv_before_mean": 0.0,
            "observer_fipo/step_adv_after_mean": 0.0,
            "observer_fipo/step_adv_before_abs_mean": 0.0,
            "observer_fipo/step_adv_after_abs_mean": 0.0,
            "observer_fipo/step_adv_weighted_raw_mean": 0.0,
            "observer_fipo/step_adv_weighted_raw_abs_mean": 0.0,
            "observer_fipo/renormalize_per_response": 0.0,
            "observer_fipo/episode_reweight_enabled": 0.0,
            "observer_fipo/episode_reweight_renormalize_per_response": 0.0,
            "observer_fipo/episode_adv_before_mean": 0.0,
            "observer_fipo/episode_adv_after_mean": 0.0,
            "observer_fipo/episode_adv_before_abs_mean": 0.0,
            "observer_fipo/episode_adv_after_abs_mean": 0.0,
            "observer_fipo/episode_adv_weighted_raw_abs_mean": 0.0,
            "observer_fipo/prefix_fkl_enabled": 0.0,
            "observer_fipo/prefix_fkl_action_count": 0.0,
            "observer_fipo/prefix_fkl_token_coverage": 0.0,
            "observer_fipo/prefix_fkl_abs_mean": 0.0,
            "observer_fipo/prefix_fkl_pos_mean": 0.0,
            "observer_fipo/prefix_fkl_neg_mean": 0.0,
            "observer_fipo/prefix_fkl_budget_ratio_mean": 0.0,
            "observer_fipo/prefix_fkl_action_adv_abs_sum_mean": 0.0,
            "observer_fipo/prefix_fkl_prefix_adv_abs_sum_mean": 0.0,
            "observer_fipo/prefix_fkl_support_action_frac": 0.0,
            "observer_fipo/prefix_fkl_suppress_action_frac": 0.0,
            "observer_fipo/prefix_fkl_skipped_no_prefix_frac": 0.0,
            "observer_fipo/prefix_fkl_skipped_low_action_delta_frac": 0.0,
            "observer_fipo/prefix_fkl_no_valid_span_frac": 0.0,
            "observer_fipo/credit_assignment_enabled": 0.0,
            "observer_fipo/credit_assignment_action_count": 0.0,
            "observer_fipo/credit_assignment_transferred_action_count": 0.0,
            "observer_fipo/credit_assignment_token_coverage": 0.0,
            "observer_fipo/credit_assignment_support_empty_frac": 0.0,
            "observer_fipo/credit_assignment_no_valid_span_frac": 0.0,
            "observer_fipo/credit_assignment_mass_before_mean": 0.0,
            "observer_fipo/credit_assignment_mass_after_mean": 0.0,
            "observer_fipo/credit_assignment_abs_mass_before_mean": 0.0,
            "observer_fipo/credit_assignment_abs_mass_after_mean": 0.0,
            "observer_fipo/credit_assignment_transfer_abs_mass_mean": 0.0,
            "observer_fipo/credit_assignment_local_alpha": 1.0,
            "observer_fipo/credit_assignment_window": 0.0,
            "observer_fipo/credit_assignment_prefix_and_action": 0.0,
            "observer_fipo/credit_assignment_preserve_mass": 1.0,
        }
    else:
        advantages_episode = advantages_episode.to(device=advantages.device, dtype=advantages.dtype)
        advantages_step = advantages_step.to(device=advantages.device, dtype=advantages.dtype)
        action_weight_map, observer_metrics = _compute_observer_future_kl_action_weight_map(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            step_token_positions=step_token_positions,
            advantages_step=advantages_step,
            advantages_episode=advantages_episode,
            config=config,
        )
        credit_assignment_mode = str(
            config.policy_loss.get("action_future_kl_credit_assignment", "off") or "off"
        ).lower()
        if credit_assignment_mode == "transfer":
            step_adv_after, credit_metrics = _compute_observer_future_kl_credit_assignment(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                response_mask=response_mask,
                step_token_positions=step_token_positions,
                advantages_step=advantages_step,
                config=config,
            )
            observer_metrics.update(credit_metrics)
        elif credit_assignment_mode in {"off", "none", "false", "0"}:
            step_adv_after = action_weight_map.to(dtype=advantages_step.dtype) * advantages_step
        else:
            raise ValueError(
                f"Unsupported action_future_kl_credit_assignment={credit_assignment_mode}. "
                "Expected one of: off, transfer."
            )
        observer_metrics["observer_fipo/step_adv_weighted_raw_mean"] = float(
            verl_F.masked_mean(step_adv_after, response_mask).detach().item()
        )
        observer_metrics["observer_fipo/step_adv_weighted_raw_abs_mean"] = float(
            verl_F.masked_mean(step_adv_after.abs(), response_mask).detach().item()
        )
        if _policy_loss_bool(config.policy_loss, "action_future_kl_renormalize_per_response", False):
            step_adv_after, renorm_metrics = _renormalize_step_advantages_per_response(
                advantages_step,
                step_adv_after,
                response_mask,
            )
            observer_metrics.update(renorm_metrics)
        else:
            observer_metrics["observer_fipo/renormalize_per_response"] = 0.0
        prefix_adv, prefix_metrics = _compute_observer_future_kl_prefix_credit(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            response_mask=response_mask,
            step_token_positions=step_token_positions,
            advantages_step=advantages_step,
            config=config,
        )
        observer_metrics.update(prefix_metrics)
        episode_adv_after = advantages_episode
        observer_metrics["observer_fipo/episode_reweight_enabled"] = 0.0
        observer_metrics["observer_fipo/episode_reweight_renormalize_per_response"] = 0.0
        observer_metrics["observer_fipo/episode_adv_before_mean"] = float(
            verl_F.masked_mean(advantages_episode, response_mask).detach().item()
        )
        observer_metrics["observer_fipo/episode_adv_before_abs_mean"] = float(
            verl_F.masked_mean(advantages_episode.abs(), response_mask).detach().item()
        )
        if _policy_loss_bool(config.policy_loss, "action_future_kl_reweight_episode", False):
            episode_adv_after = action_weight_map.to(dtype=advantages_episode.dtype) * advantages_episode
            observer_metrics["observer_fipo/episode_reweight_enabled"] = 1.0
            observer_metrics["observer_fipo/episode_adv_weighted_raw_abs_mean"] = float(
                verl_F.masked_mean(episode_adv_after.abs(), response_mask).detach().item()
            )
            if _policy_loss_bool(config.policy_loss, "action_future_kl_reweight_episode_renormalize", True):
                episode_adv_after, episode_renorm_metrics = _renormalize_step_advantages_per_response(
                    advantages_episode,
                    episode_adv_after,
                    response_mask,
                )
                observer_metrics["observer_fipo/episode_reweight_renormalize_per_response"] = float(
                    episode_renorm_metrics.get("observer_fipo/renormalize_per_response", 0.0)
                )
                for key, value in episode_renorm_metrics.items():
                    if key == "observer_fipo/renormalize_per_response":
                        continue
                    if key.startswith("observer_fipo/renorm_"):
                        new_key = "observer_fipo/episode_reweight_renorm_" + key[len("observer_fipo/renorm_") :]
                        observer_metrics[new_key] = value
        else:
            observer_metrics["observer_fipo/episode_adv_weighted_raw_abs_mean"] = 0.0
        observer_metrics["observer_fipo/episode_adv_after_mean"] = float(
            verl_F.masked_mean(episode_adv_after, response_mask).detach().item()
        )
        observer_metrics["observer_fipo/episode_adv_after_abs_mean"] = float(
            verl_F.masked_mean(episode_adv_after.abs(), response_mask).detach().item()
        )
        weighted_advantages = episode_adv_after + step_adv_after + prefix_adv.to(dtype=advantages_episode.dtype)
        observer_metrics["observer_fipo/missing_components"] = 0.0
        observer_metrics["observer_fipo/step_adv_before_mean"] = float(
            verl_F.masked_mean(advantages_step, response_mask).detach().item()
        )
        observer_metrics["observer_fipo/step_adv_after_mean"] = float(
            verl_F.masked_mean(step_adv_after, response_mask).detach().item()
        )
        observer_metrics["observer_fipo/step_adv_before_abs_mean"] = float(
            verl_F.masked_mean(advantages_step.abs(), response_mask).detach().item()
        )
        observer_metrics["observer_fipo/step_adv_after_abs_mean"] = float(
            verl_F.masked_mean(step_adv_after.abs(), response_mask).detach().item()
        )

    pg_losses1 = -weighted_advantages * ratio
    pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (weighted_advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    pg_metrics.update(observer_metrics)
    return pg_loss, pg_metrics


def _center_by_uid_masked(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    uid: Any,
) -> torch.Tensor:
    """Center token-level values by uid group under response mask."""
    if uid is None:
        valid = response_mask.to(values.dtype)
        denom = valid.sum().clamp(min=1.0)
        mean = (values * valid).sum() / denom
        return values - mean

    uid_arr = np.asarray(uid).reshape(-1)
    if uid_arr.shape[0] != values.shape[0]:
        raise ValueError(f"uid batch mismatch: {uid_arr.shape[0]} vs {values.shape[0]}")

    _, inverse = np.unique(uid_arr, return_inverse=True)
    inv = torch.as_tensor(inverse, device=values.device, dtype=torch.long)
    group_count = int(inv.max().item()) + 1 if inv.numel() > 0 else 0

    mask = response_mask.to(values.dtype)
    token_sums = (values * mask).sum(dim=-1)
    token_counts = mask.sum(dim=-1)

    group_sums = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    group_counts = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    group_sums.scatter_add_(0, inv, token_sums)
    group_counts.scatter_add_(0, inv, token_counts)
    group_means = group_sums / group_counts.clamp(min=1.0)

    return values - group_means[inv].unsqueeze(-1)


@register_policy_loss("entity_anchored_grouped_kl")  # type: ignore[arg-type]
def compute_policy_loss_entity_anchored_grouped_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
    # Entity-anchor specific tensors (passed from dp_actor.py via batch)
    anchor_positions: torch.Tensor | None = None,        # (B, max_K), -1 = pad
    anchor_group_ids: torch.Tensor | None = None,        # (B, max_K), -1 = pad
    anchor_prompt_ids: torch.Tensor | None = None,       # (B, max_K), -1 = pad — prompt-level
    segment_id_per_token: torch.Tensor | None = None,    # (B, R), -1 = past last anchor
    num_anchors: torch.Tensor | None = None,             # (B,)
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Entity-anchored future-KL with GiGPO group-norm and additive composition.

    Differs from FIPO's `future_kl` loss in 3 fundamental ways:
      1. Signal is aggregated AT ENTITY ANCHORS instead of per-token.
      2. Anchor signals are GROUP-NORMED across rollouts that share the same
         (prompt_uid, entity_id) — applying GiGPO's cross-rollout calibration
         to the FIPO-style policy-update direction signal.
      3. Composition is ADDITIVE (advantage += w × reward), allowing the signal
         to flip the advantage sign — UNLIKE FIPO's multiplicative modulator
         which can only modulate magnitude.

    Math (per rollout):
        δ_t = log π_θ(y_t) - log π_old(y_t)                         per-token raw signal
        A_m = Σ_{t in segment_m} γ^(e_m - t) × δ_t × mask           anchor backward γ-decay
        A_m_normalized = A_m - mean_over_group(A_m)                 GiGPO-style group-norm
        total_k = A_m_normalized_k + λ × total_{k+1}                cross-anchor forward decay
        reward[p] = total_{seg(p)}                                  per-token (segment-uniform)
        reward[p] = clamp(reward[p], -c, +c)                        magnitude clip
        if protect_negative and adv[p] < 0 and reward[p] > 0:
            reward[p] = 0                                           prevent flip on wrong rollouts
        weighted_adv = adv + w × reward                             additive composition

    Then standard PPO clip on weighted_adv.

    Required batch tensors (pre-computed by entity_info_only step reward computer):
        anchor_positions: (B, max_K) — token position of each anchor
        anchor_group_ids: (B, max_K) — globally unique group ID per anchor
        segment_id_per_token: (B, R) — which anchor each token belongs to
        num_anchors: (B,) — actual anchor count per rollout

    Config keys (all on `config.policy_loss`):
        anchor_decay_gamma   (float, default 0.978)  — within-segment backward decay
        anchor_decay_lambda  (float, default 0.5)    — across-anchor forward decay
        anchor_alpha         (float, default 1.0)    — reward scale
        anchor_reward_clip   (float, default 1.0)    — |reward| clip
        anchor_weight        (float, default 0.3)    — additive weight w
        anchor_protect_negative      (bool, default True)  — clip positive reward on negative-adv tokens
        anchor_size_1_strategy       (str, default 'skip') — skip / batch_mean
        anchor_groupnorm_with_std    (bool, default False) — divide by group std
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)

    # ---- Hyperparams ----
    γ = float(config.policy_loss.get("anchor_decay_gamma", 0.978))
    λ = float(config.policy_loss.get("anchor_decay_lambda", 0.5))
    α = float(config.policy_loss.get("anchor_alpha", 1.0))
    c_clip = float(config.policy_loss.get("anchor_reward_clip", 1.0))
    w = float(config.policy_loss.get("anchor_weight", 0.3))
    protect_neg = bool(config.policy_loss.get("anchor_protect_negative", True))
    # NEW: when True (default False for back-compat), also zero out NEGATIVE
    # reward on positive-advantage tokens. Combined with the existing
    # asymmetric "block positive reward on negative-advantage tokens", this
    # enforces sign(reward) == sign(advantage) — i.e. reward only modulates
    # magnitude, never flips. Eliminates the systematic rew_mean negative
    # bias caused by the asymmetric design.
    protect_symmetric = bool(config.policy_loss.get("anchor_protect_symmetric", False))
    size_1_strategy = str(config.policy_loss.get("anchor_size_1_strategy", "skip")).lower()
    groupnorm_with_std = bool(config.policy_loss.get("anchor_groupnorm_with_std", False))
    # NEW: composition mode for adding reward to advantage.
    #   "additive":       weighted_adv = advantages + w × reward            (current default)
    #   "multiplicative": weighted_adv = advantages × clamp(1 + w·tanh(r/s), 1-w_max, 1+w_max)
    # Multiplicative inherently cannot flip sign of advantage (FIPO-style
    # modulator). Best used with `protect_negative=False` since it's redundant.
    compose_mode = str(config.policy_loss.get("anchor_compose_mode", "additive")).lower()
    mult_scale = float(config.policy_loss.get("anchor_mult_scale", 1.0))
    mult_max = float(config.policy_loss.get("anchor_mult_max", 0.5))

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    # ---- Per-token signal ----
    negative_approx_kl = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    B, R = log_prob.shape
    device, dtype = log_prob.device, log_prob.dtype

    # ---- Fallback: if no entity tensors, behave like vanilla PPO ----
    have_anchors = (
        anchor_positions is not None
        and anchor_group_ids is not None
        and segment_id_per_token is not None
        and num_anchors is not None
    )

    if not have_anchors:
        weighted_adv = advantages
        reward_per_token = torch.zeros_like(advantages)
        anchor_metrics: dict[str, float] = {"anchor/no_anchor_data": 1.0}
    else:
        max_K = anchor_positions.shape[1]
        # Move to same device/dtype as log_prob.
        anchor_positions = anchor_positions.to(device=device).long()
        anchor_group_ids = anchor_group_ids.to(device=device).long()
        if anchor_prompt_ids is not None:
            anchor_prompt_ids = anchor_prompt_ids.to(device=device).long()
        segment_id_per_token = segment_id_per_token.to(device=device).long()
        num_anchors = num_anchors.to(device=device).long()

        # ---- Step 1: γ-weight per token = γ^(end_of_seg - position) ----
        # For tokens past the last anchor, segment_id = -1 → mark invalid.
        valid_seg_mask = (segment_id_per_token >= 0)                              # (B, R)
        seg_id_safe = segment_id_per_token.clamp(min=0)                            # (B, R)

        # End-of-segment position for each token: anchor_positions[i, seg_id[i, t]]
        anchor_pos_for_token = anchor_positions.gather(1, seg_id_safe)             # (B, R)
        pos_indices = torch.arange(R, device=device).unsqueeze(0).expand(B, R)     # (B, R)
        # γ^(distance), distance = anchor_pos - position (>= 0 within segment).
        distance = (anchor_pos_for_token - pos_indices).clamp(min=0)               # (B, R)
        γ_weight = torch.pow(torch.tensor(γ, device=device, dtype=dtype), distance.to(dtype))
        γ_weight = γ_weight * valid_seg_mask.to(dtype) * response_mask.to(dtype)
        weighted_δ = γ_weight * negative_approx_kl                                  # (B, R)

        # ---- Step 2: A_m = scatter_add(weighted_δ, seg_id) → (B, max_K) ----
        A = torch.zeros(B, max_K, device=device, dtype=dtype)
        # Only scatter at valid (seg >= 0) positions.
        # Use seg_id_safe; invalid positions multiply weighted_δ by 0 above.
        A.scatter_add_(1, seg_id_safe, weighted_δ)
        # Mask out anchor slots beyond num_anchors.
        anchor_idx = torch.arange(max_K, device=device).unsqueeze(0)               # (1, max_K)
        anchor_valid_mask = (anchor_idx < num_anchors.unsqueeze(1)) & (anchor_group_ids >= 0)  # (B, max_K)
        A = A * anchor_valid_mask.to(dtype)

        # ---- Step 3: GiGPO group-norm (cross-rank) ----
        # Group-norm needs to see ENOUGH rollouts per group to be statistically
        # meaningful. Within a single micro_batch (e.g. 2 rollouts/GPU), most
        # entities only appear in 1 of the 2 → group_size=1 → ``size_1_strategy=skip``
        # zeroes 99% of the signal. We fix this by all-reducing ``group_sum`` and
        # ``group_count`` across all ranks (DP world). With world_size=8 and
        # mini_batch=16, group-norm sees mini_batch × world_size = 128 rollouts
        # per backward step — large enough that high-frequency entities form
        # group_size 4-8 and the normalized signal is non-trivial.
        flat_A = A.flatten()                                                        # (B*max_K,)
        flat_group = anchor_group_ids.flatten()                                     # (B*max_K,)
        flat_valid = anchor_valid_mask.flatten()                                    # (B*max_K,)

        valid_group_ids = flat_group[flat_valid]
        valid_A = flat_A[flat_valid]

        # Detect distributed environment. We need to all-reduce only across the
        # data-parallel group; here we use the default world group (FSDP single
        # DP group). For Megatron with separate DP/TP/PP groups, this would need
        # to use the DP subgroup, but FSDP is the path that actually runs
        # entity_anchored_grouped_kl in this fork.
        try:
            import torch.distributed as _dist
            _dp_active = _dist.is_initialized() and _dist.get_world_size() > 1
        except ImportError:
            _dist = None
            _dp_active = False

        # First step: agree on the global ``max_group`` (max group_id + 1) across
        # ranks. Different ranks may see different max valid group_id, but all
        # ranks must allocate group_sum/count of the SAME size for all-reduce.
        local_max_g = (
            int(valid_group_ids.max().item()) + 1
            if valid_group_ids.numel() > 0
            else 0
        )
        if _dp_active:
            max_g_tensor = torch.tensor(local_max_g, device=device, dtype=torch.long)
            _dist.all_reduce(max_g_tensor, op=_dist.ReduceOp.MAX)
            max_group = int(max_g_tensor.item())
        else:
            max_group = local_max_g

        if max_group > 0:
            group_sum = torch.zeros(max_group, device=device, dtype=dtype)
            group_count = torch.zeros(max_group, device=device, dtype=dtype)
            if valid_group_ids.numel() > 0:
                group_sum.scatter_add_(0, valid_group_ids, valid_A)
                group_count.scatter_add_(
                    0, valid_group_ids, torch.ones_like(valid_A, dtype=dtype)
                )
            if groupnorm_with_std:
                group_sq = torch.zeros(max_group, device=device, dtype=dtype)
                if valid_group_ids.numel() > 0:
                    group_sq.scatter_add_(0, valid_group_ids, valid_A * valid_A)
            # Cross-rank reduction: SUM combines partial group stats from each
            # rank into a single global picture identical on all ranks. This is
            # autograd-friendly because all_reduce(SUM) is a linear op.
            if _dp_active:
                _dist.all_reduce(group_sum, op=_dist.ReduceOp.SUM)
                _dist.all_reduce(group_count, op=_dist.ReduceOp.SUM)
                if groupnorm_with_std:
                    _dist.all_reduce(group_sq, op=_dist.ReduceOp.SUM)
            group_mean = group_sum / group_count.clamp(min=1.0)
            if groupnorm_with_std:
                group_var = (group_sq / group_count.clamp(min=1.0)) - group_mean ** 2
                group_std = (group_var.clamp(min=1e-8)) ** 0.5
            else:
                group_std = None

            # Strategy for size-1 groups:
            #   skip:        zero out A_normalized (no signal)
            #   batch_mean:  use overall GLOBAL batch mean
            #   prompt_mean: use mean of all anchors of the SAME prompt (recommended)
            size_1_mask = group_count <= 1
            # Global batch mean derived from already-all-reduced group_sum/count.
            total_count = group_count.sum().clamp(min=1.0)
            batch_mean_value = group_sum.sum() / total_count

            # Prompt-level mean: per-prompt aggregate of all that prompt's anchors.
            # Uses the same all-reduce pattern as group_mean but keyed only by uid.
            anchor_prompt_mean_per: torch.Tensor | None = None
            if size_1_strategy == "prompt_mean" and anchor_prompt_ids is not None:
                flat_prompt = anchor_prompt_ids.flatten()
                valid_prompt_ids = flat_prompt[flat_valid]
                local_max_p = (
                    int(valid_prompt_ids.max().item()) + 1
                    if valid_prompt_ids.numel() > 0
                    else 0
                )
                if _dp_active:
                    max_p_tensor = torch.tensor(local_max_p, device=device, dtype=torch.long)
                    _dist.all_reduce(max_p_tensor, op=_dist.ReduceOp.MAX)
                    max_prompt = int(max_p_tensor.item())
                else:
                    max_prompt = local_max_p
                if max_prompt > 0:
                    prompt_sum = torch.zeros(max_prompt, device=device, dtype=dtype)
                    prompt_count = torch.zeros(max_prompt, device=device, dtype=dtype)
                    if valid_prompt_ids.numel() > 0:
                        prompt_sum.scatter_add_(0, valid_prompt_ids, valid_A)
                        prompt_count.scatter_add_(
                            0, valid_prompt_ids, torch.ones_like(valid_A, dtype=dtype)
                        )
                    if _dp_active:
                        _dist.all_reduce(prompt_sum, op=_dist.ReduceOp.SUM)
                        _dist.all_reduce(prompt_count, op=_dist.ReduceOp.SUM)
                    prompt_mean = prompt_sum / prompt_count.clamp(min=1.0)
                    anchor_prompt_safe = torch.where(
                        anchor_valid_mask,
                        anchor_prompt_ids.clamp(min=0),
                        torch.zeros_like(anchor_prompt_ids),
                    )
                    anchor_prompt_mean_per = prompt_mean[anchor_prompt_safe]

            # Center each anchor by its group mean.
            # Defensive: invalid slots may have anchor_group_ids = -1 (clamped to 0
            # by .clamp(min=0)), but valid-but-out-of-mask slots could in theory
            # carry a group_id >= max_group. We mask those to 0 here so the
            # subsequent fancy-index never goes out of bounds; the result is then
            # zeroed by ``anchor_valid_mask`` anyway.
            anchor_group_safe = torch.where(
                anchor_valid_mask,
                anchor_group_ids.clamp(min=0),
                torch.zeros_like(anchor_group_ids),
            )
            anchor_mean_per = group_mean[anchor_group_safe] * anchor_valid_mask.to(dtype)
            A_centered = (A - anchor_mean_per) * anchor_valid_mask.to(dtype)

            if groupnorm_with_std:
                anchor_std_per = group_std[anchor_group_safe] * anchor_valid_mask.to(dtype)
                A_centered = A_centered / anchor_std_per.clamp(min=1e-8)

            # Apply size-1 strategy.
            anchor_count_per = group_count[anchor_group_safe] * anchor_valid_mask.to(dtype)
            is_size_1 = (anchor_count_per <= 1) & anchor_valid_mask
            if size_1_strategy == "skip":
                # Zero out A_centered where group size <= 1.
                A_normalized = torch.where(is_size_1, torch.zeros_like(A_centered), A_centered)
            elif size_1_strategy == "prompt_mean" and anchor_prompt_mean_per is not None:
                # Re-center size-1 anchors using SAME-PROMPT mean (across that
                # prompt's rollouts and all entities). More meaningful than
                # batch_mean because it removes per-prompt difficulty bias.
                A_prompt_centered = (A - anchor_prompt_mean_per) * anchor_valid_mask.to(dtype)
                A_normalized = torch.where(is_size_1, A_prompt_centered, A_centered)
            else:  # "batch_mean" (also fallback for prompt_mean if anchor_prompt_ids missing)
                A_batch_centered = (A - batch_mean_value) * anchor_valid_mask.to(dtype)
                A_normalized = torch.where(is_size_1, A_batch_centered, A_centered)

            num_groups_total = int(max_group)
            num_size_1 = int(size_1_mask.sum().item())
            avg_group_size = float(group_count[group_count > 0].mean().item()) if num_groups_total > 0 else 0.0
        else:
            A_normalized = torch.zeros_like(A)
            num_groups_total = 0
            num_size_1 = 0
            avg_group_size = 0.0

        # ---- Step 4: Forward λ-decay across anchors ----
        # total_k = A_normalized_k + λ × total_{k+1}
        # Iterate from rightmost to leftmost.
        total = torch.zeros_like(A_normalized)
        # Initialize from the rightmost anchor.
        total[:, max_K - 1] = A_normalized[:, max_K - 1]
        for k in range(max_K - 2, -1, -1):
            total[:, k] = A_normalized[:, k] + λ * total[:, k + 1]
        # Mask beyond num_anchors.
        total = total * anchor_valid_mask.to(dtype)

        # ---- Step 5: scatter total to per-token reward ----
        reward_per_token = total.gather(1, seg_id_safe) * valid_seg_mask.to(dtype)  # (B, R)

        # ---- Step 6: Apply alpha and clip ----
        reward_per_token = reward_per_token * α
        reward_per_token = reward_per_token.clamp(min=-c_clip, max=c_clip)

        # ---- Step 7: Negative-advantage protection ----
        if protect_neg:
            # Where advantage < 0, do not allow positive reward (would lessen penalty too much
            # or flip the sign). Allow negative reward to deepen the penalty.
            neg_adv = advantages < 0
            reward_per_token = torch.where(
                neg_adv & (reward_per_token > 0),
                torch.zeros_like(reward_per_token),
                reward_per_token,
            )
        if protect_symmetric:
            # Symmetric counterpart: where advantage > 0, also block NEGATIVE reward
            # so reward sign is always aligned with advantage sign. This fixes the
            # rew_mean systematic negative bias caused by the asymmetric default.
            pos_adv = advantages > 0
            reward_per_token = torch.where(
                pos_adv & (reward_per_token < 0),
                torch.zeros_like(reward_per_token),
                reward_per_token,
            )

        # Apply response mask so reward outside response is 0.
        reward_per_token = reward_per_token * response_mask.to(dtype)

        # ---- Step 8: Composition (additive or multiplicative) ----
        if compose_mode == "multiplicative":
            # weighted_adv = advantages × clamp(1 + w·tanh(reward/scale), 1-w_max, 1+w_max)
            # Cannot flip sign of advantages — only modulates magnitude. Useful when
            # outcome reward should remain the dominant direction signal.
            modulator = torch.tanh(reward_per_token / max(mult_scale, 1e-8))
            modulator = (1.0 + w * modulator).clamp(min=1.0 - mult_max, max=1.0 + mult_max)
            weighted_adv = advantages * modulator
        else:  # "additive" (default)
            weighted_adv = advantages + w * reward_per_token

        # ---- Stats ----
        # Cross-rank diagnostics: distinguish whether group-norm now sees global
        # stats vs. just local. Key signals to verify the fix is live:
        #   - dp_active: 1 means we took the all-reduce path (cross-rank)
        #   - non_empty_groups: count of groups that received any anchor data
        #   - max_group_size_observed: largest group; should be > 1 if cross-rank works
        #   - size_1_groups_frac: fraction of (non-empty) groups that are still size 1
        #     — was 99% before fix; should drop substantially with cross-rank stats
        #   - global_total_anchors: sum of group_count across all groups (post-allreduce);
        #     should equal world_size × local_anchor_count
        if max_group > 0:
            non_empty_mask = group_count > 0
            non_empty_groups = int(non_empty_mask.sum().item())
            max_group_size_observed = (
                int(group_count.max().item()) if non_empty_groups > 0 else 0
            )
            size_1_among_nonempty = int(((group_count > 0) & (group_count <= 1)).sum().item())
            size_1_frac = (
                float(size_1_among_nonempty) / float(non_empty_groups)
                if non_empty_groups > 0 else 0.0
            )
            global_total_anchors = float(group_count.sum().item())
            local_anchor_count = float(valid_A.numel())
        else:
            non_empty_groups = 0
            max_group_size_observed = 0
            size_1_frac = 0.0
            global_total_anchors = 0.0
            local_anchor_count = 0.0

        anchor_metrics = {
            "anchor/A_raw_mean": float(A[anchor_valid_mask].mean().item()) if anchor_valid_mask.any() else 0.0,
            "anchor/A_raw_std": float(A[anchor_valid_mask].std().item()) if anchor_valid_mask.any() else 0.0,
            "anchor/A_normalized_mean": float(A_normalized[anchor_valid_mask].mean().item()) if anchor_valid_mask.any() else 0.0,
            "anchor/A_normalized_std": float(A_normalized[anchor_valid_mask].std().item()) if anchor_valid_mask.any() else 0.0,
            "anchor/total_per_anchor_mean": float(total[anchor_valid_mask].mean().item()) if anchor_valid_mask.any() else 0.0,
            "anchor/reward_mean": float(verl_F.masked_mean(reward_per_token, response_mask).item()),
            "anchor/reward_std": float((reward_per_token[response_mask.bool()]).std().item()) if response_mask.any() else 0.0,
            "anchor/reward_min": float(reward_per_token[response_mask.bool()].min().item()) if response_mask.any() else 0.0,
            "anchor/reward_max": float(reward_per_token[response_mask.bool()].max().item()) if response_mask.any() else 0.0,
            "anchor/reward_clip_high_frac": float(((reward_per_token >= c_clip - 1e-7) & response_mask.bool()).float().mean().item()),
            "anchor/reward_clip_low_frac": float(((reward_per_token <= -c_clip + 1e-7) & response_mask.bool()).float().mean().item()),
            "anchor/num_groups": float(num_groups_total),
            "anchor/avg_group_size": avg_group_size,
            "anchor/size_1_groups": float(num_size_1),
            # NEW cross-rank diagnostics
            "anchor/dp_active": 1.0 if _dp_active else 0.0,
            "anchor/non_empty_groups": float(non_empty_groups),
            "anchor/max_group_size_observed": float(max_group_size_observed),
            "anchor/size_1_among_nonempty_frac": size_1_frac,
            "anchor/global_total_anchors": global_total_anchors,
            "anchor/local_anchor_count": local_anchor_count,
            "anchor/weighted_adv_mean": float(verl_F.masked_mean(weighted_adv, response_mask).item()),
            "anchor/weighted_adv_std": float(weighted_adv[response_mask.bool()].std().item()) if response_mask.any() else 0.0,
            "anchor/sign_flip_frac": float(
                ((weighted_adv * advantages < 0) & response_mask.bool()).float().mean().item()
            ),
            # NEW: distinguish compose_mode (1.0=multiplicative, 0.0=additive)
            "anchor/compose_multiplicative": 1.0 if compose_mode == "multiplicative" else 0.0,
            "anchor/protect_symmetric_on": 1.0 if protect_symmetric else 0.0,
        }
        if protect_neg:
            anchor_metrics["anchor/protect_neg_active_frac"] = float(
                ((advantages < 0) & response_mask.bool()).float().mean().item()
            )
        if protect_symmetric:
            anchor_metrics["anchor/protect_sym_active_frac"] = float(
                ((advantages > 0) & response_mask.bool()).float().mean().item()
            )

    # ---- Standard PPO clip on weighted_adv ----
    pg_losses1 = -weighted_adv * ratio
    pg_losses2 = -weighted_adv * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_adv * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (weighted_adv < 0).float(), response_mask
    )

    pg_losses = torch.where(weighted_adv < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    pg_metrics.update(anchor_metrics)
    return pg_loss, pg_metrics


@register_policy_loss("mgpo")
def compute_policy_loss_mgpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MGPO policy loss: PPO-clipped ratio weighted by implicit-reward-adjusted advantages.

    Core formula:
        w_{i,t}   = A_centered_{i,t} - R_{i,t}     (R = β·log(π_θ/π_ref), NOT centered)
        ratio     = π_θ / π_old
        L         = -w · clip(ratio, 1-ε, 1+ε)      (standard PPO clip on the ratio)

    Key design choices vs. the previous version:
    1. Implicit reward R is NOT group-centered. Its per-token mean carries the global
       KL penalty: when the policy drifts from π_ref, R shifts uniformly, pulling w
       back and dampening updates. Centering R used to erase this signal.
    2. Uses the PPO clipped-ratio objective instead of raw weighted log-likelihood.
       The old form  -w'·log π_θ  caused systematic entropy drift because the
       cross-term E[w'·log π_θ] ≠ 0 even when E[w'] = 0.
    3. sech² gating is replaced by PPO clipping, which works reliably even when the
       within-epoch ratio barely deviates from 1 (soft sech² gate was ≈ 1 always).
    """
    assert config is not None
    assert isinstance(config, ActorConfig)

    if ref_log_prob is None:
        raise ValueError("MGPO policy loss requires ref_log_prob")

    mgpo_beta = float(config.policy_loss.get("mgpo_beta", 0.01))
    mgpo_max_w = float(config.policy_loss.get("mgpo_max_w", 5.0))
    mgpo_max_kl = float(config.policy_loss.get("mgpo_max_kl", 10.0))
    mgpo_max_entropy_ratio = float(config.policy_loss.get("mgpo_max_entropy_ratio", 3.0))

    clip_ratio = config.clip_ratio  # PPO ε, default 0.2

    # --- Importance ratio ---
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # --- Entropy / KL monitoring ---
    with torch.no_grad():
        cur_entropy = verl_F.masked_mean(-log_prob, response_mask).item()
        old_entropy = verl_F.masked_mean(-old_log_prob, response_mask).item()
        entropy_ratio = cur_entropy / max(old_entropy, 1e-6)
    kl_val = ppo_kl.detach().item()

    # --- Implicit reward (uncentered — preserves global KL signal) ---
    implicit_reward = (mgpo_beta * (log_prob - ref_log_prob)).detach()

    # --- Weight: advantages come pre-centered from GiGPO/GRPO (reward - group_mean over
    # the full n-rollout group). Do NOT re-center here: a per-micro-batch re-centering
    # (_center_by_uid_masked) subtracts the *subset* mean over the 2 samples in the
    # current micro-batch, which wipes out the signal whenever both samples fall in the
    # same correctness class (~79% of micro-batches for a 30%-accurate model), leaving
    # w ≈ 0 almost everywhere.
    w = (advantages.detach() - implicit_reward).detach()
    w = w.clamp(-mgpo_max_w, mgpo_max_w)

    # --- PPO clipped surrogate with MGPO weights ---
    pg_losses1 = -w * ratio
    pg_losses2 = -w * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    # For w > 0 (want to increase ratio): take the max (pessimistic)
    # For w < 0 (want to decrease ratio): take the max (pessimistic)
    # This is the standard PPO clip logic: max(−w·r, −w·clip(r))
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # --- Clip fraction (for monitoring) ---
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    if kl_val > mgpo_max_kl:
        print(f"[MGPO Safety] KL elevated: {kl_val:.4f} > {mgpo_max_kl}")
    if entropy_ratio > mgpo_max_entropy_ratio:
        print(f"[MGPO Safety] Entropy elevated: ratio={entropy_ratio:.2f} > {mgpo_max_entropy_ratio}")

    # --- Metrics ---
    valid_mask = response_mask.to(torch.bool)
    valid_w = torch.masked_select(w, valid_mask)
    valid_impl = torch.masked_select(implicit_reward, valid_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": torch.tensor(0.0).item(),
        "actor/mgpo/w_mean": valid_w.mean().detach().item(),
        "actor/mgpo/w_std": valid_w.std().detach().item(),
        "actor/mgpo/w_min": valid_w.min().detach().item(),
        "actor/mgpo/w_max": valid_w.max().detach().item(),
        "actor/mgpo/w_prime_mean": valid_w.mean().detach().item(),
        "actor/mgpo/w_prime_std": valid_w.std().detach().item(),
        "actor/mgpo/implicit_reward_mean": valid_impl.mean().detach().item(),
        "actor/mgpo/implicit_reward_std": valid_impl.std().detach().item(),
        "actor/mgpo/entropy_ratio": entropy_ratio,
        "actor/mgpo/kl_val": kl_val,
        "actor/mgpo/gate_mean": 1.0,
        "actor/mgpo/gate_min": 1.0,
        "actor/mgpo/ratio_clipped_frac": pg_clipfrac.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.detach().cpu().numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    ref_log_prob: torch.Tensor | None = None,
    uid: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level ("token", "sequence", "geometric", or None)
        rollout_rs_threshold: Upper threshold for rejection sampling
        rollout_rs_threshold_lower: Lower threshold for rejection sampling
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
                rollout_rs_threshold_lower=rollout_rs_threshold_lower,
                rollout_token_veto_threshold=rollout_token_veto_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


# ============================================================================
# GiGPO (Grouped Importance-weighted Generalized Policy Optimization) Functions
# ============================================================================

# Global variable to store the latest GiGPO statistics for monitoring
_last_gigpo_stats = {}


def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    traj_index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    compute_mean_std_cross_all_data: bool = True,
    advantage_scope: str = "token",
):
    """
    基于 episode 的均值/方差归一化，用于 GiGPO。
    token_level_rewards: (bs, response_length)
    response_mask:       (bs, response_length)
    index:               (bs,)
    traj_index:          (bs,)
    """
    response_length = token_level_rewards.shape[-1]
    # Use only the final valid response token as outcome reward.
    # In graph reward manager, correctness reward is written to the last valid token.
    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
    scores = torch.zeros_like(valid_lengths, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    valid_rows = valid_lengths > 0
    if torch.any(valid_rows):
        row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
        col_indices = valid_lengths[valid_rows] - 1
        scores[valid_rows] = token_level_rewards[row_indices, col_indices]

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_all_data:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                stack = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(stack)
                id2std[idx] = torch.std(stack)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        if advantage_scope == "token":
            episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        elif advantage_scope == "terminal":
            # Only apply episode-level advantage to the final valid response token.
            episode_advantages = torch.zeros_like(response_mask, dtype=scores.dtype, device=scores.device)
            if torch.any(valid_rows):
                row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
                col_indices = valid_lengths[valid_rows] - 1
                episode_advantages[row_indices, col_indices] = scores[valid_rows]
        else:
            raise ValueError(f"Unknown advantage_scope: {advantage_scope}")

    return episode_advantages


def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def build_step_group(anchor_obs: np.ndarray, index: np.ndarray):
    """
    Group flattened step-level observations by exact match within each prompt group.

    For each unique prompt index, observations with identical hashable representations
    are assigned the same UUID. This replaces the previous union-find approach with
    precise matching — only steps observing exactly the same entity get grouped.

    Args:
        anchor_obs: 1-D array of shape (N,) containing entity IDs (already flattened).
        index: 1-D array of shape (N,) indicating which prompt each step belongs to.

    Returns:
        step_group_uids: 1-D object array of shape (N,) with UUID strings.
    """
    from uuid import uuid4

    assert len(anchor_obs) == len(index), "len(anchor_obs) != len(index)"
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    unique_indices = np.unique(index)

    group_sizes = []
    for idx in unique_indices:
        positions = np.where(index == idx)[0]
        obs_to_uid = {}  # hashable_obs -> uid
        obs_to_count = defaultdict(int)

        for pos in positions:
            key = to_hashable(anchor_obs[pos])
            if key not in obs_to_uid:
                obs_to_uid[key] = str(uuid4())
            step_group_uids[pos] = obs_to_uid[key]
            obs_to_count[key] += 1

        group_sizes.extend(obs_to_count.values())

    if None in step_group_uids or np.any(step_group_uids is None):
        missing = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing}")

    if group_sizes:
        n_groups = len(group_sizes)
        avg_size = np.mean(group_sizes)
        single_count = sum(1 for s in group_sizes if s == 1)
        print(f"build_step_group: {n_groups} groups, avg_size={avg_size:.2f}, "
              f"single_groups={single_count}/{n_groups} ({single_count/n_groups*100:.1f}%)")
    return step_group_uids


def step2episode_scalar(index: np.ndarray, step_advantages: torch.Tensor, return_inverse: bool = False):
    """
    将 step_advantages 按 UUID 进行 mean 聚合，返回 [num_uuid]。
    """
    # 确保输入为 1D，并与 index 对齐
    step_advantages = step_advantages.reshape(-1)
    index = np.asarray(index).reshape(-1)
    if index.shape[0] != step_advantages.shape[0]:
        raise ValueError(f"index 和 step_advantages 的长度必须相同，got {index.shape[0]} vs {step_advantages.shape[0]}")

    unique_uuids, inverse_indices = np.unique(index, return_inverse=True)
    n = len(unique_uuids)
    device = step_advantages.device
    inverse_indices_tensor = torch.from_numpy(inverse_indices).to(device)
    if not torch.is_floating_point(step_advantages):
        step_advantages = step_advantages.float()
    sums = torch.bincount(inverse_indices_tensor, weights=step_advantages, minlength=n)
    counts = torch.bincount(inverse_indices_tensor, minlength=n)
    mean_advantages = sums / counts.clamp(min=1)
    
    if return_inverse:
        return mean_advantages, inverse_indices_tensor
    return mean_advantages


def _normalize_token_positions_for_step_adv(pos_spec):
    if pos_spec is None:
        return []
    if isinstance(pos_spec, np.ndarray):
        if pos_spec.ndim == 0:
            return [int(pos_spec.item())]
        return [int(x) for x in pos_spec.tolist()]
    if isinstance(pos_spec, (list, tuple)):
        if len(pos_spec) >= 2 and isinstance(pos_spec[0], str):
            kind = pos_spec[0].lower()
            if kind == "prefix":
                end = max(0, int(pos_spec[1]))
                return range(0, end + 1)
            if kind == "range" and len(pos_spec) >= 3:
                start = max(0, int(pos_spec[1]))
                end = max(start, int(pos_spec[2]))
                return range(start, end + 1)
        return [int(x) for x in pos_spec]
    return [int(pos_spec)]


def _position_spec_as_sequence(pos_spec):
    if isinstance(pos_spec, np.ndarray):
        if pos_spec.ndim == 0:
            return pos_spec.item()
        return pos_spec.tolist()
    return pos_spec


def _compact_range_for_step_adv(pos_spec):
    pos_spec = _position_spec_as_sequence(pos_spec)
    if isinstance(pos_spec, (list, tuple)) and len(pos_spec) >= 2 and isinstance(pos_spec[0], str):
        kind = pos_spec[0].lower()
        if kind == "prefix":
            end = max(0, int(pos_spec[1]))
            return 0, end
        if kind == "range" and len(pos_spec) >= 3:
            start = max(0, int(pos_spec[1]))
            end = max(start, int(pos_spec[2]))
            return start, end
    return None


def _backward_decay_range_for_step_adv(pos_spec):
    """Return (prefix_start, action_start, action_end, decay, local_alpha)."""
    pos_spec = _position_spec_as_sequence(pos_spec)
    if isinstance(pos_spec, (list, tuple)) and len(pos_spec) >= 6 and isinstance(pos_spec[0], str):
        kind = pos_spec[0].lower()
        if kind == "backward_decay_range":
            prefix_start = max(0, int(pos_spec[1]))
            action_start = max(prefix_start, int(pos_spec[2]))
            action_end = max(action_start, int(pos_spec[3]))
            decay = max(1e-6, min(1.0, float(pos_spec[4])))
            local_alpha = max(0.0, min(1.0, float(pos_spec[5])))
            return prefix_start, action_start, action_end, decay, local_alpha
    return None


def _decay_prefix_for_step_adv(pos_spec):
    pos_spec = _position_spec_as_sequence(pos_spec)
    if isinstance(pos_spec, (list, tuple)) and len(pos_spec) >= 3 and isinstance(pos_spec[0], str):
        kind = pos_spec[0].lower()
        if kind == "decay_prefix":
            end = max(0, int(pos_spec[1]))
            decay = max(1e-6, min(1.0, float(pos_spec[2])))
            return end, decay
    return None


def _put_max_abs_token(out: torch.Tensor, seen: torch.Tensor, row: int, col: int, value: float):
    current = out[row, col]
    if (not bool(seen[row, col])) or abs(float(value)) > abs(float(current.detach().item())):
        out[row, col] = torch.tensor(float(value), dtype=out.dtype, device=out.device)
        seen[row, col] = True


def _scatter_entity_step_scores_max_abs(scores, response_mask, sample_indices, token_positions):
    bs, response_length = response_mask.shape
    device = response_mask.device
    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).cpu().numpy()
    step_advantages = torch.zeros(bs, response_length, dtype=scores.dtype, device=device)
    seen = torch.zeros(bs, response_length, dtype=torch.bool, device=device)

    def put_range(row: int, start: int, end: int, value: float):
        valid_len = int(valid_lengths[row])
        if valid_len <= 0:
            return
        start = max(0, min(int(start), response_length - 1, valid_len - 1))
        end = max(0, min(int(end), response_length - 1, valid_len - 1))
        if start > end:
            return
        for col in range(start, end + 1):
            _put_max_abs_token(step_advantages, seen, row, col, value)

    for i in range(len(scores)):
        row = int(sample_indices[i])
        if row < 0 or row >= bs:
            continue
        valid_len = int(valid_lengths[row])
        if valid_len <= 0:
            continue
        value = float(scores[i].detach().item())
        pos_spec = token_positions[i]

        backward_decay = _backward_decay_range_for_step_adv(pos_spec)
        if backward_decay is not None:
            prefix_start, action_start, action_end, decay, local_alpha = backward_decay
            prefix_start = max(0, min(int(prefix_start), response_length - 1, valid_len - 1))
            action_start = max(0, min(int(action_start), response_length - 1, valid_len - 1))
            action_end = max(0, min(int(action_end), response_length - 1, valid_len - 1))
            if action_start <= action_end:
                action_value = value * float(local_alpha)
                put_range(row, action_start, action_end, action_value)
                prefix_len = max(0, action_start - prefix_start)
                if prefix_len > 0:
                    action_len = max(1.0, float(action_end - action_start + 1))
                    prefix_budget = value * (1.0 - float(local_alpha)) * action_len
                    decay = max(1e-6, min(1.0, float(decay)))
                    if decay >= 1.0 - 1e-8:
                        denom = float(prefix_len)
                    else:
                        denom = decay * (1.0 - decay ** float(prefix_len)) / max(1e-8, 1.0 - decay)
                    denom = max(1e-8, denom)
                    for distance in range(1, prefix_len + 1):
                        col = action_start - distance
                        prefix_value = prefix_budget * (decay ** distance) / denom
                        _put_max_abs_token(step_advantages, seen, row, col, prefix_value)
            continue

        decay_prefix = _decay_prefix_for_step_adv(pos_spec)
        if decay_prefix is not None:
            end, decay = decay_prefix
            end = max(0, min(int(end), response_length - 1, valid_len - 1))
            decay = max(1e-6, min(1.0, float(decay)))
            for col in range(0, end + 1):
                _put_max_abs_token(step_advantages, seen, row, col, value * (decay ** (end - col)))
            continue

        compact_range = _compact_range_for_step_adv(pos_spec)
        if compact_range is not None:
            start, end = compact_range
            put_range(row, start, end, value)
            continue

        for col in _normalize_token_positions_for_step_adv(pos_spec):
            if 0 <= int(col) < response_length and int(col) < valid_len:
                _put_max_abs_token(step_advantages, seen, row, int(col), value)

    return step_advantages


def _scatter_entity_step_scores(scores, response_mask, sample_indices, token_positions, pool_mode: str = "mean"):
    pool_mode = str(pool_mode or "mean").lower().replace("-", "_")
    if pool_mode in ("max", "max_abs", "maxabs"):
        return _scatter_entity_step_scores_max_abs(scores, response_mask, sample_indices, token_positions)
    if pool_mode != "mean":
        raise ValueError(f"Unsupported step scatter pool mode: {pool_mode}")

    bs, response_length = response_mask.shape
    device = response_mask.device
    valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1).cpu().numpy()
    step_advantages = torch.zeros(bs, response_length, dtype=scores.dtype, device=device)
    step_counts = torch.zeros_like(step_advantages)

    range_rows = []
    range_starts = []
    range_ends = []
    range_score_idxs = []
    point_rows = []
    point_cols = []
    point_score_idxs = []
    decay_by_value = defaultdict(list)
    backward_rows = []
    backward_prefix_starts = []
    backward_action_starts = []
    backward_action_ends = []
    backward_decays = []
    backward_local_alphas = []
    backward_score_idxs = []

    for i in range(len(scores)):
        r = int(sample_indices[i])
        valid_len = int(valid_lengths[r])
        if valid_len <= 0:
            continue
        backward_decay = _backward_decay_range_for_step_adv(token_positions[i])
        if backward_decay is not None:
            prefix_start, action_start, action_end, decay, local_alpha = backward_decay
            prefix_start = max(0, min(prefix_start, response_length - 1, valid_len - 1))
            action_start = max(0, min(action_start, response_length - 1, valid_len - 1))
            action_end = max(0, min(action_end, response_length - 1, valid_len - 1))
            if action_start <= action_end:
                backward_rows.append(r)
                backward_prefix_starts.append(min(prefix_start, action_start))
                backward_action_starts.append(action_start)
                backward_action_ends.append(action_end)
                backward_decays.append(decay)
                backward_local_alphas.append(local_alpha)
                backward_score_idxs.append(i)
            continue
        decay_prefix = _decay_prefix_for_step_adv(token_positions[i])
        if decay_prefix is not None:
            end, decay = decay_prefix
            end = max(0, min(end, response_length - 1, valid_len - 1))
            decay_by_value[decay].append((r, end, i))
            continue
        compact_range = _compact_range_for_step_adv(token_positions[i])
        if compact_range is not None:
            start, end = compact_range
            start = max(0, min(start, response_length - 1, valid_len - 1))
            end = max(0, min(end, response_length - 1, valid_len - 1))
            if start <= end:
                range_rows.append(r)
                range_starts.append(start)
                range_ends.append(end)
                range_score_idxs.append(i)
            continue

        for c in _normalize_token_positions_for_step_adv(token_positions[i]):
            if 0 <= c < response_length and c < valid_len:
                point_rows.append(r)
                point_cols.append(int(c))
                point_score_idxs.append(i)

    if range_rows:
        width = response_length + 1
        diff_scores = torch.zeros(bs * width, dtype=scores.dtype, device=device)
        diff_counts = torch.zeros_like(diff_scores)
        rows_t = torch.tensor(range_rows, dtype=torch.long, device=device)
        starts_t = torch.tensor(range_starts, dtype=torch.long, device=device)
        ends_t = torch.tensor(range_ends, dtype=torch.long, device=device)
        idx_t = torch.tensor(range_score_idxs, dtype=torch.long, device=device)
        vals = scores[idx_t]
        start_flat = rows_t * width + starts_t
        end_flat = rows_t * width + (ends_t + 1)
        diff_scores.scatter_add_(0, start_flat, vals)
        diff_scores.scatter_add_(0, end_flat, -vals)
        ones = torch.ones_like(vals)
        diff_counts.scatter_add_(0, start_flat, ones)
        diff_counts.scatter_add_(0, end_flat, -ones)
        step_advantages += diff_scores.view(bs, width).cumsum(dim=-1)[:, :response_length]
        step_counts += diff_counts.view(bs, width).cumsum(dim=-1)[:, :response_length]

    if backward_rows:
        width = response_length + 1
        rows_t = torch.tensor(backward_rows, dtype=torch.long, device=device)
        prefix_t = torch.tensor(backward_prefix_starts, dtype=torch.long, device=device)
        starts_t = torch.tensor(backward_action_starts, dtype=torch.long, device=device)
        ends_t = torch.tensor(backward_action_ends, dtype=torch.long, device=device)
        idx_t = torch.tensor(backward_score_idxs, dtype=torch.long, device=device)
        decays_t = torch.tensor(backward_decays, dtype=scores.dtype, device=device)
        alphas_t = torch.tensor(backward_local_alphas, dtype=scores.dtype, device=device)
        vals = scores[idx_t]

        action_scores = vals * alphas_t
        diff_scores = torch.zeros(bs * width, dtype=scores.dtype, device=device)
        diff_counts = torch.zeros_like(diff_scores)
        start_flat = rows_t * width + starts_t
        end_flat = rows_t * width + (ends_t + 1)
        diff_scores.scatter_add_(0, start_flat, action_scores)
        diff_scores.scatter_add_(0, end_flat, -action_scores)
        ones = torch.ones_like(action_scores)
        diff_counts.scatter_add_(0, start_flat, ones)
        diff_counts.scatter_add_(0, end_flat, -ones)
        step_advantages += diff_scores.view(bs, width).cumsum(dim=-1)[:, :response_length]
        step_counts += diff_counts.view(bs, width).cumsum(dim=-1)[:, :response_length]

        prefix_lens = (starts_t - prefix_t).clamp(min=0)
        if bool(torch.any(prefix_lens > 0)):
            action_lens = (ends_t - starts_t + 1).to(dtype=scores.dtype).clamp(min=1.0)
            prefix_budget = vals * (1.0 - alphas_t) * action_lens
            decay_is_one = decays_t >= (1.0 - 1e-8)
            prefix_lens_f = prefix_lens.to(dtype=scores.dtype)
            denom = torch.where(
                decay_is_one,
                prefix_lens_f.clamp(min=1.0),
                decays_t * (1.0 - torch.pow(decays_t, prefix_lens_f)) / (1.0 - decays_t).clamp(min=1e-8),
            ).clamp(min=1e-8)
            max_prefix_len = int(prefix_lens.max().detach().item())
            flat_view = step_advantages.view(-1)
            count_view = step_counts.view(-1)
            for distance in range(1, max_prefix_len + 1):
                active = prefix_lens >= distance
                if not bool(torch.any(active)):
                    continue
                cols = starts_t[active] - distance
                rows_active = rows_t[active]
                decay_pow = torch.pow(decays_t[active], distance)
                prefix_vals = prefix_budget[active] * decay_pow / denom[active]
                flat_idx = rows_active * response_length + cols
                flat_view.scatter_add_(0, flat_idx, prefix_vals)
                count_view.scatter_add_(0, flat_idx, torch.ones_like(prefix_vals))

    for decay, entries in decay_by_value.items():
        bins_scores = torch.zeros(bs, response_length, dtype=scores.dtype, device=device)
        bins_counts = torch.zeros_like(bins_scores)
        rows_t = torch.tensor([x[0] for x in entries], dtype=torch.long, device=device)
        ends_t = torch.tensor([x[1] for x in entries], dtype=torch.long, device=device)
        idx_t = torch.tensor([x[2] for x in entries], dtype=torch.long, device=device)
        flat_idx = rows_t * response_length + ends_t
        bins_scores.view(-1).scatter_add_(0, flat_idx, scores[idx_t])
        bins_counts.view(-1).scatter_add_(0, flat_idx, torch.ones(len(entries), dtype=scores.dtype, device=device))
        running_scores = torch.zeros(bs, dtype=scores.dtype, device=device)
        running_counts = torch.zeros_like(running_scores)
        decay_f = float(decay)
        for c in range(response_length - 1, -1, -1):
            running_scores = bins_scores[:, c] + decay_f * running_scores
            running_counts = bins_counts[:, c] + decay_f * running_counts
            step_advantages[:, c] += running_scores
            step_counts[:, c] += running_counts

    if point_rows:
        rows_t = torch.tensor(point_rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(point_cols, dtype=torch.long, device=device)
        idx_t = torch.tensor(point_score_idxs, dtype=torch.long, device=device)
        flat_idx = rows_t * response_length + cols_t
        step_advantages.view(-1).scatter_add_(0, flat_idx, scores[idx_t])
        step_counts.view(-1).scatter_add_(0, flat_idx, torch.ones(len(point_rows), dtype=scores.dtype, device=device))

    return torch.where(
        step_counts > 0,
        step_advantages / step_counts.clamp(min=1e-8),
        step_advantages,
    )


def step_norm_reward(
    flat_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    sample_indices: np.ndarray = None,
    token_positions: np.ndarray = None,
    advantage_scope: str = "token",
    length_factors: Optional[torch.Tensor] = None,
    single_step_fallback_weight: float = 0.0,
    scatter_pool_mode: str = "mean",
):
    """
    Normalize flat step-level rewards by group and scatter back to (bs, response_length).

    Args:
        flat_rewards: (N,) 1-D tensor of valid step rewards (already flattened).
        response_mask: (bs, response_length) token-level mask.
        step_group_uids: (N,) array of group UIDs from build_step_group.
        epsilon: small constant for numerical stability.
        remove_std: if True, only subtract mean; otherwise also divide by std.
        sample_indices: (N,) which sample (row) each flat entry belongs to.
        token_positions: (N,) which token column each flat entry maps to.
        advantage_scope: "entity" — place each step score at its token position;
                         "token" — average step scores per sample, broadcast to all tokens.
        length_factors: optional (bs,) float tensor of per-sample multipliers applied to
            ``scores`` AFTER group normalization. Used to compensate for the fact that longer
            responses see their step reward diluted by the token-mean PPO loss denominator.
            Set to None or an all-ones tensor to disable.
        single_step_fallback_weight: fallback multiplier for single-sample groups. The original
            GiGPO behavior is 0.0, which removes singleton groups. Values in (0, 1] retain part
            of the raw step reward when there is no within-group comparison.

    Returns:
        step_advantages: (bs, response_length) tensor.
    """
    bs, response_length = response_mask.shape
    device = response_mask.device
    N = len(flat_rewards)

    scores = flat_rewards.clone().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # --- group-wise normalization ---
        uid2indices = defaultdict(list)
        for i in range(N):
            uid2indices[step_group_uids[i]].append(i)

        single_count = 0
        multi_count = 0
        for uid, idxs in uid2indices.items():
            vals = scores[idxs]
            if len(idxs) == 1:
                single_count += 1
                scores[idxs[0]] = scores[idxs[0]] * float(single_step_fallback_weight)
            else:
                multi_count += 1
                mean = vals.mean()
                if remove_std:
                    scores[idxs] = vals - mean
                else:
                    std = vals.std()
                    scores[idxs] = (vals - mean) / (std + epsilon)

        total_groups = single_count + multi_count
        if total_groups > 0:
            print(f"step_norm_reward: {total_groups} groups, "
                  f"single={single_count} ({single_count/total_groups*100:.1f}%), "
                  f"multi={multi_count} ({multi_count/total_groups*100:.1f}%), "
                  f"single_fallback={float(single_step_fallback_weight):.3f}")
        if N > 0:
            print(f"normalized scores: min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
                  f"mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")

        # --- per-sample length compensation (applied post-normalization so it is not
        #     cancelled out by the group std). Caller is responsible for not passing factors
        #     when advantage_scope='token' (where broadcast already cancels the denominator).
        if length_factors is not None and N > 0 and sample_indices is not None:
            lf = length_factors.to(device=device, dtype=scores.dtype)
            per_score_factor = lf[torch.from_numpy(sample_indices.astype(np.int64)).to(device)]
            scores = scores * per_score_factor

        # --- scatter to (bs, response_length) ---
        step_advantages = torch.zeros(bs, response_length, dtype=scores.dtype, device=device)

        if advantage_scope == "entity":
            if token_positions is None or sample_indices is None:
                raise ValueError("advantage_scope='entity' requires sample_indices and token_positions")
            step_advantages = _scatter_entity_step_scores(
                scores, response_mask, sample_indices, token_positions, pool_mode=scatter_pool_mode
            )

        elif advantage_scope == "token":
            # average step scores per sample, broadcast to all response tokens
            if sample_indices is None:
                raise ValueError("advantage_scope='token' requires sample_indices")
            sample_sums = torch.zeros(bs, dtype=scores.dtype, device=device)
            sample_counts = torch.zeros(bs, dtype=torch.long, device=device)
            si_tensor = torch.from_numpy(sample_indices.astype(np.int64)).to(device)
            sample_sums.scatter_add_(0, si_tensor, scores)
            sample_counts.scatter_add_(0, si_tensor, torch.ones(N, dtype=torch.long, device=device))
            avg = sample_sums / sample_counts.clamp(min=1).float()
            step_advantages = avg.unsqueeze(-1).expand(-1, response_length) * response_mask

        elif advantage_scope == "terminal":
            if sample_indices is None:
                raise ValueError("advantage_scope='terminal' requires sample_indices")
            sample_sums = torch.zeros(bs, dtype=scores.dtype, device=device)
            sample_counts = torch.zeros(bs, dtype=torch.long, device=device)
            si_tensor = torch.from_numpy(sample_indices.astype(np.int64)).to(device)
            sample_sums.scatter_add_(0, si_tensor, scores)
            sample_counts.scatter_add_(0, si_tensor, torch.ones(N, dtype=torch.long, device=device))
            avg = sample_sums / sample_counts.clamp(min=1).float()
            valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
            valid_rows = valid_lengths > 0
            if torch.any(valid_rows):
                row_idx = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
                col_idx = valid_lengths[valid_rows] - 1
                step_advantages[row_idx, col_idx] = avg[valid_rows]
        else:
            raise ValueError(f"Unknown advantage_scope: {advantage_scope}")

    return step_advantages


def step_reward_to_advantage_no_norm(
    flat_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    sample_indices: np.ndarray = None,
    token_positions: np.ndarray = None,
    advantage_scope: str = "token",
    length_factors: Optional[torch.Tensor] = None,
    scatter_pool_mode: str = "mean",
):
    """Scatter raw discounted step rewards back to token space without group normalization."""
    bs, response_length = response_mask.shape
    device = response_mask.device
    N = len(flat_rewards)
    scores = flat_rewards.clone().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Per-sample length compensation (see step_norm_reward for rationale).
        if length_factors is not None and N > 0 and sample_indices is not None:
            lf = length_factors.to(device=device, dtype=scores.dtype)
            per_score_factor = lf[torch.from_numpy(sample_indices.astype(np.int64)).to(device)]
            scores = scores * per_score_factor

        step_advantages = torch.zeros(bs, response_length, dtype=scores.dtype, device=device)

        if advantage_scope == "entity":
            if token_positions is None or sample_indices is None:
                raise ValueError("advantage_scope='entity' requires sample_indices and token_positions")
            step_advantages = _scatter_entity_step_scores(
                scores, response_mask, sample_indices, token_positions, pool_mode=scatter_pool_mode
            )

        elif advantage_scope == "token":
            if sample_indices is None:
                raise ValueError("advantage_scope='token' requires sample_indices")
            sample_sums = torch.zeros(bs, dtype=scores.dtype, device=device)
            sample_counts = torch.zeros(bs, dtype=torch.long, device=device)
            si_tensor = torch.from_numpy(sample_indices.astype(np.int64)).to(device)
            sample_sums.scatter_add_(0, si_tensor, scores)
            sample_counts.scatter_add_(0, si_tensor, torch.ones(N, dtype=torch.long, device=device))
            avg = sample_sums / sample_counts.clamp(min=1).float()
            step_advantages = avg.unsqueeze(-1).expand(-1, response_length) * response_mask

        elif advantage_scope == "terminal":
            if sample_indices is None:
                raise ValueError("advantage_scope='terminal' requires sample_indices")
            sample_sums = torch.zeros(bs, dtype=scores.dtype, device=device)
            sample_counts = torch.zeros(bs, dtype=torch.long, device=device)
            si_tensor = torch.from_numpy(sample_indices.astype(np.int64)).to(device)
            sample_sums.scatter_add_(0, si_tensor, scores)
            sample_counts.scatter_add_(0, si_tensor, torch.ones(N, dtype=torch.long, device=device))
            avg = sample_sums / sample_counts.clamp(min=1).float()
            valid_lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
            valid_rows = valid_lengths > 0
            if torch.any(valid_rows):
                row_idx = torch.nonzero(valid_rows, as_tuple=False).squeeze(-1)
                col_idx = valid_lengths[valid_rows] - 1
                step_advantages[row_idx, col_idx] = avg[valid_rows]
        else:
            raise ValueError(f"Unknown advantage_scope: {advantage_scope}")

        if N > 0:
            print(f"raw step scores: min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
                  f"mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")

    return step_advantages


def compute_step_discounted_returns(step_rewards, gamma: float):
    new_step_rewards = np.zeros_like(step_rewards)
    for traj_index in range(step_rewards.shape[0]):  
        traj_rewards = step_rewards[traj_index]
        step_length = np.count_nonzero(~np.isnan(traj_rewards))
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        for t in reversed(range(len(traj_rewards))):
            if  t >= step_length:
                traj_returns[t] = np.nan
            else:
                running_return = traj_rewards[t] + gamma * running_return
                traj_returns[t] = running_return
        new_step_rewards[traj_index] = traj_returns
        
    return new_step_rewards


@register_adv_est(AdvantageEstimator.GiGPO)
def compute_gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    step_token_positions=None,
    step_reward_valid_mask: Optional[torch.Tensor] = None,
    pure_step_reward_tensor: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    config: Optional[AlgoConfig] = None,
    return_components: bool = False,
    **kwargs,
):
    """GiGPO: combine episode-level and step-level advantages with step group normalization."""

    """
    GiGPO: combine episode-level and step-level advantages.

    Flattens (bs, max_entities) step rewards to 1-D, groups by exact entity match,
    normalizes within groups, and scatters back to (bs, response_length).
    """
    # --- Convert step_rewards to numpy ---
    if isinstance(step_rewards, torch.Tensor):
        sr_cpu = step_rewards.detach().cpu()
        if sr_cpu.dtype == torch.bfloat16:
            sr_cpu = sr_cpu.to(torch.float32)
        step_rewards_np = sr_cpu.numpy()
    else:
        step_rewards_np = np.asarray(step_rewards, dtype=np.float64)

    # Mask step rewards for incorrect final answers
    if step_reward_valid_mask is not None:
        if isinstance(step_reward_valid_mask, torch.Tensor):
            valid_mask_np = step_reward_valid_mask.detach().cpu().numpy().astype(bool)
        else:
            valid_mask_np = np.asarray(step_reward_valid_mask).astype(bool)
        if valid_mask_np.shape[0] != step_rewards_np.shape[0]:
            raise ValueError(
                f"step_reward_valid_mask length mismatch: {valid_mask_np.shape[0]} vs {step_rewards_np.shape[0]}"
            )
        invalid_rows = ~valid_mask_np
        if np.any(invalid_rows):
            valid_entries = ~np.isnan(step_rewards_np)
            step_rewards_np[invalid_rows[:, None] & valid_entries] = 0.0
        print(f"step_reward_valid_mask: {int(valid_mask_np.sum())}/{len(valid_mask_np)} valid")

    # --- Read config ---
    gamma = 0.5
    advantage_scope = "token"
    scatter_pool_mode = "mean"
    if config is not None:
        if hasattr(config, "gigpo_step_scatter_pool"):
            scatter_pool_mode = str(config.gigpo_step_scatter_pool).lower()
        elif isinstance(config, dict) and "gigpo_step_scatter_pool" in config:
            scatter_pool_mode = str(config["gigpo_step_scatter_pool"]).lower()
        if hasattr(config, 'gigpo_gamma'):
            gamma = config.gigpo_gamma
        elif isinstance(config, dict) and 'gigpo_gamma' in config:
            gamma = config['gigpo_gamma']
        if hasattr(config, "gigpo_advantage_scope"):
            advantage_scope = str(config.gigpo_advantage_scope)
        elif isinstance(config, dict) and "gigpo_advantage_scope" in config:
            advantage_scope = str(config["gigpo_advantage_scope"])

    advantage_scope = advantage_scope.lower()
    if advantage_scope not in ("token", "terminal", "entity"):
        raise ValueError(f"Unsupported gigpo_advantage_scope={advantage_scope}")

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_adv_scope = "token"
    if config is not None:
        if hasattr(config, "gigpo_episode_advantage_scope"):
            episode_adv_scope = str(config.gigpo_episode_advantage_scope).lower()
        elif isinstance(config, dict) and "gigpo_episode_advantage_scope" in config:
            episode_adv_scope = str(config["gigpo_episode_advantage_scope"]).lower()
    if episode_adv_scope not in ("token", "terminal"):
        raise ValueError(f"Unsupported gigpo_episode_advantage_scope={episode_adv_scope}")

    # --- Length-norm config (only applies to step reward path) ---
    length_norm_mode = "none"
    length_norm_ref_cfg = "batch_median"
    if config is not None:
        if hasattr(config, "gigpo_step_length_norm"):
            length_norm_mode = str(config.gigpo_step_length_norm).lower()
        elif isinstance(config, dict) and "gigpo_step_length_norm" in config:
            length_norm_mode = str(config["gigpo_step_length_norm"]).lower()
        if hasattr(config, "gigpo_step_length_ref"):
            length_norm_ref_cfg = config.gigpo_step_length_ref
        elif isinstance(config, dict) and "gigpo_step_length_ref" in config:
            length_norm_ref_cfg = config["gigpo_step_length_ref"]
    if length_norm_mode not in ("none", "sqrt", "linear"):
        raise ValueError(
            f"Unsupported gigpo_step_length_norm={length_norm_mode}. Use one of: none, sqrt, linear."
        )

    single_step_fallback_weight = 0.0
    if config is not None:
        if hasattr(config, "gigpo_single_step_fallback_weight"):
            single_step_fallback_weight = config.gigpo_single_step_fallback_weight
        elif isinstance(config, dict) and "gigpo_single_step_fallback_weight" in config:
            single_step_fallback_weight = config["gigpo_single_step_fallback_weight"]
    single_step_fallback_weight = max(0.0, float(single_step_fallback_weight))

    print(
        f"GiGPO: gamma={gamma}, advantage_scope={advantage_scope}, episode_scope={episode_adv_scope}, "
        f"length_norm={length_norm_mode}, length_ref={length_norm_ref_cfg}, "
        f"single_step_fallback={single_step_fallback_weight}"
    )

    # --- Episode-level advantage: prefer outcome-only rewards when available ---
    # Subtracting pure_step_reward_tensor gives us a tensor containing only the outcome
    # (correctness) reward signal, so episode_norm_reward is no longer mixing step rewards
    # into the episode advantage. This is the Y1 design: fully decoupled episode and step paths.
    if pure_step_reward_tensor is not None:
        if pure_step_reward_tensor.shape != token_level_rewards.shape:
            raise ValueError(
                f"pure_step_reward_tensor shape {tuple(pure_step_reward_tensor.shape)} "
                f"!= token_level_rewards shape {tuple(token_level_rewards.shape)}"
            )
        outcome_only_rewards = token_level_rewards - pure_step_reward_tensor.to(
            dtype=token_level_rewards.dtype, device=token_level_rewards.device
        )
        episode_source = "outcome_only"
    else:
        # Fallback (e.g. streaming reward path): episode advantage still mixes step reward.
        # Length normalization in the step path still works, but episode path retains the legacy
        # length bias from the token-broadcasted step reward.
        print(
            "[GiGPO] pure_step_reward_tensor not provided; falling back to token_level_rewards "
            "for episode advantage. Length normalization on episode path is disabled."
        )
        outcome_only_rewards = token_level_rewards
        episode_source = "combined_fallback"

    episode_advantages = episode_norm_reward(
        outcome_only_rewards, response_mask, index, traj_index,
        epsilon, remove_std, advantage_scope=episode_adv_scope,
    )

    # --- Resolve length factors (per-sample scalar multipliers applied to step scores only) ---
    # Tensor lives on the same device as response_mask so the scatter stage can index directly.
    valid_lengths_float = response_mask.to(dtype=torch.float32).sum(dim=-1).clamp(min=1.0)
    if length_norm_mode == "none":
        length_factors = torch.ones_like(valid_lengths_float)
        length_ref_used = float("nan")
    else:
        # scope=token broadcasts the average step advantage to every response token, which
        # already cancels out the 1/L factor implicitly. Stacking a sqrt(L/L_ref) scaling on top
        # of that would reverse the effect, so we deliberately skip it. See the design doc.
        if advantage_scope == "token":
            print(
                "[GiGPO] length_norm='{mode}' requested but gigpo_advantage_scope='token' already "
                "cancels the length denominator via broadcast. Length norm is disabled for this run "
                "to avoid double correction.".format(mode=length_norm_mode)
            )
            length_factors = torch.ones_like(valid_lengths_float)
            length_ref_used = float("nan")
        else:
            if isinstance(length_norm_ref_cfg, str):
                ref_key = length_norm_ref_cfg.strip().lower()
                if ref_key in ("batch_median", "median"):
                    length_ref_tensor = torch.median(valid_lengths_float)
                elif ref_key in ("batch_mean", "mean"):
                    length_ref_tensor = torch.mean(valid_lengths_float)
                else:
                    try:
                        length_ref_tensor = torch.tensor(
                            float(length_norm_ref_cfg), device=valid_lengths_float.device
                        )
                    except ValueError:
                        raise ValueError(
                            f"Unsupported gigpo_step_length_ref={length_norm_ref_cfg!r}. "
                            "Use 'batch_median', 'batch_mean', or a numeric literal."
                        )
            else:
                length_ref_tensor = torch.tensor(
                    float(length_norm_ref_cfg), device=valid_lengths_float.device
                )
            length_ref_tensor = length_ref_tensor.clamp(min=1.0)
            ratio = valid_lengths_float / length_ref_tensor
            if length_norm_mode == "sqrt":
                length_factors = torch.sqrt(ratio)
            else:  # "linear"
                length_factors = ratio
            length_ref_used = float(length_ref_tensor.item())

    # --- Step-level: discount on (bs, max_entities), then flatten ---
    step_rewards_disc = compute_step_discounted_returns(step_rewards_np, gamma=gamma)

    # Flatten valid entries
    valid_mask = ~np.isnan(step_rewards_disc)
    rows, cols = np.where(valid_mask)
    N = len(rows)

    if N == 0:
        print("GiGPO: no valid step rewards, returning episode-only advantages")
        if return_components:
            zero_step = torch.zeros_like(episode_advantages)
            return episode_advantages, episode_advantages, episode_advantages, zero_step
        return episode_advantages, episode_advantages

    flat_rewards = torch.tensor(step_rewards_disc[valid_mask], dtype=torch.float32)
    flat_anchor_obs = anchor_obs[rows, cols]
    flat_index = index[rows]
    flat_sample_indices = rows.astype(np.int64)

    # Resolve token positions
    if step_token_positions is not None:
        if isinstance(step_token_positions, torch.Tensor):
            stp_np = step_token_positions.detach().cpu().numpy()
        else:
            stp_np = np.asarray(step_token_positions, dtype=object)
        flat_token_positions = stp_np[rows, cols]
    else:
        flat_token_positions = None

    print(f"Flattened: {N} valid steps from {step_rewards_disc.shape}")

    # --- Exact-match grouping ---
    step_group_uids = build_step_group(flat_anchor_obs, flat_index)

    # --- Group-wise normalization + scatter ---
    step_advantages = step_norm_reward(
        flat_rewards, response_mask, step_group_uids,
        epsilon, remove_std,
        sample_indices=flat_sample_indices,
        token_positions=flat_token_positions,
        advantage_scope=advantage_scope,
        length_factors=length_factors,
        single_step_fallback_weight=single_step_fallback_weight,
        scatter_pool_mode=scatter_pool_mode,
    )

    # --- Combine ---
    step_component = step_advantage_w * step_advantages
    scores = episode_advantages + step_component

    # --- Stats (exposed to wandb via metric_utils.compute_data_metrics) ---
    # Group statistics
    unique_groups, group_counts = np.unique(step_group_uids, return_counts=True)
    num_groups = int(len(unique_groups))
    avg_group_size = float(group_counts.mean()) if num_groups > 0 else 0.0
    max_group_size = int(group_counts.max()) if num_groups > 0 else 0
    single_sample_groups = int((group_counts == 1).sum())
    single_sample_ratio = float(single_sample_groups / num_groups) if num_groups > 0 else 0.0

    # Valid entry statistics
    total_entries = int(step_rewards_np.size)
    valid_entries = int(N)
    valid_ratio = float(valid_entries / total_entries) if total_entries > 0 else 0.0

    # step_reward_valid_mask gate ratio (if provided)
    gate_valid_count = int(valid_mask_np.sum()) if step_reward_valid_mask is not None else -1
    gate_total_count = int(valid_mask_np.shape[0]) if step_reward_valid_mask is not None else -1

    gigpo_stats = {
        "episode_adv_mean": episode_advantages.mean().item(),
        "episode_adv_std": episode_advantages.std().item(),
        "step_adv_mean": step_advantages.mean().item(),
        "step_adv_std": step_advantages.std().item(),
        "final_mean": scores.mean().item(),
        "final_std": scores.std().item(),
        "num_groups": num_groups,
        "avg_group_size": avg_group_size,
        "max_group_size": max_group_size,
        "single_sample_group_ratio": single_sample_ratio,
        "valid_entries": valid_entries,
        "total_entries": total_entries,
        "valid_entry_ratio": valid_ratio,
        "step_reward_gate_valid_count": gate_valid_count,
        "step_reward_gate_total_count": gate_total_count,
        "step_advantage_weight": float(step_advantage_w),
        "single_step_fallback_weight": float(single_step_fallback_weight),
        "step_scatter_pool_max_abs": 1.0 if scatter_pool_mode in ("max", "max_abs", "maxabs") else 0.0,
        "length_norm_mode": length_norm_mode,
        "length_norm_ref_used": length_ref_used,
        "length_factor_mean": float(length_factors.mean().item()),
        "length_factor_min": float(length_factors.min().item()),
        "length_factor_max": float(length_factors.max().item()),
        "episode_reward_source": 1.0 if episode_source == "outcome_only" else 0.0,
    }
    print(f"GiGPO stats: {gigpo_stats}")

    global _last_gigpo_stats
    _last_gigpo_stats = gigpo_stats

    if return_components:
        return scores, scores, episode_advantages, step_component
    return scores, scores


@register_adv_est(AdvantageEstimator.GiGPO_NO_STEP_NORM)
def compute_gigpo_no_step_norm_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.array,
    index: np.array,
    traj_index: np.array,
    step_token_positions=None,
    step_reward_valid_mask: Optional[torch.Tensor] = None,
    pure_step_reward_tensor: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    config: Optional[AlgoConfig] = None,
    return_components: bool = False,
    **kwargs,
):
    """GiGPO-family estimator that preserves step reward handling but removes step group normalization."""
    if isinstance(step_rewards, torch.Tensor):
        sr_cpu = step_rewards.detach().cpu()
        if sr_cpu.dtype == torch.bfloat16:
            sr_cpu = sr_cpu.to(torch.float32)
        step_rewards_np = sr_cpu.numpy()
    else:
        step_rewards_np = np.asarray(step_rewards, dtype=np.float64)

    if step_reward_valid_mask is not None:
        if isinstance(step_reward_valid_mask, torch.Tensor):
            valid_mask_np = step_reward_valid_mask.detach().cpu().numpy().astype(bool)
        else:
            valid_mask_np = np.asarray(step_reward_valid_mask).astype(bool)
        if valid_mask_np.shape[0] != step_rewards_np.shape[0]:
            raise ValueError(
                f"step_reward_valid_mask length mismatch: {valid_mask_np.shape[0]} vs {step_rewards_np.shape[0]}"
            )
        invalid_rows = ~valid_mask_np
        if np.any(invalid_rows):
            valid_entries = ~np.isnan(step_rewards_np)
            step_rewards_np[invalid_rows[:, None] & valid_entries] = 0.0
        print(f"step_reward_valid_mask: {int(valid_mask_np.sum())}/{len(valid_mask_np)} valid")

    gamma = 0.5
    advantage_scope = "token"
    scatter_pool_mode = "mean"
    if config is not None:
        if hasattr(config, "gigpo_step_scatter_pool"):
            scatter_pool_mode = str(config.gigpo_step_scatter_pool).lower()
        elif isinstance(config, dict) and "gigpo_step_scatter_pool" in config:
            scatter_pool_mode = str(config["gigpo_step_scatter_pool"]).lower()
        if hasattr(config, 'gigpo_gamma'):
            gamma = config.gigpo_gamma
        elif isinstance(config, dict) and 'gigpo_gamma' in config:
            gamma = config['gigpo_gamma']
        if hasattr(config, "gigpo_advantage_scope"):
            advantage_scope = str(config.gigpo_advantage_scope)
        elif isinstance(config, dict) and "gigpo_advantage_scope" in config:
            advantage_scope = str(config["gigpo_advantage_scope"])

    advantage_scope = advantage_scope.lower()
    if advantage_scope not in ("token", "terminal", "entity"):
        raise ValueError(f"Unsupported gigpo_advantage_scope={advantage_scope}")

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_adv_scope = "token"
    if config is not None:
        if hasattr(config, "gigpo_episode_advantage_scope"):
            episode_adv_scope = str(config.gigpo_episode_advantage_scope).lower()
        elif isinstance(config, dict) and "gigpo_episode_advantage_scope" in config:
            episode_adv_scope = str(config["gigpo_episode_advantage_scope"]).lower()
    if episode_adv_scope not in ("token", "terminal"):
        raise ValueError(f"Unsupported gigpo_episode_advantage_scope={episode_adv_scope}")

    # --- Length-norm config (only applies to step reward path) ---
    length_norm_mode = "none"
    length_norm_ref_cfg = "batch_median"
    if config is not None:
        if hasattr(config, "gigpo_step_length_norm"):
            length_norm_mode = str(config.gigpo_step_length_norm).lower()
        elif isinstance(config, dict) and "gigpo_step_length_norm" in config:
            length_norm_mode = str(config["gigpo_step_length_norm"]).lower()
        if hasattr(config, "gigpo_step_length_ref"):
            length_norm_ref_cfg = config.gigpo_step_length_ref
        elif isinstance(config, dict) and "gigpo_step_length_ref" in config:
            length_norm_ref_cfg = config["gigpo_step_length_ref"]
    if length_norm_mode not in ("none", "sqrt", "linear"):
        raise ValueError(
            f"Unsupported gigpo_step_length_norm={length_norm_mode}. Use one of: none, sqrt, linear."
        )

    print(
        f"GiGPO_NO_STEP_NORM: gamma={gamma}, advantage_scope={advantage_scope}, "
        f"episode_scope={episode_adv_scope}, length_norm={length_norm_mode}, "
        f"scatter_pool={scatter_pool_mode}"
    )

    # --- Episode-level advantage: prefer outcome-only rewards when available (Y1) ---
    if pure_step_reward_tensor is not None:
        if pure_step_reward_tensor.shape != token_level_rewards.shape:
            raise ValueError(
                f"pure_step_reward_tensor shape {tuple(pure_step_reward_tensor.shape)} "
                f"!= token_level_rewards shape {tuple(token_level_rewards.shape)}"
            )
        outcome_only_rewards = token_level_rewards - pure_step_reward_tensor.to(
            dtype=token_level_rewards.dtype, device=token_level_rewards.device
        )
    else:
        print(
            "[GiGPO_NO_STEP_NORM] pure_step_reward_tensor not provided; using token_level_rewards."
        )
        outcome_only_rewards = token_level_rewards

    episode_advantages = episode_norm_reward(
        outcome_only_rewards, response_mask, index, traj_index,
        epsilon, remove_std, advantage_scope=episode_adv_scope,
    )

    # --- Length factors (shared logic with compute_gigpo_outcome_advantage) ---
    valid_lengths_float = response_mask.to(dtype=torch.float32).sum(dim=-1).clamp(min=1.0)
    if length_norm_mode == "none" or advantage_scope == "token":
        length_factors = torch.ones_like(valid_lengths_float)
    else:
        if isinstance(length_norm_ref_cfg, str):
            ref_key = length_norm_ref_cfg.strip().lower()
            if ref_key in ("batch_median", "median"):
                length_ref_tensor = torch.median(valid_lengths_float)
            elif ref_key in ("batch_mean", "mean"):
                length_ref_tensor = torch.mean(valid_lengths_float)
            else:
                length_ref_tensor = torch.tensor(
                    float(length_norm_ref_cfg), device=valid_lengths_float.device
                )
        else:
            length_ref_tensor = torch.tensor(
                float(length_norm_ref_cfg), device=valid_lengths_float.device
            )
        length_ref_tensor = length_ref_tensor.clamp(min=1.0)
        ratio = valid_lengths_float / length_ref_tensor
        length_factors = torch.sqrt(ratio) if length_norm_mode == "sqrt" else ratio

    step_rewards_disc = compute_step_discounted_returns(step_rewards_np, gamma=gamma)
    valid_mask = ~np.isnan(step_rewards_disc)
    rows, cols = np.where(valid_mask)
    N = len(rows)

    if N == 0:
        print("GiGPO_NO_STEP_NORM: no valid step rewards, returning episode-only advantages")
        if return_components:
            zero_step = torch.zeros_like(episode_advantages)
            return episode_advantages, episode_advantages, episode_advantages, zero_step
        return episode_advantages, episode_advantages

    flat_rewards = torch.tensor(step_rewards_disc[valid_mask], dtype=torch.float32)
    flat_sample_indices = rows.astype(np.int64)

    if step_token_positions is not None:
        if isinstance(step_token_positions, torch.Tensor):
            stp_np = step_token_positions.detach().cpu().numpy()
        else:
            stp_np = np.asarray(step_token_positions, dtype=object)
        flat_token_positions = stp_np[rows, cols]
    else:
        flat_token_positions = None

    print(f"Flattened raw steps: {N} valid steps from {step_rewards_disc.shape}")

    step_advantages = step_reward_to_advantage_no_norm(
        flat_rewards,
        response_mask,
        sample_indices=flat_sample_indices,
        token_positions=flat_token_positions,
        advantage_scope=advantage_scope,
        length_factors=length_factors,
        scatter_pool_mode=scatter_pool_mode,
    )

    step_component = step_advantage_w * step_advantages
    scores = episode_advantages + step_component

    gigpo_stats = {
        "episode_adv_mean": episode_advantages.mean().item(),
        "episode_adv_std": episode_advantages.std().item(),
        "step_adv_mean": step_advantages.mean().item(),
        "step_adv_std": step_advantages.std().item(),
        "final_mean": scores.mean().item(),
        "final_std": scores.std().item(),
        "length_norm_mode": length_norm_mode,
        "step_scatter_pool_max_abs": 1.0 if scatter_pool_mode in ("max", "max_abs", "maxabs") else 0.0,
        "length_factor_mean": float(length_factors.mean().item()),
        "episode_reward_source": 1.0 if pure_step_reward_tensor is not None else 0.0,
    }
    print(f"GiGPO_NO_STEP_NORM stats: {gigpo_stats}")

    global _last_gigpo_stats
    _last_gigpo_stats = gigpo_stats

    if return_components:
        return scores, scores, episode_advantages, step_component
    return scores, scores
