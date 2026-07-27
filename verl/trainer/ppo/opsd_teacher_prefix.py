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

"""Teacher-prefix extension point for OPSD.

The initial OPSD integration uses the existing reference-policy forward on the
student trajectory. This builder is intentionally small: downstream projects
can subclass it and derive a privileged teacher prefix from dataset-specific
non-tensor fields without changing the policy loss implementation.
"""

from __future__ import annotations

import ast
import json
from importlib import import_module
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class OPSDTeacherPrefixBuilder:
    """Default teacher-prefix builder.

    The default implementation returns the static ``opsd_teacher_prefix`` value
    for every sample. With the default empty string, OPSD reduces to teacher
    scoring under the normal prompt context. Override ``build_one`` or
    ``build_batch`` to use task/category-specific fields from ``DataProto``.
    """

    def __init__(self, config: Any = None):
        self.config = config

    def build_one(self, sample: dict[str, Any]) -> str:
        del sample
        policy_cfg = getattr(self.config, "policy_loss", None)
        if policy_cfg is None:
            return ""
        return str(policy_cfg.get("opsd_teacher_prefix", "") or "")

    def build_batch(self, batch: Any) -> list[str]:
        batch_size = _infer_batch_size(batch)
        return [self.build_one(_sample_view(batch, idx)) for idx in range(batch_size)]


class GraphTaskAlgorithmPrefixBuilder(OPSDTeacherPrefixBuilder):
    """Build one compact algorithm prefix for each graph task.

    This builder reads ``extra_info.task`` from the RL batch and injects only the
    corresponding task's algorithm text from ``scripts.graph_task_algorithm_texts``.
    The static ``opsd_teacher_prefix`` config remains available as an optional
    global preamble, but the full 50-task table is never inserted into a single
    sample.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        policy_cfg = getattr(self.config, "policy_loss", None)
        self.static_prefix = ""
        self.max_chars = 1400
        if policy_cfg is not None:
            self.static_prefix = str(policy_cfg.get("opsd_teacher_prefix", "") or "").strip()
            self.max_chars = int(policy_cfg.get("opsd_teacher_prefix_max_chars", 1400) or 1400)

        try:
            from scripts.graph_task_algorithm_texts import GraphTaskAlgorithmTexts
        except Exception as exc:  # pragma: no cover - import failure is surfaced in build_one.
            self._algorithm_texts = None
            self._import_error = exc
        else:
            self._algorithm_texts = GraphTaskAlgorithmTexts
            self._import_error = None

    def build_one(self, sample: dict[str, Any]) -> str:
        if self._algorithm_texts is None:
            raise RuntimeError(
                "GraphTaskAlgorithmPrefixBuilder could not import scripts.graph_task_algorithm_texts"
            ) from self._import_error

        task = _extract_task_name(sample)
        blocks: list[str] = []
        if self.static_prefix:
            blocks.append(self.static_prefix)

        if task and self._algorithm_texts.has_task(task):
            blocks.append(
                "Teacher algorithm hint for OPSD scoring. Use this verifier-compatible "
                "procedure when assigning higher likelihood to the sampled reasoning path.\n"
                + self._algorithm_texts.prompt_block(task)
            )

        prefix = "\n\n".join(blocks).strip()
        if self.max_chars > 0 and len(prefix) > self.max_chars:
            prefix = prefix[: self.max_chars].rstrip()
        return prefix


def get_opsd_teacher_prefix_builder(config: Any) -> OPSDTeacherPrefixBuilder:
    """Instantiate the configured OPSD teacher-prefix builder."""

    policy_cfg = getattr(config, "policy_loss", None)
    path = ""
    if policy_cfg is not None:
        path = str(policy_cfg.get("opsd_teacher_prefix_builder", "") or "")
    if not path:
        return OPSDTeacherPrefixBuilder(config)

    module_name, _, class_name = path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            f"Invalid opsd_teacher_prefix_builder={path!r}. Expected a fully qualified class path."
        )
    module = import_module(module_name)
    builder_cls = getattr(module, class_name)
    builder = builder_cls(config)
    if not isinstance(builder, OPSDTeacherPrefixBuilder):
        required = getattr(builder, "build_batch", None)
        if required is None or not callable(required):
            raise TypeError(
                f"{path} must inherit OPSDTeacherPrefixBuilder or provide a callable build_batch(batch)."
            )
    return builder


def build_opsd_teacher_batch(
    batch: DataProto,
    tokenizer: Any,
    prefixes: list[str],
) -> tuple[DataProto, dict[str, float]]:
    """Return a copy of ``batch`` whose prompt side is prefixed for teacher scoring.

    ``responses`` are kept byte-for-byte unchanged, so the ref worker still
    returns one log-prob per sampled response token. Only the teacher/reference
    forward sees the algorithm prefix; rollout prompts and student log-probs are
    not changed.
    """

    if not prefixes:
        return batch, {}

    input_ids = batch.batch["input_ids"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    response_len = int(responses.shape[-1])
    prompt_len = int(input_ids.shape[-1] - response_len)
    if prompt_len <= 0:
        raise ValueError(
            f"Cannot build OPSD teacher batch: input length {input_ids.shape[-1]} <= response length {response_len}."
        )

    batch_size = int(input_ids.shape[0])
    if len(prefixes) != batch_size:
        raise ValueError(f"Expected {batch_size} OPSD prefixes, got {len(prefixes)}.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    prompt_ids = batch.batch.get("prompts", input_ids[:, :prompt_len])
    prompt_attention = attention_mask[:, :prompt_len]
    response_mask = batch.batch.get("response_mask", attention_mask[:, -response_len:])

    new_prompt_rows: list[torch.Tensor] = []
    new_prompt_mask_rows: list[torch.Tensor] = []
    prefix_token_counts: list[int] = []
    truncated = 0

    for idx, prefix in enumerate(prefixes):
        prefix = str(prefix or "")
        prefix_ids = _tokenize_prefix(tokenizer, prefix)
        prefix_token_counts.append(len(prefix_ids))

        valid_prompt = prompt_ids[idx][prompt_attention[idx].bool()].detach().cpu().tolist()
        if prefix_ids:
            # Keep the task algorithm hint and trim only the left side of the
            # original prompt when the fixed prompt budget is exceeded.
            room_for_prompt = prompt_len - len(prefix_ids)
            if room_for_prompt <= 0:
                merged_ids = prefix_ids[-prompt_len:]
                truncated += 1
            else:
                if len(valid_prompt) > room_for_prompt:
                    truncated += 1
                merged_ids = prefix_ids + valid_prompt[-room_for_prompt:]
        else:
            merged_ids = valid_prompt[-prompt_len:]

        pad_count = prompt_len - len(merged_ids)
        if pad_count < 0:
            merged_ids = merged_ids[-prompt_len:]
            pad_count = 0
            truncated += 1

        row = torch.tensor(
            [pad_token_id] * pad_count + merged_ids,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        mask = torch.tensor(
            [0] * pad_count + [1] * len(merged_ids),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        new_prompt_rows.append(row)
        new_prompt_mask_rows.append(mask)

    new_prompts = torch.stack(new_prompt_rows, dim=0)
    new_prompt_mask = torch.stack(new_prompt_mask_rows, dim=0)
    new_input_ids = torch.cat([new_prompts, responses], dim=-1)
    new_attention_mask = torch.cat([new_prompt_mask, response_mask.to(dtype=attention_mask.dtype)], dim=-1)
    new_position_ids = compute_position_id_with_mask(new_attention_mask)

    tensors = {key: value for key, value in batch.batch.items()}
    tensors["input_ids"] = new_input_ids
    tensors["attention_mask"] = new_attention_mask
    tensors["position_ids"] = new_position_ids
    if "prompts" in tensors:
        tensors["prompts"] = new_prompts

    teacher_batch = DataProto.from_dict(
        tensors=tensors,
        non_tensors={key: value for key, value in batch.non_tensor_batch.items()},
        meta_info={key: value for key, value in batch.meta_info.items()},
    )

    non_empty = sum(1 for prefix in prefixes if str(prefix or ""))
    metrics = {
        "opsd/teacher_prefix_nonempty_frac": float(non_empty / max(1, batch_size)),
        "opsd/teacher_prefix_token_mean": float(np.mean(prefix_token_counts)) if prefix_token_counts else 0.0,
        "opsd/teacher_prefix_truncated_frac": float(truncated / max(1, batch_size)),
        "opsd/teacher_prefix_builder_enabled": 1.0,
    }
    return teacher_batch, metrics


def _infer_batch_size(batch: Any) -> int:
    if hasattr(batch, "batch") and "responses" in batch.batch:
        return int(batch.batch["responses"].shape[0])
    if hasattr(batch, "batch") and "input_ids" in batch.batch:
        return int(batch.batch["input_ids"].shape[0])
    return 0


def _sample_view(batch: Any, idx: int) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    if hasattr(batch, "batch"):
        for key, value in batch.batch.items():
            try:
                sample[key] = value[idx]
            except Exception:
                pass
    if hasattr(batch, "non_tensor_batch"):
        for key, value in batch.non_tensor_batch.items():
            try:
                sample[key] = value[idx]
            except Exception:
                sample[key] = value
    return sample


def _tokenize_prefix(tokenizer: Any, prefix: str) -> list[int]:
    if not prefix:
        return []
    encoded = tokenizer(prefix + "\n\n", add_special_tokens=False)
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


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


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _coerce_mapping(value.item())
        if value.size == 1:
            return _coerce_mapping(value.reshape(-1)[0])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _string_or_empty(value.item())
        if value.size == 1:
            return _string_or_empty(value.reshape(-1)[0])
    text = str(value).strip()
    return text.lower()
