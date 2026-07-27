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
Single Process Actor
"""

import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # For memory-efficient embedding extraction using forward hooks
        self._target_layer_output = None
        self._forward_hook_handle = None
        self._flat_entity_keys = None
        self._flat_entity_keys_filtered = None
        self._flat_entity_keys_map = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False,
        extract_embeddings=False, embedding_layer=30, entity_positions=None, flat_entity_keys=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list | None]:
        """
        Args:
            extract_embeddings: Whether to extract hidden states from a specific layer
            embedding_layer: Which layer to extract embeddings from (default: 30)
            entity_positions: List[List[int]], entity token positions for each sample in batch
                            If provided, only extract embeddings at these positions (memory optimization)

        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            embeddings: # (bs, response_len, hidden_size) if entity_positions=None
                       # (total_entities, hidden_size) if entity_positions is provided
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            response_embeddings = None  # Initialize embeddings variable
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # Use forward hook instead of output_hidden_states to save memory
                if extract_embeddings:
                    # Map response-relative entity positions to unpadded indices if using remove padding.
                    entity_positions_for_hook = entity_positions
                    flat_entity_keys_for_hook = None
                    flat_entity_keys_map = None
                    if entity_positions is not None:
                        response_offset = seqlen - response_length - 1
                        if self.use_remove_padding:
                            # indices are flattened positions in (batch * seqlen) space.
                            indices_cpu = indices.detach().cpu().tolist()
                            index_map = {idx: j for j, idx in enumerate(indices_cpu)}
                            mapped_positions = []
                            for i, positions in enumerate(entity_positions):
                                mapped = []
                                for pos in positions:
                                    flat_pos = i * seqlen + response_offset + pos
                                    unpadded_pos = index_map.get(flat_pos)
                                    if unpadded_pos is not None:
                                        mapped.append(unpadded_pos)
                                mapped_positions.append(mapped)
                            entity_positions_for_hook = mapped_positions
                            if flat_entity_keys is not None:
                                # Build flat keys aligned to mapped positions; drop missing positions.
                                per_sample_map = [
                                    {orig: mapped for orig, mapped in zip(orig_list, mapped_list)}
                                    for orig_list, mapped_list in zip(entity_positions, mapped_positions)
                                ]
                                flat_entity_keys_for_hook = []
                                flat_entity_keys_map = [defaultdict(list) for _ in range(len(entity_positions))]
                                for (sample_idx, entity_type, entity_id, span_positions, token_pos) in flat_entity_keys:
                                    mapped_pos = per_sample_map[sample_idx].get(token_pos)
                                    if mapped_pos is not None:
                                        flat_entity_keys_for_hook.append(
                                            (sample_idx, entity_type, entity_id, span_positions, mapped_pos)
                                        )
                                        flat_entity_keys_map[sample_idx][mapped_pos].append(
                                            (sample_idx, entity_type, entity_id, span_positions, token_pos)
                                        )
                        else:
                            # Convert response-relative positions to absolute positions in the padded sequence.
                            mapped_positions = []
                            for i, positions in enumerate(entity_positions):
                                mapped_positions.append([response_offset + pos for pos in positions])
                            entity_positions_for_hook = mapped_positions
                            if flat_entity_keys is not None:
                                per_sample_map = [
                                    {orig: mapped for orig, mapped in zip(orig_list, mapped_list)}
                                    for orig_list, mapped_list in zip(entity_positions, mapped_positions)
                                ]
                                flat_entity_keys_for_hook = []
                                flat_entity_keys_map = [defaultdict(list) for _ in range(len(entity_positions))]
                                for (sample_idx, entity_type, entity_id, span_positions, token_pos) in flat_entity_keys:
                                    mapped_pos = per_sample_map[sample_idx].get(token_pos)
                                    if mapped_pos is not None:
                                        flat_entity_keys_for_hook.append(
                                            (sample_idx, entity_type, entity_id, span_positions, mapped_pos)
                                        )
                                        flat_entity_keys_map[sample_idx][mapped_pos].append(
                                            (sample_idx, entity_type, entity_id, span_positions, token_pos)
                                        )

                    # print(f"[DEBUG dp_actor] Registering embedding hook for layer {embedding_layer}")
                    self._register_embedding_hook(
                        embedding_layer,
                        entity_positions=entity_positions_for_hook,
                        flat_entity_keys=flat_entity_keys_for_hook,
                        flat_entity_keys_map=flat_entity_keys_map,
                    )
                    # print(f"[DEBUG dp_actor] Hook registered, _forward_hook_handle: {self._forward_hook_handle is not None}")

                try:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                except Exception as e:
                    # CRITICAL: Clean up hook on error to prevent hook accumulation
                    if extract_embeddings:
                        self._remove_embedding_hook()
                    raise e

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # Extract embeddings from target layer if requested
                target_layer_hidden = None
                if extract_embeddings and self._target_layer_output is not None:
                    # print(f"[DEBUG dp_actor] Extracting embeddings from target layer, shape: {self._target_layer_output.shape}")
                    # Get the captured output from forward hook
                    target_layer_hidden = self._target_layer_output

                    # If entity_positions was used, embeddings are already in final form (total_entities, hidden_size)
                    # No need to pad or extract response part
                    if entity_positions is not None:
                        # Entity embeddings: (total_entities, hidden_size)
                        # Already extracted at specific positions, no further processing needed
                        pass
                    else:
                        # Full embeddings: need to process
                        # Shape: (1, total_nnz, hidden_size) for remove_padding case
                        target_layer_hidden = target_layer_hidden.squeeze(0)  # (total_nnz, hidden_size)

                        # Gather if using ulysses sp
                        if self.use_ulysses_sp:
                            target_layer_hidden = gather_outputs_and_unpad(
                                target_layer_hidden,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                        if is_mask_all_zero:
                            target_layer_hidden = target_layer_hidden[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # Pad embeddings back if extracted (only for full embeddings, not entity embeddings)
                if extract_embeddings and target_layer_hidden is not None and entity_positions is None:
                    full_embeddings = pad_input(
                        hidden_states=target_layer_hidden,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )  # (batch, seqlen, hidden_size)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # Extract response embeddings if extracted
                if extract_embeddings and target_layer_hidden is not None:
                    if entity_positions is not None:
                        # Entity embeddings: already in final form (total_entities, hidden_size)
                        response_embeddings = target_layer_hidden
                    else:
                        # Full embeddings: extract response part
                        response_embeddings = full_embeddings[:, -response_length - 1 : -1, :]  # (bsz, response_length, hidden_size)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # Use forward hook instead of output_hidden_states to save memory
                if extract_embeddings:
                    # print(f"[DEBUG dp_actor] Registering embedding hook for layer {embedding_layer}")
                    flat_entity_keys_map = None
                    if flat_entity_keys is not None:
                        num_samples = batch_size
                        if entity_positions is not None:
                            num_samples = len(entity_positions)
                        flat_entity_keys_map = [defaultdict(list) for _ in range(num_samples)]
                        for (sample_idx, entity_type, entity_id, span_positions, token_pos) in flat_entity_keys:
                            if 0 <= sample_idx < len(flat_entity_keys_map):
                                flat_entity_keys_map[sample_idx][token_pos].append(
                                    (sample_idx, entity_type, entity_id, span_positions, token_pos)
                                )
                    self._register_embedding_hook(
                        embedding_layer,
                        entity_positions=entity_positions,
                        flat_entity_keys=flat_entity_keys,
                        flat_entity_keys_map=flat_entity_keys_map,
                    )
                    # print(f"[DEBUG dp_actor] Hook registered, _forward_hook_handle: {self._forward_hook_handle is not None}")

                try:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                except Exception as e:
                    # CRITICAL: Clean up hook on error to prevent hook accumulation
                    if extract_embeddings:
                        self._remove_embedding_hook()
                    raise e

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                # Extract embeddings from target layer if requested (non-rmpad case)
                if extract_embeddings and self._target_layer_output is not None:
                    # Get the captured output from forward hook
                    target_layer_hidden = self._target_layer_output

                    if entity_positions is not None:
                        # Entity embeddings: already in final form (total_entities, hidden_size)
                        response_embeddings = target_layer_hidden
                    else:
                        # Full embeddings: Shape (batch, seqlen, hidden_size)
                        # Extract only response part
                        response_embeddings = target_layer_hidden[:, -response_length - 1 : -1, :]  # (bsz, response_length, hidden_size)

            # CRITICAL: Save filtered_flat_entity_keys BEFORE cleaning up hook
            # This prevents losing the filtered keys when hook is removed
            filtered_flat_entity_keys = None
            if extract_embeddings:
                # print(f"[DEBUG dp_actor] Before cleanup, response_embeddings is None: {response_embeddings is None}")
                if response_embeddings is not None:
                    # print(f"[DEBUG dp_actor] response_embeddings shape: {response_embeddings.shape}")
                    pass
                # Save filtered keys before cleanup
                filtered_flat_entity_keys = self._flat_entity_keys_filtered
                self._remove_embedding_hook()

            # print(f"[DEBUG dp_actor] Returning from _forward_micro_batch, response_embeddings is None: {response_embeddings is None}")
            return entropy, log_probs, response_embeddings, filtered_flat_entity_keys

    def _register_embedding_hook(self, embedding_layer: int, entity_positions=None, flat_entity_keys=None, flat_entity_keys_map=None):
        """Register forward hook to capture specific layer output.

        This is much more memory-efficient than output_hidden_states=True,
        which stores all layers (33 layers = 462 GB for batch_size=512).
        Using hook only stores the target layer (14 GB for batch_size=512).

        With entity_positions optimization, we only store entity embeddings
        (200 MB for batch_size=512), achieving 98.6% memory reduction.

        Args:
            embedding_layer: Which layer to extract (0-indexed)
            entity_positions: List[List[int]], entity token positions for each sample
                            If provided, only extract embeddings at these positions
        """
        # Store entity positions for use in hook
        self._entity_positions = entity_positions
        self._flat_entity_keys = flat_entity_keys
        self._flat_entity_keys_filtered = None
        self._flat_entity_keys_map = flat_entity_keys_map

        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...)
            # We only need the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # If flat_entity_keys is provided, extract embeddings in that exact order
            if self._flat_entity_keys is not None:
                entity_embeddings_list = []
                filtered_keys = []
                is_rmpad = hidden_states.dim() == 3 and hidden_states.shape[0] == 1
                for sample_idx, _entity_type, _entity_id, span_positions, pos in self._flat_entity_keys:
                    if pos < hidden_states.shape[1]:
                        if is_rmpad:
                            emb = hidden_states[0, pos, :].detach().clone()
                        else:
                            emb = hidden_states[sample_idx, pos, :].detach().clone()
                        entity_embeddings_list.append(emb)
                        if self._flat_entity_keys_map is not None:
                            bucket = self._flat_entity_keys_map[sample_idx].get(pos)
                            if bucket:
                                filtered_keys.append(bucket.pop(0))
                            else:
                                filtered_keys.append((sample_idx, _entity_type, _entity_id, span_positions, pos))
                        else:
                            filtered_keys.append((sample_idx, _entity_type, _entity_id, span_positions, pos))
                if entity_embeddings_list:
                    self._target_layer_output = torch.stack(entity_embeddings_list)
                else:
                    self._target_layer_output = None
                self._flat_entity_keys_filtered = filtered_keys
            # If entity_positions is provided, only extract entity embeddings
            elif self._entity_positions is not None:
                entity_embeddings_list = []
                filtered_keys = []

                is_rmpad = hidden_states.dim() == 3 and hidden_states.shape[0] == 1
                for i, positions in enumerate(self._entity_positions):
                    if len(positions) == 0:
                        continue

                    # Extract embeddings at specified positions
                    for pos in positions:
                        if pos < hidden_states.shape[1]:  # Check bounds
                            # Extract and immediately detach + clone
                            if is_rmpad:
                                emb = hidden_states[0, pos, :].detach().clone()
                            else:
                                emb = hidden_states[i, pos, :].detach().clone()
                            entity_embeddings_list.append(emb)
                            if self._flat_entity_keys_map is not None:
                                bucket = self._flat_entity_keys_map[i].get(pos)
                                if bucket:
                                    filtered_keys.append(bucket.pop(0))

                # Stack into (total_entities, hidden_size)
                if entity_embeddings_list:
                    self._target_layer_output = torch.stack(entity_embeddings_list)
                else:
                    self._target_layer_output = None
                if self._flat_entity_keys_map is not None:
                    self._flat_entity_keys_filtered = filtered_keys
            else:
                # Original behavior: extract all tokens
                self._target_layer_output = hidden_states.detach()

        # Get the target layer
        # For Qwen/LLaMA models: model.model.layers[layer_idx]
        try:
            if hasattr(self.actor_module, 'module'):
                # FSDP wrapped
                if hasattr(self.actor_module.module, 'model'):
                    target_layer = self.actor_module.module.model.layers[embedding_layer]
                else:
                    # Some models have different structure
                    target_layer = self.actor_module.module.layers[embedding_layer]
            else:
                if hasattr(self.actor_module, 'model'):
                    target_layer = self.actor_module.model.layers[embedding_layer]
                else:
                    target_layer = self.actor_module.layers[embedding_layer]
        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to get layer {embedding_layer}: {e}")
            logger.error(f"Model structure: {type(self.actor_module)}")
            raise

        # Register hook
        self._forward_hook_handle = target_layer.register_forward_hook(hook_fn)

        if torch.distributed.get_rank() == 0:
            if self._entity_positions is not None:
                total_entities = sum(len(positions) for positions in self._entity_positions)
                logger.info(f"Registered forward hook on layer {embedding_layer} for {total_entities} entity positions (memory-optimized)")
            else:
                logger.info(f"Registered forward hook on layer {embedding_layer} for memory-efficient embedding extraction")

    def _remove_embedding_hook(self):
        """Remove the forward hook and clean up captured output."""
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None
        self._target_layer_output = None
        self._entity_positions = None  # Clean up entity positions
        self._flat_entity_keys = None
        self._flat_entity_keys_filtered = None
        self._flat_entity_keys_map = None

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False,
                        extract_embeddings=False, embedding_layer=30) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy: Whether to calculate entropy
            extract_embeddings: Whether to extract hidden states from a specific layer
            embedding_layer: Which layer to extract embeddings from (default: 30)

        Returns:
            tuple: (log_probs, entropys, embeddings) where embeddings is None if not extracted
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # Read from meta_info if available (allows overriding via DataProto)
        extract_embeddings = data.meta_info.get("extract_embeddings", extract_embeddings)
        embedding_layer = data.meta_info.get("embedding_layer", embedding_layer)
        entity_positions = data.meta_info.get("entity_positions", None)  # New: entity positions for memory optimization
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        embeddings_lst = []  # New: collect embeddings
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # Calculate entity positions for this micro batch if available
            micro_entity_positions = None
            if entity_positions is not None:
                # Get the range of samples in this micro batch
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = min(start_idx + len(micro_batch), len(entity_positions))
                micro_entity_positions = entity_positions[start_idx:end_idx]

            with torch.no_grad():
                entropy, log_probs, embeddings, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                    extract_embeddings=extract_embeddings, embedding_layer=embedding_layer,
                    entity_positions=micro_entity_positions  # New: pass entity positions
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if extract_embeddings and embeddings is not None:
                embeddings_lst.append(embeddings)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        response_embeddings = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if extract_embeddings and len(embeddings_lst) > 0:
            response_embeddings = torch.concat(embeddings_lst, dim=0)  # (batch, response_len, hidden_size)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if response_embeddings is not None:
                response_embeddings = restore_dynamic_batch(response_embeddings, batch_idx_list)

        return log_probs, entropys, response_embeddings

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_rewards(self, data: DataProto, tokenizer, reward_fn,
                                      calculate_entropy=False, extract_embeddings=False,
                                      embedding_layer=30) -> tuple:
        """
        Compute log probabilities and process rewards in a streaming fashion.
        This avoids storing large embedding tensors by computing rewards immediately.

        Args:
            data: DataProto containing input data
            tokenizer: Tokenizer for decoding responses
            reward_fn: Reward function (GraphRewardManager instance)
            calculate_entropy: Whether to calculate entropy
            extract_embeddings: Whether to extract embeddings for reward computation
            embedding_layer: Which layer to extract embeddings from

        Returns:
            tuple: (log_probs, entropys, reward_tensor, step_rewards, anchor_obs,
                    step_token_positions, reward_extra_infos, pure_step_reward_tensor)
                - log_probs: (batch_size, response_length)
                - entropys: (batch_size, response_length) or None
                - reward_tensor: (batch_size, response_length) with computed rewards
                - step_rewards: (batch_size, max_entities) for GiGPO or None
                - anchor_obs: (batch_size, max_entities) for GiGPO or None
                - step_token_positions: (batch_size, max_entities) token positions for GiGPO or None
                - reward_extra_infos: dict of per-sample reward diagnostics
                - pure_step_reward_tensor: (batch_size, response_length) step-reward-only tensor
                  used by GiGPO to compute outcome_only_rewards for the episode path, or None
        """
        # Set to eval mode
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # Read from meta_info if available
        extract_embeddings = data.meta_info.get("extract_embeddings", extract_embeddings)
        embedding_layer = data.meta_info.get("embedding_layer", embedding_layer)
        entity_positions = data.meta_info.get("entity_positions", None)
        flat_entity_keys = data.meta_info.get("flat_entity_keys", None)
        force_full_response_embeddings = data.meta_info.get("force_full_response_embeddings", False)
        # print(f"[DEBUG dp_actor] extract_embeddings={extract_embeddings}, embedding_layer={embedding_layer}")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]

        # Include all non_tensor_batch keys for reward computation
        non_tensor_select_keys = list(data.non_tensor_batch.keys()) if extract_embeddings else []
        if has_multi_modal_inputs and not extract_embeddings:
            non_tensor_select_keys = ["multi_modal_inputs"]

        # Prepare batches
        if use_dynamic_bsz:
            max_token_len = self.config.log_prob_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.select(batch_keys=select_keys,
                                       non_tensor_batch_keys=non_tensor_select_keys).split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        reward_tensor_lst = []
        step_rewards_lst = []  # For GiGPO
        anchor_obs_lst = []    # For GiGPO
        step_token_positions_lst = []  # For GiGPO entity-token updates
        pure_step_reward_tensor_lst = []  # For GiGPO outcome-only episode path decoupling
        reward_extra_infos_acc = defaultdict(list)

        # Process each micro batch
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())  # Move to GPU
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # Calculate entity positions for this micro batch if available
            micro_entity_positions = None
            if extract_embeddings and entity_positions is not None and not force_full_response_embeddings:
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = min(start_idx + len(micro_batch), len(entity_positions))
                micro_entity_positions = entity_positions[start_idx:end_idx]
            micro_flat_entity_keys = None
            if extract_embeddings and flat_entity_keys is not None and not force_full_response_embeddings:
                # Filter flat keys for this micro batch
                micro_flat_entity_keys = []
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = start_idx + len(micro_batch)
                for key in flat_entity_keys:
                    sample_idx, entity_type, entity_id, span_positions, token_pos = key
                    if start_idx <= sample_idx < end_idx:
                        micro_flat_entity_keys.append(
                            (sample_idx - start_idx, entity_type, entity_id, span_positions, token_pos)
                        )

            with torch.no_grad():
                # Forward pass to get log_probs and embeddings
                entropy, log_probs, embeddings, filtered_flat_entity_keys = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                    extract_embeddings=extract_embeddings, embedding_layer=embedding_layer,
                    entity_positions=micro_entity_positions, flat_entity_keys=micro_flat_entity_keys
                )

            if filtered_flat_entity_keys is not None:
                micro_flat_entity_keys = filtered_flat_entity_keys

            # print(f"[DEBUG dp_actor] Micro batch {len(log_probs_lst)}: embeddings is None: {embeddings is None}, extract_embeddings: {extract_embeddings}")
            if embeddings is not None:
                # print(f"[DEBUG dp_actor] Embeddings shape: {embeddings.shape}")
                pass

            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

            # Compute rewards immediately if embeddings were extracted
            if extract_embeddings and embeddings is not None:
                # Build a temporary DataProto without embeddings first (to keep batch dims consistent),
                # then attach response_embeddings.
                temp_data = DataProto.from_dict(
                    tensors={
                        "responses": micro_batch.batch["responses"],
                        "prompts": micro_batch.batch["input_ids"][:, :-micro_batch.batch["responses"].shape[-1]],
                        "attention_mask": micro_batch.batch["attention_mask"],
                    },
                    non_tensors=micro_batch.non_tensor_batch,
                )
                # Store embeddings outside of TensorDict to avoid batch size mismatch
                # For entity embeddings, shape is (total_entities, hidden_size) with no batch dimension
                temp_data.meta_info["response_embeddings"] = embeddings
                if micro_flat_entity_keys is not None:
                    # Provide flat entity keys aligned to embeddings order
                    temp_data.meta_info["flat_entity_keys"] = micro_flat_entity_keys
                    if embeddings is not None and embeddings.dim() == 2 and len(micro_flat_entity_keys) != embeddings.shape[0]:
                        print(
                            f"[streaming][warn] flat_entity_keys len={len(micro_flat_entity_keys)} "
                            f"!= embeddings rows={embeddings.shape[0]}"
                        )

                # Compute rewards using the reward function
                # This will process embeddings and return reward_tensor, step_rewards, anchor_obs
                reward_result = reward_fn(temp_data, return_dict=True)
                reward_tensor = reward_result["reward_tensor"]  # (micro_batch_size, response_len)
                # Normalize reward tensor device for consistent concatenation.
                if isinstance(reward_tensor, torch.Tensor):
                    target_device = micro_batch.batch["responses"].device
                    if reward_tensor.device != target_device:
                        reward_tensor = reward_tensor.to(target_device)

                # Extract step_rewards and anchor_obs for GiGPO
                step_rewards = reward_result.get("step_rewards", None)
                anchor_obs = reward_result.get("anchor_obs", None)
                step_token_positions = reward_result.get("step_token_positions", None)
                pure_step_reward_tensor = reward_result.get("pure_step_reward_tensor", None)
                reward_extra_info = reward_result.get("reward_extra_info", {})
                if isinstance(step_rewards, torch.Tensor):
                    if step_rewards.device != reward_tensor.device:
                        step_rewards = step_rewards.to(reward_tensor.device)
                if isinstance(pure_step_reward_tensor, torch.Tensor):
                    if pure_step_reward_tensor.device != reward_tensor.device:
                        pure_step_reward_tensor = pure_step_reward_tensor.to(reward_tensor.device)

                # Debug logging
                if step_rewards is not None:
                    # print(f"[DEBUG dp_actor] Extracted step_rewards with shape: {step_rewards.shape}")
                    pass
                else:
                    # print(f"[DEBUG dp_actor] No step_rewards in reward_result, keys: {reward_result.keys()}")
                    pass
                if anchor_obs is not None:
                    # print(f"[DEBUG dp_actor] Extracted anchor_obs with shape: {anchor_obs.shape}")
                    pass

                reward_tensor_lst.append(reward_tensor)
                if step_rewards is not None:
                    step_rewards_lst.append(step_rewards)
                if anchor_obs is not None:
                    anchor_obs_lst.append(anchor_obs)
                if step_token_positions is not None:
                    step_token_positions_lst.append(step_token_positions)
                if pure_step_reward_tensor is not None:
                    pure_step_reward_tensor_lst.append(pure_step_reward_tensor)
                else:
                    # Keep per-micro-batch alignment even when the reward manager does not
                    # produce a pure step tensor (fallback path). Use zeros so concat still works.
                    pure_step_reward_tensor_lst.append(torch.zeros_like(reward_tensor))
                for key, values in reward_extra_info.items():
                    reward_extra_infos_acc[key].append(np.asarray(values))

                # Immediately free embeddings memory
                del embeddings, temp_data
                # Avoid empty_cache every micro-batch; keep it periodic to reduce sync overhead.
                empty_cache_interval = getattr(self.config, "empty_cache_interval", 8)
                if empty_cache_interval and empty_cache_interval > 0:
                    self._empty_cache_counter = getattr(self, "_empty_cache_counter", 0) + 1
                    if self._empty_cache_counter % empty_cache_interval == 0:
                        torch.cuda.empty_cache()
            else:
                batch_size = micro_batch.batch["responses"].shape[0]
                response_len = micro_batch.batch["responses"].shape[1]

                if extract_embeddings:
                    temp_data = DataProto.from_dict(
                        tensors={
                            "responses": micro_batch.batch["responses"],
                            "prompts": micro_batch.batch["input_ids"][:, :-micro_batch.batch["responses"].shape[-1]],
                            "attention_mask": micro_batch.batch["attention_mask"],
                        },
                        non_tensors=micro_batch.non_tensor_batch,
                    )

                    reward_result = reward_fn(temp_data, return_dict=True)
                    reward_tensor = reward_result["reward_tensor"]
                    if isinstance(reward_tensor, torch.Tensor):
                        target_device = micro_batch.batch["responses"].device
                        if reward_tensor.device != target_device:
                            reward_tensor = reward_tensor.to(target_device)
                    else:
                        reward_tensor = torch.zeros(
                            (batch_size, response_len),
                            dtype=torch.float32,
                            device=micro_batch.batch["responses"].device,
                        )

                    step_rewards = reward_result.get("step_rewards", None)
                    anchor_obs = reward_result.get("anchor_obs", None)
                    step_token_positions = reward_result.get("step_token_positions", None)
                    pure_step_reward_tensor = reward_result.get("pure_step_reward_tensor", None)
                    reward_extra_info = reward_result.get("reward_extra_info", {})

                    reward_tensor_lst.append(reward_tensor)
                    if step_rewards is not None:
                        if isinstance(step_rewards, torch.Tensor) and step_rewards.device != reward_tensor.device:
                            step_rewards = step_rewards.to(reward_tensor.device)
                        step_rewards_lst.append(step_rewards)
                    else:
                        step_rewards_empty = torch.full(
                            (batch_size, 1),
                            float("nan"),
                            dtype=torch.float32,
                            device=reward_tensor.device,
                        )
                        step_rewards_lst.append(step_rewards_empty)

                    if anchor_obs is not None:
                        anchor_obs_lst.append(anchor_obs)
                    else:
                        anchor_obs_empty = np.full((batch_size, 1), None, dtype=object)
                        anchor_obs_lst.append(anchor_obs_empty)

                    if step_token_positions is not None:
                        step_token_positions_lst.append(step_token_positions)
                    else:
                        step_token_positions_empty = np.full((batch_size, 1), None, dtype=object)
                        step_token_positions_lst.append(step_token_positions_empty)

                    if isinstance(pure_step_reward_tensor, torch.Tensor):
                        if pure_step_reward_tensor.device != reward_tensor.device:
                            pure_step_reward_tensor = pure_step_reward_tensor.to(reward_tensor.device)
                        pure_step_reward_tensor_lst.append(pure_step_reward_tensor)
                    else:
                        pure_step_reward_tensor_lst.append(torch.zeros_like(reward_tensor))

                    for key, values in reward_extra_info.items():
                        reward_extra_infos_acc[key].append(np.asarray(values))
                else:
                    # No embeddings, create zero reward tensor
                    reward_tensor = torch.zeros(
                        (batch_size, response_len),
                        dtype=torch.float32,
                        device=micro_batch.batch["responses"].device,
                    )
                    reward_tensor_lst.append(reward_tensor)
                    # Keep the pure-step list aligned with reward_tensor_lst even in this branch.
                    pure_step_reward_tensor_lst.append(torch.zeros_like(reward_tensor))

        # Concatenate results
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        reward_tensor = torch.concat(reward_tensor_lst, dim=0)

        pure_step_reward_tensor = None
        if len(pure_step_reward_tensor_lst) > 0:
            # All entries in pure_step_reward_tensor_lst share the same (mb, response_len) layout
            # as reward_tensor_lst, so a plain cat is safe.
            pure_step_reward_tensor = torch.concat(pure_step_reward_tensor_lst, dim=0)

        # Concatenate step_rewards and anchor_obs for GiGPO
        step_rewards = None
        anchor_obs = None
        step_token_positions = None
        reward_extra_infos = {}
        # print(f"[DEBUG dp_actor] step_rewards_lst length: {len(step_rewards_lst)}, anchor_obs_lst length: {len(anchor_obs_lst)}")
        if len(step_rewards_lst) > 0:
            # Find the maximum number of entities across all micro batches
            max_entities = max(sr.shape[1] for sr in step_rewards_lst)
            # print(f"[DEBUG dp_actor] Max entities across micro batches: {max_entities}")

            # Pad all tensors to the same size before concatenation
            padded_step_rewards_lst = []
            for sr in step_rewards_lst:
                if sr.shape[1] < max_entities:
                    # Pad with NaN to match max_entities
                    padding = torch.full((sr.shape[0], max_entities - sr.shape[1]),
                                        float('nan'),
                                        dtype=sr.dtype,
                                        device=sr.device)
                    sr_padded = torch.cat([sr, padding], dim=1)
                    padded_step_rewards_lst.append(sr_padded)
                else:
                    padded_step_rewards_lst.append(sr)

            step_rewards = torch.concat(padded_step_rewards_lst, dim=0)
            # print(f"[DEBUG dp_actor] Concatenated step_rewards shape: {step_rewards.shape}")
        if len(anchor_obs_lst) > 0:
            # Find the maximum number of entities across all micro batches
            max_entities = max(ao.shape[1] for ao in anchor_obs_lst)

            # Pad all arrays to the same size before concatenation
            padded_anchor_obs_lst = []
            for ao in anchor_obs_lst:
                if ao.shape[1] < max_entities:
                    # Pad with NaN to match max_entities
                    padding = np.full((ao.shape[0], max_entities - ao.shape[1]),
                                     np.nan,
                                     dtype=ao.dtype)
                    ao_padded = np.concatenate([ao, padding], axis=1)
                    padded_anchor_obs_lst.append(ao_padded)
                else:
                    padded_anchor_obs_lst.append(ao)

            anchor_obs = np.concatenate(padded_anchor_obs_lst, axis=0)
            # print(f"[DEBUG dp_actor] Concatenated anchor_obs shape: {anchor_obs.shape}")

        if len(step_token_positions_lst) > 0:
            max_entities = max(pos.shape[1] for pos in step_token_positions_lst)
            padded_step_token_positions_lst = []
            for pos in step_token_positions_lst:
                if pos.shape[1] < max_entities:
                    padding = np.full((pos.shape[0], max_entities - pos.shape[1]), None, dtype=object)
                    pos_padded = np.concatenate([pos.astype(object), padding], axis=1)
                    padded_step_token_positions_lst.append(pos_padded)
                else:
                    padded_step_token_positions_lst.append(pos.astype(object))
            step_token_positions = np.concatenate(padded_step_token_positions_lst, axis=0)

        for key, values_lst in reward_extra_infos_acc.items():
            if values_lst:
                reward_extra_infos[key] = np.concatenate(values_lst, axis=0)

        # Restore dynamic batch if needed
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            reward_tensor = restore_dynamic_batch(reward_tensor, batch_idx_list)
            if pure_step_reward_tensor is not None:
                pure_step_reward_tensor = restore_dynamic_batch(pure_step_reward_tensor, batch_idx_list)
            if step_rewards is not None:
                step_rewards = restore_dynamic_batch(step_rewards, batch_idx_list)
            if anchor_obs is not None:
                # restore_dynamic_batch only supports torch tensor; anchor_obs is numpy object array.
                indices = [idx for sub in batch_idx_list for idx in sub]
                reverse_idx = np.argsort(np.asarray(indices))
                anchor_obs = anchor_obs[reverse_idx]
            if step_token_positions is not None:
                indices = [idx for sub in batch_idx_list for idx in sub]
                reverse_idx = np.argsort(np.asarray(indices))
                step_token_positions = step_token_positions[reverse_idx]
            for key, values in reward_extra_infos.items():
                indices = [idx for sub in batch_idx_list for idx in sub]
                reverse_idx = np.argsort(np.asarray(indices))
                reward_extra_infos[key] = values[reverse_idx]

        return (
            log_probs,
            entropys,
            reward_tensor,
            step_rewards,
            anchor_obs,
            step_token_positions,
            reward_extra_infos,
            pure_step_reward_tensor,
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss or loss_mode in ("mgpo", "entity_anchored_grouped_kl", "opsd", "opsd_future_kl"):
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if loss_mode in {"observer_future_kl", "opsd_future_kl"}:
            for key in ("advantages_episode", "advantages_step"):
                if key in data.batch.keys():
                    select_keys.append(key)
        # entity_anchored_grouped_kl: anchor tensors live in batch.batch (preferred path)
        # and numpy mirrors in non_tensor_batch (fallback). Whitelist both — without
        # this, data.select() strips them and dp_actor sees MISSING.
        if loss_mode == "entity_anchored_grouped_kl":
            for ak in ("anchor_positions", "anchor_group_ids",
                       "anchor_prompt_ids",
                       "segment_id_per_token", "num_anchors"):
                if ak in data.batch.keys():
                    select_keys.append(ak)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if loss_mode in ("mgpo", "entity_anchored_grouped_kl"):
            non_tensor_select_keys.append("uid")
        if loss_mode in {"observer_future_kl", "opsd_future_kl"} and "step_token_positions" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("step_token_positions")
        if loss_mode == "entity_anchored_grouped_kl":
            for ak in ("anchor_positions_np", "anchor_group_ids_np",
                       "anchor_prompt_ids_np",
                       "segment_id_per_token_np", "num_anchors_np"):
                if ak in data.non_tensor_batch.keys():
                    non_tensor_select_keys.append(ak)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # Note: embeddings not needed in update_policy, so we don't extract them
                    entropy, log_prob, _, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                        extract_embeddings=False  # No need for embeddings during training
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    ref_log_prob = model_inputs.get("ref_log_prob", None)
                    uid = micro_batch.non_tensor_batch.get("uid", None)

                    # Optional: entity-anchor tensors for `entity_anchored_grouped_kl` loss.
                    # Tensors are written to BOTH batch.batch AND non_tensor_batch (as np mirrors)
                    # by ray_trainer. The np path reliably survives Ray DataProto dispatch; the
                    # tensor path may not. We try tensor first, fall back to numpy mirror.
                    anchor_kwargs = {}
                    if loss_mode == "entity_anchored_grouped_kl":
                        import numpy as _np
                        target_device = (
                            model_inputs.get("old_log_probs", None).device
                            if model_inputs.get("old_log_probs", None) is not None
                            else (advantages.device if advantages is not None else "cpu")
                        )
                        for key in ("anchor_positions", "anchor_group_ids",
                                    "anchor_prompt_ids",
                                    "segment_id_per_token", "num_anchors"):
                            tensor = model_inputs.get(key, None)
                            if tensor is None and hasattr(micro_batch, "batch") and key in micro_batch.batch:
                                tensor = micro_batch.batch[key]
                            if tensor is None:
                                # Numpy fallback (the reliable path).
                                np_key = f"{key}_np"
                                np_arr = micro_batch.non_tensor_batch.get(np_key, None)
                                if np_arr is None and hasattr(micro_batch, "batch") and np_key in micro_batch.batch:
                                    np_arr = micro_batch.batch[np_key]
                                if np_arr is not None:
                                    if isinstance(np_arr, _np.ndarray):
                                        if np_arr.dtype == object:
                                            # array of arrays — stack into homogeneous 2D
                                            np_arr = _np.stack([_np.asarray(x) for x in np_arr])
                                        tensor = torch.from_numpy(np_arr).to(target_device).long()
                            anchor_kwargs[key] = tensor
                    if loss_mode in {"observer_future_kl", "opsd_future_kl"}:
                        anchor_kwargs["advantages_episode"] = model_inputs.get("advantages_episode", None)
                        anchor_kwargs["advantages_step"] = model_inputs.get("advantages_step", None)
                        anchor_kwargs["step_token_positions"] = micro_batch.non_tensor_batch.get(
                            "step_token_positions", None
                        )

                    if batch_idx == 0 and not hasattr(self, '_mgpo_loss_logged'):
                        print(f"[dp_actor] loss_mode={loss_mode}, policy_loss_fn={policy_loss_fn.__name__}, ref_log_prob={'present' if ref_log_prob is not None else 'MISSING'}, uid={'present' if uid is not None else 'MISSING'}")
                        if anchor_kwargs:
                            print(f"[dp_actor] anchor tensors: " + ", ".join(
                                f"{k}={'present' if v is not None else 'MISSING'}" for k, v in anchor_kwargs.items()
                            ))
                        self._mgpo_loss_logged = True

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        ref_log_prob=ref_log_prob,
                        uid=uid,
                        **anchor_kwargs,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
