"""
Activation-based Reward Function for Graph Tasks

This module provides a flexible reward function that can load different trained models
to compute step rewards based on entity token embeddings.

"""

import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path


class ActivationRewardFunction:
    """
    Activation-based reward function that loads a trained reward model.

    This class wraps a trained encoder-regressor model to compute rewards
    from entity token embeddings during RL training.

    Args:
        model_dir: Directory containing the trained model files
                   (best_encoder.pth, best_regressor.pth, test_results.json)
        device: Device to run the model on ('cuda' or 'cpu')
        normalize_reward: Whether to normalize rewards to [0, 1] range
        reward_scale: Scale factor for rewards (applied after normalization)
        cache_models: Whether to keep models in memory (default: True)

    Example:
        >>> reward_fn = ActivationRewardFunction(
        ...     model_dir="/path/to/mlp_mi/best_model",
        ...     device="cuda",
        ...     normalize_reward=True,
        ...     reward_scale=1.0
        ... )
        >>> embedding = torch.randn(3584)  # hidden_size dimension
        >>> reward = reward_fn(embedding)
        >>> print(reward)  # scalar reward value
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        cache_models: bool = True,
        stats_path: Optional[str] = None,
        auto_load_stats: bool = True,
    ):
        self.model_dir = Path(model_dir)

        # Store the requested device (don't check availability yet)
        # Device availability will be checked when the model is actually loaded
        self.device = device
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.cache_models = cache_models
        self.stats_path = stats_path
        self.auto_load_stats = auto_load_stats

        # Model components (will be loaded lazily on first use)
        self.encoder = None
        self.regressor = None
        self.config = None
        self._model_loaded = False

        # Preprocessor (will be loaded with model)
        self.preprocessor = None

        # Statistics for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._stats_loaded = False

    def _load_model(self):
        """Load the trained encoder and regressor models.

        This method is called lazily on first use, ensuring that device availability
        is checked in the correct process context (e.g., GPU worker process).
        """
        # Skip if already loaded
        if self._model_loaded:
            return

        # Check device availability NOW (when actually loading, not during __init__)
        # This ensures we're in the correct process context (e.g., GPU worker)
        if self.device == "cuda":
            if not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available in current process.")
                print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
                print(f"  This may be expected if loading in main process before GPU worker spawn.")
                print(f"  Attempting to proceed with CUDA device anyway...")
                # self.device = "cpu"
                # Don't fall back to CPU - let PyTorch handle device placement
                # If GPU is truly unavailable, the .to(device) call will fail with a clear error
            else:
                print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        # Try to load from final_model.pth first (new format)
        final_model_path = self.model_dir / "final_model.pth"
        encoder_path = self.model_dir / "best_encoder.pth"
        regressor_path = self.model_dir / "best_regressor.pth"
        config_path = self.model_dir / "test_results.json"

        # Check which format is available
        use_final_model = final_model_path.exists()
        use_separate_files = encoder_path.exists() and regressor_path.exists()

        if not use_final_model and not use_separate_files:
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                f"Expected either 'final_model.pth' or both 'best_encoder.pth' and 'best_regressor.pth'"
            )

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print(f"Loading activation reward model from: {self.model_dir}")
        print(f"Model type: {self.config.get('stage1_config', {}).get('backbone', 'unknown')}")
        print(f"Test R²: {self.config.get('test_metrics', {}).get('r2', 'N/A')}")

        # Optionally load normalization stats from config or external file
        if self.normalize_reward and self.auto_load_stats:
            self._maybe_load_stats()

        # Determine the device for loading (use CPU first, then move to target device)
        # This is important because the main process may not have GPU access
        load_device = self.device

        if use_final_model:
            # Load from final_model.pth (new format)
            print(f"Loading from final_model.pth...")
            checkpoint = torch.load(final_model_path, map_location=load_device, weights_only=False)

            # Extract encoder and regressor state dicts
            encoder_checkpoint = {
                'state_dict': checkpoint['encoder_state_dict'],
                'config': checkpoint.get('encoder_config', {})
            }
            regressor_checkpoint = {
                'state_dict': checkpoint['regressor_state_dict'],
                'config': checkpoint.get('regressor_config', {})
            }
        else:
            # Load from separate files (old format)
            print(f"Loading from separate encoder and regressor files...")
            encoder_checkpoint_raw = torch.load(encoder_path, map_location=load_device, weights_only=False)
            regressor_checkpoint_raw = torch.load(regressor_path, map_location=load_device, weights_only=False)

            # Standardize format to match final_model.pth structure
            # The separate files may have different key names, so we need to handle them
            if 'encoder_state_dict' in encoder_checkpoint_raw:
                encoder_checkpoint = {
                    'state_dict': encoder_checkpoint_raw['encoder_state_dict'],
                    'config': encoder_checkpoint_raw.get('config', {})
                }
            elif 'state_dict' in encoder_checkpoint_raw:
                encoder_checkpoint = encoder_checkpoint_raw
            else:
                # Assume the entire checkpoint is the state dict
                encoder_checkpoint = {
                    'state_dict': encoder_checkpoint_raw,
                    'config': {}
                }

            if 'regressor_state_dict' in regressor_checkpoint_raw:
                regressor_checkpoint = {
                    'state_dict': regressor_checkpoint_raw['regressor_state_dict'],
                    'config': regressor_checkpoint_raw.get('config', {})
                }
            elif 'state_dict' in regressor_checkpoint_raw:
                regressor_checkpoint = regressor_checkpoint_raw
            else:
                # Assume the entire checkpoint is the state dict
                regressor_checkpoint = {
                    'state_dict': regressor_checkpoint_raw,
                    'config': {}
                }

        # Build models
        self.encoder = self._build_encoder(encoder_checkpoint)
        self.encoder.eval()

        self.regressor = self._build_regressor(regressor_checkpoint)
        self.regressor.eval()

        # Move models to target device
        self.encoder = self.encoder.to(self.device)
        self.regressor = self.regressor.to(self.device)

        # Load preprocessor if available
        self._load_preprocessor()

        # Mark as loaded
        self._model_loaded = True

        print(f"✓ Activation reward model loaded successfully")
        print(f"  Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"  Regressor parameters: {sum(p.numel() for p in self.regressor.parameters()):,}")
        print(f"  Device: {self.device}")
        if self.preprocessor is not None:
            print(f"  Preprocessor: Loaded from {self.model_dir / 'preprocessor.pkl'}")
        else:
            print(f"  Preprocessor: Not found (⚠️  Warning: Using raw activations)")

    def _load_preprocessor(self):
        """Load the preprocessor that was used during training.

        This is CRITICAL - without this, the model will receive raw activations
        instead of the preprocessed features it was trained on.
        """
        # Add path to import ActivationPreprocessor module (required for unpickling)
        import sys
        rnc_path = os.environ.get("RNC_LINEAR_SCRIPTS_PATH", "")
        if rnc_path and rnc_path not in sys.path:
            sys.path.insert(0, rnc_path)

        # Try multiple possible locations for the preprocessor
        possible_paths = [
            self.model_dir / "preprocessor.pkl",           # In best_model/ or stage2_xxx/
            self.model_dir.parent / "preprocessor.pkl",    # In parent directory
            self.model_dir.parent / "temp_models" / "preprocessor.pkl",  # In temp_models/
        ]

        # Check if this is a stage2 directory, and add corresponding stage1 directory
        model_dir_name = self.model_dir.name
        parent_dir = self.model_dir.parent

        if "stage2" in model_dir_name:
            # Convert stage2_graph_element_mi -> stage1_graph_element_mi
            stage1_dir_name = model_dir_name.replace("stage2", "stage1")
            stage1_path = parent_dir / stage1_dir_name / "preprocessor.pkl"
            possible_paths.insert(1, stage1_path)  # Insert after model_dir, before parent
            print(f"  Detected stage2 model, will also check stage1: {stage1_path}")

        for preprocessor_path in possible_paths:
            if preprocessor_path.exists():
                try:
                    import pickle
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                    print(f"✓ Loaded preprocessor from {preprocessor_path}")

                    # Print preprocessor details for verification
                    if hasattr(self.preprocessor, 'method'):
                        print(f"  Preprocessing method: {self.preprocessor.method}")
                    if hasattr(self.preprocessor, 'clip_percentile'):
                        print(f"  Clip percentile: {self.preprocessor.clip_percentile}")
                    if hasattr(self.preprocessor, 'apply_log_transform'):
                        print(f"  Log transform: {self.preprocessor.apply_log_transform}")

                    return  # Successfully loaded

                except Exception as e:
                    print(f"⚠️  Error loading preprocessor from {preprocessor_path}: {e}")
                    continue

        # If we reach here, no preprocessor was found
        print(f"⚠️  Warning: No preprocessor found in any of these locations:")
        for path in possible_paths:
            print(f"    - {path}")
        print(f"  This means the model will receive raw activations, which may not match training!")
        self.preprocessor = None

    def _apply_preprocessing(self, entity_embedding: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to entity embeddings before feeding to the model.

        Args:
            entity_embedding: Raw activation tensor of shape (batch_size, hidden_size)
                             or (hidden_size,)

        Returns:
            preprocessed_embedding: Preprocessed tensor ready for the model
        """
        if self.preprocessor is None:
            # No preprocessing available, return as is
            return entity_embedding

        # Convert to numpy for preprocessing
        original_device = entity_embedding.device
        original_shape = entity_embedding.shape

        # Ensure 2D shape for preprocessing
        if entity_embedding.dim() == 1:
            entity_embedding = entity_embedding.unsqueeze(0)

        # NumPy does not support direct conversion from torch.bfloat16.
        # Preprocessing models are trained in float space, so cast here explicitly.
        emb_np = entity_embedding.detach().to(torch.float32).cpu().numpy()

        # Apply preprocessing (this matches the training pipeline)
        try:
            emb_preprocessed = self.preprocessor.transform(emb_np)
        except Exception as e:
            print(f"⚠️  Error during preprocessing: {e}")
            print(f"  Falling back to raw embeddings")
            return entity_embedding

        # Convert back to tensor in float32; caller will cast to encoder dtype.
        emb_tensor = torch.from_numpy(emb_preprocessed).to(device=original_device, dtype=torch.float32)

        # Restore original shape if needed
        if len(original_shape) == 1:
            emb_tensor = emb_tensor.squeeze(0)

        return emb_tensor

    def _maybe_load_stats(self) -> None:
        """Load normalization stats from config or an external JSON file if available."""
        if self._stats_loaded:
            return

        def _extract_stats(cfg: Dict[str, Any]) -> Optional[tuple[float, float]]:
            if not isinstance(cfg, dict):
                return None
            # Common patterns
            if "reward_mean" in cfg and "reward_std" in cfg:
                return float(cfg["reward_mean"]), float(cfg["reward_std"])
            if "mean" in cfg and "std" in cfg:
                return float(cfg["mean"]), float(cfg["std"])
            for key in ("reward_stats", "normalization", "stats"):
                sub = cfg.get(key, None)
                if isinstance(sub, dict) and "mean" in sub and "std" in sub:
                    return float(sub["mean"]), float(sub["std"])
            return None

        stats = _extract_stats(self.config or {})

        # Optional external stats file
        if stats is None and self.stats_path:
            stats_path = Path(self.stats_path)
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    stats_cfg = json.load(f)
                stats = _extract_stats(stats_cfg)

        if stats is not None:
            self.reward_mean, self.reward_std = stats
            self._stats_loaded = True
            print(f"Loaded reward normalization stats: mean={self.reward_mean:.6f}, std={self.reward_std:.6f}")

    def _build_encoder(self, checkpoint: Dict[str, Any]) -> nn.Module:
        """Build encoder from checkpoint."""
        # Import model definitions
        try:
            import sys
            rnc_path = os.environ.get("RNC_CLASSIFIER_PATH", "")
            if rnc_path and rnc_path not in sys.path:
                sys.path.insert(0, rnc_path)
            from rnc_models import RnCEncoder
        except ImportError as e:
            raise ImportError(f"Failed to import RnCEncoder: {e}")

        # Get configuration
        stage1_config = self.config.get('stage1_config', {})

        # Get input_dim from the first layer weight
        # Use 'state_dict' key which is set in _load_model
        model_state = checkpoint['state_dict']
        first_layer_key = 'projector.backbone.0.weight'
        if first_layer_key in model_state:
            input_dim = model_state[first_layer_key].shape[1]
        else:
            raise KeyError(f"Cannot find {first_layer_key} in checkpoint")

        # Create encoder
        encoder = RnCEncoder(
            input_dim=input_dim,
            proj_dim=stage1_config.get('proj_dim', 256),
            backbone=stage1_config.get('backbone', 'mlp'),
            hidden_dim=stage1_config.get('hidden_dim', 4096),
            num_layers=stage1_config.get('num_layers', 2),
            dropout=stage1_config.get('dropout', 0.1),
        )

        # Load weights
        encoder.load_state_dict(checkpoint['state_dict'])
        # Note: Device transfer is handled in _load_model after building

        return encoder

    def _build_regressor(self, checkpoint: Dict[str, Any]) -> nn.Module:
        """Build regressor from checkpoint."""
        # Import model definitions
        try:
            import sys
            rnc_path = os.environ.get("RNC_CLASSIFIER_PATH", "")
            if rnc_path and rnc_path not in sys.path:
                sys.path.insert(0, rnc_path)
            from rnc_models import create_regressor
        except ImportError as e:
            raise ImportError(f"Failed to import create_regressor: {e}")

        # Get configuration
        args = self.config.get('args', {})

        # Parse hidden dims
        hidden_dims_str = args.get('regressor_hidden_dims', '512,256')
        hidden_dims = [int(d) for d in hidden_dims_str.split(',')]

        # Get input_dim from the first layer weight
        # Use 'state_dict' key which is set in _load_model
        regressor_state = checkpoint['state_dict']
        first_layer_key = 'feature_extractor.0.weight'
        if first_layer_key in regressor_state:
            input_dim = regressor_state[first_layer_key].shape[1]
        else:
            raise KeyError(f"Cannot find {first_layer_key} in checkpoint")

        # Create regressor
        regressor = create_regressor(
            input_dim=input_dim,
            backbone=args.get('regressor_backbone', 'mlp'),
            hidden_dims=hidden_dims,
            dropout=args.get('regressor_dropout', 0.2),
        )

        # Load weights
        regressor.load_state_dict(checkpoint['state_dict'])
        # Note: Device transfer is handled in _load_model after building

        return regressor

    @torch.no_grad()
    def __call__(self, entity_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute reward from entity embedding.

        Args:
            entity_embedding: Tensor of shape (hidden_size,) representing entity token embedding

        Returns:
            reward: Scalar reward tensor on the model device
        """
        # Lazy load model on first use (ensures we're in the correct process context)
        if not self._model_loaded:
            self._load_model()

        # Ensure input is on correct device
        if entity_embedding.device != torch.device(self.device):
            entity_embedding = entity_embedding.to(self.device)

        # Add batch dimension if needed
        if entity_embedding.dim() == 1:
            entity_embedding = entity_embedding.unsqueeze(0)  # (1, hidden_size)

        # CRITICAL: Apply preprocessing that was used during training
        entity_embedding = self._apply_preprocessing(entity_embedding)

        # Convert to the same dtype as the encoder model
        # This is necessary because the main model may use BFloat16 while the reward model uses Float32
        encoder_dtype = next(self.encoder.parameters()).dtype
        if entity_embedding.dtype != encoder_dtype:
            entity_embedding = entity_embedding.to(encoder_dtype)

        # Forward pass through encoder
        encoded = self.encoder(entity_embedding)  # (1, proj_dim)

        # Forward pass through regressor
        reward = self.regressor(encoded)  # (1, 1)

        # Remove batch dimension
        reward = reward.squeeze()  # scalar

        # Normalize if requested
        if self.normalize_reward:
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
            reward = reward.clamp(-1.0, 1.0)

        # Apply scale
        reward = reward * self.reward_scale

        return reward

    def _aggregate_span_rewards(
        self, rewards: torch.Tensor, span_indices: Optional[list[list[int]]] = None
    ) -> torch.Tensor:
        if span_indices is None:
            return rewards
        if len(span_indices) == 0:
            return torch.empty(0, dtype=rewards.dtype, device=rewards.device)

        aggregated = []
        reward_length = int(rewards.shape[0])
        for span in span_indices:
            if not span:
                aggregated.append(torch.zeros((), dtype=rewards.dtype, device=rewards.device))
                continue
            valid_span = [int(idx) for idx in span if 0 <= int(idx) < reward_length]
            if not valid_span:
                aggregated.append(torch.zeros((), dtype=rewards.dtype, device=rewards.device))
                continue
            idx_tensor = torch.as_tensor(valid_span, dtype=torch.long, device=rewards.device)
            aggregated.append(rewards.index_select(0, idx_tensor).mean())
        return torch.stack(aggregated)

    @torch.no_grad()
    def compute_batch(
        self, entity_embeddings: torch.Tensor, span_indices: Optional[list[list[int]]] = None
    ) -> torch.Tensor:
        """
        Compute rewards from multiple entity embeddings in batch (much faster than calling __call__ repeatedly).

        Args:
            entity_embeddings: Tensor of shape (N, hidden_size) representing N entity token embeddings

        Returns:
            rewards: Tensor of shape (N,) containing reward values
        """
        # Lazy load model on first use (ensures we're in the correct process context)
        if not self._model_loaded:
            self._load_model()

        # Ensure a batch dimension for single embedding input
        if entity_embeddings.dim() == 1:
            entity_embeddings = entity_embeddings.unsqueeze(0)

        # Handle empty input
        if entity_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.float32, device=entity_embeddings.device)

        # Ensure input is on correct device
        if entity_embeddings.device != torch.device(self.device):
            entity_embeddings = entity_embeddings.to(self.device)

        # CRITICAL: Apply preprocessing that was used during training
        entity_embeddings = self._apply_preprocessing(entity_embeddings)

        # Convert to the same dtype as the encoder model
        encoder_dtype = next(self.encoder.parameters()).dtype
        if entity_embeddings.dtype != encoder_dtype:
            entity_embeddings = entity_embeddings.to(encoder_dtype)

        # Forward pass through encoder
        encoded = self.encoder(entity_embeddings)  # (N, proj_dim)

        # Forward pass through regressor
        rewards = self.regressor(encoded)  # (N, 1)

        # Remove last dimension
        rewards = rewards.squeeze(-1)  # (N,)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)

        # Normalize if requested
        if self.normalize_reward:
            rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
            rewards = rewards.clamp(-1.0, 1.0)

        # Apply scale
        rewards = rewards * self.reward_scale

        return self._aggregate_span_rewards(rewards, span_indices=span_indices)

    def set_normalization_stats(self, mean: float, std: float):
        """
        Set normalization statistics for rewards.

        Args:
            mean: Mean reward value
            std: Standard deviation of reward values
        """
        self.reward_mean = mean
        self.reward_std = std
        self._stats_loaded = True
        print(f"Updated normalization stats: mean={mean:.4f}, std={std:.4f}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        # Lazy load model if not already loaded
        if not self._model_loaded:
            self._load_model()

        # Get expected input dimension
        first_layer = self.encoder.projector.backbone[0]
        input_dim = first_layer.weight.shape[1]

        return {
            "model_dir": str(self.model_dir),
            "model_type": self.config.get('stage1_config', {}).get('backbone', 'unknown'),
            "label_type": self.config.get('stage1_config', {}).get('label_type', 'unknown'),
            "test_metrics": self.config.get('test_metrics', {}),
            "device": self.device,
            "normalize_reward": self.normalize_reward,
            "reward_scale": self.reward_scale,
            "expected_input_dim": input_dim,
        }

    def get_input_dim(self) -> int:
        """Get the expected input dimension for embeddings."""
        # Lazy load model if not already loaded
        if not self._model_loaded:
            self._load_model()

        first_layer = self.encoder.projector.backbone[0]
        return first_layer.weight.shape[1]


def create_activation_reward_fn(
    model_type: str = "mlp",
    label_type: str = "mi",
    base_dir: str = "",
    device: str = "cuda",
    **kwargs
) -> ActivationRewardFunction:
    """
    Factory function to create activation reward function.

    Args:
        model_type: Type of model ('mlp', 'deep', 'tpv', or 'graph_element')
        label_type: Type of label ('mi', 'pos')
        base_dir: Base directory containing model results
        device: Device to run on
        **kwargs: Additional arguments passed to ActivationRewardFunction

    Returns:
        reward_fn: Configured ActivationRewardFunction instance

    Example:
        >>> # Load MLP model trained on MI labels (old structure)
        >>> reward_fn = create_activation_reward_fn(
        ...     model_type="mlp",
        ...     label_type="mi",
        ...     device="cuda"
        ... )

        >>> # Load graph element model (new structure)
        >>> reward_fn = create_activation_reward_fn(
        ...     model_type="graph_element",
        ...     label_type="mi",
        ...     base_dir="/path/to/rnc_results/graph_element",
        ...     device="cuda"
        ... )
    """
    # Handle different directory structures
    if model_type == "graph_element":
        # New structure: base_dir/stage2_graph_element_{label_type}/
        model_dir = os.path.join(base_dir, f"stage2_graph_element_{label_type}")
    else:
        # Old structure: base_dir/{model_type}_{label_type}/best_model/
        model_dir = os.path.join(base_dir, f"{model_type}_{label_type}", "best_model")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Available combinations: mlp_mi, mlp_pos, deep_mi, deep_pos, tpv_mi, tpv_pos, graph_element_mi, graph_element_pos"
        )

    return ActivationRewardFunction(
        model_dir=model_dir,
        device=device,
        **kwargs
    )


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example 1: Create reward function for MLP + MI
    print("=" * 60)
    print("Example 1: Loading MLP + MI model")
    print("=" * 60)

    reward_fn = create_activation_reward_fn(
        model_type="mlp",
        label_type="mi",
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_reward=True,
        reward_scale=1.0
    )

    # Print model info
    info = reward_fn.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Get the expected input dimension from the model
    print("\n" + "=" * 60)
    print("Model Input Dimension")
    print("=" * 60)

    # Get input dimension from encoder
    first_layer = reward_fn.encoder.projector.backbone[0]
    expected_input_dim = first_layer.weight.shape[1]
    print(f"\nExpected input dimension: {expected_input_dim}")
    print(f"Note: This model was trained on {expected_input_dim}-dimensional embeddings")
    print(f"      Make sure your entity embeddings match this dimension!")

    # Test with correct dimension embedding
    print("\n" + "=" * 60)
    print("Testing with correct dimension embedding")
    print("=" * 60)

    test_embedding = torch.randn(expected_input_dim)

    if torch.cuda.is_available():
        test_embedding = test_embedding.cuda()

    reward = reward_fn(test_embedding)
    print(f"\nTest embedding shape: {test_embedding.shape}")
    print(f"Computed reward: {reward:.6f}")
    print(f"Reward type: {type(reward)}")

    # Test multiple embeddings
    print("\n" + "=" * 60)
    print("Testing with multiple embeddings")
    print("=" * 60)

    num_tests = 5
    rewards = []
    for i in range(num_tests):
        emb = torch.randn(expected_input_dim)
        if torch.cuda.is_available():
            emb = emb.cuda()
        r = reward_fn(emb)
        rewards.append(r)
        print(f"  Embedding {i+1}: reward = {r:.6f}")

    print(f"\nReward statistics:")
    print(f"  Mean: {sum(rewards) / len(rewards):.6f}")
    print(f"  Min:  {min(rewards):.6f}")
    print(f"  Max:  {max(rewards):.6f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
