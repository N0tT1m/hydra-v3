"""Partial model loading - load only specific layers for distributed inference."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
import gc
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
import structlog

log = structlog.get_logger()


class PartialTransformer(nn.Module):
    """A transformer that only contains a subset of layers."""

    def __init__(
        self,
        config: Any,
        layer_start: int,
        layer_end: int,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.config = config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.dtype = dtype
        self.device = device

        self.hidden_size = config.hidden_size
        self.num_layers = layer_end - layer_start

        self.embed_tokens: Optional[nn.Embedding] = None
        self.layers: nn.ModuleList = nn.ModuleList()
        self.norm: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Linear] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """Forward pass through partial layers."""
        if self.has_embedding and hidden_states.dtype in (torch.long, torch.int):
            hidden_states = self.embed_tokens(hidden_states)

        if self.dtype in (torch.float16, torch.bfloat16, torch.float32):
            hidden_states = hidden_states.to(self.dtype)

        if past_key_values is None and use_cache:
            past_key_values = [None] * self.num_layers

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                new_past_key_values.append(layer_outputs[1])

        if self.has_lm_head:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states, new_past_key_values


def parse_dtype(dtype_str: str) -> Tuple[torch.dtype, bool, int]:
    """Parse dtype string to (torch dtype, is_quantized, bits)."""
    dtype_map = {
        "float16": (torch.float16, False, 16),
        "bfloat16": (torch.bfloat16, False, 16),
        "float32": (torch.float32, False, 32),
        "int8": (torch.float16, True, 8),
        "int4": (torch.bfloat16, True, 4),
        "fp8": (torch.float16, True, 8),
    }
    return dtype_map.get(dtype_str, (torch.bfloat16, False, 16))


class PartialModelLoader:
    """Load only specific layers from a model for distributed inference.

    Uses selective weight loading - only loads weights for assigned layers,
    not the full model. This enables true distributed loading across GPUs.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cuda"),
        dtype: Union[str, torch.dtype] = "bfloat16",
    ):
        self.device = device

        if isinstance(dtype, str):
            self.dtype, self.quantize, self.quant_bits = parse_dtype(dtype)
            self.dtype_str = dtype
        else:
            self.dtype = dtype
            self.quantize = False
            self.quant_bits = 16
            self.dtype_str = str(dtype).split(".")[-1]

        self.original_model_path = model_path

        # Download or locate model
        local_path = Path(model_path)
        if local_path.exists():
            self.model_path = local_path
        else:
            log.info("Downloading model from HuggingFace", model=model_path)
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            try:
                self.model_path = Path(snapshot_download(
                    model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
                    ignore_patterns=["*.bin", "*.pt", "*.gguf"],
                ))
                log.info("Model downloaded", path=str(self.model_path))
            except Exception as e:
                log.error("Failed to download model", error=str(e))
                raise

        # Load config
        self.config = AutoConfig.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.arch = self._detect_architecture()
        self.is_moe = self._is_moe_model()

        # Build weight index
        self.weight_files = self._find_weight_files()
        self.weight_index = self._build_weight_index()

        log.info(
            "Initialized partial loader",
            model=model_path,
            arch=self.arch,
            is_moe=self.is_moe,
            num_layers=self.config.num_hidden_layers,
            dtype=self.dtype_str,
            weight_files=len(self.weight_files),
        )

    def _detect_architecture(self) -> str:
        """Detect model architecture from config."""
        model_type = getattr(self.config, "model_type", "").lower()

        if "llama" in model_type:
            return "llama"
        elif "mistral" in model_type and "moe" not in model_type:
            return "mistral"
        elif "mixtral" in model_type or ("mistral" in model_type and "moe" in model_type):
            return "mixtral"
        elif "qwen2_moe" in model_type or "qwen3" in model_type:
            return "qwen2_moe"
        elif "qwen" in model_type:
            return "qwen2"
        elif "phi" in model_type:
            return "phi3"
        else:
            log.warning(f"Unknown model type {model_type}, will use auto-detection")
            return "auto"

    def _is_moe_model(self) -> bool:
        """Check if model is Mixture of Experts."""
        if hasattr(self.config, "num_experts"):
            return self.config.num_experts > 1
        if hasattr(self.config, "num_local_experts"):
            return self.config.num_local_experts > 1
        if hasattr(self.config, "num_experts_per_tok"):
            return True
        if "moe" in self.arch or "mixtral" in self.arch:
            return True
        return False

    def _find_weight_files(self) -> List[Path]:
        """Find all safetensors files."""
        if self.model_path.is_file():
            return [self.model_path]

        files = list(self.model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files in {self.model_path}")

        return sorted(files)

    def _build_weight_index(self) -> Dict[str, Path]:
        """Build index of weight name -> file path."""
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            return {
                name: self.model_path / filename
                for name, filename in index["weight_map"].items()
            }

        # Build index by scanning files
        index = {}
        for file_path in self.weight_files:
            with safe_open(file_path, framework="pt") as f:
                for name in f.keys():
                    index[name] = file_path

        return index

    def _get_layer_prefix(self) -> str:
        """Get the weight name prefix for layers based on architecture."""
        # Most models use model.layers.X
        return "model.layers."

    def _load_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Load a single tensor by name."""
        if name not in self.weight_index:
            return None

        file_path = self.weight_index[name]
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(self.dtype)

    def _get_layer_weight_names(self, layer_idx: int) -> List[str]:
        """Get all weight names for a specific layer."""
        prefix = f"{self._get_layer_prefix()}{layer_idx}."
        return [name for name in self.weight_index.keys() if name.startswith(prefix)]

    def load_partial_model(
        self,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> Tuple[PartialTransformer, AutoTokenizer]:
        """Load a partial model with only specified layers.

        Uses selective loading - only loads weights for the assigned layers,
        not the full model. This enables distributed loading across multiple GPUs.
        """
        log.info(
            "Loading partial model (selective)",
            layers=f"{layer_start}-{layer_end}",
            embedding=include_embedding,
            lm_head=include_lm_head,
            dtype=self.dtype_str,
        )

        # Import the correct model class
        model_class = self._get_model_class()

        # Create partial model structure
        partial = PartialTransformer(
            config=self.config,
            layer_start=layer_start,
            layer_end=layer_end,
            has_embedding=include_embedding,
            has_lm_head=include_lm_head,
            dtype=self.dtype,
            device=self.device,
        )

        # Load embedding if needed
        if include_embedding:
            embed_weight = self._load_tensor("model.embed_tokens.weight")
            if embed_weight is not None:
                partial.embed_tokens = nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                partial.embed_tokens.weight.data.copy_(embed_weight.to(self.device))
                del embed_weight
                log.info("Loaded embedding layer")

        # Load transformer layers selectively
        for layer_idx in range(layer_start, layer_end):
            layer = self._create_and_load_layer(layer_idx, model_class)
            partial.layers.append(layer)
            log.info(f"Loaded layer {layer_idx}")

            # Clear CUDA cache periodically
            if torch.cuda.is_available() and (layer_idx - layer_start) % 4 == 0:
                torch.cuda.empty_cache()

        log.info(f"Loaded {len(partial.layers)} transformer layers")

        # Load norm and lm_head if needed
        if include_lm_head:
            norm_weight = self._load_tensor("model.norm.weight")
            if norm_weight is not None:
                partial.norm = self._create_rms_norm()
                partial.norm.weight.data.copy_(norm_weight.to(self.device))
                partial.norm = partial.norm.to(self.device)
                del norm_weight

            lm_head_weight = self._load_tensor("lm_head.weight")
            if lm_head_weight is not None:
                partial.lm_head = nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                partial.lm_head.weight.data.copy_(lm_head_weight.to(self.device))
                del lm_head_weight

            log.info("Loaded norm and lm_head")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return partial, tokenizer

    def _get_model_class(self):
        """Get the appropriate model/layer class for this architecture."""
        model_type = getattr(self.config, "model_type", "").lower()

        # Return config so we can create layers with it
        return self.config

    def _create_rms_norm(self):
        """Create RMSNorm layer."""
        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            return LlamaRMSNorm(
                self.config.hidden_size,
                eps=getattr(self.config, "rms_norm_eps", 1e-5)
            )
        except ImportError:
            # Fallback to manual implementation
            return RMSNorm(self.config.hidden_size, getattr(self.config, "rms_norm_eps", 1e-5))

    def _create_and_load_layer(self, layer_idx: int, model_class) -> nn.Module:
        """Create a layer and load its weights."""
        model_type = getattr(self.config, "model_type", "").lower()

        # Get all weights for this layer
        weight_names = self._get_layer_weight_names(layer_idx)
        weights = {}
        for name in weight_names:
            short_name = name.replace(f"{self._get_layer_prefix()}{layer_idx}.", "")
            tensor = self._load_tensor(name)
            if tensor is not None:
                # Apply quantization if requested
                if self.quantize and tensor.numel() > 1024:  # Only quantize larger tensors
                    tensor = self._quantize_tensor(tensor)
                weights[short_name] = tensor

        # Create the appropriate layer based on model type
        if "qwen" in model_type:
            layer = self._create_qwen_layer(layer_idx, weights)
        elif "llama" in model_type:
            layer = self._create_llama_layer(layer_idx, weights)
        elif "mistral" in model_type or "mixtral" in model_type:
            layer = self._create_mistral_layer(layer_idx, weights)
        else:
            # Generic fallback - try to auto-detect from weight names
            layer = self._create_generic_layer(layer_idx, weights)

        return layer.to(self.device)

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to int8 or int4."""
        if self.quant_bits == 8:
            # Simple int8 quantization
            scale = tensor.abs().max() / 127.0
            if scale == 0:
                return tensor.to(self.dtype)
            quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            # Store as float for now (proper int8 inference needs special kernels)
            return (quantized.float() * scale).to(self.dtype)
        elif self.quant_bits == 4:
            # Simple int4-like quantization (stored as int8)
            scale = tensor.abs().max() / 7.0
            if scale == 0:
                return tensor.to(self.dtype)
            quantized = (tensor / scale).round().clamp(-7, 7).to(torch.int8)
            return (quantized.float() * scale).to(self.dtype)
        else:
            return tensor.to(self.dtype)

    def _create_qwen_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Qwen/Qwen2 layer."""
        model_type = getattr(self.config, "model_type", "").lower()

        # Ensure config has required attributes with defaults
        self._patch_config_defaults()

        if self.is_moe or "moe" in model_type or "qwen3" in model_type:
            # Qwen2 MoE / Qwen3 MoE
            try:
                from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer
                layer = Qwen3MoeDecoderLayer(self.config, layer_idx)
            except (ImportError, AttributeError) as e:
                log.debug(f"Qwen3MoeDecoderLayer failed: {e}, trying Qwen2Moe")
                try:
                    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
                    layer = Qwen2MoeDecoderLayer(self.config, layer_idx)
                except (ImportError, AttributeError) as e:
                    log.warning(f"Qwen MoE layer not available ({e}), using generic")
                    return self._create_generic_layer(layer_idx, weights)
        else:
            # Dense Qwen2
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
                layer = Qwen2DecoderLayer(self.config, layer_idx)
            except (ImportError, AttributeError):
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                layer = LlamaDecoderLayer(self.config, layer_idx)

        self._load_weights_into_layer(layer, weights)
        return layer

    def _patch_config_defaults(self):
        """Add missing config attributes with sensible defaults."""
        defaults = {
            "qkv_bias": True,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "rope_scaling": None,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": 0,
        }
        for attr, default in defaults.items():
            if not hasattr(self.config, attr):
                setattr(self.config, attr, default)

    def _create_llama_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Llama layer."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        layer = LlamaDecoderLayer(self.config, layer_idx)
        self._load_weights_into_layer(layer, weights)
        return layer

    def _create_mistral_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Mistral/Mixtral layer."""
        if self.is_moe:
            try:
                from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
                layer = MixtralDecoderLayer(self.config, layer_idx)
            except ImportError:
                return self._create_generic_layer(layer_idx, weights)
        else:
            try:
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                layer = MistralDecoderLayer(self.config, layer_idx)
            except ImportError:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                layer = LlamaDecoderLayer(self.config, layer_idx)

        self._load_weights_into_layer(layer, weights)
        return layer

    def _create_generic_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a generic layer using Llama as fallback."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

        # Create a LlamaConfig from the model config
        llama_config = LlamaConfig(
            hidden_size=self.config.hidden_size,
            intermediate_size=getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
            rms_norm_eps=getattr(self.config, "rms_norm_eps", 1e-5),
            rope_theta=getattr(self.config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(self.config, "max_position_embeddings", 4096),
        )

        layer = LlamaDecoderLayer(llama_config, layer_idx)
        self._load_weights_into_layer(layer, weights)
        return layer

    def _load_weights_into_layer(self, layer: nn.Module, weights: Dict[str, torch.Tensor]):
        """Load weights into a layer, handling name mismatches."""
        state_dict = layer.state_dict()
        loaded = 0
        missing = []

        for weight_name, weight_tensor in weights.items():
            # Try direct match
            if weight_name in state_dict:
                if state_dict[weight_name].shape == weight_tensor.shape:
                    state_dict[weight_name].copy_(weight_tensor)
                    loaded += 1
                else:
                    log.warning(f"Shape mismatch for {weight_name}: "
                              f"expected {state_dict[weight_name].shape}, got {weight_tensor.shape}")
            else:
                missing.append(weight_name)

        layer.load_state_dict(state_dict, strict=False)

        if missing and len(missing) < 10:  # Only log if not too many
            log.debug(f"Layer weights not matched: {missing[:5]}...")

    def estimate_memory(self, layer_start: int, layer_end: int) -> int:
        """Estimate GPU memory needed for layers in bytes."""
        bytes_per_param = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

        hidden = self.config.hidden_size
        intermediate = getattr(self.config, "intermediate_size", hidden * 4)

        if self.is_moe:
            num_experts = getattr(self.config, "num_experts", getattr(self.config, "num_local_experts", 8))
            mlp_params = num_experts * 3 * hidden * intermediate
        else:
            mlp_params = 3 * hidden * intermediate

        attn_params = 4 * hidden * hidden
        norm_params = 2 * hidden

        params_per_layer = attn_params + mlp_params + norm_params
        num_layers = layer_end - layer_start

        return int(num_layers * params_per_layer * bytes_per_param)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states
