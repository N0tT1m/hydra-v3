"""Partial model loading - load only specific layers for distributed inference."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
import structlog

log = structlog.get_logger()


class PartialTransformer(nn.Module):
    """A transformer that only contains a subset of layers.

    This module wraps layers [layer_start, layer_end) from a full model,
    plus optionally the embedding layer and lm_head.
    """

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

        # These will be populated by load_weights
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
        """Forward pass through partial layers.

        Args:
            hidden_states: Input embeddings or hidden states from previous node
                          Shape: [batch, seq_len, hidden_size] or [batch, seq_len] for tokens
            position_ids: Position IDs for RoPE
            attention_mask: Attention mask
            past_key_values: KV cache from previous forward passes
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: Output hidden states (or logits if has_lm_head)
            past_key_values: Updated KV cache
        """
        # If we have embedding layer and input is token IDs
        if self.has_embedding and hidden_states.dtype in (torch.long, torch.int):
            hidden_states = self.embed_tokens(hidden_states)

        # Ensure correct dtype
        hidden_states = hidden_states.to(self.dtype)

        # Initialize past_key_values if needed
        if past_key_values is None and use_cache:
            past_key_values = [None] * self.num_layers

        new_past_key_values = [] if use_cache else None

        # Forward through our layers
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            # Standard transformer layer forward
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

        # Apply final norm and lm_head if this is the last node
        if self.has_lm_head:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states, new_past_key_values


class PartialModelLoader:
    """Load only specific layers from a model for distributed inference."""

    # Weight name patterns for different architectures
    WEIGHT_PATTERNS = {
        "llama": {
            "embed": "model.embed_tokens.weight",
            "layer_prefix": "model.layers.",
            "norm": "model.norm.weight",
            "lm_head": "lm_head.weight",
        },
        "mistral": {
            "embed": "model.embed_tokens.weight",
            "layer_prefix": "model.layers.",
            "norm": "model.norm.weight",
            "lm_head": "lm_head.weight",
        },
        "qwen2": {
            "embed": "model.embed_tokens.weight",
            "layer_prefix": "model.layers.",
            "norm": "model.norm.weight",
            "lm_head": "lm_head.weight",
        },
        "phi3": {
            "embed": "model.embed_tokens.weight",
            "layer_prefix": "model.layers.",
            "norm": "model.norm.weight",
            "lm_head": "lm_head.weight",
        },
    }

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self.original_model_path = model_path

        # Check if it's a local path or HuggingFace model ID
        local_path = Path(model_path)
        if local_path.exists():
            self.model_path = local_path
        else:
            # Download from HuggingFace
            log.info("Downloading model from HuggingFace", model=model_path)
            cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            try:
                # Download only config and safetensors files first
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
        self.patterns = self.WEIGHT_PATTERNS.get(self.arch, self.WEIGHT_PATTERNS["llama"])

        # Find safetensors files
        self.weight_files = self._find_weight_files()
        self.weight_index = self._build_weight_index()

        log.info(
            "Initialized partial loader",
            model=model_path,
            arch=self.arch,
            num_layers=self.config.num_hidden_layers,
            files=len(self.weight_files),
        )

    def _detect_architecture(self) -> str:
        """Detect model architecture from config."""
        model_type = getattr(self.config, "model_type", "").lower()

        if "llama" in model_type:
            return "llama"
        elif "mistral" in model_type or "mixtral" in model_type:
            return "mistral"
        elif "qwen" in model_type:
            return "qwen2"
        elif "phi" in model_type:
            return "phi3"
        else:
            log.warning(f"Unknown model type {model_type}, defaulting to llama patterns")
            return "llama"

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
        # Check for index file
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

    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load all weights for a specific layer."""
        prefix = f"{self.patterns['layer_prefix']}{layer_idx}."
        weights = {}

        for name, file_path in self.weight_index.items():
            if name.startswith(prefix):
                # Remove prefix for cleaner names
                short_name = name[len(prefix):]
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    weights[short_name] = f.get_tensor(name).to(self.dtype)

        return weights

    def load_partial_model(
        self,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> Tuple[PartialTransformer, AutoTokenizer]:
        """Load a partial model with only specified layers.

        Args:
            layer_start: First layer index (inclusive)
            layer_end: Last layer index (exclusive)
            include_embedding: Whether to include embedding layer
            include_lm_head: Whether to include final norm and lm_head

        Returns:
            Tuple of (partial model, tokenizer)
        """
        log.info(
            "Loading partial model",
            layers=f"{layer_start}-{layer_end}",
            embedding=include_embedding,
            lm_head=include_lm_head,
        )

        # Create partial model structure
        model = PartialTransformer(
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
            embed_name = self.patterns["embed"]
            if embed_name in self.weight_index:
                with safe_open(self.weight_index[embed_name], framework="pt", device="cpu") as f:
                    embed_weight = f.get_tensor(embed_name).to(self.dtype)

                model.embed_tokens = nn.Embedding(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                model.embed_tokens.weight.data.copy_(embed_weight.to(self.device))
                log.info("Loaded embedding layer")

        # Load transformer layers
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

        for layer_idx in range(layer_start, layer_end):
            # Create layer with correct config
            layer_config = LlamaConfig(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
                rms_norm_eps=getattr(self.config, "rms_norm_eps", 1e-5),
                rope_theta=getattr(self.config, "rope_theta", 10000.0),
                max_position_embeddings=self.config.max_position_embeddings,
            )

            layer = LlamaDecoderLayer(layer_config, layer_idx).to(self.device, self.dtype)

            # Load weights
            weights = self.get_layer_weights(layer_idx)
            self._load_layer_weights(layer, weights)

            model.layers.append(layer)
            log.debug(f"Loaded layer {layer_idx}")

        log.info(f"Loaded {len(model.layers)} transformer layers")

        # Load norm and lm_head if needed
        if include_lm_head:
            # Final norm
            norm_name = self.patterns["norm"]
            if norm_name in self.weight_index:
                with safe_open(self.weight_index[norm_name], framework="pt", device="cpu") as f:
                    norm_weight = f.get_tensor(norm_name).to(self.dtype)

                from transformers.models.llama.modeling_llama import LlamaRMSNorm
                model.norm = LlamaRMSNorm(self.config.hidden_size, eps=getattr(self.config, "rms_norm_eps", 1e-5))
                model.norm.weight.data.copy_(norm_weight)
                model.norm = model.norm.to(self.device, self.dtype)

            # LM head
            lm_head_name = self.patterns["lm_head"]
            if lm_head_name in self.weight_index:
                with safe_open(self.weight_index[lm_head_name], framework="pt", device="cpu") as f:
                    lm_head_weight = f.get_tensor(lm_head_name).to(self.dtype)

                model.lm_head = nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=False,
                    device=self.device,
                    dtype=self.dtype,
                )
                model.lm_head.weight.data.copy_(lm_head_weight.to(self.device))

            log.info("Loaded norm and lm_head")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return model, tokenizer

    def _load_layer_weights(self, layer: nn.Module, weights: Dict[str, torch.Tensor]):
        """Load weights into a transformer layer."""
        state_dict = layer.state_dict()

        for name, param in weights.items():
            if name in state_dict:
                state_dict[name].copy_(param.to(self.device))
            else:
                # Try to match with slight naming differences
                matched = False
                for state_name in state_dict.keys():
                    if name.replace(".", "_") in state_name or state_name.replace(".", "_") in name:
                        state_dict[state_name].copy_(param.to(self.device))
                        matched = True
                        break

                if not matched:
                    log.warning(f"Weight {name} not found in layer state dict")

        layer.load_state_dict(state_dict, strict=False)

    def estimate_memory(self, layer_start: int, layer_end: int) -> int:
        """Estimate GPU memory needed for layers in bytes."""
        bytes_per_param = 2 if self.dtype == torch.float16 else 4

        # Per-layer params (approximate)
        hidden = self.config.hidden_size
        intermediate = self.config.intermediate_size
        num_heads = self.config.num_attention_heads

        # Attention: Q, K, V, O
        attn_params = 4 * hidden * hidden
        # MLP: gate, up, down
        mlp_params = 3 * hidden * intermediate
        # Norms
        norm_params = 2 * hidden

        params_per_layer = attn_params + mlp_params + norm_params
        num_layers = layer_end - layer_start

        return num_layers * params_per_layer * bytes_per_param
