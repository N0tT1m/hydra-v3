"""Partial model loading - load only specific layers for distributed inference."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
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
    """Load only specific layers from a model for distributed inference.

    Uses transformers native classes for proper architecture support.
    """

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

        log.info(
            "Initialized partial loader",
            model=model_path,
            arch=self.arch,
            is_moe=self.is_moe,
            num_layers=self.config.num_hidden_layers,
        )

    def _detect_architecture(self) -> str:
        """Detect model architecture from config."""
        model_type = getattr(self.config, "model_type", "").lower()

        if "llama" in model_type:
            return "llama"
        elif "mistral" in model_type:
            return "mistral"
        elif "mixtral" in model_type:
            return "mixtral"
        elif "qwen2_moe" in model_type or "qwen3" in model_type:
            return "qwen2_moe"
        elif "qwen" in model_type:
            return "qwen2"
        elif "phi" in model_type:
            return "phi3"
        else:
            log.warning(f"Unknown model type {model_type}, defaulting to llama")
            return "llama"

    def _is_moe_model(self) -> bool:
        """Check if model is Mixture of Experts."""
        # Check various config attributes that indicate MoE
        if hasattr(self.config, "num_experts"):
            return self.config.num_experts > 1
        if hasattr(self.config, "num_local_experts"):
            return self.config.num_local_experts > 1
        if hasattr(self.config, "num_experts_per_tok"):
            return True
        if "moe" in self.arch or "mixtral" in self.arch:
            return True
        return False

    def load_partial_model(
        self,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> Tuple[PartialTransformer, AutoTokenizer]:
        """Load a partial model with only specified layers.

        Uses transformers' native model loading for proper architecture support.

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

        # Load full model with low memory usage
        log.info("Loading model via transformers (this may take a moment)...")

        full_model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            device_map="cpu",  # Load to CPU first
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        log.info("Full model loaded to CPU, extracting layers...")

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

        # Get the model's internal structure
        # Most models use model.model.layers or model.transformer.layers
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"):
            # Llama, Mistral, Qwen style
            inner_model = full_model.model
            layers = inner_model.layers
            embed_tokens = inner_model.embed_tokens
            norm = inner_model.norm
        elif hasattr(full_model, "transformer") and hasattr(full_model.transformer, "h"):
            # GPT style
            inner_model = full_model.transformer
            layers = inner_model.h
            embed_tokens = inner_model.wte
            norm = inner_model.ln_f
        else:
            raise ValueError(f"Unknown model structure for {self.arch}")

        # Copy embedding if needed
        if include_embedding:
            partial.embed_tokens = embed_tokens.to(self.device)
            log.info("Loaded embedding layer")

        # Copy only the layers we need
        for idx in range(layer_start, layer_end):
            layer = layers[idx].to(self.device)
            partial.layers.append(layer)
            log.info(f"Loaded layer {idx}")

        log.info(f"Loaded {len(partial.layers)} transformer layers")

        # Copy norm and lm_head if needed
        if include_lm_head:
            partial.norm = norm.to(self.device)
            partial.lm_head = full_model.lm_head.to(self.device)
            log.info("Loaded norm and lm_head")

        # Free the full model from memory
        del full_model
        del layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        log.info("Cleaned up full model from memory")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return partial, tokenizer

    def estimate_memory(self, layer_start: int, layer_end: int) -> int:
        """Estimate GPU memory needed for layers in bytes."""
        bytes_per_param = 2 if self.dtype == torch.float16 else 4

        # Per-layer params (approximate)
        hidden = self.config.hidden_size
        intermediate = getattr(self.config, "intermediate_size", hidden * 4)

        # Check for MoE
        if self.is_moe:
            num_experts = getattr(self.config, "num_experts", getattr(self.config, "num_local_experts", 8))
            # MoE has multiple expert MLPs
            mlp_params = num_experts * 3 * hidden * intermediate
        else:
            # Dense: gate, up, down projections (for SwiGLU)
            mlp_params = 3 * hidden * intermediate

        # Attention: Q, K, V, O
        attn_params = 4 * hidden * hidden
        # Norms
        norm_params = 2 * hidden

        params_per_layer = attn_params + mlp_params + norm_params
        num_layers = layer_end - layer_start

        return num_layers * params_per_layer * bytes_per_param
