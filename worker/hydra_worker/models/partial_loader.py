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

# Try to import bitsandbytes for proper int8/int4 quantization
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    log.warning("bitsandbytes not available - quantization will use fallback method")


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
        self.rotary_emb: Optional[nn.Module] = None

        # Preallocated causal mask; grown on demand. Reusing one tensor
        # avoids allocating a fresh [T, total] tensor on every forward,
        # which matters in the tight decode loop.
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_cache_dtype: Optional[torch.dtype] = None

    def _get_causal_mask(
        self, seq_len: int, total_len: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Return a (1,1,seq_len,total_len) causal mask, reusing one buffer.

        We build a single upper-bound max-length mask lazily, and slice it for
        each call. The expensive `triu` + allocation only happen when the
        buffer needs to grow.
        """
        need = max(seq_len, total_len)
        need_dtype = dtype

        cached = self._mask_cache
        if (
            cached is None
            or cached.shape[-1] < need
            or cached.shape[-2] < need
            or cached.dtype != need_dtype
            or cached.device != device
        ):
            # Allocate slightly larger than requested to amortize growth.
            grow_to = max(need, 64)
            if cached is not None:
                grow_to = max(grow_to, cached.shape[-1] * 2)
            base = torch.full(
                (grow_to, grow_to),
                torch.finfo(need_dtype).min,
                device=device,
                dtype=need_dtype,
            )
            # Upper-triangular above the main diagonal are the "future"
            # positions we want masked out. We select a causal slice per
            # call below rather than baking in a specific past_len.
            self._mask_cache = torch.triu(base, diagonal=1)
            self._mask_cache_dtype = need_dtype

        # Slice: rows are query positions (last `seq_len` of the total),
        # columns are key positions (first `total_len`).
        mask = self._mask_cache[
            total_len - seq_len : total_len, :total_len
        ]
        return mask[None, None, :, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass through partial layers.

        `past_key_values` is threaded as a single Cache object across all
        layers (transformers `DynamicCache` convention). Each layer uses its
        own stored `layer_idx` to read/write its slot, so the caller can
        reuse the same cache for every generation step. None means "no
        history yet" — the layers will populate the cache on this pass.

        For compatibility with older tuple-format caches, callers may pass
        a list of per-layer tuples; we detect and index per layer in that
        case.
        """
        if hidden_states.device != self.device:
            hidden_states = hidden_states.to(self.device)
        if position_ids is not None and position_ids.device != self.device:
            position_ids = position_ids.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        if self.has_embedding and hidden_states.dtype in (torch.long, torch.int):
            hidden_states = self.embed_tokens(hidden_states)

        if self.dtype in (torch.float16, torch.bfloat16, torch.float32):
            hidden_states = hidden_states.to(self.dtype)

        # Decide cache shape. A list means "old tuple-per-layer" — pass
        # layer_past[i] to layer i. Otherwise treat as a DynamicCache-style
        # object and pass it whole to each layer; layers mutate it in place
        # keyed by their stored layer_idx.
        per_layer_list = isinstance(past_key_values, list)

        position_embeddings = None
        if self.rotary_emb is not None and position_ids is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Current cache length drives both the causal mask extent and the
        # cache_position tensor layers now require.
        past_len_cached = _cache_seq_length(past_key_values, self.layer_start)
        seq_len = hidden_states.shape[1]
        total_len = past_len_cached + seq_len

        # Build (or grow) a cached causal attention mask when the caller
        # didn't supply one. Without this, attention defaults to "attend to
        # everything" which blows up quality for generation.
        if attention_mask is None:
            attention_mask = self._get_causal_mask(
                seq_len, total_len, hidden_states.dtype, hidden_states.device
            )

        # cache_position: absolute positions in the cache that this forward
        # writes to. Required by transformers 5.x layer attention paths.
        cache_position = torch.arange(
            past_len_cached, total_len, device=hidden_states.device
        )

        for i, layer in enumerate(self.layers):
            if per_layer_list:
                layer_past = past_key_values[i] if i < len(past_key_values) else None
            else:
                layer_past = past_key_values  # shared Cache object

            # Transformers 5.0 renamed this kwarg from `past_key_value` to
            # `past_key_values` (plural) AND the return type changed from a
            # tuple to a bare tensor. Passing the old name was a silent no-op
            # that broke KV-cache accumulation.
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=layer_past,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if per_layer_list and use_cache and len(layer_outputs) > 1:
                    if i < len(past_key_values):
                        past_key_values[i] = layer_outputs[1]
                    else:
                        past_key_values.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

        if self.has_lm_head:
            # Only the last position's logits matter for autoregressive
            # sampling. Running lm_head on all T prompt positions produces
            # a [B, T, vocab] tensor that's often hundreds of MB for large
            # vocabularies (Qwen2.5 vocab=152k); computing + moving that
            # across the MPS/CPU boundary for a 41-token prompt costs tens
            # of seconds for zero benefit. Slice first.
            hidden_states = hidden_states[:, -1:, :]
            hidden_states = self.norm(hidden_states)
            if self.lm_head.weight.dtype != hidden_states.dtype:
                hidden_states = hidden_states.to(self.lm_head.weight.dtype)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states, past_key_values


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
            dtype = self._maybe_downgrade_dtype(dtype, device)
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
        # Log quantization status
        quant_info = ""
        if self.quantize:
            if HAS_BITSANDBYTES:
                quant_info = f" with {self.quant_bits}-bit quantization (bitsandbytes)"
            else:
                quant_info = f" (quantization unavailable - bitsandbytes not installed)"

        log.info(
            "Loading partial model (selective)",
            layers=f"{layer_start}-{layer_end}",
            embedding=include_embedding,
            lm_head=include_lm_head,
            dtype=self.dtype_str,
            quantization=f"{self.quant_bits}-bit" if self.quantize else "none",
            bitsandbytes=HAS_BITSANDBYTES,
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

        # Create rotary embedding module (needed for position embeddings)
        # ALL workers need this, not just the first one
        model_type = getattr(self.config, "model_type", "").lower()
        try:
            if "qwen3" in model_type or (self.is_moe and "qwen" in model_type):
                from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        partial.rotary_emb = Qwen3MoeRotaryEmbedding(self.config).to(self.device)
                else:
                    partial.rotary_emb = Qwen3MoeRotaryEmbedding(self.config).to(self.device)
                log.info("Created rotary embedding (Qwen3MoE)")
            else:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        partial.rotary_emb = Qwen2RotaryEmbedding(self.config).to(self.device)
                else:
                    partial.rotary_emb = Qwen2RotaryEmbedding(self.config).to(self.device)
                log.info("Created rotary embedding (Qwen2)")
        except ImportError as e:
            log.warning(f"Could not create rotary embedding: {e}")

        # Load transformer layers selectively
        for layer_idx in range(layer_start, layer_end):
            layer = self._create_and_load_layer(layer_idx, model_class)
            partial.layers.append(layer)

            # Log memory usage (device-aware)
            if self.device.type == "cuda":
                dev_idx = self.device.index or 0
                mem_used = torch.cuda.memory_allocated(dev_idx) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(dev_idx) / (1024**3)
                log.info(
                    f"Loaded layer {layer_idx}",
                    device=str(self.device),
                    mem_used_gb=f"{mem_used:.2f}",
                    mem_reserved_gb=f"{mem_reserved:.2f}",
                )
            else:
                log.info(f"Loaded layer {layer_idx}", device=str(self.device))

            # Clear CUDA cache periodically (on the correct device)
            if self.device.type == "cuda" and (layer_idx - layer_start) % 4 == 0:
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

            # Many modern small LMs (Qwen2.5, Llama-3.2, Gemma) set
            # tie_word_embeddings=True and ship without a separate lm_head
            # weight — the lm_head is supposed to use embed_tokens.weight.
            # If we don't fall back to the embedding, partial.lm_head stays
            # None and the forward pass blows up on `lm_head.weight.dtype`.
            lm_head_weight = self._load_tensor("lm_head.weight")
            tie_embeddings = getattr(self.config, "tie_word_embeddings", False)
            if lm_head_weight is None and tie_embeddings:
                if partial.embed_tokens is not None:
                    lm_head_weight = partial.embed_tokens.weight.data
                    log.info("lm_head absent; reusing embed_tokens (tied)")
                else:
                    # We own lm_head but not embedding — load the embedding
                    # weight directly for reuse.
                    lm_head_weight = self._load_tensor("model.embed_tokens.weight")
                    if lm_head_weight is not None:
                        log.info("lm_head absent; loaded embed_tokens.weight for reuse (tied)")

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
            else:
                # We claimed has_lm_head but can't materialize the weight —
                # flip the flag so forward() doesn't NPE, and log loudly.
                log.error(
                    "include_lm_head=True but no lm_head weight could be found "
                    "(checked lm_head.weight and tied embeddings). Disabling "
                    "has_lm_head; sampling will fail."
                )
                partial.has_lm_head = False

            log.info("Loaded norm and lm_head", has_lm_head=partial.has_lm_head)

        # Cleanup
        gc.collect()
        if self.device.type == "cuda":
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

        # Get all weights for this layer (load in float16/bfloat16 first)
        weight_names = self._get_layer_weight_names(layer_idx)
        weights = {}
        for name in weight_names:
            short_name = name.replace(f"{self._get_layer_prefix()}{layer_idx}.", "")
            tensor = self._load_tensor(name)
            if tensor is not None:
                weights[short_name] = tensor

        # Create the appropriate layer based on model type (on CPU)
        if "qwen" in model_type:
            layer = self._create_qwen_layer(layer_idx, weights)
        elif "llama" in model_type:
            layer = self._create_llama_layer(layer_idx, weights)
        elif "mistral" in model_type or "mixtral" in model_type:
            layer = self._create_mistral_layer(layer_idx, weights)
        else:
            # Generic fallback - try to auto-detect from weight names
            layer = self._create_generic_layer(layer_idx, weights)

        # Apply quantization (replaces linear layers with bitsandbytes versions)
        if self.quantize:
            layer = self._quantize_layer(layer)
            log.debug(f"Layer {layer_idx} quantized to {self.quant_bits}-bit")

        # Move to target device with proper context guard
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                return layer.to(self.device)
        return layer.to(self.device)

    def _quantize_layer(self, layer: nn.Module) -> nn.Module:
        """Replace linear layers with quantized versions using bitsandbytes."""
        if not self.quantize:
            return layer

        if not HAS_BITSANDBYTES:
            log.warning("Quantization requested but bitsandbytes not available")
            return layer

        replaced_linears = 0
        replaced_moe = 0
        failed = 0

        def _replace_linears(module: nn.Module, prefix: str = ""):
            nonlocal replaced_linears, replaced_moe, failed
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name

                # Check for MoE experts module (has gate_up_proj and down_proj as 3D tensors)
                if self._is_moe_experts_module(child):
                    success = self._quantize_moe_experts(child)
                    if success:
                        replaced_moe += 1
                    else:
                        failed += 1
                    continue

                if isinstance(child, nn.Linear):
                    # Replace with bitsandbytes quantized linear
                    if self.quant_bits == 8:
                        quantized_linear = self._create_int8_linear(child)
                    elif self.quant_bits == 4:
                        quantized_linear = self._create_int4_linear(child)
                    else:
                        continue

                    if quantized_linear is not None:
                        setattr(module, name, quantized_linear)
                        replaced_linears += 1
                    else:
                        failed += 1
                else:
                    # Recursively quantize nested modules
                    _replace_linears(child, full_name)

        _replace_linears(layer)

        if replaced_linears > 0 or replaced_moe > 0 or failed > 0:
            log.debug(
                f"Quantization: {replaced_linears} linears, {replaced_moe} MoE experts, {failed} failed"
            )

        return layer

    def _is_moe_experts_module(self, module: nn.Module) -> bool:
        """Check if module is an MoE experts container (Qwen3MoeExperts, etc.)."""
        # Check for 3D weight tensors that indicate MoE experts
        has_gate_up = hasattr(module, 'gate_up_proj') and isinstance(
            getattr(module, 'gate_up_proj', None), (nn.Parameter, torch.Tensor)
        )
        has_down = hasattr(module, 'down_proj') and isinstance(
            getattr(module, 'down_proj', None), (nn.Parameter, torch.Tensor)
        )

        if has_gate_up and has_down:
            gate_up = module.gate_up_proj
            down = module.down_proj
            # MoE experts have 3D weight tensors: (num_experts, out_features, in_features)
            return len(gate_up.shape) == 3 and len(down.shape) == 3

        return False

    def _quantize_moe_experts(self, experts_module: nn.Module) -> bool:
        """Quantize MoE expert weight tensors in-place."""
        try:
            # Get the expert weight tensors
            gate_up = experts_module.gate_up_proj.data
            down = experts_module.down_proj.data

            num_experts = gate_up.shape[0]
            original_size = (gate_up.numel() + down.numel()) * gate_up.element_size()

            if self.quant_bits == 8:
                # Quantize to int8 with per-expert scale
                gate_up_q, gate_up_scale = self._quantize_tensor_int8(gate_up)
                down_q, down_scale = self._quantize_tensor_int8(down)

                # Delete original tensors to free memory
                del gate_up, down
                gc.collect()

                # Store quantized versions
                experts_module.register_buffer('gate_up_proj_q', gate_up_q)
                experts_module.register_buffer('down_proj_q', down_q)
                experts_module.register_buffer('gate_up_scale', gate_up_scale.view(num_experts, 1, 1))
                experts_module.register_buffer('down_scale', down_scale.view(num_experts, 1, 1))

                # Replace original params with dummy to free memory
                experts_module.gate_up_proj = nn.Parameter(torch.empty(0), requires_grad=False)
                experts_module.down_proj = nn.Parameter(torch.empty(0), requires_grad=False)

                experts_module._quantized = True
                experts_module._quant_bits = 8

            elif self.quant_bits == 4:
                # 4-bit quantization
                gate_up_q, gate_up_scale = self._quantize_tensor_int4(gate_up)
                down_q, down_scale = self._quantize_tensor_int4(down)

                del gate_up, down
                gc.collect()

                experts_module.register_buffer('gate_up_proj_q', gate_up_q)
                experts_module.register_buffer('down_proj_q', down_q)
                experts_module.register_buffer('gate_up_scale', gate_up_scale.view(num_experts, 1, 1))
                experts_module.register_buffer('down_scale', down_scale.view(num_experts, 1, 1))

                experts_module.gate_up_proj = nn.Parameter(torch.empty(0), requires_grad=False)
                experts_module.down_proj = nn.Parameter(torch.empty(0), requires_grad=False)

                experts_module._quantized = True
                experts_module._quant_bits = 4

            # Patch the module's forward to handle quantized weights
            self._wrap_moe_experts_forward(experts_module)

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            quantized_size = (
                experts_module.gate_up_proj_q.numel() * experts_module.gate_up_proj_q.element_size() +
                experts_module.down_proj_q.numel() * experts_module.down_proj_q.element_size()
            )

            log.debug(
                f"Quantized MoE experts: {num_experts} experts, "
                f"{original_size / 1e6:.1f}MB -> {quantized_size / 1e6:.1f}MB"
            )
            return True

        except Exception as e:
            log.warning(f"Failed to quantize MoE experts: {e}", exc_info=True)
            return False

    # Scales are returned as float16, so the floor has to be representable
    # there: 1e-8 underflows to exactly 0.0 in fp16, and a zero scale turns
    # dequantization into a multiply-by-zero. finfo.tiny is the smallest
    # normal fp16 value.
    _SCALE_FLOOR = float(torch.finfo(torch.float16).tiny)

    def _quantize_tensor_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor to int4 (stored as int8) with per-expert scale."""
        if len(tensor.shape) == 3:
            scales = tensor.abs().amax(dim=(1, 2), keepdim=True) / 7.0
            scales = scales.clamp(min=self._SCALE_FLOOR)
        else:
            scales = (tensor.abs().max() / 7.0).clamp(min=self._SCALE_FLOOR).reshape(1)

        quantized = (tensor / scales).round().clamp(-7, 7).to(torch.int8)
        return quantized, scales.to(torch.float16).squeeze()

    def _wrap_moe_experts_forward(self, experts_module: nn.Module):
        """Wrap MoE experts to dequantize weights during forward pass.

        Only dequantizes the selected experts (typically 2-8 out of 128),
        not all experts, to avoid memory explosion.
        """
        # Store original forward
        if not hasattr(experts_module, '_orig_forward'):
            experts_module._orig_forward = experts_module.forward

        def dequantized_forward(hidden_states, selected_experts, routing_weights):
            """Forward with selective on-the-fly dequantization."""
            if not hasattr(experts_module, '_quantized') or not experts_module._quantized:
                return experts_module._orig_forward(hidden_states, selected_experts, routing_weights)

            # Get quantized weights
            gate_up_q = experts_module.gate_up_proj_q
            down_q = experts_module.down_proj_q
            gate_up_scale = experts_module.gate_up_scale
            down_scale = experts_module.down_scale

            # Get unique selected experts to minimize dequantization
            unique_experts = selected_experts.unique()
            log.debug(f"MoE forward: {len(unique_experts)} unique experts, input shape {hidden_states.shape}")

            # Process each token through its selected experts
            # hidden_states: (batch * seq_len, hidden_size) after reshaping
            # selected_experts: (batch * seq_len, num_experts_per_tok)
            # routing_weights: (batch * seq_len, num_experts_per_tok)

            batch_size = hidden_states.shape[0]
            hidden_size = hidden_states.shape[1]
            num_experts_per_tok = selected_experts.shape[1]

            # Get intermediate size from gate_up weights (it's 2x because gate and up are concatenated)
            intermediate_size = gate_up_q.shape[1] // 2

            # Output accumulator
            final_hidden_states = torch.zeros_like(hidden_states)

            # Process each unique expert
            for expert_idx in unique_experts:
                expert_idx = expert_idx.item()

                # Find which tokens use this expert and their positions
                expert_mask = (selected_experts == expert_idx)  # (batch, num_experts_per_tok)

                # Get the routing weights for this expert
                expert_weights = routing_weights.masked_fill(~expert_mask, 0.0)  # Zero out non-selected
                expert_weights_sum = expert_weights.sum(dim=1, keepdim=True)  # Sum across expert slots

                # Skip if no tokens use this expert
                if expert_weights_sum.sum() == 0:
                    continue

                # Dequantize only this expert's weights
                gate_up_w = gate_up_q[expert_idx].to(hidden_states.dtype) * gate_up_scale[expert_idx].to(hidden_states.dtype)
                down_w = down_q[expert_idx].to(hidden_states.dtype) * down_scale[expert_idx].to(hidden_states.dtype)

                # Forward through expert MLP: x -> gate_up -> split -> silu(gate) * up -> down
                intermediate = torch.nn.functional.linear(hidden_states, gate_up_w)
                gate, up = intermediate.chunk(2, dim=-1)
                intermediate = torch.nn.functional.silu(gate) * up
                expert_out = torch.nn.functional.linear(intermediate, down_w)

                # Weight by routing weights and accumulate
                final_hidden_states += expert_out * expert_weights_sum

                # Free intermediate tensors
                del gate_up_w, down_w, intermediate, gate, up, expert_out

            return final_hidden_states

        experts_module.forward = dequantized_forward

    def _quantize_tensor_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor to int8 with per-tensor scale."""
        # Compute scale per expert (first dimension)
        if len(tensor.shape) == 3:
            # Per-expert scaling for MoE
            scales = tensor.abs().amax(dim=(1, 2), keepdim=True) / 127.0
            scales = scales.clamp(min=self._SCALE_FLOOR)
        else:
            # clamp, not max(): builtin max on (0-dim tensor, float) returns
            # the *float* when the tensor is smaller, and a float has no
            # .to(). An all-zero weight tensor took exactly that path.
            scales = (tensor.abs().max() / 127.0).clamp(min=self._SCALE_FLOOR)

        quantized = (tensor / scales).round().clamp(-127, 127).to(torch.int8)
        return quantized, scales.to(torch.float16).squeeze()

    def _create_int8_linear(self, linear: nn.Linear) -> Optional[nn.Module]:
        """Create an int8 quantized linear layer from a regular linear layer."""
        try:
            has_bias = linear.bias is not None

            # Create Int8 linear layer
            int8_linear = bnb.nn.Linear8bitLt(
                linear.in_features,
                linear.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,  # Outlier threshold for mixed-precision
            )

            # Copy weights (bitsandbytes will quantize them on first forward pass or when moved to GPU)
            int8_linear.weight = bnb.nn.Int8Params(
                linear.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )

            if has_bias:
                int8_linear.bias = nn.Parameter(linear.bias.data.clone())

            return int8_linear
        except Exception as e:
            log.warning(f"Failed to create int8 linear: {e}")
            return None

    def _create_int4_linear(self, linear: nn.Linear) -> Optional[nn.Module]:
        """Create a 4-bit quantized linear layer from a regular linear layer."""
        try:
            has_bias = linear.bias is not None

            # Use bitsandbytes 4-bit quantization (NF4)
            int4_linear = bnb.nn.Linear4bit(
                linear.in_features,
                linear.out_features,
                bias=has_bias,
                compute_dtype=self.dtype,
                compress_statistics=True,
                quant_type="nf4",  # Normal Float 4-bit
            )

            # Copy weights
            int4_linear.weight = bnb.nn.Params4bit(
                linear.weight.data.clone(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )

            if has_bias:
                int4_linear.bias = nn.Parameter(linear.bias.data.clone())

            return int4_linear
        except Exception as e:
            log.warning(f"Failed to create int4 linear: {e}")
            return None

    def _create_qwen_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Qwen/Qwen2 layer on CPU, load weights, then move to GPU."""
        model_type = getattr(self.config, "model_type", "").lower()

        # Ensure config has required attributes with defaults
        self._patch_config_defaults()

        # Create layer on CPU to avoid double GPU memory allocation
        with torch.device('cpu'):
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

        # Load weights on CPU
        self._load_weights_into_layer(layer, weights)

        # Now move to GPU (single transfer, no double allocation)
        return layer

    def _patch_config_defaults(self):
        """Add missing config attributes with sensible defaults."""
        attn_impl = self._pick_attn_impl()
        defaults = {
            "qkv_bias": True,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "rope_scaling": None,
            "use_sliding_window": False,
            "sliding_window": None,
            "max_window_layers": 0,
            "_attn_implementation": attn_impl,
            # MoE expert dispatch is more fragile across transformers
            # versions — keep it eager even when attention is sped up.
            "_experts_implementation": "eager",
        }
        for attr, default in defaults.items():
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                setattr(self.config, attr, default)

    @staticmethod
    def _maybe_downgrade_dtype(requested: str, device: torch.device) -> str:
        """bf16 on MPS has gaps in kernel coverage (many ops fall back to
        CPU), so inference runs at a small fraction of fp16 speed. Downgrade
        silently with a warning so users don't hit a footgun."""
        if device.type == "mps" and requested == "bfloat16":
            log.warning(
                "MPS: downgrading bfloat16 -> float16 (MPS bf16 kernels are "
                "incomplete; many ops silently fall back to CPU)."
            )
            return "float16"
        return requested

    def _pick_attn_impl(self) -> str:
        """Select the best attention implementation for this device.

        Priority:
          flash_attention_2  — CUDA + flash-attn installed. 3-5x on long
                               prompts, minimal memory overhead.
          sdpa               — torch>=2 fused SDPA; good on CUDA and MPS.
          eager              — fallback.

        Respect an env-var override (HYDRA_ATTN_IMPL) for debugging.
        """
        override = os.environ.get("HYDRA_ATTN_IMPL", "").strip()
        if override:
            log.info("Using attention impl from HYDRA_ATTN_IMPL", impl=override)
            return override

        if self.device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
                log.info("Selected flash_attention_2 (CUDA + flash-attn available)")
                return "flash_attention_2"
            except ImportError:
                pass

        # SDPA exists from torch 2.0+. Covers CPU/CUDA/MPS with hardware
        # fused paths where available, otherwise falls back internally.
        try:
            import torch.nn.functional as F
            if hasattr(F, "scaled_dot_product_attention"):
                log.info("Selected sdpa (torch fused attention)")
                return "sdpa"
        except Exception:
            pass

        log.info("Selected eager attention (no fused path available)")
        return "eager"

    def _create_llama_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Llama layer on CPU, load weights, then ready for GPU transfer."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        with torch.device('cpu'):
            layer = LlamaDecoderLayer(self.config, layer_idx)
        self._load_weights_into_layer(layer, weights)
        return layer

    def _create_mistral_layer(self, layer_idx: int, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create a Mistral/Mixtral layer on CPU."""
        with torch.device('cpu'):
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
        """Create a generic layer using Llama as fallback, on CPU."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig

        # Create a LlamaConfig from the model config
        num_heads = self.config.num_attention_heads
        llama_config = LlamaConfig(
            hidden_size=self.config.hidden_size,
            intermediate_size=getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(self.config, "num_key_value_heads", num_heads),
            rms_norm_eps=getattr(self.config, "rms_norm_eps", 1e-5),
            rope_theta=getattr(self.config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(self.config, "max_position_embeddings", 4096),
            # Some families (Gemma-7B, for one) set a head_dim that is not
            # hidden_size // num_attention_heads. LlamaConfig would derive the
            # wrong one, and the layer's rotary dimensions would then disagree
            # with the rotary_emb built from the real config.
            head_dim=getattr(self.config, "head_dim", None) or self.config.hidden_size // num_heads,
        )

        with torch.device('cpu'):
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
        # Base bytes per param based on dtype
        if self.quantize:
            if self.quant_bits == 4:
                bytes_per_param = 0.5  # 4-bit = 0.5 bytes per param
            elif self.quant_bits == 8:
                bytes_per_param = 1.0  # 8-bit = 1 byte per param
            else:
                bytes_per_param = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        else:
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


def _cache_seq_length(past_key_values: Any, layer_start: int) -> int:
    """Return how many past tokens are already cached for this sequence.

    Works across the two cache shapes we thread through forward():
      * DynamicCache-style object: expose `get_seq_length()`.
      * List of per-layer tuples (k,v): use the first populated slot's k shape.
      * None: no history yet.
    """
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length(layer_start))
        except TypeError:
            # Some versions don't accept a layer_idx argument.
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                return 0
        except Exception:
            return 0
    if isinstance(past_key_values, list):
        for slot in past_key_values:
            if slot is None:
                continue
            # Tuple of (key, value); key is [batch, n_heads, past_len, head_dim].
            if isinstance(slot, tuple) and len(slot) >= 1 and hasattr(slot[0], "shape"):
                try:
                    return int(slot[0].shape[-2])
                except Exception:
                    return 0
    return 0


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
