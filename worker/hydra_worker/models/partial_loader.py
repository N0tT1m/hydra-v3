"""Partial model loading - load only specific layers for distributed inference."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
import gc
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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


def parse_dtype(dtype_str: str) -> Tuple[torch.dtype, Optional[BitsAndBytesConfig]]:
    """Parse dtype string and return torch dtype and optional quantization config."""
    if dtype_str == "float16":
        return torch.float16, None
    elif dtype_str == "bfloat16":
        return torch.bfloat16, None
    elif dtype_str == "float32":
        return torch.float32, None
    elif dtype_str == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        return torch.float16, bnb_config  # Base dtype for non-quantized parts
    elif dtype_str == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        return torch.bfloat16, bnb_config
    elif dtype_str == "fp8":
        # FP8 via bitsandbytes or native PyTorch
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=True,
            )
            return torch.float16, bnb_config
        except Exception:
            log.warning("FP8 not fully supported, falling back to int8")
            return parse_dtype("int8")
    else:
        log.warning(f"Unknown dtype {dtype_str}, defaulting to bfloat16")
        return torch.bfloat16, None


class PartialModelLoader:
    """Load only specific layers from a model for distributed inference."""

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cuda"),
        dtype: Union[str, torch.dtype] = "bfloat16",
    ):
        self.device = device

        # Parse dtype string
        if isinstance(dtype, str):
            self.dtype, self.quantization_config = parse_dtype(dtype)
            self.dtype_str = dtype
        else:
            self.dtype = dtype
            self.quantization_config = None
            self.dtype_str = str(dtype).split(".")[-1]

        self.original_model_path = model_path

        # Check if it's a local path or HuggingFace model ID
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

        self.config = AutoConfig.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.arch = self._detect_architecture()
        self.is_moe = self._is_moe_model()

        log.info(
            "Initialized partial loader",
            model=model_path,
            arch=self.arch,
            is_moe=self.is_moe,
            num_layers=self.config.num_hidden_layers,
            dtype=self.dtype_str,
            quantized=self.quantization_config is not None,
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
            log.warning(f"Unknown model type {model_type}, defaulting to auto")
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

    def load_partial_model(
        self,
        layer_start: int,
        layer_end: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
    ) -> Tuple[PartialTransformer, AutoTokenizer]:
        """Load a partial model with only specified layers."""
        log.info(
            "Loading partial model",
            layers=f"{layer_start}-{layer_end}",
            embedding=include_embedding,
            lm_head=include_lm_head,
            dtype=self.dtype_str,
        )

        # Build loading kwargs
        load_kwargs = {
            "pretrained_model_name_or_path": str(self.model_path),
            "device_map": "auto",  # Let transformers handle device placement
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Add quantization or dtype
        if self.quantization_config:
            load_kwargs["quantization_config"] = self.quantization_config
            log.info("Loading with quantization", config=str(self.quantization_config))
        else:
            load_kwargs["torch_dtype"] = self.dtype

        log.info("Loading model via transformers...")

        try:
            full_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        except Exception as e:
            log.error("Failed to load model", error=str(e))
            raise

        log.info("Model loaded, extracting layers...")

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
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"):
            inner_model = full_model.model
            layers = inner_model.layers
            embed_tokens = inner_model.embed_tokens
            norm = inner_model.norm
        elif hasattr(full_model, "transformer") and hasattr(full_model.transformer, "h"):
            inner_model = full_model.transformer
            layers = inner_model.h
            embed_tokens = inner_model.wte
            norm = inner_model.ln_f
        else:
            raise ValueError(f"Unknown model structure for {self.arch}")

        # Copy embedding if needed
        if include_embedding:
            partial.embed_tokens = embed_tokens
            log.info("Loaded embedding layer")

        # Copy only the layers we need
        for idx in range(layer_start, layer_end):
            layer = layers[idx]
            partial.layers.append(layer)
            log.info(f"Loaded layer {idx}")

        log.info(f"Loaded {len(partial.layers)} transformer layers")

        # Copy norm and lm_head if needed
        if include_lm_head:
            partial.norm = norm
            partial.lm_head = full_model.lm_head
            log.info("Loaded norm and lm_head")

        # Clear references to free memory for unused layers
        # Note: With device_map="auto", layers are already on the right device
        del full_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Cleaned up unused model parts")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return partial, tokenizer

    def estimate_memory(self, layer_start: int, layer_end: int) -> int:
        """Estimate GPU memory needed for layers in bytes."""
        # Determine bytes per parameter based on quantization
        if self.dtype_str == "int4":
            bytes_per_param = 0.5
        elif self.dtype_str in ("int8", "fp8"):
            bytes_per_param = 1
        elif self.dtype in (torch.float16, torch.bfloat16):
            bytes_per_param = 2
        else:
            bytes_per_param = 4

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
