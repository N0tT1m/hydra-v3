"""Inference engine - handles generation with sampling."""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import structlog

log = structlog.get_logger()


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    eos_token_id: int = 2
    pad_token_id: int = 0


@dataclass
class GenerationState:
    """State for an ongoing generation."""
    sequence_id: str
    input_ids: torch.Tensor
    generated_ids: List[int] = field(default_factory=list)
    past_key_values: Optional[List] = None
    finished: bool = False
    finish_reason: Optional[str] = None


class TokenSampler:
    """Token sampling with various strategies."""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: Raw logits [batch, vocab_size]
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Penalty for repeated tokens
            past_tokens: Previously generated tokens for repetition penalty

        Returns:
            Sampled token IDs [batch]
        """
        # Apply repetition penalty
        if repetition_penalty != 1.0 and past_tokens is not None:
            logits = TokenSampler._apply_repetition_penalty(
                logits, past_tokens, repetition_penalty
            )

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            logits = TokenSampler._top_k_filter(logits, top_k)

        # Apply top-p (nucleus sampling)
        if top_p < 1.0:
            logits = TokenSampler._top_p_filter(logits, top_p)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        for i in range(logits.shape[0]):
            for token_id in past_tokens[i].unique():
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        return logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filter to top-k tokens."""
        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filter using nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits


class InferenceEngine:
    """
    Inference engine for single-node or distributed generation.

    For single node: Runs full model locally
    For distributed: Coordinates with pipeline nodes
    """

    def __init__(
        self,
        model: "PartialTransformer",
        tokenizer,
        device: torch.device,
        is_distributed: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_distributed = is_distributed

        # Active generations
        self.generations: Dict[str, GenerationState] = {}

        # Callbacks for distributed mode
        self.on_forward_complete: Optional[Callable] = None

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        sequence_id: Optional[str] = None,
    ) -> str:
        """Synchronous generation (single node only)."""
        if self.is_distributed:
            raise RuntimeError("Use generate_async for distributed mode")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generate
        output_ids = self._generate_loop(input_ids, config)

        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        sequence_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Streaming generation - yields tokens as they're generated."""
        import uuid
        sequence_id = sequence_id or str(uuid.uuid4())

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Initialize state
        state = GenerationState(
            sequence_id=sequence_id,
            input_ids=input_ids,
        )
        self.generations[sequence_id] = state

        try:
            # Generate token by token
            current_ids = input_ids
            past_key_values = None

            for _ in range(config.max_new_tokens):
                # Forward pass
                with torch.no_grad():
                    if past_key_values is not None:
                        # Only pass last token with KV cache
                        model_input = current_ids[:, -1:]
                        position_ids = torch.tensor(
                            [[current_ids.shape[1] - 1]],
                            device=self.device
                        )
                    else:
                        model_input = current_ids
                        position_ids = torch.arange(
                            current_ids.shape[1],
                            device=self.device
                        ).unsqueeze(0)

                    outputs, past_key_values = self.model(
                        model_input,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Get logits for last position
                logits = outputs[:, -1, :]

                # Sample next token
                if config.do_sample:
                    past_tokens = current_ids if config.repetition_penalty != 1.0 else None
                    next_token = TokenSampler.sample(
                        logits,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                        past_tokens=past_tokens,
                    )
                else:
                    next_token = torch.argmax(logits, dim=-1)

                next_token_id = next_token.item()
                state.generated_ids.append(next_token_id)

                # Check for EOS
                if next_token_id == config.eos_token_id:
                    state.finished = True
                    state.finish_reason = "stop"
                    break

                # Decode and yield token
                token_text = self.tokenizer.decode([next_token_id])
                yield token_text

                # Update current_ids
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

                # Allow other tasks to run
                await asyncio.sleep(0)

            if not state.finished:
                state.finished = True
                state.finish_reason = "length"

        finally:
            # Cleanup
            del self.generations[sequence_id]

    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """Synchronous generation loop."""
        current_ids = input_ids
        past_key_values = None

        for _ in range(config.max_new_tokens):
            with torch.no_grad():
                if past_key_values is not None:
                    model_input = current_ids[:, -1:]
                    position_ids = torch.tensor(
                        [[current_ids.shape[1] - 1]],
                        device=self.device
                    )
                else:
                    model_input = current_ids
                    position_ids = torch.arange(
                        current_ids.shape[1],
                        device=self.device
                    ).unsqueeze(0)

                outputs, past_key_values = self.model(
                    model_input,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs[:, -1, :]

            if config.do_sample:
                past_tokens = current_ids if config.repetition_penalty != 1.0 else None
                next_token = TokenSampler.sample(
                    logits,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    past_tokens=past_tokens,
                )
            else:
                next_token = torch.argmax(logits, dim=-1)

            if next_token.item() == config.eos_token_id:
                break

            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return current_ids


class DistributedInferenceEngine:
    """
    Inference engine for distributed generation across pipeline.

    This runs on the coordinator and manages the full generation loop
    by sending requests through the pipeline of workers.
    """

    def __init__(self, broker: "zmq.Broker", tokenizer):
        self.broker = broker
        self.tokenizer = tokenizer

        # Pending generations
        self.pending: Dict[str, GenerationState] = {}
        self.result_queues: Dict[str, asyncio.Queue] = {}

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
        first_node_id: str,
        last_node_id: str,
    ) -> AsyncIterator[str]:
        """Stream generation through distributed pipeline."""
        import uuid
        sequence_id = str(uuid.uuid4())

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()

        # Create result queue
        self.result_queues[sequence_id] = asyncio.Queue()

        # Initialize state
        state = GenerationState(
            sequence_id=sequence_id,
            input_ids=torch.tensor([input_ids]),
        )
        self.pending[sequence_id] = state

        try:
            # Send initial tokens to first node
            await self._send_forward_request(
                first_node_id,
                sequence_id,
                token_ids=input_ids,
                past_len=0,
            )

            # Generation loop
            generated_tokens = []
            for _ in range(config.max_new_tokens):
                # Wait for logits from last node
                try:
                    result = await asyncio.wait_for(
                        self.result_queues[sequence_id].get(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    log.error("Timeout waiting for logits")
                    break

                logits = torch.tensor(result["logits"])

                # Sample next token
                if config.do_sample:
                    past_tokens = torch.tensor([input_ids + generated_tokens]) if config.repetition_penalty != 1.0 else None
                    next_token = TokenSampler.sample(
                        logits[-1:],
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                        past_tokens=past_tokens,
                    )
                else:
                    next_token = torch.argmax(logits[-1], dim=-1)

                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)

                # Check for EOS
                if next_token_id == config.eos_token_id:
                    break

                # Decode and yield
                token_text = self.tokenizer.decode([next_token_id])
                yield token_text

                # Send next token through pipeline
                await self._send_forward_request(
                    first_node_id,
                    sequence_id,
                    token_ids=[next_token_id],
                    past_len=len(input_ids) + len(generated_tokens) - 1,
                )

        finally:
            # Cleanup
            del self.pending[sequence_id]
            del self.result_queues[sequence_id]

            # Clear KV cache on all nodes
            await self._clear_kv_cache(sequence_id)

    async def _send_forward_request(
        self,
        node_id: str,
        sequence_id: str,
        token_ids: List[int],
        past_len: int,
    ):
        """Send forward request to first node."""
        from hydra_worker.comm.tensor_protocol import TensorSerializer

        await self.broker.SendTo(node_id, "forward", {
            "sequence_id": sequence_id,
            "token_ids": token_ids,
            "past_len": past_len,
        })

    async def _clear_kv_cache(self, sequence_id: str):
        """Clear KV cache for sequence on all nodes."""
        await self.broker.Broadcast("clear_kv_cache", {
            "sequence_id": sequence_id,
        })

    def handle_forward_result(self, msg: Dict):
        """Handle forward result from last node."""
        sequence_id = msg.get("sequence_id")
        if sequence_id in self.result_queues:
            self.result_queues[sequence_id].put_nowait(msg)
