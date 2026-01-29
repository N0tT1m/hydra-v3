#!/usr/bin/env python3
"""
Example: Test partial model loading on a single node.

This tests the partial model loading without the full distributed setup.
Useful for verifying that layer extraction works correctly.
"""

import torch
from pathlib import Path


def test_partial_loading(model_path: str, layer_start: int, layer_end: int):
    """Test loading a subset of layers from a model."""
    from hydra_worker.models.partial_loader import PartialModelLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model: {model_path}")
    print(f"Layers: {layer_start} to {layer_end}")

    # Initialize loader
    loader = PartialModelLoader(model_path, device, dtype)

    print(f"\nModel config:")
    print(f"  Architecture: {loader.arch}")
    print(f"  Total layers: {loader.config.num_hidden_layers}")
    print(f"  Hidden size: {loader.config.hidden_size}")
    print(f"  Vocab size: {loader.config.vocab_size}")

    # Estimate memory
    mem_bytes = loader.estimate_memory(layer_start, layer_end)
    print(f"  Estimated memory: {mem_bytes / 1024**3:.2f} GB")

    # Load partial model
    print(f"\nLoading layers {layer_start} to {layer_end}...")
    is_first = layer_start == 0
    is_last = layer_end == loader.config.num_hidden_layers

    model, tokenizer = loader.load_partial_model(
        layer_start=layer_start,
        layer_end=layer_end,
        include_embedding=is_first,
        include_lm_head=is_last,
    )

    print(f"  Loaded {len(model.layers)} layers")
    print(f"  Has embedding: {model.has_embedding}")
    print(f"  Has lm_head: {model.has_lm_head}")

    # Test forward pass
    print("\nTesting forward pass...")

    if is_first:
        # First node: input is tokens
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        print(f"  Input: {input_text}")
        print(f"  Token IDs: {input_ids.tolist()}")

        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        with torch.no_grad():
            output, kv_cache = model(input_ids, position_ids=position_ids)

        print(f"  Output shape: {output.shape}")
    else:
        # Middle/last node: input is hidden states
        batch_size = 1
        seq_len = 10
        hidden_states = torch.randn(
            batch_size, seq_len, model.hidden_size,
            device=device, dtype=dtype
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        print(f"  Input hidden states: {hidden_states.shape}")

        with torch.no_grad():
            output, kv_cache = model(hidden_states, position_ids=position_ids)

        print(f"  Output shape: {output.shape}")

    if is_last:
        # Check logits
        print(f"  Logits: {output.shape} (vocab_size={loader.config.vocab_size})")

        # Get top tokens
        probs = torch.softmax(output[0, -1], dim=-1)
        top_k = torch.topk(probs, 5)
        print("  Top 5 tokens:")
        for prob, idx in zip(top_k.values, top_k.indices):
            token = tokenizer.decode([idx])
            print(f"    {idx.item()}: '{token}' ({prob.item():.4f})")

    print("\nPartial loading test passed!")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python single_node_test.py <model_path> [layer_start] [layer_end]")
        print("\nExamples:")
        print("  # Test first 8 layers of a 32-layer model")
        print("  python single_node_test.py meta-llama/Llama-2-7b-hf 0 8")
        print()
        print("  # Test middle layers")
        print("  python single_node_test.py meta-llama/Llama-2-7b-hf 8 16")
        print()
        print("  # Test last layers (includes lm_head)")
        print("  python single_node_test.py meta-llama/Llama-2-7b-hf 24 32")
        return

    model_path = sys.argv[1]
    layer_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    layer_end = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    test_partial_loading(model_path, layer_start, layer_end)


if __name__ == "__main__":
    main()
