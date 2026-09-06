#!/usr/bin/env python3
"""
Example: Distributed inference across multiple GPUs/nodes.

This example shows how to:
1. Load a model split across multiple workers
2. Run inference through the pipeline

Setup:
    # Terminal 1: Start coordinator
    cd hydra-v3 && make run-dev

    # Terminal 2: Start worker 1 (will get layers 0-15)
    cd hydra-v3/worker && hydra-worker start --node-id worker-1 --coordinator tcp://localhost:5555

    # Terminal 3: Start worker 2 (will get layers 16-31)
    cd hydra-v3/worker && hydra-worker start --node-id worker-2 --coordinator tcp://localhost:5555

    # Terminal 4: Run this example
    python examples/distributed_inference.py --model-path /path/to/model
"""

import asyncio
import httpx
import json
import argparse


async def main(model_path: str, model_id: str, total_layers: int):
    base_url = "http://localhost:8080"

    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Check health
        print("Checking health...")
        try:
            resp = await client.get(f"{base_url}/health")
            print(f"  Health: {resp.json()}")
        except httpx.ConnectError:
            print("  ERROR: Cannot connect to coordinator. Is it running?")
            print("  Start with: cd hydra-v3 && make run-dev")
            return

        # 2. Check cluster status
        print("\nChecking cluster status...")
        resp = await client.get(f"{base_url}/api/cluster/status")
        status = resp.json()
        print(f"  Nodes: {status.get('total_nodes', 0)}")
        print(f"  Healthy: {status.get('healthy_nodes', 0)}")
        print(f"  Total VRAM: {status.get('total_vram_gb', 0):.1f} GB")

        if status.get('nodes'):
            print("  Workers:")
            for node in status['nodes']:
                print(f"    - {node.get('id')}: {node.get('vram_gb', 0):.1f} GB, "
                      f"layers {node.get('layer_start', '?')}-{node.get('layer_end', '?')}, "
                      f"healthy={node.get('is_healthy')}")

        if status.get("healthy_nodes", 0) < 1:
            print("\n  WARNING: No workers available. Start workers first:")
            print("  cd hydra-v3/worker && hydra-worker start --node-id worker-1")
            print("\n  Continuing anyway for demonstration...")

        # 3. Load model (this will distribute across workers)
        print(f"\nLoading model '{model_id}' from '{model_path}'...")
        resp = await client.post(
            f"{base_url}/api/models/load",
            json={
                "model_path": model_path,
                "model_id": model_id,
                "total_layers": total_layers,
            }
        )
        result = resp.json()
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Status: {result.get('status')}")
            print(f"  Message: {result.get('message')}")

        # Wait a bit for model to load
        print("\nWaiting for model to load on workers...")
        await asyncio.sleep(3)

        # 4. Run chat completion
        print("\nRunning inference...")
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            }
        )

        result = resp.json()
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"  Response: {content}")
            usage = result.get('usage', {})
            print(f"  Usage: {usage.get('prompt_tokens', 0)} prompt + "
                  f"{usage.get('completion_tokens', 0)} completion = "
                  f"{usage.get('total_tokens', 0)} total tokens")

        # 5. Stream response
        print("\nStreaming inference...")
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "Tell me a very short story in 2-3 sentences."}
                ],
                "max_tokens": 200,
                "stream": True,
            }
        ) as response:
            print("  ", end="", flush=True)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
            print("\n")

        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test distributed inference")
    parser.add_argument(
        "--model-path",
        default="meta-llama/Llama-2-7b-hf",
        help="Path to model (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--model-id",
        default="llama-7b",
        help="ID to assign to loaded model",
    )
    parser.add_argument(
        "--total-layers",
        type=int,
        default=32,
        help="Total number of layers in the model",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model_path, args.model_id, args.total_layers))
