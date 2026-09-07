# TODO

Known issues, ordered by severity. Each entry states the symptom, the
evidence, and what "done" looks like. Findings come from running a two-node
cluster (CUDA + Apple MPS) against Qwen2.5-7B and Qwen3.5-27B.

---

## P0 — correctness

### 1. Hidden states are corrupted crossing a bf16 → fp16 node boundary

**Symptom.** With `temperature=0`, the same prompt produces *different*
outputs run to run — coherent English on some passes, another language or
mojibake on others. Determinism is violated, so this is numerical corruption,
not sampling.

**Evidence.** 12 generations of one prompt at `temperature=0` diverged. The
boundary dtypes are confirmed asymmetric:

```
sending node:   dtype=torch.bfloat16
receiving node: dtype=torch.float16     # MPS downgrades; bf16 kernels incomplete
```

fp16 saturates at 65504; bf16 carries fp32's exponent range (~3.4e38). Large
activations that are fine in bf16 overflow to `inf` in fp16 and propagate as
`NaN`. Reproduced on Qwen3.5-27B (hidden 5120, 64 layers); Qwen2.5-7B looked
clean but has **not** been verified for determinism.

**Status.** Hypothesis. The dtype asymmetry and the nondeterminism are both
confirmed; the overflow itself is not yet instrumented.

**Do first:** log `hidden_states.abs().max()` on both sides of the boundary
and count `isinf`/`isnan` after the cast. That settles it either way.

**Candidate fixes** (preference order):
1. Carry fp32 on the wire between nodes, cast at each end. Numerically safe,
   costs 2× bandwidth on the link.
2. Scale before the bf16 → fp16 cast, unscale on arrival.
3. Run the MPS node in fp32 (slower; likely will not fit at 27B).
4. At minimum, detect non-finite values at the boundary and fail loudly
   instead of emitting garbage.

**Done when:** a fixed prompt at `temperature=0` yields byte-identical output
across ≥5 consecutive runs, on every supported node dtype combination, for
both a small and a large model. Add this as a test.

### 2. A failed forward pass hangs the client instead of erroring

**Symptom.** When a worker's forward raises, the coordinator never surfaces
it. The client blocks until its own timeout — observed as a 120s hang, and
earlier a 10-minute stall, with the error visible only in the worker log.

**Done when:** a worker-side exception terminates the sequence and returns an
error response promptly; covered by a test that injects a forward failure.

---

## P1 — capacity and safety

### 3. Layer distribution uses a fixed per-layer memory constant

**Symptom.** OOM on any non-trivial prompt after a large model loads:

```
CUDA out of memory. Tried to allocate 12.00 MiB.
GPU has 31.39 GiB capacity, of which 8.19 MiB is free.
```

**Cause.** `memory_per_layer_gb` is a static config value (default 0.5) applied
to every model regardless of size. A Qwen3.5-27B layer is ~0.76 GB, so the
allocator believed 38 layers needed ~19 GB when they actually needed ~29 GB.
It filled the card to 94% and left 8 MiB for activations and KV cache. The
default is roughly correct for a 7B by coincidence, and wrong for anything
larger.

**Fix.** Derive per-layer bytes from the model config (params × dtype width) —
`PartialModelLoader.estimate_memory()` already does most of this — and feed it
into distribution instead of the constant. Separately reserve headroom for the
KV cache as a function of context length and batch size, not a flat
`reserved_vram_gb`.

**Done when:** loading a model that would not leave room for its own KV cache
either redistributes or refuses with a clear message, rather than loading
successfully and OOMing on the first long prompt.

### 4. No end-to-end correctness test across a real split

The suite covers units well but nothing asserts that a model split across two
nodes produces the *same* tokens as the same model unsplit. That is the
property most likely to break, and both P0 issues would have been caught by it.

**Done when:** CI (or a documented manual job) runs a tiny model single-node
and two-node and asserts identical greedy output.

---

## P2 — performance

### 5. Every token round-trips through the coordinator

Last node → coordinator → first node adds two process hops per token. A direct
ring (last → first) would remove them. Measured at ~60 ms/token unaccounted
for by compute and network on a 27B run; some fraction is this.

### 6. Measure a single-node baseline

Claims that the split is "N× slower than one GPU" are currently based on spec
expectations, not measurement. Benchmark the same model on one node and
publish the comparison.

### 7. Quantization that does not require bitsandbytes

`nn.Linear` int8/int4 currently needs bitsandbytes (CUDA-only, a no-op
elsewhere), so a mixed CUDA+MPS cluster cannot quantize at all. Pure-torch
int8 already exists for MoE experts; extending it to dense linears would let
large models fit on fewer nodes — usually a bigger win than splitting them.

---

## P3 — housekeeping

### 8. De-brand the default logging

`den-den-mushi` and the default app tag `robin-hydra` appear in ~20 places
across `cmd/hydra/main.go`, `internal/config/config.go`, and
`worker/hydra_worker/cli.py`. The mechanism (JSON lines on stdout) is
ordinary; the vocabulary is private and means nothing to an outside reader.

### 9. Document the operational footguns

Worth a short `docs/operations.md`:

- Remote workers need `setsid` and `</dev/null`, or they die with the SSH
  session.
- `pkill -f <pattern>` over SSH kills its own shell when the command line also
  contains the pattern. Keep kill and start in separate invocations.
- On macOS, run the worker under `caffeinate -dims` or sleep drops the node.
- Seed large models to a slow-linked node over LAN from a fast-linked one;
  a direct download can be ~50× slower.
