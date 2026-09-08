# TODO

Known issues, ordered by severity. Each entry states the symptom, the
evidence, and what "done" looks like. Findings come from running a two-node
cluster (CUDA + Apple MPS) against Qwen2.5-7B and Qwen3.5-27B.

---

## P0 — correctness

### 1. ~~Hidden states are corrupted crossing a bf16 → fp16 node boundary~~ — FIXED (needs cluster verification)

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

**Fixed.** The MPS bf16 downgrade is now a runtime probe rather than an
assumption — it dated from builds that could not run bf16, whereas on torch
2.14 / M3 Max bf16 runs natively and measures *faster* than fp16 (5755 vs
3913 GFLOP/s). Keeping bf16 removes the mixed-dtype pipeline that caused this.
Narrowing casts are additionally guarded: they are checked for non-finite
results and raise with the offending magnitude and sequence id rather than
emitting `inf`.

**Still to verify.** The fix is covered by unit tests and the MPS capability
benchmark, but has **not** been re-run end to end on a real two-node cluster.
The original acceptance criterion still stands: a fixed prompt at
`temperature=0` yielding byte-identical output across ≥5 consecutive runs, on
both a small and a large model. Item 4 below is the test that would prove it.

### 2. ~~A failed forward pass hangs the client instead of erroring~~ — FIXED

**Symptom.** When a worker's forward raises, the coordinator never surfaces
it. The client blocks until its own timeout — observed as a 120s hang, and
earlier a 10-minute stall, with the error visible only in the worker log.

**Done when:** a worker-side exception terminates the sequence and returns an
error response promptly; covered by a test that injects a forward failure.

**Fixed.** A `forward_error` message type carries the failure from worker to
coordinator, which ends that one sequence with a `finish_reason` the caller can
read. Only the named sequence is failed — a forward can fail for reasons
specific to one request, and killing every in-flight generation would be worse
than the hang it replaces.

All four silent returns in `_handle_forward` now report, and the dispatch is
wrapped so a raised exception does too. Four existing worker tests asserted
`sent == []` on those paths — they had encoded the bug as intended behaviour —
and now assert the error is sent. 7 coordinator tests and 7 worker tests cover
it.

---

## P1 — capacity and safety

### 3. ~~Layer distribution uses a fixed per-layer memory constant~~ — FIXED

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

**Fixed.** Per-layer cost is now derived from the model config (params ×
dtype width), embedding and lm_head are charged to the nodes that hold them,
a KV reserve is sized from `cluster.kv_reserve_tokens`, and assignments are
clamped to node capacity. A model that cannot fit is refused with the
arithmetic shown instead of OOMing later.

**Remaining:** the dense parameter formula under-counts hybrid decoders
(Qwen3.5's linear-attention layers have convolution and gating projections it
does not model), covered today by a 1.30 safety factor. Reading exact tensor
sizes from the safetensors headers would remove the guess. Also note this now
refuses Qwen3.5-27B on a 31.4 + 22.3 GiB cluster — correctly, since it only
ever fit by leaving 8 MiB for activations — so an explicit
`--allow-oversubscribe` escape hatch may be worth adding for short-prompt use.

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

### 8. ~~De-brand the default logging~~ — PARTLY DONE

The default app tag was `robin-hydra`, which names nothing in this repository.
It is now `hydra` (`hydra-worker` on the worker side), matching the binary and
the module path.

`den-den-mushi` references remain and are correct: it is the name of the log
collector these lines are shaped for, and the comments describe a real external
system rather than decorating this one.

### 9. Document the operational footguns

Worth a short `docs/operations.md`:

- Remote workers need `setsid` and `</dev/null`, or they die with the SSH
  session.
- `pkill -f <pattern>` over SSH kills its own shell when the command line also
  contains the pattern. Keep kill and start in separate invocations.
- On macOS, run the worker under `caffeinate -dims` or sleep drops the node.
- Seed large models to a slow-linked node over LAN from a fast-linked one;
  a direct download can be ~50× slower.
