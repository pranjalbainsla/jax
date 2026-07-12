# 06 — Performance pitfalls

Pitfalls below are tied to the mental model (tracing, jaxpr, compile cache, async), not a random laundry list.

Official companions: [Sharp Bits](../docs/notebooks/Common_Gotchas_in_JAX.ipynb), [FAQ](../docs/faq.rst), [docs/benchmarking.md](../docs/benchmarking.md), [docs/buffer_donation.md](../docs/buffer_donation.md).

## 1. Recompilation / retracing

**Symptom:** Every step (or many steps) is slow; first-batch slowness never goes away; compile-time dominates.

**Mental model:** `jit` caches executables by a signature: traced shapes/dtypes, sharding, and **static argument values**.

Common causes:

| Cause | Fix |
|-------|-----|
| Input shapes change every batch | Pad to fixed shapes; or use shape polymorphism carefully |
| Boolean / int flags passed as traced args when they should be static | `static_argnums` / `static_argnames` |
| Huge objects or arrays marked static | Don’t; only small hashable config |
| Closing over changing Python values | Pass as arguments; or accept recompile when config changes |
| Unique `jit` wrappers created in a loop | Define jitted fn once at module scope |

```python
# Bad: new Python bool closed over each call pattern → cache churn
@partial(jax.jit, static_argnames=["train"])
def step(params, batch, train: bool):
  ...
```

**Debug:** log when compiles happen; compare `make_jaxpr` across calls; ensure batch shapes are stable.

## 2. Python loops vs `vmap` / `lax.scan`

| Pattern | Cost |
|---------|------|
| Python `for` calling a jitted body per example | Dispatch / compile / launch overhead |
| `vmap` over examples | One batched program |
| `lax.scan` over time/layers | One program with carry; XLA-friendly |

**Rule:** If the iteration count is large or on the training hot path, push the loop into JAX (`vmap`/`scan`) instead of Python.

## 3. Host–device sync in the hot path

Anything that forces a concrete host value can stall the pipeline:

- `print(array)` / formatting arrays
- `.item()`, casting to Python `int`/`float`
- Data-dependent Python `if` on device values
- Naive host callbacks every step

Use `jax.debug.print` for device-side debugging; remove before serious benchmarks.

## 4. Benchmarking without `block_until_ready`

```python
# Wrong: measures enqueue
t0 = time.time(); y = f(x); dt = time.time() - t0

# Right: measures completion
y = f(x).block_until_ready()
```

Also warm up once to exclude compile time from steady-state metrics. See [05-arrays-and-execution.md](05-arrays-and-execution.md).

## 5. Rematerialization vs OOM

**Problem:** Activations for reverse-mode AD blow HBM on deep nets.

**Tool:** `jax.checkpoint` / `jax.remat` — recompute forward pieces during backward.

**Trade:** more FLOPs, less memory. Place remat around transformer blocks / large residuals, not randomly on tiny ops.

Compose with training step:

```python
loss = remat(big_forward_loss)
train_step = jit(value_and_grad(loss))
```

Official: remat / autodiff rematerialization docs under advanced guides.

## 6. Buffer donation

`donate_argnums` lets the compiled function reuse input buffers for outputs (classic: update `params` in place at buffer level).

**Pitfalls:**

- Using a donated array again → error or undefined
- Forgetting donation on the step that updates large trees → extra memory traffic

Read [docs/buffer_donation.md](../docs/buffer_donation.md) when optimizing.

## 7. Precision, dtypes, and “wrong” numerics

- Default **float32**; float64 opt-in.
- XLA may rewrite algebra (`log(exp(x))` etc.) — FAQ: jit can change numerics.
- Matmul precision settings affect GPU TensorCores / TPU.

Don’t chase bit-identical NumPy under `jit`. Chase training stability and metric parity with a reference.

## 8. Unnecessary work inside `jit`

- Building huge Python structures during trace
- Embedding datasets in the jaxpr (constants balloon compile and memory)
- Re-`jit`ing tiny helpers that should stay fused in one outer `jit`

Keep data as **arguments**. Keep compile units large enough to fuse, small enough to not compile forever.

## 9. Multi-device footguns (preview)

- Unintended replication vs sharding → memory × devices
- Collectives on wrong mesh axes
- Host-side Python loops over devices instead of SPMD `jit`

Learn sharding deliberately via [docs/parallel.md](../docs/parallel.md) after single-device fluency.

## Prioritized “fix this first” list

When training is slow or unstable:

1. Fixed batch shapes? Recompiling?
2. Measuring with sync?
3. Syncing every step?
4. Python loop that should be `vmap`/`scan`?
5. OOM → remat / donation / smaller activation footprint?
6. Only then: XLA flags, kernels, Pallas

## What not to do

- Premature Pallas/custom kernels before the above
- Reading [compiler.py](../jax/_src/compiler.py) to “optimize” a model
- Disabling `jit` permanently to “go faster” (eager is slower for real workloads)

Next: [07-debugging.md](07-debugging.md).
