# 07 — Debugging strategies

Staging systems fail differently from eager ones. Debug **where in the pipeline** you are (Python → trace → jaxpr → compile → execute).

Architecture reminder: [01-architecture.md](01-architecture.md).

## Strategy 0: Localize the stage

```text
1. Run the function WITHOUT jit          → pure Python / numerics / shapes?
2. make_jaxpr(...)                       → tracing / concretization / capture?
3. jit(...).lower(...)                   → lowering / sharding?
4. .compile() / run                      → XLA / runtime / async?
```

Most “JAX is broken” reports die at step 1 or 2.

## Concretization errors

**Meaning:** Something needed a concrete Python value (branch, shape that depends on data, conversion to `int`, etc.) but saw a Tracer.

**What to inspect:**

1. Which value is traced? (`x` from batch vs static hyperparameter)
2. Can it be `static_argnums`?
3. Should control flow be `lax.cond` / `scan`?
4. Are you calling NumPy / Python on JAX arrays under `jit`?

```python
# See what was captured
print(jax.make_jaxpr(f)(*args))
```

Chapter: [03-tracing-and-jaxpr.md](03-tracing-and-jaxpr.md). Official: [docs/tracing.md](../docs/tracing.md), [docs/control-flow.md](../docs/control-flow.md).

## Shape / dtype errors

- Print `jax.tree.map(lambda a: (a.shape, a.dtype), pytree)`
- Ensure `vmap` `in_axes` match intended batch dims
- Watch broadcasting silently changing shapes before a matmul

## “Wrong” jaxpr (looks too small / too big)

| Observation | Likely cause |
|-------------|--------------|
| Ops missing | Dead-code elimination; constants folded; you didn’t pass what you think |
| Giant const | Closed over big arrays; embedded data |
| Unexpected static | `static_argnums` or Python const baked in |
| No batch dim | Forgot `vmap` or wrong `in_axes` |

`make_jaxpr` is truth for what transforms saw — not your mental Python.

## Debug printing under transforms

- Eager: normal `print`
- Under `jit`: `jax.debug.print("x={x}", x=x)` (device-side; ordering caveats)
- Avoid host `print` of device arrays in hot loops ([06](06-performance-pitfalls.md))

Also explore `jax.debug` helpers and docs under [docs/debugging.md](../docs/debugging.md) / `docs/debugging/`.

## AD bugs

| Symptom | Try |
|---------|-----|
| Gradients all zero / None-like | Wrong `argnums`; discrete ops; `stop_gradient`; non-differentiable path |
| NaN grads | Check forward NaNs; precision; missing where masks |
| Custom op wrong | `custom_jvp`/`vjp` tests with `jvp`/`vjp` finite checks |

Use `jax.value_and_grad` with `has_aux` to return intermediates for inspection.

## Parallel / sharding bugs

- Start single-device
- Check shardings with array sharding properties / debugging docs
- Ensure mesh axis names match collective/`shard_map` expectations

## When jaxpr looks “wrong” but isn’t

XLA and jaxpr construction may:

- Constant-fold,
- DCE unused ops,
- Reorder for purity,

So a jaxpr is not a perfect diary of your Python lines. It is the **program that will be lowered**. Trust it for semantics; don’t expect 1:1 line mapping.

## Optional tools

- `jax.explain` / transformer explainers when available in your version — for transform composition confusion
- AOT: `.lower().as_text()` style inspection (see [docs/aot.md](../docs/aot.md))
- Profilers: [docs](../docs) profiling guides under 201 / advanced — after correctness

## What to skip while debugging models

- Stepping through [partial_eval.py](../jax/_src/interpreters/partial_eval.py)
- Reading MLIR dumps by default
- Patching XLA

If you need a custom primitive, *then* read primitives docs — not while fixing a training step.

## Quick playbook card

```text
Error under jit?
  → disable jit; fix eager
  → make_jaxpr; fix tracing/control flow
  → re-enable jit; check static args & shapes
  → block_until_ready for timing/NaN investigation
```

Next: [08-reading-repos.md](08-reading-repos.md).
