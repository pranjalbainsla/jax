# Glossary

Short definitions for the mental model. Official: [docs/glossary.rst](../docs/glossary.rst).

| Term | Meaning |
|------|---------|
| **Abstract value** | Shape/dtype (etc.) without concrete data; e.g. `ShapedArray` |
| **AOT** | Ahead-of-time stages: trace → lower → compile → execute |
| **Batch rule** | How a primitive behaves under `vmap` |
| **ClosedJaxpr** | Jaxpr plus constants closed over at trace time |
| **Concretization** | Forcing a Tracer to a concrete Python value; fails if still abstract |
| **Donation** | Allowing `jit` to reuse an input buffer for outputs (`donate_argnums`) |
| **Jaxpr** | JAX’s typed ANF IR: binders, equations of primitives, outputs |
| **JVP / VJP** | Forward- and reverse-mode AD building blocks |
| **Leaf** | Array (or array-like) at the bottom of a pytree |
| **Lowering** | Jaxpr → StableHLO/MLIR (user sees this via `.lower()`) |
| **Mesh** | Named grid of devices for SPMD sharding |
| **Primitive** | Atomic op with rules (impl, abstract eval, AD, batch, lower) |
| **PRNG key** | Explicit RNG state (`jax.random`); split rather than mutate globals |
| **Pytree** | Nested containers with array leaves; uniform walk for transforms |
| **Remat / checkpoint** | Recompute forward activations in backward to save memory |
| **Sharding** | How a logical `jax.Array` is partitioned across devices |
| **StableHLO** | Portable compiler IR consumed by XLA |
| **Static arg** | `jit` argument treated as compile-time constant; part of cache key |
| **Trace** | Interpreter context that records/rewrites ops during a transform |
| **Tracer** | Placeholder for an array under a Trace (carries abstract value) |
| **Transform** | Function→function rewrite: `jit`, `grad`, `vmap`, … |
| **XLA** | Compiler stack that turns lowered IR into device executables |
| **`jax.Array`** | Current array type (values + devices + sharding) |
| **`lax`** | Lower-level ops / control-flow primitives closer to XLA |
| **`pmap`** | Legacy multi-device map; prefer mesh + `jit` / `shard_map` for new code |
| **`shard_map`** | Map a function over shards with explicit multi-device semantics |
| **`vmap`** | Automatic batching transform |

## Related chapter map

| Term cluster | Chapter |
|--------------|---------|
| Purity, vs PyTorch | [00](00-philosophy.md) |
| Stack, XLA, primitives | [01](01-architecture.md) |
| Pytree, leaf | [02](02-pytrees.md) |
| Tracer, jaxpr, concretization | [03](03-tracing-and-jaxpr.md) |
| Transforms, remat | [04](04-transformations.md) |
| Array, async, sharding | [05](05-arrays-and-execution.md) |
