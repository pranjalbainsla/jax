# Source map: ~25 files that matter (and what to skip)

Tiers for **ML research / systems users**, not JAX contributors.

| Tier | Meaning |
|------|---------|
| **Must** | Read (docs) or skim carefully (short API files) |
| **Skim** | Concepts / class headers only — do not read end-to-end |
| **Skip** | Safe to ignore forever for this career path |

Paths are relative to the repo root.

---

## Official docs (Must)

| Path | Why |
|------|-----|
| [docs/notebooks/thinking_in_jax.ipynb](../docs/notebooks/thinking_in_jax.ipynb) | Primary “how to think in JAX” |
| [docs/notebooks/Common_Gotchas_in_JAX.ipynb](../docs/notebooks/Common_Gotchas_in_JAX.ipynb) | Sharp Bits |
| [docs/key-concepts.md](../docs/key-concepts.md) | Transforms, tracing, jaxpr, pytrees, PRNG |
| [docs/pytrees.md](../docs/pytrees.md) | Pytree tutorial |
| [docs/tracing.md](../docs/tracing.md) | Tracing mental model |
| [docs/jaxpr.md](../docs/jaxpr.md) | Reading jaxprs |
| [docs/jit-compilation.md](../docs/jit-compilation.md) | `jit` |
| [docs/automatic-vectorization.md](../docs/automatic-vectorization.md) | `vmap` |
| [docs/automatic-differentiation.md](../docs/automatic-differentiation.md) | `grad` |
| [docs/control-flow.md](../docs/control-flow.md) | `cond` / `scan` / … |
| [docs/aot.md](../docs/aot.md) | Trace → lower → compile |
| [docs/parallel.md](../docs/parallel.md) | Multi-device (systems track) |
| [docs/notebooks/autodiff_cookbook.md](../docs/notebooks/autodiff_cookbook.md) | Research AD patterns |

**Strongly recommended later:** [docs/the-training-cookbook.md](../docs/the-training-cookbook.md), [docs/migrate_pmap.md](../docs/migrate_pmap.md), [docs/custom_pytrees.md](../docs/custom_pytrees.md).

**Optional deep dive (not required):** [docs/autodidax.md](../docs/autodidax.md) — “JAX from scratch”; excellent, but contributor-shaped.

---

## Code: Must / Skim

| Path | Tier | Why |
|------|------|-----|
| [jax/tree.py](../jax/tree.py) | **Must** | Modern `jax.tree` API (tiny) |
| [jax/tree_util.py](../jax/tree_util.py) | **Must** (registration) | Custom pytrees / dataclasses |
| [jax/_src/api.py](../jax/_src/api.py) | **Skim** | `jit`/`grad`/`vmap`/`make_jaxpr` signatures & docs |
| [jax/_src/core.py](../jax/_src/core.py) | **Skim** | `Tracer`, `Jaxpr`, `Primitive`, `ShapedArray` headers only |
| [jax/_src/basearray.py](../jax/_src/basearray.py) | **Skim** | `jax.Array` protocol story |
| [jax/_src/stages.py](../jax/_src/stages.py) | **Skim** | AOT stage objects |

That’s enough source contact for proficiency. Prefer docs over reading `_src`.

---

## Code: Skip (contributor / compiler)

| Path | Why you can ignore it |
|------|------------------------|
| [jax/_src/interpreters/mlir.py](../jax/_src/interpreters/mlir.py) | Jaxpr → StableHLO lowering |
| [jax/_src/compiler.py](../jax/_src/compiler.py) | XLA compile handoff |
| [jax/_src/dispatch.py](../jax/_src/dispatch.py) | Runtime dispatch |
| [jax/_src/pjit.py](../jax/_src/pjit.py) | Real `jit` implementation (know the *name* only) |
| [jax/_src/interpreters/partial_eval.py](../jax/_src/interpreters/partial_eval.py) | Trace → jaxpr machinery |
| [jax/_src/interpreters/ad.py](../jax/_src/interpreters/ad.py) | AD rule engine (unless custom AD research) |
| [jax/_src/interpreters/batching.py](../jax/_src/interpreters/batching.py) | `vmap` rule engine |
| [jax/_src/pmap.py](../jax/_src/pmap.py) | Legacy `pmap` internals |
| [jax/_src/lax/lax.py](../jax/_src/lax/lax.py) | Huge op surface — don’t read through |
| [jaxlib/pytree.cc](../jaxlib/pytree.cc) | C++ pytree engine |
| Most of `jaxlib/`, `build/`, Bazel files | Packaging / native builds |

---

## Examples (Must practice)

| Path | Tier | Why |
|------|------|-----|
| [examples/mnist_classifier_fromscratch.py](../examples/mnist_classifier_fromscratch.py) | **Must** | Minimal `jit`+`grad` loop |
| [examples/differentially_private_sgd.py](../examples/differentially_private_sgd.py) | **Must** | `vmap` per-example grads |
| [examples/advi.py](../examples/advi.py) | Skim | `vmap`+`jit`+`grad` composition |
| [examples/spmd_mnist_classifier_fromscratch.py](../examples/spmd_mnist_classifier_fromscratch.py) | Systems track | SPMD / donation |

---

## Count check (curated set)

**Docs Must:** 13 · **Code Must/Skim:** 6 · **Examples Must:** 2 · **Skip list:** 10+  

Together with this `onboarding/` curriculum ≈ the **20–30 concepts/files** that define proficiency.

---

## Reading order vs opening files

Follow [README.md](README.md) weeks 1–4. Open `SOURCE_MAP` entries only when a chapter points at them. Do not start proficiency by browsing `jax/_src/` alphabetically.

Diagram: [diagrams/dependency-map.md](diagrams/dependency-map.md).
