# 08 — Reading JAX research repositories

Goal: open a paper repo and find the **training spine** in minutes, then understand transforms and sharding without drowning in framework code.

## The spine almost every repo has

```text
data batch
   → loss(params, batch)           # pure function or apply_fn
   → value_and_grad / grad
   → optimizer update (optax)
   → jit(train_step)
   → optional: mesh / sharding
   → checkpointing (orbax / custom)
```

Hunt in this order:

1. **`loss` / `compute_metrics`** — what is differentiated?
2. **`train_step` / `update`** — where is `jit`? `value_and_grad`?
3. **Optimizer** — Optax `init`/`update` + `tree` maps
4. **Model apply** — Flax `apply` / Equinox `__call__` / Haiku `transform`
5. **Sharding / mesh** — only if multi-device
6. **Config** — static flags that affect `jit` cache keys

## Pattern recognition cheat sheet

| You see… | It usually means… |
|----------|-------------------|
| `TrainState` | Params + opt_state (+ apply_fn) bundled as a pytree |
| `apply_fn(params, x)` | Separated weights from code (Flax-style) |
| `eqx.filter_grad` / `filter_jit` | Equinox filtering static vs array leaves |
| `optax.apply_updates` | Pytree add of updates onto params |
| `@jax.jit` on `train_step` | Compiled fused step |
| `jax.checkpoint` / `remat` | Memory optimization on blocks |
| `NamedSharding` / `PartitionSpec` | SPMD layout |
| `pmap` | Older multi-device code — map mentally to mesh+jit |
| `donate_argnums=(0,)` | Buffer donation on state |

## Ecosystem: when to leave pure JAX

| Library | Role | Learn enough to… |
|---------|------|------------------|
| **Optax** | Optimizers, schedules, grad clipping | Read `update` + tree updates |
| **Flax (NNX/Linen)** | Modules, apply, train state | Find params pytree + apply |
| **Equinox** | Modules as pytrees | Read filtered transforms |
| **Orbax** | Checkpointing | Restore shapes/shardings |
| **RLax / Distrax / …** | Domain libs | Same JAX rules underneath |

Pure JAX is enough for small models ([examples/mnist_classifier_fromscratch.py](../examples/mnist_classifier_fromscratch.py)). Real papers usually pick Flax or Equinox — **the mental model does not change**: still pytrees, still `jit(grad(...))`.

## In-repo examples worth reading

| File | Patterns |
|------|----------|
| [examples/mnist_classifier_fromscratch.py](../examples/mnist_classifier_fromscratch.py) | `jit` + `grad` training |
| [examples/mnist_classifier.py](../examples/mnist_classifier.py) | Same with stax-style helpers |
| [examples/differentially_private_sgd.py](../examples/differentially_private_sgd.py) | `vmap` per-example grads + `jit` |
| [examples/advi.py](../examples/advi.py) | `vmap` + `jit` + `grad` |
| [examples/spmd_mnist_classifier_fromscratch.py](../examples/spmd_mnist_classifier_fromscratch.py) | SPMD / donate |

## How to read a foreign `train_step`

Ask:

1. What is the pytree being differentiated?
2. What is static vs traced?
3. Where could recompiles come from (variable shapes, Python flags)?
4. Is there remat? Where?
5. Is RNG threaded explicitly (`key` split) or hidden?

If something is unclear, `jax.make_jaxpr(train_step)(state, batch)` on tiny shapes — even in a foreign codebase — reveals the real program.

## Mapping old → new

| Old repo idiom | Modern reading |
|----------------|----------------|
| `pmap(train_step)` | SPMD `jit` + shardings |
| `DeviceArray` | `jax.Array` |
| `jax.tree_util.tree_map` | `jax.tree.map` |
| Haiku `hk.transform` | init/apply pair; params pytree |

## Suggested practice

1. Read MNIST fromscratch end-to-end; reimplement the step from memory.
2. Open a small Flax or Equinox example outside this repo; find the spine.
3. Open one paper repo you care about; write a half-page “map” of loss → step → shard.

## Tie-back to this curriculum

- Pytrees: [02](02-pytrees.md)
- Transforms: [04](04-transformations.md)
- Pitfalls while reproducing: [06](06-performance-pitfalls.md)
- File map for *this* JAX tree: [SOURCE_MAP.md](SOURCE_MAP.md)

Official systems-oriented companion: [docs/the-training-cookbook.md](../docs/the-training-cookbook.md).
