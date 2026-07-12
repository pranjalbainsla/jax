# 02 — Pytrees: the container model for ML state

**Why this abstraction exists:** Every transform (`jit`, `grad`, `vmap`, …) must treat nested parameter structures *uniformly*. Pytrees are the contract: “walk any tree of containers; arrays are leaves.”

Without pytrees, you would flatten models by hand before every `grad`, and libraries like Flax/Equinox/Optax could not share one update story.

## Mental model

```text
pytree = nested containers + array leaves

examples:
  params = {"linear": {"w": Array[...], "b": Array[...]}}
  (params, opt_state, batch)
  MyModule(submodules..., arrays...)   # if registered
```

| Piece | Role |
|-------|------|
| **Leaf** | Usually a `jax.Array` (sometimes a scalar that JAX treats as array-like) |
| **Node** | `tuple`, `list`, `dict`, custom registered type |
| **Structure** | The shape of the tree ignoring leaf values |
| **Flatten / unflatten** | How transforms and `jax.tree.map` walk your data |

`grad(loss)(params, batch)` differentiates w.r.t. array leaves in `params`. Nested dicts “just work” because they are already pytrees.

## User API you should memorize

Prefer the modern namespace:

```python
import jax

leaves = jax.tree.leaves(params)
params2 = jax.tree.map(lambda x: x * 0.0, params)
struct = jax.tree.structure(params)
```

Older / fuller utilities live in `jax.tree_util` (registration, path-aware APIs).

| Task | API |
|------|-----|
| Map over leaves | `jax.tree.map(f, tree, ...)` |
| Reduce / all leaves | `jax.tree.leaves`, `jax.tree.reduce` |
| Compare structures | `jax.tree.structure` |
| Flatten | `jax.tree.flatten` / `unflatten` |
| Register custom type | `jax.tree_util.register_pytree_node` or `register_dataclass` |

**Must read (tiny):** [jax/tree.py](../jax/tree.py)  
**Must know for custom containers:** [jax/tree_util.py](../jax/tree_util.py)  
**Skip forever:** [jaxlib/pytree.cc](../jaxlib/pytree.cc) (C++ engine)

Official tutorial: [docs/pytrees.md](../docs/pytrees.md), [docs/custom_pytrees.md](../docs/custom_pytrees.md).

## Why ML code is full of `tree.map`

Optax-style updates are “same tree, new leaves”:

```python
updates, opt_state = optimizer.update(grads, opt_state, params)
params = jax.tree.map(lambda p, u: p + u, params, updates)  # schematic
```

Checkpointing, EMA, freezing subsets, and multi-optimizer setups are all **tree surgery**: select leaves by path, map differently on different subtrees.

When reading repos, `jax.tree_*` / `optax` / `flax.traverse_util` are usually “pytree choreography,” not deep JAX magic.

## Registration: when you need it

Built-ins (`tuple`, `list`, `dict`, some stdlib types) work out of the box. You register when:

- You invent a `class Params: ...` or dataclass you want `grad`/`jit` to treat as a tree.
- A library module type must flatten into arrays + aux metadata.

Equinox/Flax register their module types for you. If you write a bare `@dataclass` of arrays, prefer `jax.tree_util.register_dataclass` (or follow the custom pytrees doc) so you do not accidentally treat the whole object as a leaf.

**Pitfall:** If a type is *not* registered, JAX may treat the object as a leaf (or error), which breaks `grad` and produces baffling jit errors. Symptom: “seems like my module isn’t differentiating.”

## Interaction with transforms

```mermaid
flowchart LR
  args[Nested_args]
  flat[Flatten_to_leaves]
  tr[Trace_or_transform_leaves]
  rebuild[Unflatten_to_same_structure]
  out[Nested_outputs]

  args --> flat --> tr --> rebuild --> out
```

`static_argnums` / `static_argnames` mark arguments that are **not** traced as arrays — often Python ints, bools, or hashable config. Those still participate in the *cache key* for `jit`. Putting a large array in `static_argnums` is almost always wrong.

Details: [04-transformations.md](04-transformations.md), [06-performance-pitfalls.md](06-performance-pitfalls.md).

## Exercises (mental, then in a REPL)

1. Build `params = {"w": jnp.ones((2, 2)), "b": jnp.zeros((2,))}` and write `jax.tree.map` that adds noise to every leaf.
2. Flatten and unflatten; confirm structure equality with a second tree of different values.
3. Pass `params` into `jax.grad(lambda p: jnp.sum(p["w"]))` and inspect the gradient tree shape.

## What to skip

- Implementing a new pytree backend
- Path hashing internals
- Anything under `jaxlib/` related to pytrees

Next: [03-tracing-and-jaxpr.md](03-tracing-and-jaxpr.md).
