# 00 — Philosophy: why JAX exists

**Audience:** ML engineers and researchers who will *use* JAX, not maintain it.  
**Goal:** Build the right priors before you touch transforms or IR.

## The one-sentence thesis

JAX is a **system for composing pure array programs with program transformations**, then compiling those programs with XLA. The product is not “NumPy on GPU”; it is *`jit(vmap(grad(f)))`* as a first-class way to write research code.

## Historical context (enough, not a museum tour)

| Era | Idea | What to keep |
|-----|------|--------------|
| Autograd | Transform Python functions (`grad`, `jacobian`) by tracing | Transforms as the API |
| XLA | Compile array graphs for accelerators | Staging + fusion + portability |
| SysML 2018 JAX paper | Autograd-style transforms + XLA | Philosophy; **not** today’s API surface |
| Modern JAX | Unified `jax.Array`, mesh sharding, `jit`≈pjit path, StableHLO | What you actually learn |

You do **not** need the SysML paper or Autodidax to be productive. Treat them as optional depth *after* this curriculum.

Companion in-repo: [docs/about.md](../docs/about.md), [README.md](../README.md).

## Functional core (non-negotiable mental model)

JAX wants your ML code to look like:

```text
params, opt_state, key  →  loss, new_params, new_opt_state, new_key
```

not:

```text
module.forward(x); loss.backward(); optimizer.step()  # mutating objects
```

Three pillars:

1. **Purity under transforms** — A jitted / differentiated function should not rely on side effects, global mutation, or “whatever was printed last time.” Captured Python values can be baked into the compiled executable.
2. **Immutability of arrays** — Updates are functional: `x.at[i].set(v)`, not `x[i] = v`. This matches how XLA thinks about buffers and how transforms rewrite programs.
3. **Explicit state and PRNG** — Parameters and RNG keys are *arguments and return values* (usually pytrees), not hidden fields that Autograd discovers by walking a module graph.

If you fight these three, every later chapter (tracing, recompiles, multi-device) will feel hostile. If you lean into them, the rest of JAX becomes predictable.

## Why “transforms compose” is the product

In many frameworks, vectorization, differentiation, and compilation are separate product features with separate restrictions. In JAX they are the **same kind of thing**: functions that take a function and return a new function, implemented by rewriting an intermediate representation (jaxpr).

```python
# Per-example gradients, compiled — a research-shaped one-liner
grads = jax.jit(jax.vmap(jax.grad(loss)))(params, batch)
```

Composition is why:

- You can differentiate through a `vmap`’d loss.
- You can `vmap` a custom JVP.
- You can rematerialize (`checkpoint`) a block and still `jit` + `grad` it.

You will spend much of your career choosing *order* of transforms, not inventing new ones. See [04-transformations.md](04-transformations.md) and [diagrams/transform-composition.md](diagrams/transform-composition.md).

## JAX vs PyTorch (fundamental, not cosmetic)

| Dimension | PyTorch (typical) | JAX |
|-----------|-------------------|-----|
| Execution | Eager by default; `torch.compile` optional | Eager for exploration; **staging** (`jit`) for real training |
| Autodiff | Tape / dynamic graph on a forward run | Transform of a function → another function (`grad`, `jvp`, `vjp`) |
| Parameters | Mutable `nn.Module` + Parameter tensors | Immutable pytrees of arrays (+ library wrappers) |
| Batching | Implicit batch dims in layers; `vmap` exists but secondary | **`vmap` is a primary tool** for per-example grads, ensembles, etc. |
| Parallelism | DDP / FSDP / DeviceMesh (evolving) | Mesh + sharding specs + `jit` / `shard_map` (SPMD) |
| Control flow | Python `if`/`for` fine in eager | Under `jit`, value-dependent control flow needs `lax.cond` / `scan` / … |
| RNG | Global generator or generator objects | Explicit `PRNGKey` threading / splitting |
| Debugging | Stack traces on eager ops | Stage with `jit` off first; then `make_jaxpr`, `jax.debug` |

**When PyTorch feels better:** highly dynamic Python control flow, interactive debugging of every op, huge ecosystem of eager-first libraries, “just run it.”

**When JAX feels better:** you want a single, composable story for grad × batch × compile × shard; functional training steps; reproducible paper code with explicit state; TPU/SPMD-shaped workloads.

Neither is “more scientific.” They encode different defaults. Switching frameworks without switching mental models is the main failure mode.

## What “highly proficient” means here

You are proficient when you can:

1. Explain why a `ConcretizationTypeError` happened in terms of tracers.
2. Predict whether a change will **recompile**.
3. Read a paper repo and find `loss` → `value_and_grad` → `jit` step → sharding in minutes.
4. Choose between Python loops, `vmap`, and `lax.scan` deliberately.
5. Know which files in *this* repo are worth opening — and which to ignore forever as a user.

That last skill is [SOURCE_MAP.md](SOURCE_MAP.md).

## How to use the rest of this guide

| Chapter | Question it answers |
|---------|---------------------|
| [01-architecture](01-architecture.md) | How does Python become an executable? |
| [02-pytrees](02-pytrees.md) | Why are params trees, and why does every API take them? |
| [03-tracing-and-jaxpr](03-tracing-and-jaxpr.md) | What is a Tracer / Jaxpr, and why do they constrain Python? |
| [04-transformations](04-transformations.md) | What do jit/grad/vmap/parallel actually rewrite? |
| [05-arrays-and-execution](05-arrays-and-execution.md) | What is a `jax.Array` and when does work really run? |
| [06-performance-pitfalls](06-performance-pitfalls.md) | What burns wall-clock and HBM? |
| [07-debugging](07-debugging.md) | How do I localize staging bugs? |
| [08-reading-repos](08-reading-repos.md) | How do I navigate real research codebases? |

Official companions after this chapter: [Thinking in JAX](../docs/notebooks/thinking_in_jax.ipynb), [Key concepts](../docs/key-concepts.md).

## Explicitly out of scope (for now)

- Writing new primitives or MLIR lowerings
- Building jaxlib / XLA
- Autodidax as required reading ([docs/autodidax.md](../docs/autodidax.md) is optional deep dive)
- Becoming a Flax/Equinox expert (we only teach enough to *read* them)

Next: [01-architecture.md](01-architecture.md).
