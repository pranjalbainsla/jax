# JAX onboarding for ML engineers

A **mental-model-first** curriculum for people who will build models, reproduce papers, write research code, and read JAX repositories — **not** for contributing to JAX itself.

Official docs already teach the mechanics. This folder adds what they don’t package as one path: architecture narrative, transform composition maps, PyTorch contrast, a curated file map with explicit skip lists, debugging playbooks, and a progressive schedule.

```text
onboarding/
  README.md                 
  00-philosophy.md
  01-architecture.md
  02-pytrees.md
  03-tracing-and-jaxpr.md
  04-transformations.md
  05-arrays-and-execution.md
  06-performance-pitfalls.md
  07-debugging.md
  08-reading-repos.md
  SOURCE_MAP.md             ← ~25 must / skim / skip files
  GLOSSARY.md
  diagrams/                 ← Mermaid sources (also inlined in chapters)
```



## How to use this guide

1. Read chapters **in order** (or follow the 4-week plan below).
2. After each chapter, do the linked **official companion** for drills — this guide is the map; docs are the terrain.
3. Open source files only when a chapter or [SOURCE_MAP.md](SOURCE_MAP.md) says **Must** / **Skim**.
4. Treat anything labeled **Skip** as out of scope for proficiency.

**Default posture:** understand *why* pytrees, tracing, jaxpr, and transforms exist. Do **not** start in `mlir.py`, `compiler.py`, or Autodidax.

## What you will be able to do

- Explain Python → Tracer → Jaxpr → StableHLO → XLA → `jax.Array`
- Compose `jit` / `grad` / `vmap` (and modern parallel APIs) deliberately
- Predict recompiles, concretization failures, and async timing bugs
- Navigate Flax/Equinox/Optax-style repos via the training spine
- Ignore ~90% of `jax/_src` without guilt



## Four-week reading order



### Week 1 — Mental model

- [ ] [00-philosophy.md](00-philosophy.md) — purity, transforms-as-product, JAX vs PyTorch
- [ ] [01-architecture.md](01-architecture.md) — stack + execution pipeline
- [ ] [02-pytrees.md](02-pytrees.md) — params / opt_state as trees
- [ ] Companions: [Thinking in JAX](../docs/notebooks/thinking_in_jax.ipynb), [key-concepts](../docs/key-concepts.md), [pytrees](../docs/pytrees.md)
- [ ] Diagrams: [stack](diagrams/stack.md), [dependency-map](diagrams/dependency-map.md)



### Week 2 — Staging and IR

- [ ] [03-tracing-and-jaxpr.md](03-tracing-and-jaxpr.md)
- [ ] [04-transformations.md](04-transformations.md) — focus on `jit` / `grad` / `vmap`
- [ ] Companions: [tracing](../docs/tracing.md), [jaxpr](../docs/jaxpr.md), [Sharp Bits](../docs/notebooks/Common_Gotchas_in_JAX.ipynb), JAX 101 transform guides ([jit](../docs/jit-compilation.md), [vmap](../docs/automatic-vectorization.md), [grad](../docs/automatic-differentiation.md))
- [ ] Diagrams: [tracing](diagrams/tracing.md), [transform-composition](diagrams/transform-composition.md)
- [ ] Practice: [examples/mnist_classifier_fromscratch.py](../examples/mnist_classifier_fromscratch.py)



### Week 3 — Execution and scale

- [ ] [05-arrays-and-execution.md](05-arrays-and-execution.md)
- [ ] [04-transformations.md](04-transformations.md) — parallel section (`pmap` → mesh / `shard_map`)
- [ ] [06-performance-pitfalls.md](06-performance-pitfalls.md)
- [ ] Companions: [aot](../docs/aot.md), [parallel](../docs/parallel.md), [training cookbook](../docs/the-training-cookbook.md)
- [ ] Diagram: [execution-flow](diagrams/execution-flow.md)



### Week 4 — Practice

- [ ] [07-debugging.md](07-debugging.md)
- [ ] [08-reading-repos.md](08-reading-repos.md)
- [ ] [SOURCE_MAP.md](SOURCE_MAP.md) + [GLOSSARY.md](GLOSSARY.md)
- [ ] Companions: [autodiff cookbook](../docs/notebooks/autodiff_cookbook.md)
- [ ] Practice: [differentially_private_sgd.py](../examples/differentially_private_sgd.py) (`vmap` grads); map one external paper repo’s `train_step`



## Chapter index


| #   | Chapter                                            | Core question                          |
| --- | -------------------------------------------------- | -------------------------------------- |
| 00  | [Philosophy](00-philosophy.md)                     | Why does JAX exist this way?           |
| 01  | [Architecture](01-architecture.md)                 | How does Python become an executable?  |
| 02  | [Pytrees](02-pytrees.md)                           | Why is all state a tree?               |
| 03  | [Tracing & Jaxpr](03-tracing-and-jaxpr.md)         | What do transforms actually see?       |
| 04  | [Transformations](04-transformations.md)           | How do jit/grad/vmap/parallel compose? |
| 05  | [Arrays & execution](05-arrays-and-execution.md)   | What runs when, and where?             |
| 06  | [Performance pitfalls](06-performance-pitfalls.md) | What burns time and HBM?               |
| 07  | [Debugging](07-debugging.md)                       | How do I localize staging bugs?        |
| 08  | [Reading repos](08-reading-repos.md)               | How do I navigate research code?       |




## Explicitly skip (until you need them)


| Topic                                                                                    | When (if ever)                           |
| ---------------------------------------------------------------------------------------- | ---------------------------------------- |
| [Autodidax](../docs/autodidax.md)                                                        | Optional deep dive after week 4          |
| [jax/_src/interpreters/mlir.py](../jax/_src/interpreters/mlir.py)                        | Writing custom lowerings                 |
| [jax/_src/compiler.py](../jax/_src/compiler.py) / [dispatch.py](../jax/_src/dispatch.py) | Contributing to runtime                  |
| [jax/_src/pjit.py](../jax/_src/pjit.py)                                                  | Know “jit uses pjit path”; don’t read it |
| [jaxlib/pytree.cc](../jaxlib/pytree.cc)                                                  | Never for ML work                        |
| Full [partial_eval.py](../jax/_src/interpreters/partial_eval.py)                         | Extending JAX                            |


Full table: [SOURCE_MAP.md](SOURCE_MAP.md).

## Relationship to official docs

```text
This onboarding/     →  why + architecture + curriculum + skip lists
docs/ (JAX 101, …)   →  tutorials and API detail
jax/_src/            →  implementation (mostly skip)
```

Beginner hub in-tree: [docs/beginner_guide.rst](../docs/beginner_guide.rst). Published site: [https://docs.jax.dev/en/latest/](https://docs.jax.dev/en/latest/)

## Diagrams


| Diagram               | File                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| Stack                 | [diagrams/stack.md](diagrams/stack.md)                                 |
| Execution flow        | [diagrams/execution-flow.md](diagrams/execution-flow.md)               |
| Tracing → jaxpr       | [diagrams/tracing.md](diagrams/tracing.md)                             |
| Transform composition | [diagrams/transform-composition.md](diagrams/transform-composition.md) |
| Dependency map        | [diagrams/dependency-map.md](diagrams/dependency-map.md)               |




## One-line thesis (pin this)

> JAX turns pure array functions into staged programs (jaxprs) that transformations rewrite and XLA executes — proficiency is knowing that pipeline and its constraints, not memorizing `_src`.

