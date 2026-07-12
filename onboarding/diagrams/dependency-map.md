# Dependency map: user surface → internals

What you touch as an ML engineer vs what you can ignore.

```mermaid
flowchart TD
  subgraph user [You_write]
    jnp[jax.numpy]
    api[jax.jit_grad_vmap]
    tree[jax.tree]
    random[jax.random]
  end

  subgraph mental [Know_exists]
    core[core_Tracer_Jaxpr_Primitive]
    stages[stages_AOT]
    arr[jax_Array]
  end

  subgraph skip [Skip_for_ML_work]
    pe[partial_eval]
    mlir[mlir_lowering]
    compiler[compiler_XLA]
    dispatch[dispatch]
    pjit_impl[pjit_internals]
    cplusplus[jaxlib_C++]
  end

  jnp --> core
  api --> core
  tree --> arr
  api --> stages
  stages --> mlir
  mlir --> compiler
  compiler --> dispatch
  api -.->|"jit implemented via"| pjit_impl
```

## Reading direction

1. **Docs + this guide** for mental models.
2. **Public APIs** (`jax/tree.py`, docstrings in `jax/_src/api.py`) for signatures.
3. **Skim** `core.py` class headers (`Tracer`, `Jaxpr`, `Primitive`) only if debugging IR.
4. **Never required** for research: MLIR, compiler, dispatch, jaxlib C++, full `pjit.py`.
