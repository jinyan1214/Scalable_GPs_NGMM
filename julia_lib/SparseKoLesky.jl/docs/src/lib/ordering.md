# Ordering

## Index

```@index
Pages = ["ordering.md"]
```

```@meta
CurrentModule = SparseKoLesky.Ordering
```

```@docs
Ordering
```

```@docs
KernelDist
KernelDist(p, q)
```

```@docs
maximin_ordering(
    x::AbstractMatrix,
    k_neighbors=1;
    init_distances,
    Tree,
    metric::Metric=Euclidean(),
)
maximin_ordering(
    kernel::Kernel,
    x::AbstractMatrix,
    k_neighbors=1;
    init_distances,
    Tree,
)
```
