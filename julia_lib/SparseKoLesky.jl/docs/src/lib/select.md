# Select

## Index

```@index
Pages = ["select.md"]
```

```@meta
CurrentModule = SparseKoLesky.Select
```

```@docs
Select
```

## Public interface

```@docs
select(kernel::Kernel, X_train, X_test, s)
select(kernel::Kernel, X, train, test, s; budget)
```

## Internal functions

```@docs
select_single
select_mult
select_nonadj
select_budget
```

### Helper functions

```@docs
covariance!
cholupdate!(L, i, k, cond_var; rows)
cholupdate!(L, i, k, cond_var, cond_cov)
```

```@docs
choldowndate!
cholinsert!
insertindex
selectpoint!
scoresupdate!
```
