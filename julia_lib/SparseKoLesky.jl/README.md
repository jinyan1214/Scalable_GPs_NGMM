# SparseKoLesky

This is a fork of [KoLesky.jl](https://github.com/f-t-s/KoLesky.jl)
tailored to applications in Gaussian process regression (e.g.
[EarthquakeGPs.jl](https://github.com/stephen-huan/EarthquakeGPs.jl)).

Documentation may be found on the project
[homepage](https://kolesky.cgdct.moe).

## Installation

Use the [Pkg](https://pkgdocs.julialang.org/v1/managing-packages/) interface
```text
pkg> add https://github.com/stephen-huan/SparseKoLesky.jl
```

## Running

Tests may be ran through the standard `Pkg.test()` interface
```text
pkg> test
```

The files in `examples` may be ran like
```bash
julia --project="@." examples/main.jl
```
