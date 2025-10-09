# SparseKolesky.jl

SparseKolesky.jl is a Julia package for the sparse Cholesky
factorization of dense kernel ("covariance") matrices.

Our method enjoys a number of advantageous properties including

- Near-linear computational time and space efficiency
- Specifically, ``\epsilon``-accuracy in time complexity ``\mathcal{O}(N
  \log^{2d} (N/\epsilon))`` using just ``\mathcal{O}(N \log^{d} (N/\epsilon))``
  nonzero entries (and kernel function evaluations) where ``d`` is the
  intrinsic dimensionality of the data
- Embarrassingly parallel factorization and simple implementation
- Optimal Kullback–Leibler divergence for a fixed sparsity pattern
- Guaranteed positive definiteness of the resulting factor

This is a fork of [KoLesky.jl](https://github.com/f-t-s/KoLesky.jl)
tailored to applications in Gaussian process regression (e.g.
[EarthquakeGPs.jl](https://github.com/stephen-huan/EarthquakeGPs.jl)).
