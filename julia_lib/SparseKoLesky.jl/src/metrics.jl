"""
    Metrics

A collection of accuracy metrics.
"""
module Metrics

export mse, rmse
export coverage

using Statistics: mean

using ..Utils: norminvcdf

"""
    mse(u, v)

Mean squared error (MSE) between collections of vectors `u` and `v`.

The MSE between vectors ``u`` and ``v`` of length ``n`` is defined as
```math
\\mathsf{mse}(u, v) \\coloneqq \\frac{1}{n} \\lVert u - v \\rVert^2
```
where ``\\lVert \\cdot \\rVert`` is the Euclidean norm.

See also: [`rmse`](@ref).
"""
mse(u, v) = mean((u - v) .^ 2; dims=1)

"""
    rmse(u, v)

Root-mean-square error (RMSE) between collections of vectors `u` and `v`.

The RMSE is the square root of the mean squared error.

See also: [`mse`](@ref).
"""
rmse(u, v) = sqrt.(mse(u, v))

"""
    coverage(y_test, mean_pred, var_pred; alpha=0.9)

Percentage of points within a `alpha`-% confidence interval around `mean_pred`.
"""
function coverage(y_test, mean_pred, var_pred; alpha=0.9)
    std = sqrt.(var_pred)
    # symmetric coverage centered around mean
    delta = norminvcdf((1 + alpha) / 2) * std
    counts = @. (mean_pred - delta <= y_test) & (y_test <= mean_pred + delta)
    return vec(mean(counts; dims=2))
end

end
