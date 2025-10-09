"""
    GP

Implements routines for statistical inference in Gaussian processes.
"""
module GP

export sample
export estimate

using Random: AbstractRNG

using KernelFunctions: Kernel, kernelmatrix, kernelmatrix_diag

using ..Utils: chol
import ..SparseKoLesky: sample

"""
    sample(rng::AbstractRNG, kernel::Kernel, x, mean=zeros(length(x)); draws=1)

Draw samples from the multivariate Gaussian with specified `mean` and `kernel`.

That is, draw ``y \\sim \\mathcal{N}(\\mu(x), k(x, x))`` where
``\\mu`` is the mean and ``k(\\cdot, \\cdot)`` is the kernel function.
"""
function sample(
    rng::AbstractRNG, kernel::Kernel, x, mean=zeros(length(x)); draws=1
)
    L = chol(kernelmatrix(kernel, x)).L
    z = randn(rng, length(x), draws)
    return L * z .+ mean
end

"""
    estimate(
        kernel::Kernel,
        x_train,
        y_train,
        x_test,
        mean_train=zeros(length(x_train)),
        mean_test=zeros(length(x_test));
        full_cov=false,
        indices=1:length(x_train),
    )

Estimate the posterior mean and variance with Gaussian process regression.

Returns only the diagonal of the covariance matrix when `full_cov` is false.
"""
function estimate(
    kernel::Kernel,
    x_train,
    y_train,
    x_test,
    mean_train=zeros(length(x_train)),
    mean_test=zeros(length(x_test));
    full_cov=false,
    indices=1:length(x_train),
)
    @views x_train_ind = x_train[indices]
    @views y_train_ind = y_train[indices, :]
    @views mean_ind = mean_train[indices]
    k_tt = kernelmatrix(kernel, x_train_ind)
    k_tp = kernelmatrix(kernel, x_train_ind, x_test)
    k_pt_tt = (chol(k_tt) \ k_tp)'

    mean_pred = mean_test .+ k_pt_tt * (y_train_ind .- mean_ind)
    cov_pred = if full_cov
        k_pp = kernelmatrix(kernel, x_test)
        k_pp - k_pt_tt * k_tp
    else
        k_pp = kernelmatrix_diag(kernel, x_test)
        k_pp - vec(sum(k_pt_tt' .* k_tp; dims=1))
    end
    return (mean_pred, cov_pred)
end

end
