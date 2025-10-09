"""
    Ordering

Implements the maximin ordering and other geometric algorithms.
"""
module Ordering

export KernelDist
export maximin_ordering

using KernelFunctions: Kernel, kernelmatrix
using Distances: Metric, Euclidean
using NearestNeighbors: KDTree, BallTree, inrange, knn

using ..SparseKoLesky: SparseKoLesky as KL, MutableHeap
import ..SparseKoLesky: maximin_ordering

"""
    KernelDist{K<:Kernel} <: Metric

Correlation distance based on a kernel function for covariances.

This implements the [Distances.jl](https://github.com/JuliaStats/Distances.jl)
`Metric` interface.

# Arguments
- `kernel::K`: base kernel function.
   This should be a [KernelFunctions.jl]\
   (https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/)
   `Kernel`.
"""
struct KernelDist{K<:Kernel} <: Metric
    kernel::K
end

"""
    (k::KernelDist)(p, q)

Distance between points `p` and `q` using the [`KernelDist`](@ref) metric.

The correlation distance for a kernel function
``k(\\cdot, \\cdot)`` is defined as
```math
\\mathsf{dist}(p, q) \\coloneqq
    \\sqrt{1 - \\left \\lvert \\rho \\right \\rvert}
```
where the correlation coefficient ``\\rho`` is defined as
```math
\\rho(p, q) \\coloneqq \\frac{k(p, q)}{\\sqrt{k(p, p) k(q, q)}}.
```
"""
function (k::KernelDist)(p, q)
    pp, pq, _, qq = kernelmatrix(k.kernel, [p, q])
    corr = pq / sqrt(pp * qq)
    return sqrt(1 - abs(corr))
end

"""
    maximin_ordering(
        x::AbstractMatrix,
        k_neighbors=1;
        init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))),
        Tree=KDTree,
        metric::Metric=Euclidean(),
    )

Maximin ordering of `x` using `k_neighbors` for robustness.

The current implementation is directly copied
from the [`maximin_ordering`](@ref) of
[KoLesky.jl](https://github.com/f-t-s/KoLesky.jl/blob/master/src/MaximinNN.jl).
"""
function maximin_ordering(
    x::AbstractMatrix,
    k_neighbors=1;
    init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))),
    Tree=KDTree,
    metric::Metric=Euclidean(),
)
    # constructing the tree
    N = size(x, 2)
    tree = Tree(x, metric)
    nearest_distances = copy(init_distances)
    @assert size(nearest_distances) == (k_neighbors, N)
    for k in 1:N
        sort!(vec(view(nearest_distances, :, k)); rev=true)
    end
    heap = MutableHeap(vec(nearest_distances[1, :]))
    ℓ = Vector{eltype(init_distances)}(undef, N)
    P = Vector{Int64}(undef, N)
    for k in 1:N
        pivot = KL.top_node!(heap)
        ℓ[k] = KL.getval(pivot)
        P[k] = KL.getid(pivot)
        # a little clunky, since inrange doesn't have an option
        # to return range and we want to avoid introducing a
        # distance measure separate from the NearestNeighbors
        number_in_range = length(inrange(tree, x[:, P[k]], ℓ[k]))
        ids, dists = knn(tree, x[:, P[k]], number_in_range)
        for (id, dist) in zip(ids, dists)
            if id != KL.getid(pivot)
                # update the distance as stored in nearest_distances
                new_dist = KL._update_distances!(nearest_distances, id, dist)
                # decreases the distance as stored in the heap
                KL.update!(heap, id, new_dist)
            end
        end
    end
    # returns the maximin ordering P together with the distance vector
    return P, ℓ
end

"""
    maximin_ordering(
        kernel::Kernel,
        x::AbstractMatrix,
        k_neighbors=1;
        init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))),
        Tree=BallTree,
    )

Maximin ordering with the [`KernelDist`](@ref) metric.

See also: [`maximin_ordering`](@ref).
"""
function maximin_ordering(
    kernel::Kernel,
    x::AbstractMatrix,
    k_neighbors=1;
    init_distances=fill(typemax(eltype(x)), (k_neighbors, size(x, 2))),
    Tree=BallTree,  # `KDTree`s only work with axis-aligned metrics
)
    metric = KernelDist(kernel)
    return maximin_ordering(x, k_neighbors; init_distances, Tree, metric)
end

end
