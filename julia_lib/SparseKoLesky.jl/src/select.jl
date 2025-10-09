"""
    Select

Provides efficient algorithms for information-theoretic sparsity selection.
"""
module Select

export select

using LinearAlgebra: LinearAlgebra, axpby!, mul!

using KernelFunctions: Kernel, kernelmatrix, kernelmatrix!, kernelmatrix_diag

"""
    covariance!(kernel::Kernel, X_train, X_test, L, i, k)

Fill the factor `L`'s `i`-th column by the covariance with the `k`-th point.
"""
function covariance!(kernel::Kernel, X_train, X_test, L, i, k)
    n = length(X_train)
    m = length(X_test)
    # if k > n, then it's a test point rather than a train point
    point = @views (k <= n) ? X_train[k:k] : X_test[(k - n):(k - n)]
    @views kernelmatrix!(L[begin:n, i:i], kernel, X_train, point)
    # optionally include testing points if L is big enough
    if size(L, 1) == n + m
        @views kernelmatrix!(L[(n + 1):end, i:i], kernel, X_test, point)
    end
    return nothing
end

"""
    cholupdate!(L, i, k, cond_var; rows=size(L, 1))

Conditions the `i`-th column of the Cholesky factor `L` by the `k`-th point.

This function modifies both `L` and `cond_var` in-place.

See also [`cholupdate!(L, i, k, cond_var, cond_cov)`](@ref).
"""
function cholupdate!(L, i, k, cond_var; rows=size(L, 1))
    n = length(cond_var)
    # update Cholesky factor by left looking
    col = @views L[begin:rows, i]
    @views mul!(
        col, L[begin:rows, begin:(i - 1)], L[k, begin:(i - 1)], -1.0, 1.0
    )
    @views @. col /= sqrt(L[k, i])
    # update conditional variance
    @views @. cond_var -= L[begin:n, i]^2
    # mark as selected
    k <= n && (cond_var[k] = -1.0)
    return nothing
end

"""
    cholupdate!(L, i, k, cond_var, cond_cov)

Additionally updates `cond_cov` in-place if it is provided.

See also [`cholupdate!(L, i, k, cond_var; rows)`](@ref).
"""
function cholupdate!(L, i, k, cond_var, cond_cov)
    cholupdate!(L, i, k, cond_var)
    @views @. cond_cov -= L[begin:(end - 1), i] * L[end, i]
    return nothing
end

# basic selection

"""
    select_single(kernel::Kernel, X_train, X_test, s)

Select the `s` points from `X_train` most informative to `X_test` greedily.

This function assumes `length(X_test) == 1`.

See also [`select`](@ref), [`select_mult`](@ref).
"""
function select_single(kernel::Kernel, X_train, X_test, s)
    n = length(X_train)
    s = min(s, n)
    # initialization
    indices = zeros(typeof(s), s)
    cond_cov = vec(kernelmatrix(kernel, X_train, X_test))
    cond_var = kernelmatrix_diag(kernel, X_train)
    L = Array{eltype(cond_var)}(undef, n + 1, s)

    for i in 1:s
        # pick best entry
        k, best = -1, 0.0
        for j in 1:n
            score = cond_cov[j]^2 / cond_var[j]
            if score > best
                k, best = j, score
            end
        end
        # didn't select anything, abort early
        k == -1 && return indices[begin:(i - 1)]
        indices[i] = k
        covariance!(kernel, X_train, X_test, L, i, k)
        cholupdate!(L, i, k, cond_var, cond_cov)
    end
    return indices
end

"""
    select_mult(kernel::Kernel, X_train, X_test, s)

Select the `s` points from `X_train` most informative to `X_test` greedily.

This function assumes each point of `X_test` can be conditioned by `X_train`.

See also [`select`](@ref), [`select_single`](@ref).
"""
function select_mult(kernel::Kernel, X_train, X_test, s)
    n = length(X_train)
    m = length(X_test)
    s = min(s, n)
    # initialization
    indices = zeros(typeof(s), s)
    cond_var = kernelmatrix_diag(kernel, X_train)
    cond_var_pr = copy(cond_var)
    L = Array{eltype(cond_var)}(undef, n, s)
    L_pr = Array{eltype(cond_var)}(undef, n + m, s + m)
    # pre-condition on the m prediction points
    for i in 1:m
        covariance!(kernel, X_train, X_test, L_pr, i, n + i)
        cholupdate!(L_pr, i, n + i, cond_var_pr)
    end

    for i in 1:s
        # pick best entry
        k, best = -1, 0.0
        for j in 1:n
            score = cond_var[j] / cond_var_pr[j]
            if cond_var[j] > 0 && cond_var_pr[j] > 0 && score > best
                k, best = j, score
            end
        end
        # didn't select anything, abort early
        k == -1 && return indices[begin:(i - 1)]
        indices[i] = k
        covariance!(kernel, X_train, X_test, L, i, k)
        @views @. L_pr[begin:n, i + m] = L[:, i]
        cholupdate!(L, i, k, cond_var)
        cholupdate!(L_pr, i + m, k, cond_var_pr; rows=n)
    end
    return indices
end

# non-adjacent selection

"""
    choldowndate!(L, u, order; start=1, finish=size(L, 2))

Computes the rank-one downdate of `L` by `u`, modifying `L` in-place.

Given a Cholesky factor ``L``, a rank-one downdate by a vector ``u`` computes
```math
L \\to \\mathsf{chol}(L L^{\\top} - u u^{\\top}).
```
This function only computes the columns from `start` to `finish`.

See also: [`cholinsert!`](@ref).
"""
function choldowndate!(L, u, order; start=1, finish=size(L, 2))
    for i in start:finish
        k = order[i]
        c1, c2 = L[k, i], u[k]
        dp = sqrt((c1 + c2) * (c1 - c2))  # sqrt(c1^2 - c2^2)
        c1, c2 = c1 / dp, c2 / dp
        # @. @views L[:, i] = c1*L[:, i] - c2*u
        @views axpby!(-c2, u, c1, L[:, i])
        # @. @views u = (1/c1)*u - (c2/c1)*L[:, i]
        @views axpby!(-c2 / c1, L[:, i], 1.0 / c1, u)
    end
    return nothing
end

"""
    cholinsert!(L, index, k, order, i)

Insert the `k`-th point at the `index` column in the Cholesky factor `L`.

See also: [`choldowndate!`](@ref).
"""
function cholinsert!(L, index, k, order, i)
    # use last column as temporary working space
    @views mul!(
        L[:, end], L[:, begin:(index - 1)], L[k, begin:(index - 1)], -1.0, 1.0
    )
    @views @. L[:, end] /= sqrt(L[k, end])
    # move columns over to make space at index
    for col in (i + 1):-1:(index + 1)
        @views @. L[:, col] = L[:, col - 1]
    end
    # copy conditional covariance from temporary space
    @views @. L[:, index] = L[:, end]
    # update downstream Cholesky factor by rank-one downdate
    @views choldowndate!(L, L[:, end], order; start=index + 1, finish=i + 1)
    return nothing
end

"""
    insertindex(k, order, locations, i; start=1)

Finds the index to insert index `k` into `order`.
"""
function insertindex(k, order, locations, i; start=1)
    index = 0
    for outer index in start:i
        # bigger than current value, insertion spot
        if locations[k] >= locations[order[index]]
            return index
        end
    end
    return index + 1
end

"""
    selectpoint!(kernel::Kernel, points, L, k, var, order, locations, i)

Add the `k`-th point to the Cholesky factor `L`.

This function modifies `L` and `var` in-place.
"""
function selectpoint!(kernel::Kernel, points, L, k, var, order, locations, i)
    index = insertindex(k, order, locations, i)
    # shift values over to make room for k at index
    for col in (i + 1):-1:(index + 1)
        order[col] = order[col - 1]
    end
    order[index] = k
    # insert covariance with k-th point into Cholesky factor
    # use last column as temporary working space
    @views kernelmatrix!(L[:, end:end], kernel, points, points[k:k])
    cholinsert!(L, index, k, order, i)
    # mark as selected
    k <= length(var) && (var[k] = -1.0)
    return nothing
end

"""
    scoresupdate!(scores, L, var, pos, vars, order, locations, i)

Update `scores` in-place for all candidates.
"""
function scoresupdate!(scores, L, var, pos, vars, order, locations, i)
    n = length(var)
    for j in 1:n
        pos[j] = insertindex(j, order, locations, i; start=pos[j])
    end

    @. scores = 1.0
    @. vars = var
    p = 1
    for col in 1:i
        k = order[col]
        # add log conditional variance of prediction point
        if k > n
            while p < n && pos[p + 1] <= col
                p += 1
            end
            @views @. scores[begin:p] *= (
                1.0 - L[begin:p, col]^2 / vars[begin:p]
            )
        end
        @views @. vars -= L[begin:(begin + n - 1), col]^2
    end

    best_k, best_score = -1, 0.0
    # pick best entry
    for j in 1:n
        # selected already, don't consider as candidate
        if var[j] <= 0.0
            continue
        end

        scores[j] = 1.0 / scores[j]
        if scores[j] > best_score
            best_k, best_score = j, scores[j]
        end
    end
    return best_k
end

"""
    select_nonadj(kernel::Kernel, X, train, test, s)

Select the `s` points from `X_train` most informative to `X_test` greedily.

See also [`select(kernel::Kernel, X, train, test, s)`](@ref),
[`select_budget`](@ref).
"""
function select_nonadj(kernel::Kernel, X, train, test, s)
    n = length(train)
    m = length(test)
    s = min(s, n)
    trainp = sortperm(train; rev=true)
    locations = [train[trainp]; sort(test; rev=true)]
    points = X[locations]
    # initialization
    indices = zeros(typeof(s), s)
    order = zeros(typeof(s), s + m)
    @views var = kernelmatrix_diag(kernel, points[begin:n])
    scores = Array{eltype(var)}(undef, n)
    pos = ones(typeof(s), n)
    vars = copy(var)
    L = Array{eltype(var)}(undef, n + m, s + m + 1)
    # pre-condition on the m prediction points
    for i in 1:m
        selectpoint!(kernel, points, L, n + i, var, order, locations, i - 1)
    end

    for i in 1:s
        # pick best entry
        k = scoresupdate!(
            scores, L, var, pos, vars, order, locations, m + i - 1
        )
        # didn't select anything, abort early
        k == -1 && return indices[begin:(i - 1)]
        indices[i] = trainp[k]
        # update Cholesky factor
        selectpoint!(kernel, points, L, k, var, order, locations, m + i - 1)
    end
    return indices
end

"""
    select_budget(kernel::Kernel, X, train, test, s)

Select the `s` points from `X_train` most informative to `X_test` greedily.

See also [`select(kernel::Kernel, X, train, test, s)`](@ref),
[`select_nonadj`](@ref).
"""
function select_budget(kernel::Kernel, X, train, test, s)
    n = length(train)
    m = length(test)
    s = min(s, n)
    trainp = sortperm(train; rev=true)
    locations = [train[trainp]; sort(test; rev=true)]
    points = X[locations]
    # allow each selected point to condition all the prediction points
    budget = m * s
    max_sel = min(budget, n)
    # initialization
    indices = zeros(typeof(s), max_sel)
    order = zeros(typeof(s), max_sel + m)
    @views var = kernelmatrix_diag(kernel, points[begin:n])
    scores = Array{eltype(var)}(undef, n)
    pos = ones(typeof(s), n)
    vars = copy(var)
    L = Array{eltype(var)}(undef, n + m, max_sel + m + 1)
    num_cond = zeros(typeof(s), n)
    for i in 1:n
        index = locations[i]
        count = 0
        for j in 1:m
            count += test[j] < index
        end
        num_cond[i] = count
    end
    # pre-condition on the m prediction points
    for i in 1:m
        selectpoint!(kernel, points, L, n + i, var, order, locations, i - 1)
    end

    for i in 1:budget
        # pick best entry
        k, best = -1, 0.0
        scoresupdate!(scores, L, var, pos, vars, order, locations, m + i - 1)
        for j in 1:n
            # selected already, don't consider as candidate
            if var[j] <= 0.0 || num_cond[j] == 0
                continue
            end
            score = log2(scores[j]) / num_cond[j]
            if score > best
                k, best = j, score
            end
        end
        # didn't select anything, abort early
        k == -1 && return indices[begin:(i - 1)]
        indices[i] = trainp[k]
        # subtract number conditioned from budget
        budget -= num_cond[k]
        budget <= 0 && return indices[begin:(i - (budget != 0))]
        # update Cholesky factor
        selectpoint!(kernel, points, L, k, var, order, locations, m + i - 1)
    end
    return indices
end

"""
    select(kernel::Kernel, X_train, X_test, s)

Select the `s` points from `X_train` most informative to `X_test` greedily.

This function dispatches to [`select_single`](@ref) or [`select_mult`](@ref)
depending on whether `length(X_test) == 1`. This function
assumes each point of `X_test` can be conditioned by `X_train`.

See also [`select(kernel::Kernel, X, train, test, s)`](@ref).
"""
function select(kernel::Kernel, X_train, X_test, s)
    return if length(X_test) == 1
        select_single(kernel, X_train, X_test, s)
    else
        select_mult(kernel, X_train, X_test, s)
    end
end

"""
    select(kernel::Kernel, X, train, test, s; budget=true)

Select the `s` points from `X[train]` most informative to `X[test]` greedily.

This function dispatches to [`select_single`](@ref) if
`length(X_test) == 1`, otherwise [`select_budget`](@ref) or
[`select_nonadj`](@ref) depending on whether `budget` is true.

See also [`select(kernel::Kernel, X_train, X_test, s)`](@ref).
"""
function select(kernel::Kernel, X, train, test, s; budget=true)
    return if length(test) == 1
        @views select_single(kernel, X[train], X[test], s)
    elseif budget
        select_budget(kernel, X, train, test, s)
    else
        select_nonadj(kernel, X, train, test, s)
    end
end

end
