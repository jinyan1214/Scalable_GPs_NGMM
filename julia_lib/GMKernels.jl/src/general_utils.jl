
using Base.Iterators: product
using FastGaussQuadrature: gausslobatto

export norm2

# General Utility Functions
#-------------------------------------------------
function norm2(x::AbstractArray{T}) where T <: Real
    return sum(x -> x^2, x)
end

"""
    dotsub(x, y, u, v)

Compute the inner product ``\\langle x - y, u - v \\rangle``.
"""
@inline function dotsub(x::AbstractArray{T}, y::AbstractArray{T}, 
                        u::AbstractArray{T}, v::AbstractArray{T}) where T <: Real
    
    dot(x,u) - dot(y,u) - dot(x,v) + dot(y,v)
end

"""
    psqrt(x)

Compute `sqrt(relu(x))`.
"""
@inline psqrt(x::T) where T <: Real = sqrt(abs(x)) 

"""
    euclidean_dist(x1, x2)  

Compute the Euclidean length of vector with origin x1 and end x2.
"""
@inline function euclidean_len(x1::AbstractVector{T}, x2::AbstractVector{T}) where T<:Real
    acc = zero(T)
    @inbounds @simd for i in eachindex(x1, x2)
        Δ = x2[i] - x1[i]
        acc += Δ*Δ
    end
    return sqrt(acc)
end

"""
    getdist(x1, x2, y1, y2)

Return a function of `t` and `s` to compute the quantity below.
```julia
norm(x1 + t * (x2 - x1) - (y1 + s * (y2 - y1)))
```
"""
@inline function getdist(x1::AbstractArray{T}, x2::AbstractArray{T}, 
                         y1::AbstractArray{T}, y2::AbstractArray{T}) where T <: Real

    t2 = dotsub(x2, x1, x2, x1)
    s2 = dotsub(y2, y1, y2, y1)
    t1 = 2.0 * dotsub(x1, y1, x2, x1)
    s1 = 2.0 * dotsub(y1, x1, y2, y1)
    ts = 2.0 * dotsub(x2, x1, y2, y1)
    c = dotsub(x1, y1, x1, y1)

    let t2 = t2, s2 = s2, t1 = t1, s1 = s1, ts = ts, c = c
        """
            dist(t, s)

        Compute a distance quantity given `t` and `s`.
        """
        function dist(t::Real, s::Real)
            return psqrt(
                t^2 * t2 + t * t1 + s^2 * s2 + s * s1 - t * s * ts + c
            )
        end
    end
end

"""
    dist(x1, x2, y1, y2, t, s)

Compute the distance between two paths given `t` and `s`.
"""
function dist(x1::AbstractVector{T}, x2::AbstractVector{T}, 
              y1::AbstractVector{T}, y2::AbstractVector{T},
              t::AbstractVector{T}, s::AbstractVector{T}) where T <: Real 

    x = x1' .+ t .* (x2 - x1)' 
    y = y1' .+ s .* (y2 - y1)'

    return sqrt.(sum((x-y).^2; dims=2))
end

"""
    trapezoid1d(f, a, b, n)

Integrate the 1-d function `f` over [`a`, `b`] with `n` points.
"""
function trapezoid1d(f, a::T, b::T, n::Int)::T where T <:Real
    x = range(a, b; length=n)[(begin + 1):(end - 1)]
    return ((f(a) + f(b)) / 2.0 + sum(f, x; init=0.0)) / (n - 1)
end

"""
    trapezoid2d(f, a1, b1, a2, b2, n)

Integrate 2-d function `f` over [`a1`, `b1`] x [`a2`, `b2`] by ``n^2`` points.
"""
function trapezoid2d(f, a1::T, b1::T, a2::T, b2::T, n::Int)::T where T <: Real
    #x1 = range(a1, b1; length=n)[(begin + 1):(end - 1)]
    #x2 = range(a2, b2; length=n)[(begin + 1):(end - 1)]
    x1 = [a1 + i*(b1-a1)/(n-1) for i in 1:n-2]
    x2 = [a1 + i*(b1-a1)/(n-1) for i in 1:n-2]
    # corners
    value = (f(a1, a2) + f(a1, b2) + f(b1, a2) + f(b1, b2)) / 4.0
    # edges
    for x in x1
        value += (f(x, a2) + f(x, b2)) / 2.0
    end
    for y in x2
        value += (f(a1, y) + f(b1, y)) / 2.0
    end
    # interior
    for x in x1, y in x2
        value += f(x, y)
    end
    return value / (n - 1)^2
end

"""
    gaussquad1d(n_int)

Gauss quadrature 1D integration in [0,1]
"""
function gaussquad1d(n_int::Int)

    #compute quadrature points
    t, w = gausslobatto(n_int)

    #adjust for 0.0 - 1.0 domain
    t  = 0.5 .+ 0.5*t
    w = 0.5*w
    w = w / sum(w)

    return t, w
end

"""
    gaussquad2d(n_int)

Gauss quadrature 2D integration in [[0,1] x [0,1]]
"""
function gaussquad2d(n_int::Int)

    #compute 1D quadrature points
    t1D, w1D = gaussquad1d(n_int)

    #compute 2D quadrature points
    w2D  = prod.([Iterators.product(w1D,w1D)...])
    w2D  = w2D / sum(w2D)
    ts2D = [Iterators.product(t1D,t1D)...]
    t2D  = [ts[1] for ts in ts2D]
    s2D  = [ts[2] for ts in ts2D]

    return t2D, s2D, w2D
end

"""
    gaussdiff2d(diffst, w; digits)

Compute the sum of weights for gauss quadrature location `diffst` difference with `digits` significant digits.
"""
function gaussdiff2d(diffst::AbstractVector{T}, w::AbstractVector{T}; digits::Int=9) where T <: Real

    #dictionary to store rounded values and their sums
    sums_dict = Dict{T, T}()

    #iterate over the values
    for (dsti, wi) in zip(diffst, w)
        #round xi to digits decimal places
        key = round(dsti, sigdigits=digits)
        #add new weight
        sums_dict[key] = get(sums_dict, key, 0.0) + wi
    end

    #sort keys
    idx = sortperm(collect(keys(sums_dict)))

    #convert dictionary to two arrays
    return collect(keys(sums_dict))[idx], collect(values(sums_dict))[idx]
end

"""
    allcombs_matrix(arr1, arr2)
Compute the Cartesian product of two arrays and return as a matrix.
"""
function allcombs_matrix(arr1, arr2)
    combs = collect(product(arr1, arr2))
    return copy( reshape(hcat([collect(c) for c in combs]...), (2, length(arr1) * length(arr2)))' )
end