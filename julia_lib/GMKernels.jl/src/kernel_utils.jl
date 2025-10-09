# Auxilary Functions for Kernel Evaluation
#------------------------------------------------
using KernelFunctions

"""
    kappa(kernel, d)    

Compute square exponential kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.SqExponentialKernel, d::T) where T<:Real
    return exp(-(d^2))
end

"""
    kappa(kernel, d)

Compute exponential kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.ExponentialKernel, d::T) where T<:Real
    return exp(-d )
end

"""
    kappa(kernel, d)

Compute Matern32 kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.Matern32Kernel, d::T) where T<:Real
    sqrt3 = sqrt(3.)
    return (1. + sqrt3*d) * exp(-sqrt3*d)
end

#fallback if needed (generic)
"""
    kappa(kernel, d)

Compute generic kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.Kernel, d::T) where T<:Real
    
    return KernelFunctions.kappa(kernel, d)
end

"""
    compute_kernel_integral(kernel, d, w)

Compute integral kernel parametrized by distance `d` and weights `w`.
"""
@inline function compute_kernel_integral_d(kernel::KernelFunctions.Kernel, 
                                           d::AbstractVector{T}, 
                                           w::AbstractVector{T}) where T <: Real

    acc = zero(T)
    @inbounds @simd for i in eachindex(d)
        acc += w[i] * kappa(kernel, d[i])
    end
    return acc
end

"""
    compute_kernel_integral(kernel, len, t, w)

Compute integral kernel parametrized by lenght `len`, integration points `t` and weights `w`.
"""
@inline function compute_kernel_integral_t(kernel::KernelFunctions.Kernel, 
                                           len::T,
                                           t::AbstractVector{T}, 
                                           w::AbstractVector{T}) where T <: Real

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kappa(kernel, len * t[i])
    end
    return acc
end

"""
    compute_kernel_integral(kernel, len, t, w)

Compute integral kernel parametrized by lenght `len`, integration points `t` and weights `w`.
"""
@inline function compute_kernel_integral_t(kernel::Function, 
                                           len::T,
                                           t::AbstractVector{T}, 
                                           w::AbstractVector{T}) where T <: Real

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kernel(len * t[i])
    end
    return acc
end

"""
    compute_kernel_integral(kernel, fun_dist, t, s, w)

Compute integral `kernel` function parametrized by distance function `fun_dist(t,s)` 
    integration points `t` and `s`, and weights `w`.
"""
@inline function compute_kernel_integral_ts(kernel::KernelFunctions.Kernel, 
                                            fun_dist::F,
                                            t::AbstractVector{T}, 
                                            s::AbstractVector{T}, 
                                            w::AbstractVector{T}) where {T <: Real, F <: Function}

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kappa(kernel, fun_dist(t[i], s[i]))
    end
    return acc
end

"""
    compute_kernel_integral(kernel, fun_dist, t, s, w)

Compute integral `kernel` function parametrized by distance function `fun_dist(t,s)` 
    integration points `t` and `s`, and weights `w`.
"""
@inline function compute_kernel_integral_ts(kernel::Function, 
                                            fun_dist::F,
                                            t::AbstractVector{T}, 
                                            s::AbstractVector{T}, 
                                            w::AbstractVector{T}) where {T <: Real, F <: Function}

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kernel(fun_dist(t[i], s[i]))
    end
    return acc
end
