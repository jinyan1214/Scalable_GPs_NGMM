# Kernel Functions for GMM Prediction
using KernelFunctions
using FastGaussQuadrature: gausslobatto
include("kernel_functions_path.jl")

# Aleatory Kernels
# ---   ---   ---   ---   ---
"""
   Within event kernel function
"""
function kernel_aleat_wevent(ϕ₀²::T, dims::AbstractArray{Int}=[1]) where T <: Real

    #define kernel
    κₐ = ϕ₀² * KernelFunctions.WhiteKernel()

    return κₐ
end

"""
   Between event kernel function
"""
function kernel_aleat_bevent(τ₀²::T, dims::AbstractArray{Int}=[1]) where T <: Real

    #define kernel
    κᵦ = τ₀² * KernelFunctions.WhiteKernel() ∘ KernelFunctions.SelectTransform(dims)

    return κᵦ
end

# Non-ergodic Kernels
# ---   ---   ---   ---   ---
# individual non-ergodic kernels
# - - - - - - - - - - - - - - 
"""
    Source non-ergodic kernel function
"""
function kernel_nerg_source(ωₗ²::T, λₗ::T, 
                            dims::AbstractArray{Int}=[1,2]; 
                            κ=KernelFunctions.Matern32Kernel) where T <: Real

    #define kernel
    κₗ = ωₗ² * κ(; metric=KernelFunctions.Euclidean()) ∘ KernelFunctions.ScaleTransform(λₗ) ∘ KernelFunctions.SelectTransform(dims)

    return κₗ
end

"""
    Path non-ergodic kernel function
"""
function kernel_nerg_path(ωₚ²::T, λₚ::T, 
                          dims::AbstractArray{Int}=[1,2,3,4]; 
                          n_integ_pt::Int=5, flag_normalize::Bool=true, 
                          κ=KernelFunctions.Matern32Kernel) where T<:Real
    #integration points
    t, s, w = gaussquad2d(n_integ_pt) #compute integration weights

    # #define kernel
    κₚ = ωₚ² * PathKernel(n_integ_pt, 
                          κ( ;metric=KernelFunctions.Euclidean()), 
                          flag_normalize) ∘ KernelFunctions.ScaleTransform(λₚ) ∘ KernelFunctions.SelectTransform(dims)
    # κₚ = PathKernel(n_integ_pt, 
    #                 d -> ωₚ² * exp(-d*λₚ), 
    #                 flag_normalize, t, s, w) ∘ SelectTransform(dims)
    
    return κₚ
end

"""
    Site non-ergodic kernel function
"""
function kernel_nerg_site(ωₛ²::T, λₛ::T, 
                          dims::AbstractArray{Int}=[1,2]; 
                          κ=KernelFunctions.Matern32Kernel) where T <: Real

    #define kernel
    κₛ = ωₛ² * κ(; metric=KernelFunctions.Euclidean()) ∘ KernelFunctions.ScaleTransform(λₛ) ∘ KernelFunctions.SelectTransform(dims)

    return κₛ
end

# composite non-ergodic kernels
# - - - - - - - - - - - - - - 
"""
    Source and Site non-ergodic kernel function
"""
function kernel_nerg_sourcesite(ωₗ²::T, λₗ::T, ωₛ²::T, λₛ::T, 
                                dims_event::AbstractArray{Int}=[1,2], 
                                dims_site::AbstractArray{Int}=[3,4]; 
                                κ=KernelFunctions.Matern32Kernel) where T <: Real

    #define kernels
    κₗ = kernel_nerg_source(ωₗ², λₗ, dims_event; κ=κ)
    κₛ = kernel_nerg_site(ωₛ², λₛ,   dims_site;  κ=κ)

    return κₗ + κₛ
end

"""
    Path and site non-ergodic kernel function
"""
function kernel_nerg_sourcesite(ωₚ²::T, λₚ::T, ωₛ²::T, λₛ::T, 
                                dims_event::AbstractArray{Int}=[1,2], 
                                dims_site::AbstractArray{Int}=[3,4];
                                κ=KernelFunctions.Matern32Kernel) where T <: Real

    #define kernels
    κₚ = kernel_nerg_path(ωₚ², λₚ, vcat(dims_event,dims_site); κ=κ)
    κₛ = kernel_nerg_site(ωₛ², λₛ, dims_site;                  κ=κ)

    return κₚ + κₛ
end


"""
    Source, path and site non-ergodic kernel function
"""
function kernel_nerg_sourcesite(ωₗ²::T, λₗ::T, ωₚ²::T, λₚ::T, ωₛ²::T, λₛ::T, 
                                dims_event=[1,2], 
                                dims_site=[3,4];
                                κ=KernelFunctions.Matern32Kernel) where T <: Real

    #define kernels
    κₗ = kernel_nerg_source(ωₗ², λₗ, dims_event;                 κ=κ)
    κₚ = kernel_nerg_path(ωₚ², λₚ,   vcat(dims_event,dims_site); κ=κ)
    κₛ = kernel_nerg_site(ωₛ², λₛ,   dims_site;                  κ=κ)

    return κₗ + κₚ + κₛ
end

# Hybrid Kernels
# ---   ---   ---   ---   ---
"""
   Hybrid simulation empirical kernel function
"""
function kernel_hybrid_sim(dimₕ::AbstractArray{Int}=[1],
                           dimₙ::AbstractArray{Int}=[1,2,3,4];
                           κₗ, κₚ, κₛ)

    #define kernels
    κₗₕ = (κₗ ∘ KernelFunctions.SelectTransform(dimₙ)) * (KernelFunctions.LinearKernel() ∘ KernelFunctions.SelectTransform(dimₕ))
    κₕₚ =  κₚ ∘ KernelFunctions.SelectTransform(dimₙ)
    κₕₛ = (κₛ ∘ KernelFunctions.SelectTransform(dimₙ)) * (KernelFunctions.WhiteKernel() ∘ KernelFunctions.SelectTransform(dimₕ))

    return κₗₕ + κₕₚ + κₕₛ
end