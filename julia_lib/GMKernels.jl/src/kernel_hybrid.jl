export HybridKernel

"""
    HybridDSKernel

Define kernel for hybrid dataset.

# Arguments
- `θ`: kernel parameters
- `fselect`: selection function
"""
struct HybridKernel <: KernelFunctions.Kernel
    # #kernel parameters
    θ::Vector{Real}
    # kernel
    kernel::KernelFunctions.Kernel    
    #selection function
    fselect::Function
end

"""Constructor for Hybrid Dataset Kernel"""
function(κ::HybridKernel)(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real

    # # #hybrid kernel scale
    # # θ = [κ.θ[κ.fselect(x[1], y[1])]]

    # println("selection: ", κ.fselect(x[1], y[1]))
    # println("scale: ", κ.θ[κ.fselect(x[1], y[1])])
    # println("kernel: ", κ.kernel(x[2:end], y[2:end]))
    # println("total: ", κ.θ[κ.fselect(x[1], y[1])] * κ.kernel(x[2:end], y[2:end]))

    #evaluate kernel
    return (κ.θ[κ.fselect(x[1], y[1])] * κ.kernel(x[2:end], y[2:end]))
end


"""kernelmatrix overload for HybridKernel and matrix X and Y"""
function KernelFunctions.kernelmatrix(κ::HybridKernel, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    n = size(X, 2)
    m = size(Y, 2)
    K = Matrix{T}(undef, n, m)
    @inbounds for i in 1:n, j in 1:m
        K[i, j] = @views κ(X[:, i], Y[:, j])
    end
    return K
end

"""kernelmatrix overload for HybridKernel and matrix X"""
function KernelFunctions.kernelmatrix(κ::HybridKernel, X::AbstractVector{T}) where T <: Real
    n = size(X, 2)
    K = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        K[i, i] = @views κ(X[i], X[i])
        @inbounds for  j in (i+1):n
            K[i, j] = @views κ(X[:, i], Y[:, j])
            K[j, i] = K[i, j]
        end
    end
    return K
end