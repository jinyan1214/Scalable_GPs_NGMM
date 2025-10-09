# Path Kernel Function
#------------------------------

export PathKernel

"""
    PathKernel

Derived kernel based on integration.

# Arguments
- `n`: number of points to integrate
- `kernel`: base kernel
- `normalized`: flag for normalized variant w.r.t. path length
- `t`: gauss integration points (1st axis)
- `s`: gauss integration points (2nd axis)
- `w`: gauss weights 
"""
struct PathKernel <: KernelFunctions.Kernel
    #integration points
    n::Int         
    # kernel
    kernel::KernelFunctions.Kernel
    normalized::Bool  
    #2d integration parameters
    t1D::AbstractVector{<:Real}
    w1D::AbstractVector{<:Real}
    #2d integration parameters
    t2D::AbstractVector{<:Real}
    s2D::AbstractVector{<:Real}
    w2D::AbstractVector{<:Real}
    #2d difference integration parameters
    td2D::AbstractVector{<:Real}
    wd2D::AbstractVector{<:Real}

    #constructor
    function PathKernel(n::Int, 
                        kernel::KernelFunctions.Kernel, 
                        normalized::Bool)

        #compute 1D quadrature points
        t1D, w1D = gaussquad1d(n)
        #compute 2D quadrature points
        t2D, s2D, w2D = gaussquad2d(n)
        #compute 2D difference quadrature points
        td2D, wd2D = gaussdiff2d(abs.(t2D - s2D), w2D)
        
        new(n, kernel, normalized, t1D, w1D, t2D, s2D, w2D, td2D, wd2D)
    end
end

"""Constructor for Path kernel"""
function (κ::PathKernel)(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real

    #parse coordinates, assuming x = [x1; x2] and y = [y1; y2]
    n = div(length(x), 2) #dimension
    x1 = @view x[begin:n]
    x2 = @view x[(n + 1):end]
    y1 = @view y[begin:n]
    y2 = @view y[(n + 1):end]
    #path lengths
    x_len = euclidean_len(x1, x2)
    y_len = euclidean_len(y1, y2)

    #distance between points on paths
    fun_dist = getdist(x1, x2, y1, y2)

    #compute distance
    d = fun_dist.(κ.t2D, κ.s2D)

    #integrate path kernel
    # K = compute_kernel_integral(κ.kernel, d, κ.w2D) 
    K = compute_kernel_integral_ts(κ.kernel, fun_dist, κ.t2D, κ.s2D, κ.w2D) 

    #normalization
    if κ.normalized
        #normalize kernel
        K /= sqrt( compute_kernel_integral_t(κ.kernel, x_len, κ.td2D, κ.wd2D) * 
                   compute_kernel_integral_t(κ.kernel, y_len, κ.td2D, κ.wd2D) )
    else
        K *= x_len * y_len
    end

    return K
end

"""kernelmatrix overload for PathKernel and matrix X and Y"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    n = size(X, 2)
    m = size(Y, 2)
    K = Matrix{T}(undef, n, m)
    @inbounds for i in 1:n, j in 1:m
        K[i, j] = @views κ(X[:, i], Y[:, j])  #compute kernel for each pair (X[:, i], Y[:, j])
    end
    return K
end

"""kernelmatrix overload for PathKernel and matrix X"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::AbstractMatrix{T}) where T <: Real
    n = size(X, 2)
    K = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        K[i, i] = @views κ(X[:, i], X[:, i])
        @inbounds for  j in (i+1):n
            K[i, j] = @views κ(X[:, i], X[:, j])  #compute kernel for each pair (X[:, i], Y[:, j])
            K[j, i] = K[i, j]
        end
    end
    return K
end

"""kernelmatrix overload for PathKernel on RowVecs X and Y"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::RowVecs, Y::RowVecs)
    #convert RowVecs X and Y to a matrices
    X_matrix = hcat(X...)
    Y_matrix = hcat(Y...)
    return KernelFunctions.kernelmatrix(κ, X_matrix, Y_matrix)
end

"""kernelmatrix overload for PathKernel on RowVecs X"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::RowVecs)
    #convert RowVecs to a matrix
    X_matrix = hcat(X...)
    #compute the kernel matrix
    return KernelFunctions.kernelmatrix(κ, X_matrix)
end


