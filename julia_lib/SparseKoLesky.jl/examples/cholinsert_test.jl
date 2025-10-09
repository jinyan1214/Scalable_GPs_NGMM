using LinearAlgebra: cholesky, norm
using Random: Random

using PyPrint: pprint, @pprint
using KernelFunctions: KernelFunctions as Kernels

using SparseKoLesky: SparseKoLesky as KL

Random.seed!(1)

d = 2
n = 1000

points = Kernels.ColVecs(rand(d, n))
kernel = Kernels.Matern12Kernel()
theta = Kernels.kernelmatrix(kernel, points)

L = cholesky(theta)
order = 1:n
u = theta[:, end] / 2

Lp = cholesky(theta - u * u')
@time cholesky(theta - u * u')
La = Matrix(L.L)
Lacopy = copy(La)
KL.Select.choldowndate!(La, u, order)
@time KL.Select.choldowndate!(Lacopy, u, order)

@pprint norm(Lp.L .- L.L)
@pprint norm(Lp.L .- La)
@assert norm(Lp.L .- La) < 1e-9 "error too large"

point = 500
index = 200
s = 300

theta = Kernels.kernelmatrix(kernel, points)
L = cholesky(theta).L[:, begin:s]

order_new = collect(1:n)
value = popat!(order_new, point)
insert!(order_new, index, value)
theta_new = Kernels.kernelmatrix(kernel, points[order_new])
L_new = cholesky(theta_new).L[invperm(order_new), begin:(s + 1)]
@time cholesky(theta_new)

L_space = zeros(n, (s + 2))
L_space[:, begin:s] .= L
L_space[:, end] .= theta[:, point]
L_space_copy = copy(L_space)
KL.Select.cholinsert!(L_space, index, point, order_new, s + 1)
@time KL.Select.cholinsert!(L_space_copy, index, point, order_new, s + 1)

@pprint norm(L_space[:, begin:(s + 1)] .- L_new)
@assert norm(L_space[:, begin:(s + 1)] .- L_new) < 1e-9 "error too large"
