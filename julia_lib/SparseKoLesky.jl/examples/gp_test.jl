using Random: Random
using LinearAlgebra: diag
using Statistics: mean

using PyPrint: pprint, @pprint
using KernelFunctions: KernelFunctions as Kernels, ColVecs

using SparseKoLesky: SparseKoLesky as KL

rng = Random.seed!(1)

d = 3
n = 1024
m = 128
s = 64
draws = 10000

x_train = ColVecs(rand(d, n))
x_test = ColVecs(rand(d, m))
kernel = Kernels.Matern12Kernel()

x = ColVecs([x_train.X x_test.X])
y = KL.sample(rng, kernel, x; draws)
@views y_train = y[begin:n, :]
@views y_test = y[(n + 1):end, :]

mean_pred, var_pred = KL.estimate(kernel, x_train, y_train, x_test)
_, cov_pred = KL.estimate(kernel, x_train, y_train, x_test; full_cov=true)
@assert var_pred ≈ diag(cov_pred) "variance not the same as diagonal"

loss = mean(KL.mse(mean_pred, y_test))
@assert isapprox(loss, mean(var_pred); rtol=1e-3) "mse not posterior var"

alpha = 0.9
cover = KL.coverage(y_test, mean_pred, var_pred; alpha)
exact = fill(alpha, length(cover))
@assert isapprox(cover, exact; rtol=1e-2) "coverage not close"

indices = KL.select(kernel, x_train, x_test, s)
mean_pred, _ = KL.estimate(kernel, x_train, y_train, x_test; indices)

println(" direct rmse: $loss")
println("sampled rmse: $(mean(KL.mse(mean_pred, y_test)))")

