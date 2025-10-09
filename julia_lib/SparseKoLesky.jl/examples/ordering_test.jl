using Random: Random

using PyPrint: pprint, @pprint
using KernelFunctions: KernelFunctions as Kernels

using SparseKoLesky: SparseKoLesky as KL

Random.seed!(1)

d = 2
n = 100

points = rand(d, n)
kernel = Kernels.Matern12Kernel()

kernelmetric = KL.KernelDist(kernel)
point = points[:, begin]
@assert kernelmetric(point, point) == 0.0 "distance to self not zero"

order, _ = KL.maximin_ordering(points)
order_kernel, _ = KL.maximin_ordering(kernel, points)
@assert order == order_kernel "order different"
