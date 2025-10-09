using Random: Random
using Profile: Profile, @profile

using PyPrint: pprint, @pprint
using KernelFunctions: KernelFunctions as Kernels, ColVecs

using SparseKoLesky: SparseKoLesky as KL

Random.seed!(1)

d = 2

n = 5000
m = 500
s = 500

points = ColVecs(rand(d, n))
target = ColVecs(rand(d, 1))
targets = ColVecs(rand(d, m))
kernel = Kernels.Matern12Kernel()

