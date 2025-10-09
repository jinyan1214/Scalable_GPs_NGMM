using Random: Random
using Profile: Profile, @profile

using PyPrint: pprint, @pprint

using SparseKoLesky: SparseKoLesky as KL

Random.seed!(1)

n = 10^9

f(x) = (x >> 9) << 9 == x
g(x) = mod(x, 512) == 0

ans1 = f.(1:n)
@time f.(1:n)

ans2 = g.(1:n)
@time g.(1:n)

@assert ans1 == ans2 "not the same"

x = rand(n)

logproduct(x) = sum(log2.(x))

ans1 = logproduct(x)
@time logproduct(x)

ans2 = KL.Utils.log2prod(x)
@time KL.Utils.log2prod(x)

@assert ans1 ≈ ans2 "not the same"
