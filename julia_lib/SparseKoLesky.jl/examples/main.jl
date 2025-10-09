using Random: Random
using LinearAlgebra: norm

using PyPrint: pprint, @pprint

using SparseKoLesky: SparseKoLesky as KL

# set random seed for reproducibility
Random.seed!(1)

n = 10   # number of points
ρ = 3.0  # factor density
k = 1    # nearest neighbors

points = rand(2, n)
measurements = KL.point_measurements(points)

kernel = KL.MaternCovariance5_2(0.5)

implicit_factor = KL.ImplicitKLFactorization(kernel, measurements, ρ, k)
@time explicit_factor = KL.ExplicitKLFactorization(implicit_factor)

# comparing to true result

KM = zeros(n, n)
kernel(KM, measurements, measurements)

@pprint KM
@pprint explicit_factor.U

approx_KM = KL.assemble_covariance(explicit_factor)

@show norm(approx_KM - KM) / norm(KM)
