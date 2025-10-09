using Profile: Profile, @profile

using PyPrint: pprint, @pprint
using CSV: CSV
using KernelFunctions: KernelFunctions as Kernels, RowVecs, ColVecs

using SparseKoLesky: SparseKoLesky as KL

"""
    readcsv(path; flatten=false)

Read data from `path`. If `flatten` is true, interpret as a vector.
"""
function readcsv(path; flatten=false)
    data = CSV.read("data/$path", CSV.Tables.matrix; header=false)
    return flatten ? vec(data) .+ 1 : data
end

# load pre-generated data
train = RowVecs(readcsv("train_points.csv"))
targets = RowVecs(readcsv("target_points.csv"))
target = RowVecs(targets.X[begin:begin, :])
points = RowVecs([train.X; targets.X])

train_ind = readcsv("train_indices.csv"; flatten=true)
target_ind = readcsv("target_indices.csv"; flatten=true)

select_single = readcsv("select_single.csv"; flatten=true)
select_mult = readcsv("select_mult.csv"; flatten=true)
select_nonadj = readcsv("select_nonadj.csv"; flatten=true)
select_chol = readcsv("select_chol.csv"; flatten=true)

# start comparison
kernel = Kernels.Matern12Kernel()
s = 2^10

single_ind = KL.select(kernel, train, target, s)
Profile.clear_malloc_data()
# python: 1.431
# julia: 1.055521 seconds (19 allocations: 32.266 MiB)
@time KL.select(kernel, train, target, s)

@assert single_ind == select_single "single indices not the same"

n = length(train)
single_ind = KL.select(kernel, points, 1:n, n + 1:n + 1, s)
Profile.clear_malloc_data()
# julia: 1.004957 seconds (17 allocations: 32.141 MiB)
@time KL.select(kernel, points, 1:n, n + 1:n + 1, s)

@assert single_ind == select_single "nonadj single indices not the same"

mult_ind = KL.select(kernel, train, targets, s)
Profile.clear_malloc_data()
# python: 3.329
# julia: 2.615765 seconds (11 allocations: 74.602 MiB, 0.15% gc time)
@time KL.select(kernel, train, targets, s)

@assert mult_ind == select_mult "multiple indices not the same"

s = 840

nonadj_ind = KL.select(kernel, points, train_ind, target_ind, s; budget=false)
Profile.clear_malloc_data()
# python: 11.916
# julia: 5.718952 seconds (25 allocations: 36.728 MiB)
@time KL.select(kernel, points, train_ind, target_ind, s; budget=false)
# @profile KL.select(kernel, points, train_ind, target_ind, s; budget=false)
# Profile.print()
# Profile.print(format=:flat, sortedby=:count)

@assert nonadj_ind == select_nonadj[begin:s] "nonadj indices not the same"

s = 512

chol_ind = KL.select(kernel, points, train_ind, target_ind, s; budget=true)
Profile.clear_malloc_data()
# python: 12.292
# julia: 5.226606 seconds (30 allocations: 144.924 MiB, 0.05% gc time)
@time KL.select(kernel, points, train_ind, target_ind, s; budget=true)

n = length(chol_ind)
@assert chol_ind == select_chol[begin:n] "chol indices not the same"
