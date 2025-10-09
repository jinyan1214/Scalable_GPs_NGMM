module SparseKoLesky

include("Measurements.jl")
include("CovarianceFunctions.jl")
include("SuperNodes.jl")
include("MutableHeap.jl")
include("MaximinNN.jl")
include("Factors.jl")
include("KLMinimization.jl")

include("utils.jl")
using .Utils: Utils

include("gp.jl")
include("ichol.jl")
include("metrics.jl")
include("ordering.jl")
include("select.jl")

using .GP
using .Metrics
using .Ordering
using .Select

end
