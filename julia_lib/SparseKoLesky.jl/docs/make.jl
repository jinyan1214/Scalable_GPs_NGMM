using Documenter

using SparseKoLesky

makedocs(;
    sitename="SparseKoLesky.jl",
    modules=[
        SparseKoLesky,
        SparseKoLesky.GP,
        SparseKoLesky.Metrics,
        SparseKoLesky.Ordering,
        SparseKoLesky.Select,
        SparseKoLesky.Utils,
    ],
    #! format: off
    pages=[
        "index.md",
        "Guide" => [
            "man/start.md",
            "man/kernels.md",
        ],
        "SparseKoLesky" => [
            "lib/gp.md",
            "lib/metrics.md",
            "lib/ordering.md",
            "lib/select.md",
            "lib/utils.md",
        ],
        "contributing.md",
        "resources.md",
    ]
    #! format: on
)
