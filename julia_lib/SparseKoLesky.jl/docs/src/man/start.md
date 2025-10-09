# Getting started

Julia can be [installed](https://julialang.org/downloads/) on many
common operating systems. The official documentation can be found
[here](https://docs.julialang.org/en/v1/manual/getting-started/).

## Environment setup

The first step is to [create a
package](https://pkgdocs.julialang.org/v1/creating-packages/) which helps
with installing the right dependencies and reproducibility. This is roughly
equivalent to a virtual environment or conda environment in Python.

First, enter the Julia REPL with the `julia`
command. A screen like the below should be shown.

```text
stephenhuan@neko ~> julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.0 (2023-05-07)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```

From here, pressing the `]` key brings one into the [package
management](https://docs.julialang.org/en/v1/stdlib/Pkg/)
interface. The cursor changes to

```text
(@v1.9) pkg>
```

Now use the `generate` command to make a new project
(here, `EarthquakeGPs`) in the current directory.

```text
(@v1.9) pkg> generate EarthquakeGPs
  Generating  project EarthquakeGPs:
    EarthquakeGPs/Project.toml
    EarthquakeGPs/src/EarthquakeGPs.jl
```

Navigate to the newly created `EarthquakeGPs` directory.

From now on, we will need to activate the project with the `activate` command.

```
(@v1.9) pkg> activate .
  Activating project at `~/programs/research/surf/EarthquakeGPs`

(EarthquakeGPs) pkg>
```

The environment indicator `(@v1.9)` changes to
`(EarthquakeGPs)` to indicate that we are in the local project.

When running files outside of the REPL like `julia file.jl`, one should use the
`--project` flag to set the environment. The special value `@.` indicates that
Julia should search for a suitable `Project.toml`. For convenience, one can
[alias](https://tldp.org/LDP/abs/html/aliases.html) `julia` to automatically
include this flag on the `bash` shell with the command

```bash
alias julia='command julia --project="@." "$@"'
```

Consult the documentation for your particular shell, for example, on
the [fish shell](https://fishshell.com/docs/current/cmds/alias.html)
the equivalent command is

```bash
alias --save julia='command julia --project="@."'
```

Now it is no longer necessary to explicitly run the `activate` command.

Packages can be added with the `add` command. Adding
[SparseKoLesky.jl](https://github.com/stephen-huan/SparseKoLesky.jl)
to the project,

```text
(EarthquakeGPs) pkg> add https://github.com/stephen-huan/SparseKoLesky.jl
    Updating git-repo `https://github.com/stephen-huan/SparseKoLesky.jl`
   Resolving package versions...
    Updating `~/programs/research/surf/EarthquakeGPs.jl/Project.toml`
  [169a2823] + SparseKoLesky v0.1.0 `https://github.com/stephen-huan/SparseKoLesky.jl#master`
    Updating `~/programs/research/surf/EarthquakeGPs.jl/Manifest.toml`
  [a4c015fc] + ANSIColoredPrinters v0.0.1
  [d1d4a3ce] + BitFlags v0.1.7
  [944b1d66] + CodecZlib v0.7.1
...
  [8e850b90] + libblastrampoline_jll v5.8.0+0
  [8e850ede] + nghttp2_jll v1.48.0+0
  [3f19e933] + p7zip_jll v17.4.0+0
        Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated -m`
Precompiling project...
  1 dependency successfully precompiled in 1 seconds. 49 already precompiled.
```

We need to explicitly provide a GitHub URL because SparseKoLesky.jl is not
in the [default repository](https://github.com/JuliaRegistries/General).
For [packages](https://juliapackages.com/packages) in the
default repository, for example, the popular plotting library
[Makie.jl](https://docs.makie.org/stable/), can instead be installed like

```text
(EarthquakeGPs) pkg> add CairoMakie
```

## Running code in the REPL

We can now [import](https://docs.julialang.org/en/v1/manual/modules/)
the library with the `using` command.

```@repl
using SparseKoLesky
```

This brings every name defined in the library into the current
namespace. For namespace hygiene reasons, and particularly
in scripts, I personally prefer the qualified import

```@repl
using SparseKoLesky: SparseKoLesky as KL
```

which brings the module into the name `KL`. For
example, to put points into the maximin ordering:

```@repl
using SparseKoLesky: SparseKoLesky as KL # hide
using Random: Random, rand
rng = Random.seed!(1);
points = rand(rng, 2, 10)
order, ℓ = KL.maximin_ordering(points);
order'
points = points[:, order];
```

## Running a code in a file

While the REPL can be used to quickly test an idea out,
for anything long-term it's better to put it in a file.

Make a new file called `main.jl` and add the following content.

```@example
using Random: Random
using LinearAlgebra: norm
using SparseKoLesky: SparseKoLesky as KL
using PyPrint: pprint, @pprint

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
```

This script depends on
[PyPrint.jl](https://github.com/stephen-huan/PyPrint.jl), a simple
library I wrote to make printing better in non-interactive environments.

```text
(EarthquakeGPs) pkg> add https://github.com/stephen-huan/PyPrint.jl
```

Although [Random](https://docs.julialang.org/en/v1/stdlib/Random/) and
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
are part of Julia's standard library, it's good practice (although not
strictly necessary) to declare them as dependencies as well.

```text
(EarthquakeGPs) pkg> add Random LinearAlgebra
```

The script can now be ran with `julia main.jl`.
The result should look something like

```text
[multithreading] using 1 threads
  1.081415 seconds (1.93 M allocations: 129.869 MiB, 3.12% gc time, 99.86% compilation time)
:KM = 10×10 Matrix{Float64}:
 1.0       0.698221  0.179363  0.292702  …  0.202044  0.171523  0.303929
 0.698221  1.0       0.246329  0.193833     0.245759  0.277551  0.279048
 0.179363  0.246329  1.0       0.404741     0.964642  0.928095  0.713812
 ⋮                                       ⋱
 0.202044  0.245759  0.964642  0.51511      1.0       0.825623  0.833711
 0.171523  0.277551  0.928095  0.2899       0.825623  1.0       0.559382
 0.303929  0.279048  0.713812  0.803345     0.833711  0.559382  1.0
:(explicit_factor.U) = 10×10 SparseArrays.SparseMatrixCSC{Float64, Int64} with 38 stored entries:
 1.0  -0.101019  -0.289727  -0.984688  …    ⋅         -3.13298    ⋅
  ⋅    1.00509   -0.274628  -0.125339     -0.929279     ⋅       -1.37067
  ⋅     ⋅         1.08414    0.050591      0.0164099    ⋅         ⋅
 ⋮                                     ⋱
  ⋅     ⋅          ⋅          ⋅            2.15045      ⋅       -3.30184
  ⋅     ⋅          ⋅          ⋅             ⋅          4.51707    ⋅
  ⋅     ⋅          ⋅          ⋅             ⋅           ⋅        9.83628
norm(approx_KM - KM) / norm(KM) = 0.039522347963394405
```

Note that the line `kernel = KL.MaternCovariance5_2(0.5)` creates a Matérn
kernel with smoothness $ \nu = 5/2 $ and length scale $ \ell = 0.5 $.
The parameter $ \rho $ controls the density of the resulting factor.

In Julia it's customary to use Unicode variable names to mimic mathematical
equations. In order to type $ \rho $, enter the REPL and type `\rho`
and then press the `<tab>` key. The `\rho` will change in-place to `ρ`.

```text
julia> \rho
julia> ρ
```

There is a popular [Julia plugin](https://www.julia-vscode.org/) for Visual
Studio Code that supports this as well as many other features. There are also
plugins provided by [JuliaEditorSupport](https://github.com/JuliaEditorSupport)
for vim, emacs, and other popular editors.

If we increase $ \rho $ from `3.0` to `6.0`, the resulting
factor will be denser and have necessarily better accuracy.


```text
[multithreading] using 1 threads
  1.105339 seconds (1.93 M allocations: 129.874 MiB, 5.43% gc time, 99.80% compilation time)
:KM = 10×10 Matrix{Float64}:
 1.0       0.698221  0.179363  0.292702  …  0.202044  0.171523  0.303929
 0.698221  1.0       0.246329  0.193833     0.245759  0.277551  0.279048
 0.179363  0.246329  1.0       0.404741     0.964642  0.928095  0.713812
 ⋮                                       ⋱
 0.202044  0.245759  0.964642  0.51511      1.0       0.825623  0.833711
 0.171523  0.277551  0.928095  0.2899       0.825623  1.0       0.559382
 0.303929  0.279048  0.713812  0.803345     0.833711  0.559382  1.0
:(explicit_factor.U) = 10×10 SparseArrays.SparseMatrixCSC{Float64, Int64} with 51 stored entries:
 1.0  -0.101019  -0.289727  -0.984688  …   0.320589    -3.16951     ⋅
  ⋅    1.00509   -0.274628  -0.125339     -0.976323    -0.122531  -1.08413
  ⋅     ⋅         1.08414    0.050591      0.00527681   0.051026  -0.262772
 ⋮                                     ⋱
  ⋅     ⋅          ⋅          ⋅            2.25232      0.225252  -3.64138
  ⋅     ⋅          ⋅          ⋅             ⋅           4.55964     ⋅
  ⋅     ⋅          ⋅          ⋅             ⋅            ⋅        11.6285
norm(approx_KM - KM) / norm(KM) = 0.0019967963847806417
```

Running the script again after the change confirms our hypothesis.
