# Contributing

## Installation

Use the [Pkg](https://pkgdocs.julialang.org/v1/managing-packages/) interface

```text
pkg> add https://github.com/stephen-huan/SparseKoLesky.jl
```

## Running

Tests may be ran through the standard `Pkg.test()` interface

```text
pkg> test
```

The files in `examples` may be ran like

```bash
julia --project="@." examples/main.jl
```

## Formatting

This project is formatted in the [Blue](https://github.com/invenia/BlueStyle)
style with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl).
The current configuration is

```toml
style = "blue"
margin = 79
```

The entire project may be styled with a command like

```bash
julia \
  --project="@." \
  --eval "using JuliaFormatter; format(\".\")"
```

or the provided `/bin/juliafmt` utility.

```bash
./bin/juliafmt .
```

## Documentation

Documentation is based on Julia
[docstrings](https://docs.julialang.org/en/v1/manual/documentation/)
and [Documenter.jl](https://documenter.juliadocs.org/stable/).
To locally preview the documentation, run

```bash
./bin/build && ./bin/serve
```

and open `http://localhost:8000/` in a web browser.

To push the built site to the `cloudflare-pages` branch, run

```bash
./bin/publish
```

Markdown files should be styled with [Prettier](https://prettier.io/).
