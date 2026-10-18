# PDEStudioCore.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22893504.svg)](https://doi.org/10.5281/zenodo.22893504)
[![Build Status](https://github.com/blhackslash/PDEStudioCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/blhackslash/PDEStudioCore.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/blhackslash/PDEStudioCore.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blhackslash/PDEStudioCore.jl)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://blhackslash.github.io/PDEStudioCore.jl/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://blhackslash.github.io/PDEStudioCore.jl/stable/)

**PDEStudioCore.jl** is a headless Julia backend for running, storing and analysing PDE simulations. You write the solver; PDEStudioCore runs it over parameter sweeps, caches every result on disk under a hash of its parameters, and computes statistics (norms, errors against a reference, conservation) for you.

It is the engine behind the interactive frontend [PDEStudio.jl](https://github.com/blhackslash/PDEStudio.jl), but works on its own, for example on a cluster without a display.

## Features

- **One data type for any dimension:** Eulerian grids (`ESimData`) and Lagrangian particles (`LSimData`) in 1D, 2D, 3D or more, with or without time, in any floating point precision. Conversions between both representations are built in and cached.
- **Declarative sweeps:** shared parameters, per-method overrides and swept values (`SimulationConfig`) expand into the Cartesian product of runs; duplicate runs are removed.
- **Reproducible caching:** results are stored as JLD2 files named by a deterministic SHA-256 hash of the parameters, written atomically. Re-running a configuration only computes what is missing.
- **Statistics pipeline:** register a statistic once (`register_stat!`) with the dimensions it keeps, implement `calc_stat`, and it is evaluated (multithreaded) for every dataset, including errors against an analytical reference. `data.stats[:Solution]` always *is* the solution.
- **Safe serialization:** parameters are written and read with a literal parser (`val2str`/`str2val`) that never evaluates code.

## Installation

```julia
pkg> add PDEStudioCore
```

## Quick start

```julia
using PDEStudioCore
using StaticArrays

# 1. A solver: the resolved parameters of one run in, a dataset out
function advection_1d(params::ParamDict)
    N, c, T, cfl = params[:N], params[:c], params[:T], params[:cfl]
    x  = collect(range(0.0, 1.0, length = N + 1))[1:N]          # periodic grid
    dt = cfl * (x[2] - x[1]) / c
    t  = collect(0.0:dt:T)
    u  = zeros(SVector{1, Float64}, N, length(t))
    u[:, 1] .= SVector.(sin.(2π .* x))
    for n in 1:length(t) - 1, i in 1:N
        u_l, u_c, u_r = u[mod1(i - 1, N), n][1], u[i, n][1], u[mod1(i + 1, N), n][1]
        u[i, n + 1] = SVector(params[:scheme] == "upwind" ? u_c - cfl * (u_c - u_l) :
                                                     0.5 * (u_r + u_l) - 0.5 * cfl * (u_r - u_l))
    end
    return create_sim_data(x, u, t, params)                     # dims (:x, :t)
end

# 2. An exact solution (optional): a factory params -> ((x, t) -> u), enables error statistics
exact_advection(params) = xt -> SVector(sin(2π * (xt[1] - params[:c] * xt[2])))

# 3. The configuration
shared  = create_param_dict(:N => 100, :c => 1.0, :T => 2.0, :cfl => 0.5, :sim_func_name => :advection_1d)
methods = create_method_dict(:upwind         => create_param_dict(:scheme => "upwind"),
                             :lax_friedrichs => create_param_dict(:scheme => "lax_friedrichs"))
config  = SimulationConfig(shared, methods, [:upwind, :lax_friedrichs];      # active methods (`defaults`)
                           varied = create_varied_dict(:cfl => [0.3, 0.5, 0.8]),
                           reference = :exact_advection)

# 4. Run: 2 methods × 3 CFL numbers = 6 runs, with statistics
set_stat_preset!("hyperbolic")             # mass, L1/L2 norms and errors, wave position, …
set_save_path!("results")
run_all_simulations(config; parallel = true, calculate_stats = true)

# 5. Use the results
p   = assemble_params(config.shared, config.methods, :upwind)   # parameters of one run…
p[:cfl] = 0.5
sim = load_sim_data(p, :raw)
sim.stats[:l2error]                        # one value per time step
```

More complete scripts are in [`examples/`](examples/): `advection_1d.jl` and the minimal `dummy.jl`.

## Concepts at a glance

| `SimulationConfig` field | Meaning |
| :--- | :--- |
| `shared` | Parameters of every run, including `:sim_func_name` (the solver). |
| `methods` | Per-method overrides, e.g. the scheme. |
| `defaults` | The methods that are run. |
| `varied` | The sweep; a swept key must also exist in `shared` or a method. |
| `reference` | Name of a reference factory (exact solution) for error statistics. |
| `postprocess` | Name of a function `data -> Bool` deciding whether a result is saved. |
| `sources` | Files included before running (e.g. the file defining the solver). |

Functions are referenced by name and resolved in the module set with `set_target_module!` (default: `Main`).

## Documentation

The [documentation](https://blhackslash.github.io/PDEStudioCore.jl/) covers the data structures, the `SimulationConfig` in depth, storage and conversions, custom statistics and the full API. Upgrading from an earlier version? This is a breaking release without deprecated aliases; see the [Migration Guide](https://blhackslash.github.io/PDEStudioCore.jl/stable/migration/).

## Citing

If you use PDEStudioCore.jl in your research, please cite it via its Zenodo DOI: [10.5281/zenodo.22893504](https://doi.org/10.5281/zenodo.22893504).

---

Portions of this codebase and documentation were drafted with the assistance of large language models (LLMs). All code has been human-reviewed, verified, and tested. If you notice any inaccuracies or unexpected behavior, please open an issue.
