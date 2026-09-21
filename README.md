# PDEStudioCore.jl

[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://blhackslash.github.io/PDEStudioCore.jl/)
[![Build Status](https://github.com/blhackslash/PDEStudioCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/blhackslash/PDEStudioCore.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/blhackslash/PDEStudioCore.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blhackslash/PDEStudioCore.jl)

**PDEStudioCore.jl** is a robust, headless-safe Julia backend engineered for the execution, management, and statistical analysis of Partial Differential Equation (PDE) simulations. 

Designed with reproducible academic research in mind, it provides a unified framework for generating strictly typed, precision-agnostic Eulerian and Lagrangian datasets. It features deterministic cryptographic hashing for simulation parameters, automated disk caching, and a highly optimized multithreaded statistical integration pipeline. 

## Installation

From the Julia REPL, type `]` to enter the Pkg prompt and run:

```julia
pkg> add PDEStudioCore
```

## Documentation

For a detailed overview of Core Data Structures, the `SimulationConfig`, custom statistical integration, and the full API reference, please visit the [Stable Documentation](https://blhackslash.github.io/PDEStudioCore.jl/).

## Examples

To help you get started, here is a complete example for running a simulation:

```julia
# examples/dummy.jl
using PDEStudioCore
using StaticArrays

# 1. Define a simple user simulation function
function my_wave_sim(params::ParamDict)
    N  = params[:N]
    dt = params[:dt]
    c  = params[:wave_speed]

    # Simple 1D Space + 1D Time Grid
    x = collect(range(0.0, 1.0, length=N))
    t = collect(range(0.0, 2.0, step=dt))

    # Generate mock data: u(x,t) = sin(x - c*t)
    u = zeros(SVector{1, Float64}, length(x), length(t))
    for (j, t_val) in enumerate(t)
        for (i, x_val) in enumerate(x)
            u[i, j] = SVector{1, Float64}(sin(x_val - c * t_val))
        end
    end

    return create_sim_data(x, u, t, params; time_dim=:t)
end

# 2. Setup Dictionaries
shared_params = create_param_dict(
    :N => 100,
    :wave_speed => 1.5,
    :dt => 0.1
    )

methods = create_method_dict(
    :upwind => create_param_dict(:solver => "upwind"),
    :lax_wendroff => create_param_dict(:solver => "lax_wendroff")
    )

varied_params = create_varied_dict(
    :dt => [0.1, 0.05]
    )

# 3. Create the Configuration
config = SimulationConfig(
    :my_wave_sim,
    shared_params,
    methods,
    [:upwind, :lax_wendroff];
    varied_params = varied_params
    )

# 4. Run the Pipeline!
set_save_path!(joinpath(@__DIR__, "results"))
run_all_simulations(config, parallel=true, calculate_stats=true)
```
