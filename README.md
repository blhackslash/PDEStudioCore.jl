# PDEStudioCore.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22893504.svg)](https://doi.org/10.5281/zenodo.22893504)
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

# examples/advection_1d.jl
using PDEStudioCore
using StaticArrays

# 1. Define the 1D Linear Advection solver
function advection_1d(params::ParamDict)
    # Extract shared physics and grid settings
    N   = params[:N]
    c   = params[:c]
    T   = params[:T]
    cfl = params[:cfl]
    scheme = params[:scheme]

    # Calculate step sizes based on the CFL condition
    x = collect(range(0.0, 1.0, length=N))
    dx = x[2] - x[1]
    dt = cfl * dx / abs(c)
    t = collect(0.0:dt:T)

    # Allocate the 2D Spacetime Tensor (Space x Time)
    u = zeros(SVector{1, Float64}, N, length(t))
    
    # Initial Condition: Sine wave
    for i in 1:N
        u[i, 1] = SVector{1, Float64}(sin(2 * pi * x[i]))
    end

    # Time integration loop with periodic boundary conditions
    for j in 1:(length(t)-1)
        for i in 1:N
            # Periodic index wrapping
            i_prev = i == 1 ? N : i - 1
            i_next = i == N ? 1 : i + 1

            u_curr = u[i, j][1]
            u_prev = u[i_prev, j][1]
            u_next = u[i_next, j][1]

            # Route to the specific numerical scheme
            if scheme == "upwind"
                # Standard first-order upwind (assuming c > 0)
                val = u_curr - cfl * (u_curr - u_prev)
            elseif scheme == "lax_friedrichs"
                # Lax-Friedrichs central difference
                val = 0.5 * (u_next + u_prev) - 0.5 * cfl * (u_next - u_prev)
            else
                error("Unknown scheme: $scheme")
            end

            u[i, j+1] = SVector{1, Float64}(val)
        end
    end

    return create_sim_data(x, u, t, params; time_dim=:t)
end

# 2. Setup Dictionaries
shared_params = create_param_dict(
    :N => 100,
    :c => 1.0,
    :T => 2.0,
    :cfl => .5, # Define the default to be overwritten by the varied parameters
)

# Define the methods (these inject the :scheme parameter into the solver)
methods = create_method_dict(
    :upwind => create_param_dict(:scheme => "upwind"),
    :lax_friedrichs => create_param_dict(:scheme => "lax_friedrichs")
)

# Parameter sweeps: Test both methods across different CFL limits
varied_params = create_varied_dict(
    :cfl => [0.3, 0.5, 0.8, 1.0]
)

# 3. Create the Configuration
config = SimulationConfig(
    :advection_1d,            # Pass the function name
    shared_params,
    methods,
    [:upwind, :lax_friedrichs];
    varied_params = varied_params
)

# 4. Run the Pipeline!
set_save_path!(joinpath(@__DIR__, "results"))
run_all_simulations(config, parallel=true, calculate_stats=true)
```
