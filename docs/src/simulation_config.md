# Orchestrating with SimulationConfig

[`SimulationConfig`](@ref) is the central orchestration structure of **PDEStudioCore.jl**. Instead of writing nested loops over parameters, numerical methods and resolutions, you describe the batch once. The backend expands it into the exact Cartesian product of your parameter space, hashes every parameter set and runs only what is not on disk yet.

---

## Step 1: Setting the Target Module

PDEStudioCore is a serialised, headless backend: functions are referenced by **name** (a `Symbol` or `String`), never as closures, so that every run is fully described by plain data. First, tell the backend in which module these names are looked up (usually `Main` or your own module):

```julia
using PDEStudioCore
using StaticArrays

PDEStudioCore.set_target_module!(@__MODULE__)
set_save_path!(joinpath(@__DIR__, "results"))   # data goes to results/data/
```

## Step 2: Defining the Solver

The solver receives one argument, the [`ParamDict`](@ref) of a run, and returns a dataset built with [`create_sim_data`](@ref) (or `NoSimData()` if it has nothing to store). It receives a copy of the parameters; the result is always stored under the parameters of the task.

```julia
function my_wave_solver(params::ParamDict)
    N, cfl, scheme = params[:N], params[:cfl], params[:scheme]
    x = collect(range(0.0, 1.0, length = N))
    t = collect(range(0.0, params[:T], length = 50))

    u = zeros(N, length(t))   # ... run your numerical scheme here ...

    return create_sim_data(x, u, t, params)   # numeric arrays are scalar fields (M = 1)
end
```

## Step 3: The Shared Parameters

The first layer of the hierarchy holds the values that every run shares. The solver is selected with the `:sim_func_name` parameter; since it is a parameter, it is part of the hash, and methods may override it to use different solvers.

```julia
shared = Dict(
    :sim_func_name => :my_wave_solver,
    :L => 1.0,     # domain length
    :T => 5.0,     # final time
    :N => 100,     # default resolution (swept below)
)
```

## Step 4: Method Overrides

The second layer maps method names to parameter overrides; they replace matching shared keys. Use `:ignore => [:key1, :key2]` to drop shared keys for a single method.

```julia
methods = Dict(
    :upwind       => Dict(:scheme => "Upwind",       :cfl => 0.5),
    :lax_wendroff => Dict(:scheme => "Lax-Wendroff", :cfl => 0.8),
)
```

## Step 5: The Sweep

`varied` lists the values to sweep. A varied key is applied to every method whose parameters contain that key (here: through `shared`).

```julia
varied = Dict(:N => [50, 100, 200])
```

## Step 6: Assembly and Execution

```julia
config = SimulationConfig(shared, methods, [:upwind, :lax_wendroff]; varied = varied)

config                    # prints a summary of the configuration
generate_tasks(config)    # the 6 parameter sets (2 methods × 3 resolutions)

run_all_simulations(config)                               # runs what is missing
results = [load_sim_data(p) for p in generate_tasks(config)]
```

The third positional argument (`defaults`, the methods to run) is optional and defaults to all methods of `methods`. Without any methods, the shared parameters are run on their own: `SimulationConfig(shared; varied = varied)`.

---

## Reference: Fields and Arguments

| Field | Constructor argument | Meaning |
| :--- | :--- | :--- |
| `shared` | 1st positional | Baseline parameters shared by all runs. |
| `methods` | 2nd positional | Per-method overrides (`MethodDict`). |
| `defaults` | 3rd positional | Methods executed by `run_all_simulations` (default: all). |
| `varied` | keyword `varied` | The sweep (`VariedDict`). |
| `reference` | keyword `reference` | Name of the reference factory, see [Advanced Config](advanced_config.md). |
| `postprocess` | keyword `postprocess` | Name of the post-processing function. |
| `sources` | keyword `sources` | Files included into the target module before running. |

All dictionaries may use `String` or `Symbol` keys, and assigning a field (e.g. `config.reference = "exact_factory"`) accepts the same loose input as the constructor.

Unknown keywords and field names raise an error.
