# Advanced Simulation Configurations

Beyond running a few methods, you will often want to sweep parameter spaces, compare against an exact solution, post-process every dataset, or make a batch reproducible from its source files. All of this is configured through the keyword arguments of [`SimulationConfig`](@ref).

## 1. Varied Parameters (Sweeps)

`varied` maps parameter names to vectors (or ranges) of values. The backend runs the Cartesian product of all vectors for every active method:

```julia
config = SimulationConfig(shared, methods, [:upwind];
    varied = Dict(:N => [50, 100, 200], :cfl => 0.1:0.2:0.9),
)
```

A varied key is applied to a method only if the method's parameters (shared plus overrides) contain that key. This lets you sweep a parameter that only some methods use (e.g. `:theta` of a θ-scheme) without duplicating the runs of the others. Duplicate parameter sets are removed, and a key that no active method uses triggers a warning (usually a typo).

### Tuple Sweeps

If your solver expects a tuple (e.g. a 2D resolution `:Ns => (100, 100)`), sweep its entries independently by appending `__x`, `__y`, `__z` or `__1`, `__2`, … to the base key. The tuple is reconstructed (as `Tuple{Int, Int}` for integers) before it reaches your solver:

```julia
varied = Dict(
    :Ns__x => [50, 100],   # first entry
    :Ns__y => [50, 100],   # second entry
)
```

This sweeps the entries independently (here 2 × 2 runs). To change several entries *together*, e.g. for a
convergence study on square grids, address them with one key: `__xy` (`__xz`, `__yz`, `__xyz`) sets the named
entries and `__all` every entry of the tuple to the same value. The swept values are then plain numbers, which
frontends can use directly as slider positions and axis values:

```julia
varied = Dict(:Ns__all => [50, 100, 200])   # (50, 50), (100, 100), (200, 200)
```

`__all` works for tuples of any length (the shape comes from the tuple in the parameters), so one sweep
definition serves 1D, 2D and 3D setups. The generated parameters—and therefore the data files—are identical to
those of an explicit tuple sweep such as `:Ns => [(50, 50), (100, 100), (200, 200)]`.

## 2. Analytical Reference Functions

Error statistics (L1/L2 errors, …) need the exact solution. Provide the **name of a reference factory**: a function that receives the run's parameters and returns the exact solution as a function of a spacetime coordinate `x::SVector{D}` (`x[end]` is the time for transient data). The factory lets the exact solution depend on the parameters (wave speed, initial condition, …):

```julia
exact_wave_factory(params) = x -> SVector(sin(2π * (x[1] - params[:c] * x[end])))

config = SimulationConfig(shared, methods, [:upwind];
    reference = :exact_wave_factory,
)
run_all_simulations(config; calculate_stats = true)
```

The reference may return an `SVector`, a tuple/vector or (for scalar fields) a plain number. It is evaluated once at every grid node or particle, in parallel. Statistics that come out entirely `NaN` (for example errors without a reference) are dropped.

Methods whose name contains a word starting with `analytic`, `reference`, `exact`, `baseline` or `true` (e.g. `:analytic`, `:exact_solution`) are treated as reference methods and skipped by the pipeline; frontends can generate them on demand with [`generate_reference_simdata`](@ref).

## 3. Custom Post-Processing

`postprocess` names a function that receives every dataset after the statistics pass. Mutate the data as you like and return `true` to save the changes; any other return value (including `nothing`) leaves the file untouched.

```julia
function add_maximum(data)
    add_stat!(data, :maximum, maximum(v -> v[1], data.u), Symbol[])
    return true   # save the modified dataset
end

config = SimulationConfig(shared, methods, [:upwind]; postprocess = :add_maximum)
```

## 4. Source Files

`sources` lists files that are included into the target module at the start of [`run_all_simulations`](@ref), so your solvers and reference functions are always defined and up to date. For reproducibility, the SHA-256 digest of every source file is stored in each newly written data file (entry `"sources"`, mapping the absolute path to its digest).

```julia
config = SimulationConfig(shared, methods, [:upwind];
    sources = ["src/wave_equations.jl", "src/boundary_conditions.jl"],
)
```

## 5. Execution Options

```julia
run_all_simulations(config;
    force_overwrite = false,   # rerun and recompute everything
    calculate_stats = false,   # pass 2: evaluate registered statistics
    force_stats = false,       # recompute all statistics (solvers are not rerun)
    post_process    = true,    # pass 3: run `config.postprocess`
    parallel        = false,   # pass 1 over Julia threads
)
```

- Existing data is never recomputed unless `force_overwrite = true`; statistics are only evaluated for datasets that miss them (or whose kept dimensions changed, or that were evaluated without a reference that is now configured). `force_stats = true` recomputes the statistics only.
- A failing solver, statistic or post-processing step is logged with its parameters and the batch continues.
- Solvers may return `NoSimData()` (or `nothing`) when there is nothing to store.
- If a solver modifies its parameters, the result is still stored under the parameters of the task (with a warning), so it can always be found again.
- Functions defined while the pipeline runs (e.g. through `sources`) are called in the latest world age.
