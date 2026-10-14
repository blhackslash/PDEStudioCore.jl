# Statistics and Analytics

**PDEStudioCore.jl** features a multithreaded statistics pipeline. Instead of writing loops that average data over space or time, you register what a statistic keeps; the backend slices your Eulerian tensors or Lagrangian particle series accordingly, evaluates your function on every slice and stores the reduced data with the dataset.

Every dataset exposes its statistics through `data.stats`, a dictionary view in which `:Solution` is always present and always refers to `data.u`. Derived statistics live next to it:

```julia
data.stats[:Solution] === data.u   # true
data.stats[:l2error]                # a derived statistic
copy(data.stats)                    # a plain Dict snapshot
```

The global statistics registry starts empty, so the backend makes no assumptions about your model.

## 1. Loading Presets

Predefined suites of statistics live in the `presets/` directory. For hyperbolic PDEs (mass, L1/L2 errors, …):

```julia
using PDEStudioCore
set_stat_preset!("hyperbolic")
```

## 2. Automated Integration (via `calc_stat`)

### Step A: Register the Statistic

Register the statistic's name and the dimensions to **retain**; all other dimensions are integrated out:

```julia
register_stat!(:custom_energy, :time)   # one value per time step
```

Valid kept dimensions are the aliases `:all` (pointwise field), `:space` (integrates out time), `:time` (integrates out space), a single dimension name such as `:x`, or a vector of names such as `[:x, :t]` (`Symbol[]` gives a single scalar). The statistic's array always has its axes in domain order, and [`get_kept_dims`](@ref) reports them in that order.

### Step B: Overload `calc_stat`

```julia
function PDEStudioCore.calc_stat(::Val{:custom_energy}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:custom_energy, domain)   # e.g. dx * dy
    return sum(abs2.(u)) * measure
end
```

**Arguments**
*   `::Val{:name}`: the dispatch token of your statistic.
*   `fixed_coords`: the coordinates of the kept dimensions for the current slice (an `SVector`).
*   `u`: the numerical values of the slice (a flat vector of `SVector`s).
*   `ana`: the reference values of the slice (`NaN`s if no reference is available).
*   `domain`: the [`DomainInfo`](@ref), giving access to the spacing, bounds and registry.

The function must return an `SVector` with one entry per field component.

### Step C: Evaluate

```julia
calculate_all_stats!(data, ref_func)     # ref_func(x::SVector{D}) -> exact value, or nothing
```

[`calculate_all_stats!`](@ref) evaluates every statistic of the global registry that the dataset does not have yet (all of them with `force_overwrite = true`) and saves the dataset (`save = false` keeps it in memory only). Statistics registered *after* a dataset was created are picked up, and a statistic whose kept dimensions changed is recomputed. Results that are entirely `NaN` (typically errors without a reference) are dropped and remembered as evaluated, so the pipeline does not reload the data for them again. The pipeline also remembers whether a reference was available, so these statistics are evaluated again automatically once `config.reference` is set; `run_all_simulations(config; calculate_stats = true, force_stats = true)` recomputes every registered statistic without rerunning the solvers.

Converted and reference datasets (from `load_sim_data(p, :lagrangian)`, `resample_eulerian`, `generate_reference_simdata`, …) share the parameters of the native run. Their statistics are computed as usual, but they are never saved over the native data: `calculate_all_stats!` warns and keeps them in memory. Pass `save = false` for such datasets to skip the check.

### Lagrangian Data

On particle data, statistics can keep
- no dimension: one value over all particles and time steps,
- the time dimension: one value per time step,
- all dimensions: one value per particle (`u` and `ana` are 1-tuples).

Other combinations are skipped for Lagrangian data.

## 3. Direct Injection (via `add_stat!`)

Some statistics do not fit the integration pipeline: the run time of a solver, a global quantity that needs the entire dataset, or values computed elsewhere. Attach them directly with [`add_stat!`](@ref), declaring the dimensions they keep:

```julia
add_stat!(data, :runtime, elapsed_seconds, Symbol[])          # scalar
add_stat!(data, :energy_history, energy_vector, :time)        # one value per time step
```

Numbers and arrays of numbers are converted to `SVector`s with the dataset's component count. Shapes are checked: against the kept axes for Eulerian data, and against the time steps and particle counts for Lagrangian data. `calculate_all_stats!` leaves statistics added this way alone, unless the same name is also registered globally (you get a warning if the kept dimensions differ). `delete_stat!(data, :runtime)` removes one again, and `delete_stat!(:name)` removes a statistic from the global registry.

Assigning `data.stats[:name] = value` also works, but it does not record kept dimensions; prefer `add_stat!`.

### Using `add_stat!` in Post-Processing

A natural place for custom metrics is the post-processing function of a [`SimulationConfig`](@ref), which runs after the statistics pass:

```julia
function my_post_process(data)
    add_stat!(data, :global_maximum, maximum(v -> v[1], data.u), Symbol[])
    return true   # save the dataset
end

config = SimulationConfig(shared, methods, [:upwind]; postprocess = :my_post_process)
```

> **Note:** The kept dimensions recorded by `register_stat!`/`add_stat!` tell conversions and the **PDEStudio.jl** frontend how to slice and display a statistic.
