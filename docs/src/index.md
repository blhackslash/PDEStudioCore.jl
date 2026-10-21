# PDEStudioCore.jl

**PDEStudioCore.jl** is a headless-safe Julia backend for running, storing and analysing Partial Differential Equation (PDE) simulations.

Designed with reproducible research in mind, it provides strictly typed, precision-agnostic Eulerian and Lagrangian datasets of any dimension, deterministic parameter hashing with atomic on-disk caching, declarative parameter sweeps and a multithreaded statistics pipeline.

---

## 🏗️ Core Data Structures

All simulation outputs are wrapped in strongly typed structures that map the mathematical domain of the data.

| Type | Description | Main fields |
| :--- | :--- | :--- |
| `ESimData{D, DS, M, T}` | Eulerian data: fields on a structured tensor grid. | `axes`, `u`, `stats` |
| `LSimData{D, DS, M, T}` | Lagrangian data: particles recorded at discrete times. | `t`, `x`, `u`, `stats` |

`D` is the number of spacetime dimensions, `DS` the number of spatial dimensions (`D == DS` for static data), `M` the number of field components and `T` the floating point precision.

*Example: an Eulerian simulation in 2D space over time with 4 state variables in `Float64` precision is an `ESimData{3, 2, 4, Float64}`.*

Datasets of any dimension are created with a single function, [`create_sim_data`](@ref):

```julia
create_sim_data(x, u, t, params)               # 1D + time
create_sim_data((x, y), u, t, params)          # 2D + time
create_sim_data((x, y, z), ρ, params)          # static 3D field
create_sim_data(x_traj, u_traj, t, params)     # particles over time
```

### The solution is a statistic, by construction

`data.stats` is a dictionary view ([`StatsView`](@ref)) in which `:Solution` is always present and always *is* `data.u`:

```julia
sim = create_sim_data(x, u, t, params)
sim.stats[:Solution] === sim.u    # true: the same array, never a copy
sim.u = u_new                     # :Solution follows automatically
keys(sim.stats)                   # :Solution plus every derived statistic
```

This holds for every dataset, however it was produced: `create_sim_data`, the `ESimData`/`LSimData` constructors, conversions, resampling, reference generation or loading from disk. `:Solution` cannot be deleted, is never stored twice on disk, and its kept dimensions are always all dimensions.

---

## 🧩 The Simulation Configuration (`SimulationConfig`)

[`SimulationConfig`](@ref) is the blueprint of a batch of runs. It binds your solver (referenced by name) to its parameter space:

| Field | Meaning |
| :--- | :--- |
| `shared` | Baseline parameters of every run (including `:sim_func_name`, the solver's name). |
| `methods` | Per-method parameter overrides (e.g. `:upwind`, `:lax_wendroff`). |
| `defaults` | The methods that are executed (default: all methods). |
| `varied` | The parameter sweep; the Cartesian product of all vectors is run. |
| `reference` | Name of a reference factory `params -> (x -> u_exact)` used for error statistics. |
| `postprocess` | Name of a function `data -> Bool` run on every dataset (`true` saves it). |
| `sources` | Files included into the target module before running. |

```julia
config = SimulationConfig(shared, methods; varied = Dict(:N => [50, 100, 200]))
run_all_simulations(config; calculate_stats = true)
```

---

## 📖 The Configuration Dictionaries

*   **`ParamDict` (`Dict{Symbol, Any}`):** the flat, resolved parameters of a single run (e.g. `:N => 100`, `:cfl => 0.2`). It is passed to your solver, and its normalised content is hashed to locate the run on disk.
*   **`MethodDict` (`Dict{Symbol, ParamDict}`):** method-specific overrides on top of the shared parameters. `:ignore => [:key1, :key2]` removes shared keys for one method.
*   **`VariedDict` (`Dict{Symbol, Vector}`):** the sweep definition. Keys such as `:Ns__x` sweep a single entry of a tuple parameter, `:Ns__all` all of them together.

String keys are accepted everywhere and converted to `Symbol`s.

---

## ⚙️ Main API Functions

*   **[`run_all_simulations`](@ref)`(config; kwargs...)`:** runs every task of the sweep that is not on disk yet, then (optionally) computes statistics and runs your post-processing. Failures of single tasks are logged and do not abort the batch.
*   **[`generate_tasks`](@ref)`(config)`:** the parameter sets of a sweep, e.g. to load its results: `[load_sim_data(p) for p in generate_tasks(config)]`.
*   **[`load_sim_data`](@ref)`(params[, kind, res])`:** loads the native data, or an Eulerian/Lagrangian representation that is converted (and optionally cached) on demand.
*   **[`calculate_all_stats!`](@ref)`(data, ref_func)`:** evaluates all registered statistics (mass, L1/L2 errors, …) against an exact reference, on grids and on particles.

---

## 📊 Custom Statistics

Register a statistic with the dimensions it *keeps* and implement [`calc_stat`](@ref) for it; the backend integrates out the others:

```julia
register_stat!(:my_error, :time)   # one value per time step

function PDEStudioCore.calc_stat(::Val{:my_error}, fixed_coords, u, ana, domain::DomainInfo)
    return sum(abs.(u .- ana)) * get_integration_measure(:my_error, domain)
end
```

Pre-computed values are attached with [`add_stat!`](@ref). See [Statistics and Analytics](statistics.md) for details.

---

## 🛠️ Dimension Keys (`dim_keys`)

The `dim_keys` tuple of a [`DomainInfo`](@ref) is the *sole identifier* of the axes, e.g. `(:x, :y, :t)`. Names and order matter for interpolation, statistical slicing and plotting. The time axis, if present, is always the last one. You can name the axes freely (`dims = (:r, :θ)`), but keep the layout of a model stable once data has been written to disk.

---

## 🚀 Examples

Runnable examples live in the `examples/` directory of the repository; the [Simulation Config](simulation_config.md) page walks through a complete setup step by step. If you are upgrading, see the [Migration Guide](migration.md).
