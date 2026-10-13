# Data: Creation, Storage and Conversion

## 1. Creating Datasets

[`create_sim_data`](@ref) builds Eulerian and Lagrangian datasets of any dimension. The global statistic registry is copied into the dataset, the parameters are copied, and `:Solution` is coupled to `u`.

### Eulerian (grids)

```julia
# 1D + time: u[i, n]
create_sim_data(x, u, t, params)

# 2D + time from axis vectors: u[i, j, n]
create_sim_data((x, y), u, t, params)

# static 3D field with custom axis names
create_sim_data((r, θ, φ), ρ, params; dims = (:r, :θ, :φ))

# ndgrid-style coordinate arrays (x_grid[i, j] == x[i])
create_sim_data((x_grid, y_grid), u, t, params)
```

The solution may be given as
- an array of `SVector`s (`M` components),
- an array of numbers (a scalar field, `M = 1`),
- an array of numbers with one extra trailing dimension holding the components (`u[i, j, n, c]`),
- an array of vectors.

Integer data is stored as `Float64`; `Float32` data stays `Float32`. Axes must be sorted. Beyond three spatial dimensions the default axis names are `:x1, :x2, …`; time is always the last axis (`time_dim = :t` by default).

`xmins`, `xmaxs`, `tmin` and `tmax` override the bounding box (e.g. the physical domain `[0, 1]` of a cell-centred grid). The integration measure (`domain.spacing`) is always the mean node distance of the axes, independent of these overrides.

### Lagrangian (particles)

```julia
# x_traj[n][p]::SVector{DS}, u_traj[n][p]::SVector{M} (or numbers), one entry per time step
create_sim_data(x_traj, u_traj, t, params; dims = (:x, :y))

# static particles (no time axis)
create_sim_data(positions, values, params)
```

Particle counts may differ between time steps. The bounding box defaults to the extent of all positions.

### Direct construction

The `ESimData`/`LSimData` constructors validate their input (matching axes and solution sizes, consistent dimensions, time as the last axis) and couple `:Solution` themselves, so derived datasets never need any bookkeeping:

```julia
ESimData(params, domain, axes, u)          # DS is inferred from the domain
LSimData(params, domain, t, x, u)
```

## 2. The Statistics View

`data.stats` returns a [`StatsView`](@ref): a dictionary whose `:Solution` entry is `data.u` itself.

| Operation | Effect |
| :--- | :--- |
| `data.stats[:Solution]` | returns `data.u` |
| `data.stats[:Solution] = v` / `data.u = v` | replaces the solution (size-checked) |
| `delete!(data.stats, :Solution)` | error: the solution cannot be removed |
| `data.stats[:name] = v`, `delete!(data.stats, :name)` | edit derived statistics |
| `keys`, `values`, `pairs`, `haskey`, `get`, `length` | as for a `Dict`, including `:Solution` |
| `copy(data.stats)` | plain `Dict` snapshot |

On disk, the solution is stored once (in `u`); files written by earlier versions, which also stored it among the statistics, are read transparently.

## 3. Storage Layout

Data lives in `<save path>/data/<timestamp>_<sha256>.jld2`. The hash is computed from the normalised parameters ([`calculate_hash`](@ref)), so a run is found again from its parameters alone. Each file contains:

| Entry | Content |
| :--- | :--- |
| `"raw"` | the native dataset |
| `"native"`, `"params"`, `"stat_keys"`, `"registry"`, `"shape"` | fast metadata, read without loading the data (`"shape"` is `size(u)`, or time steps and total particle count for Lagrangian data) |
| `"evaluated_stats"` | statistics already evaluated for this data |
| `"sources"` | SHA-256 digests of the source files (pipeline runs with `sources`) |
| `"conv_L"`, `"conv_E_<n1>x<n2>…"` | cached conversions |

Writes are atomic: a file is written to a temporary path and renamed into place, so an interrupted run never corrupts existing data, and rewriting a file does not grow it. Rewriting the raw data drops stale conversion caches but keeps any other entries (including groups) you stored yourself.

Converted and reference datasets carry the parameters of the native run, so they map to the same file. [`calculate_all_stats!`](@ref) detects this (by representation and shape) and does not save them over the native data; an explicit `save_sim_data(...; overwrite = true)` of such a dataset does replace it.

## 4. Loading and Converting

```julia
load_sim_data(params)                         # native data
load_sim_data(params, :eulerian, (100, 50))   # grid at spacetime resolution (Nx, Nt)
load_sim_data(params, :lagrangian)            # particles
enable_cache!(true)                           # store conversions in the file
```

- **Eulerian → Eulerian** ([`resample_eulerian`](@ref)): linear interpolation in space, nearest-neighbour in time (no cross-fading of snapshots); statistics are resampled along the dimensions they keep. Axes whose resolution does not change (e.g. adaptive time steps) are kept as they are.
- **Lagrangian → Eulerian** ([`convert_to_eulerian`](@ref)): inverse-distance scatter of particles and per-particle statistics; empty cells become `NaN`. Output times take the nearest recorded snapshot.
- **Eulerian → Lagrangian** ([`convert_to_lagrangian`](@ref)): grid nodes become stationary particles (sharing one position vector); fields become per-particle statistics.

The resolution `res` always lists every axis: the spatial resolutions followed by the number of time slices.

## 5. Managing Data

```julia
does_sim_data_exist(params)                         # without loading the data
delete_sim_data(Dict(:scheme => "upwind"); dry_run = true)   # preview, then delete
rehash_sim_data(joinpath(get_save_path(), "data"); remove_keys = [:obsolete])
load_sim_data("a1b2c3"; key = "raw")                # by hash prefix (REPL inspection)
check_data(data)                                    # min/max/mean/NaN summary as a DataFrame
```

`delete_sim_data` refuses an empty filter. `rehash_sim_data` writes the renamed file before removing the old one.

## 6. Parameter Normalisation

Before hashing and saving, parameters are normalised: closures become `"Closure"`, custom structs become a descriptive string and other values round-trip through [`val2str`](@ref)/[`str2val`](@ref). Strings are always kept verbatim, so a parameter such as `:ic => "sin"` stays a string. [`str2val`](@ref) only parses literals (numbers, strings, symbols, types, tuples, vectors, `Dict`s and arithmetic like `2π`); it never executes code. Inspect the stored form with [`print_clean_params`](@ref).
