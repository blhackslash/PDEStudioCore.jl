# Migration Guide

This is a **breaking release**. There are no deprecated aliases: old names raise errors, so every place that needs
changing is found on the first run. Parameter hashes are unchanged, so data files are still *found*, but files
written by earlier versions lack the metadata this version needs and are recomputed (see [Data files](#Data-files)).

## `SimulationConfig` fields and keywords

| Old | New |
| :--- | :--- |
| `shared_params` | `shared` |
| `methods_dict` | `methods` |
| `active_methods` | `defaults` |
| `varied_params` (field & keyword) | `varied` |
| `reference_name` / `ref_func_name` | `reference` |
| `post_process_name` (field & keyword) | `postprocess` |
| `source_files` (field & keyword) | `sources` |

```julia
# before
SimulationConfig(shared, methods, [:upwind]; varied_params = v, ref_func_name = :exact, source_files = ["s.jl"])
# now
SimulationConfig(shared, methods, [:upwind]; varied = v, reference = :exact, sources = ["s.jl"])
```

Unknown keywords raise a `MethodError`, unknown fields an error. The third positional argument (`defaults`) is
optional (default: all methods). Assigning a field accepts the same loose input as the constructor
(`config.defaults = :upwind`, `config.reference = "exact"`).

## `:Solution`

`data.stats` is a [`StatsView`](@ref) that always contains `:Solution` as `data.u`, for every dataset. You no longer
need `create_sim_data` (or manual `stats[:Solution] = u` / `registry[:Solution] = :all` bookkeeping) to get it. If you
relied on `data.stats` being a `Dict`, use `copy(data.stats)`.

## `create_sim_data`

One N-dimensional front end. Name the spatial axes with `dims`; the single-axis keywords `x_dim`/`y_dim`/`z_dim` and
the five-argument form `(x_grid, y_grid, u, t, params)` were removed:

```julia
# before
create_sim_data(x, u, t, params; time_dim = :t, x_dim = :r)
create_sim_data(x_grid, y_grid, u, t, params)
# now
create_sim_data(x, u, t, params; dims = :r)
create_sim_data((x_grid, y_grid), u, t, params)      # or ((x, y), u, t, params) with axis vectors
```

New: tuples of axes for any dimension, static data, numeric arrays as scalar fields, components-last arrays,
Lagrangian data in any dimension and static particles. Dimension names must be unique (also against `time_dim`).
The integration measure is derived from the axes, not from overridden bounds; this only changes results if you passed
`xmins`/`xmaxs`/`tmin`/`tmax` that differ from the axis extent.

## Data files

Files now carry metadata (`native`, `shape`) that loading relies on. Files written by an earlier version count as
missing: `run_all_simulations` recomputes and replaces them (with an info message), and `does_sim_data_exist` returns
`false` for them. Loading one directly raises an `ArgumentError` naming the file. The undocumented internal helpers `_normalize_spatial_res`, `_get_lsim_bounds` and
`_get_native_type` were removed.

## Behaviour changes

- [`get_kept_dims`](@ref) returns the kept dimensions in domain order (the order of the statistic's array axes), also when the registry lists them in another order (e.g. `register_stat!(:s, [:t, :x])` on an `(x, t)` domain gives `[:x, :t]`).
- [`str2val`](@ref) no longer evaluates arbitrary code; strings are kept verbatim during normalisation. String parameters that look like Julia names (e.g. `"sin"`, `"pi"`) were previously turned into functions/numbers, which broke cache lookups; they are now found correctly.
- Statistics registered after a dataset was created are computed by `calculate_all_stats!`; `force_overwrite = true` no longer deletes statistics added with `add_stat!`.
- `calculate_all_stats!` no longer saves converted or reference datasets over the native data of a run (previously, calling it on `load_sim_data(p, :lagrangian)` replaced the native Eulerian data on disk).
- Parameters equal to `nothing` stay `nothing` after normalisation (previously they became the type `Nothing`; the hash is unchanged).
- `run_all_simulations` deduplicates tasks, logs and skips failing tasks in every pass, actually loads `sources`, and stores results under the task parameters.
- [`is_reference_method`](@ref) matches words of the method name (e.g. `:inexact_newton` is no longer skipped).
- `delete_sim_data` rejects an empty filter and returns the deleted paths; `rehash_sim_data` returns `old => new` pairs.
- `load_sim_data(hash_prefix)` matches the hash part of file names only (not timestamps); `key = "raw"` returns a single entry.
- `check_data` columns have a fixed order (`Time`, then `C<c>_Min`, `_Max`, `_Mean`, `_NaNs`).
- `save_sim_data` returns the file path and writes atomically.
- Exports of undefined names (`FixedDict`, `resolve_simulation_function`, `resolve_reference_function`) were removed; `delete_stat!` is now implemented.
