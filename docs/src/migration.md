# Migration Guide

This release keeps existing data files and parameter hashes valid. Most code keeps working; deprecated names emit a one-time warning.

## Renamed `SimulationConfig` fields and keywords

| Old | New |
| :--- | :--- |
| `shared_params` | `shared` |
| `methods_dict` | `methods` |
| `active_methods` | `defaults` |
| `varied_params` (field & keyword) | `varied` |
| `reference_name` / `ref_func_name` | `reference` |
| `post_process_name` (field & keyword) | `postprocess` |
| `source_files` (field & keyword) | `sources` |

Old field names still work as properties, and the old keywords (`varied_params`, `ref_func_name`, `post_process_name`, `source_files`) still work in the constructor, with a deprecation warning. Unknown keywords now raise an error instead of being ignored. The third positional argument (`defaults`) is now optional.

## `:Solution`

`data.stats` is now a [`StatsView`](@ref) that always contains `:Solution` as `data.u`, for every dataset. You no longer need `create_sim_data` (or manual `stats[:Solution] = u` / `registry[:Solution] = :all` bookkeeping) to get it. If you relied on `data.stats` being a `Dict`, use `copy(data.stats)`.

## `create_sim_data`

The old call forms still work, including `x_dim`/`y_dim` and the `(x_grid, y_grid, u, t, params)` form. New: tuples of axes for any dimension, static data, numeric arrays as scalar fields, components-last arrays, Lagrangian data in any dimension and static particles. The integration measure is now derived from the axes, not from overridden bounds; this only changes results if you passed `xmins`/`xmaxs`/`tmin`/`tmax` that differ from the axis extent.

## Behaviour changes

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
