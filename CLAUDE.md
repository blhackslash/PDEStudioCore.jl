# PDEStudioCore.jl

Headless backend for running, storing and analysing PDE simulations: Eulerian
(`ESimData`) and Lagrangian (`LSimData`) datasets of any dimension, `SimulationConfig`
parameter sweeps, deterministic SHA-256 hashing with atomic JLD2 caching, conversions
between representations, and a multithreaded statistics pipeline. Must run without a
display (clusters, CI); no Makie or GUI dependencies here.

## Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                 # full suite
julia --project=test -e 'using Pkg; Pkg.instantiate()'      # once
julia -t 8 --project=test test/runtests.jl                  # same suite, faster to iterate
julia --project=docs docs/make.jl                           # docs (instantiate docs env first)
```

The suite also runs `examples/advection_1d.jl` and `examples/dummy.jl`; keep them working
and quick.

## Layout

| Path | Contents |
|:-----|:---------|
| `src/Structs.jl`, `src/SimData.jl` | Types: `ParamDict`, `SimulationConfig`, `ESimData`/`LSimData`/`NoSimData`, `DomainInfo`, `StatsView` |
| `src/Config.jl`, `src/Constructors.jl` | `SimulationConfig` behaviour, `create_*` constructors |
| `src/Serialization.jl` | `val2str`/`str2val`, parameter normalisation, `calculate_hash` |
| `src/IOUtils.jl`, `src/DataManagement.jl` | Save/load, atomic writes, cache lookup, rehash/delete |
| `src/ConversionUtils.jl` | Eulerian ↔ Lagrangian conversion, resampling |
| `src/StatCalculation.jl` | Stat registry, `calc_stat` evaluation, references |
| `src/Simulations.jl` | Task generation, `run_all_simulations`, `run_smart_simulation` |
| `presets/` | Stat presets (`hyperbolic.jl`), loaded with `set_stat_preset!` |

## Invariants (treat changes as breaking)

- **Hash stability.** Cache file names are `calculate_hash` of the normalised parameters,
  which depends on `val2str` and `_normalize_params`. Any change there renames every
  user's cached result. Don't change it unless asked; if you must, say so prominently in
  the PR and mention `rehash_sim_data`. `test/test_serialization.jl` pins the formats:
  never "fix" a failing hash test by updating the expected value without asking.
- **`str2val` never evaluates code.** It's a literal parser (CSV/config values come from
  files). Don't add `eval`, `Meta.parse` + `eval`, or `include` to it.
- **Atomic writes.** Data is written to a temporary file and renamed into place; keep it
  that way for every new writer.
- **Name resolution.** Solvers, references and post-processing functions are referenced
  by name and resolved in the module set with `set_target_module!` (default `Main`;
  HyperCloud sets itself). Don't replace that with function objects in stored configs.
- **Stat names and presets** (`:l1error`, `:mass`, … in `presets/hyperbolic.jl`) are used
  downstream (HyperCloud, HyperCloudCore tests, PDEStudio); don't rename them.
- `data.stats[:Solution]` is always the solution itself.

## Conventions

- Requires Julia 1.12.
- Breaking releases have no deprecated aliases (see `docs/src/migration.md`). If a change
  the user asked for is breaking, add it to the migration guide.
- New statistics: `register_stat!(name, kept_dims)` plus a `calc_stat(::Val{name}, …)`
  method; test them on a case with a known exact value.
- Threaded code uses Polyester `@batch` / `Threads`; results must not depend on the thread
  count (test with `-t 1` and `-t 8` when touching it).
- Tests use `set_save_path!(mktempdir())`; never write into the repo or the user's data.
- Every export needs a docstring that appears in the docs (`checkdocs = :exports`).
- Commit subjects: short and imperative, describing the behaviour change
  (e.g. `Hash enum parameters by value`).
