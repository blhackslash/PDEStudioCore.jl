# ==============================================================================
# --- Data health checks, deletion & rehashing ---
# ==============================================================================

function _aggregate_component(iterator)
    min_v, max_v = Inf, -Inf
    sum_v = 0.0
    valid_count, nan_count = 0, 0
    for val in iterator
        if isnan(val)
            nan_count += 1
        else
            min_v = min(min_v, val)
            max_v = max(max_v, val)
            sum_v += val
            valid_count += 1
        end
    end
    valid_count == 0 && return (NaN, NaN, NaN, Float64(nan_count))
    return (Float64(min_v), Float64(max_v), sum_v / valid_count, Float64(nan_count))
end

_component_summary(values, M::Int) =
    reduce(vcat, [collect(_aggregate_component(v[c] for v in values)) for c in 1:M])

function _health_frame(times::AbstractVector, rows::Vector{Vector{Float64}}, M::Int)
    columns = Pair{Symbol, Vector{Float64}}[:Time => Float64.(collect(times))]
    names = [Symbol("C$(c)_$(s)") for c in 1:M for s in ("Min", "Max", "Mean", "NaNs")]
    for (j, name) in enumerate(names)
        push!(columns, name => Float64[row[j] for row in rows])
    end
    return DataFrame(columns...)
end

"""
    check_data(data::ESimData; per_step = false)
    check_data(data::LSimData)

Summarises the numerical health of a dataset as a `DataFrame` with the columns `Time`, then
`C<c>_Min`, `C<c>_Max`, `C<c>_Mean`, `C<c>_NaNs` for every component `c`. Eulerian data is
aggregated into a single row (`Time = NaN`) unless `per_step = true`; Lagrangian data gets one row
per time step.
"""
function check_data(data::ESimData{D, DS, M, T}; per_step::Bool = false) where {D, DS, M, T}
    if per_step && has_time(data)
        rows = [_component_summary(selectdim(data.u, D, n), M) for n in 1:size(data.u, D)]
        return _health_frame(data.axes[D], rows, M)
    end
    return _health_frame([NaN], [_component_summary(data.u, M)], M)
end

function check_data(data::LSimData{D, DS, M, T}) where {D, DS, M, T}
    rows = Vector{Float64}[_component_summary(step, M) for step in data.u]
    return _health_frame(data.t, rows, M)
end

# Parameters stored in a data file (the lightweight "params" entry), or `nothing` if they cannot be read.
function _file_params(path::AbstractString)
    try
        return jldopen(f -> haskey(f, "params") ? f["params"] : nothing, path, "r")
    catch e
        @warn "Could not read the parameters of $(basename(path))" exception = e
        return nothing
    end
end

# `path => params` of every data file (newest first) whose stored parameters contain all pairs of
# `match_params` (keys may be strings or symbols; values are normalised like stored parameters).
function _files_with_params(match_params::AbstractDict)
    wanted = Dict(Symbol(k) => _normalize_param(v) for (k, v) in match_params)
    matched = Pair{String, Any}[]
    for path in _sort_newest!(_data_files())
        file_params = _file_params(path)
        file_params === nothing && continue
        all(((k, v),) -> haskey(file_params, k) && isequal(file_params[k], v), wanted) || continue
        push!(matched, path => file_params)
    end
    return matched
end

_param_summary(params, skip) = join(("$k = $(val2str(v))" for (k, v) in sort(collect(params), by = p -> string(first(p)))
                                     if !(Symbol(k) in skip)), ", ")

"""
    find_sim_data(match_params = Dict(); io = stdout, verbose = true)

Lists the data files whose stored parameters contain all the given key-value pairs, i.e. any subset of a
run's parameters (keys may be strings or symbols; an empty filter lists every file). Only the lightweight
`params` metadata is read. Prints one line per file (newest first) with the parameters that were not part
of the filter, and returns the file paths.

```julia-repl
julia> find_sim_data(Dict("N" => 100));
2 data files with N = 100:
  2026-10-21_17-45-00_123_3f9a….jld2   cfl = 0.5, scheme = upwind, sim_func_name = :advection_solver_1d
  2026-10-21_17-44-58_907_b02c….jld2   cfl = 0.5, scheme = lax_friedrichs, sim_func_name = :advection_solver_1d
```
"""
function find_sim_data(match_params::AbstractDict = Dict{Symbol, Any}(); io::IO = stdout, verbose::Bool = true)
    matched = _files_with_params(match_params)
    paths = String[first(m) for m in matched]
    verbose || return paths
    n = length(paths)
    filter_str = isempty(match_params) ? "" : " with " * _param_summary(match_params, Set{Symbol}())
    println(io, n, " data file", n == 1 ? "" : "s", filter_str, n == 0 ? "." : ":")
    skip = Set(Symbol(k) for k in keys(match_params))
    for (path, params) in matched
        extra = _param_summary(params, skip)
        println(io, "  ", basename(path), isempty(extra) ? "" : "   " * extra)
    end
    return paths
end

"""
    delete_sim_data(match_params::AbstractDict; dry_run = false)

Deletes every data file whose stored parameters contain all the given key-value pairs (the files
[`find_sim_data`](@ref) lists). Returns the paths of the (would-be) deleted files.
An empty filter is rejected because it would match every file.

```julia-repl
julia> delete_sim_data(Dict(:N => 100, :scheme => "upwind"))
```
"""
function delete_sim_data(match_params::AbstractDict; dry_run::Bool = false)
    isempty(match_params) && throw(ArgumentError("refusing to delete with an empty filter (it would match every file)"))
    matched = String[]
    for (path, _) in _files_with_params(match_params)
        push!(matched, path)
        if dry_run
            @info "Would delete $(basename(path))"
        else
            @warn "Deleting file matching criteria: $(basename(path))"
            rm(path)
        end
    end
    return matched
end

"""
    rehash_sim_data(target_path; delete_old = true, filter_pairs = nothing, remove_keys = Symbol[], prompt_keys = Symbol[])

Re-normalises the parameters of a `.jld2` file (or of every `.jld2` file below a directory),
recomputes the hash and stores the data under its new name. Use it after renaming or removing
parameters, or after changes to the hashing.

# Keywords
- `delete_old`: remove the old file once the new one has been written.
- `filter_pairs`: only process files whose parameters contain these exact pairs.
- `remove_keys`: parameter keys to drop before rehashing.
- `prompt_keys`: interactively ask (in the REPL) for new values of these keys for every file.

Returns a vector of `old_path => new_path` pairs.
"""
function rehash_sim_data(target_path::AbstractString; delete_old::Bool = true, filter_pairs = nothing,
                         remove_keys::AbstractVector = Symbol[], prompt_keys::AbstractVector = Symbol[])
    paths = if isfile(target_path) && endswith(target_path, ".jld2")
        [String(target_path)]
    elseif isdir(target_path)
        [joinpath(root, f) for (root, _, files) in walkdir(target_path) for f in files if endswith(f, ".jld2")]
    else
        @warn "Path is neither a .jld2 file nor a directory: $target_path"
        String[]
    end
    renamed = Pair{String, String}[]
    for path in paths
        result = _rehash_single_file(path, delete_old, filter_pairs, Symbol.(remove_keys), Symbol.(prompt_keys))
        result === nothing || push!(renamed, result)
    end
    return renamed
end

function _prompt_param_values!(params::ParamDict, prompt_keys::Vector{Symbol}, name::AbstractString)
    println("\n--- Current parameters of $name ---")
    for (k, v) in sort(collect(params), by = x -> string(x[1]))
        println("  $k => $(val2str(v))")
    end
    println("-" ^ 40)
    for key in prompt_keys
        print("Enter value for '$key' (Julia expression, leave blank to skip): ")
        input = strip(readline())
        if isempty(input)
            @info "  Skipped '$key'."
            continue
        end
        try
            # Interactive input is trusted: evaluate it in the user's target module.
            params[key] = Core.eval(get_target_module(), Meta.parse(input))
            @info "  Set $key = $(params[key])"
        catch e
            @warn "  Failed to evaluate '$input'; '$key' is unchanged." exception = e
        end
    end
    return params
end

function _rehash_single_file(path::String, delete_old::Bool, filter_pairs, remove_keys::Vector{Symbol},
                             prompt_keys::Vector{Symbol})
    raw = try
        jldopen(f -> haskey(f, "raw") ? f["raw"] : nothing, path, "r")
    catch e
        @warn "Failed to open $(basename(path))" exception = e
        nothing
    end
    raw isa Union{ESimData, LSimData} || return nothing
    params = raw.params

    if filter_pairs !== nothing
        all(((k, v),) -> haskey(params, Symbol(k)) && isequal(params[Symbol(k)], _normalize_param(v)), filter_pairs) ||
            return nothing
    end
    bname = basename(path)
    @info "Processing matched file: $bname"

    for key in remove_keys
        haskey(params, key) || continue
        delete!(params, key)
        @info "  Removed key :$key"
    end
    isempty(prompt_keys) || _prompt_param_values!(params, prompt_keys, bname)

    normalized = _normalize_params(params)
    empty!(params)
    merge!(params, normalized)

    m = match(r"^(.+)_[0-9a-fA-F]{64}\.jld2$", bname)
    stamp = m === nothing ? _timestamp() : m.captures[1]
    new_path = joinpath(dirname(path), "$(stamp)_$(calculate_hash(params)).jld2")
    same_file = abspath(new_path) == abspath(path)
    if !same_file && isfile(new_path)
        @warn "$(basename(new_path)) already exists and is replaced by the rehashed $bname."
    end

    try
        _write_sim_file(new_path, raw, Dict{String, Any}(), path)
    catch e
        @error "Failed to write $(basename(new_path))" exception = e
        return nothing
    end

    if same_file
        @info "Rehashed in place (hash unchanged): $bname"
    elseif delete_old
        rm(path; force = true)
        @info "Rehashed and renamed: $bname -> $(basename(new_path))"
    else
        @info "Rehashed copy (old file kept): $bname -> $(basename(new_path))"
    end
    return path => new_path
end
