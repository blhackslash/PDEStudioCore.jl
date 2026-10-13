# ==============================================================================
# --- Disk layout, saving & loading ---
# ==============================================================================
#
# data/<timestamp>_<sha256>.jld2
#   "raw"        the native dataset (ESimData or LSimData)
#   "native"     :eulerian | :lagrangian            (fast metadata)
#   "params"     normalised parameters              (fast metadata)
#   "stat_keys"  available statistics (incl. :Solution)
#   "registry"   kept dimensions of every statistic
#   "shape"      size of `u` (Eulerian only)
#   "evaluated_stats"  statistics already evaluated for this raw data (incl. dropped all-NaN ones)
#   "conv_*"     cached conversions ("conv_L", "conv_E_<n1>x<n2>...")

const _ENABLE_CONV_CACHE = Ref{Bool}(false)
# Entries tied to the content of "raw": never carried over when the raw data is rewritten.
const _MANAGED_KEYS = ("raw", "native", "params", "stat_keys", "registry", "shape", "evaluated_stats")

"""
    enable_cache!(val::Bool)

Toggles whether on-the-fly Eulerian/Lagrangian conversions are cached inside the data file.
"""
enable_cache!(val::Bool) = (_ENABLE_CONV_CACHE[] = val)

"""
    SimFileNotFoundError <: Exception

Thrown when no data file matches the requested parameters (or hash), or a requested key is missing.
"""
struct SimFileNotFoundError <: Exception
    message::String
end
Base.showerror(io::IO, e::SimFileNotFoundError) = print(io, "SimFileNotFoundError: ", e.message)

# Cache key of an Eulerian conversion at a given spacetime resolution.
get_conv_key(res::Tuple) = "conv_E_" * join(res, "x")

"""
    set_save_path!(path::AbstractString)

Sets the root directory for saved data and creates its `data` and `figures` subdirectories.
Returns `true` on success.
"""
function set_save_path!(path::AbstractString)
    abs_path = abspath(path)
    try
        mkpath(joinpath(abs_path, "data"))
        mkpath(joinpath(abs_path, "figures"))
    catch e
        @error "Failed to create the save directory $abs_path" exception = e
        return false
    end
    _SAVE_ROOT_PATH[] = abs_path
    @info "Module save root path set to: $abs_path"
    return true
end

"""
    get_save_path()

Returns the root directory used for saving (defaults to the working directory at load time).
"""
get_save_path() = _SAVE_ROOT_PATH[]

_data_dir() = joinpath(get_save_path(), "data")
_timestamp() = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS_sss")

const _HASH_REGEX = r"(?:^|_)([0-9a-f]{64})(?:_|\.jld2$)"

function _filename_hash(name::AbstractString)
    m = match(_HASH_REGEX, name)
    return m === nothing ? nothing : String(m.captures[1])
end

function _data_files()
    dir = _data_dir()
    isdir(dir) || return String[]
    return [joinpath(dir, f) for f in readdir(dir) if endswith(f, ".jld2")]
end

_sort_newest!(paths::Vector{String}) = sort!(paths, by = mtime, rev = true)
_files_with_hash(hash_val::AbstractString) = _sort_newest!(filter(p -> _filename_hash(basename(p)) == hash_val, _data_files()))
_param_hash(params::AbstractDict) = calculate_hash(_normalize_params(params))

function _find_file(params::AbstractDict)
    files = _files_with_hash(_param_hash(params))
    return isempty(files) ? nothing : first(files)
end

"""
    get_file_name(params::AbstractDict)

Returns the path of the newest `.jld2` file whose hash matches the (normalised) parameters.
Throws a [`SimFileNotFoundError`](@ref) if there is none.
"""
function get_file_name(params::AbstractDict)
    isdir(_data_dir()) || throw(SimFileNotFoundError("Data directory $(_data_dir()) does not exist."))
    path = _find_file(params)
    path === nothing && throw(SimFileNotFoundError("No file with matching parameters found in $(_data_dir())."))
    return path
end

function _files_matching(prefix::AbstractString)
    clean = lowercase(replace(basename(String(prefix)), r"\.jld2$" => ""))
    isempty(clean) && return String[]
    matches = filter(_data_files()) do path
        h = _filename_hash(basename(path))
        return lowercase(replace(basename(path), r"\.jld2$" => "")) == clean || (h !== nothing && startswith(h, clean))
    end
    return _sort_newest!(matches)
end

_has_key(path::AbstractString, key::AbstractString) = jldopen(file -> haskey(file, key), path, "r")

function _load_key(path::AbstractString, key::AbstractString)
    return jldopen(path, "r") do file
        haskey(file, key) || throw(SimFileNotFoundError("Key '$key' not found in $(basename(path))."))
        return file[key]
    end
end

function _append_key!(path::AbstractString, key::AbstractString, value)
    jldopen(path, "a+") do file
        haskey(file, key) || (file[key] = value)
    end
    return nothing
end

function _set_key!(path::AbstractString, key::AbstractString, value)
    jldopen(path, "a+") do file
        if haskey(file, key)
            isequal(file[key], value) && return nothing  # unchanged: avoid growing the file
            delete!(file, key)
        end
        file[key] = value
    end
    return nothing
end

# "shape" identifies the native representation: the size of `u` (Eulerian) or
# (time steps, total particle count) (Lagrangian).
_data_kind(::ESimData) = :eulerian
_data_kind(::LSimData) = :lagrangian
_data_shape(data::ESimData) = size(data.u)
_data_shape(data::LSimData) = (length(data.t), sum(length, data.x; init = 0))

function _write_metadata!(file, data::AbstractSimData)
    file["native"] = _data_kind(data)
    file["params"] = data.params
    file["stat_keys"] = collect(keys(data.stats))
    file["registry"] = data.domain.stat_registry
    file["shape"] = collect(_data_shape(data))
    return nothing
end

# Copies an entry (recursively for groups) from one open JLD2 file to another.
function _copy_entry!(dest, src, key::AbstractString)
    value = src[key]
    if value isa JLD2.Group
        for sub in keys(value)
            _copy_entry!(dest, src, string(key, "/", sub))
        end
    else
        dest[key] = value
    end
    return nothing
end

# Writes a complete file to a temporary path and renames it into place, so an interrupted write never
# corrupts existing data and rewrites do not grow the file. Entries of `previous` that are not managed
# here (and are not stale conversion caches) are carried over.
function _write_sim_file(path::AbstractString, data::AbstractSimData, metadata::AbstractDict,
                         previous::Union{Nothing, AbstractString})
    tmp = path * ".tmp"
    try
        jldopen(tmp, "w") do file
            file["raw"] = data
            _write_metadata!(file, data)
            for (k, v) in metadata
                file[String(k)] = v
            end
            if previous !== nothing && isfile(previous)
                jldopen(previous, "r") do old
                    for k in keys(old)
                        (k in _MANAGED_KEYS || startswith(k, "conv_") || haskey(file, k)) && continue
                        try
                            _copy_entry!(file, old, k)
                        catch e
                            @warn "Could not carry over entry '$k' of $(basename(previous))" exception = e
                        end
                    end
                end
            end
        end
        mv(tmp, path; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return path
end

"""
    save_sim_data(sim_data::AbstractSimData; overwrite = false, verbose = true, metadata = Dict())

Saves the dataset under the hash of its (normalised) parameters and returns the file path.
An existing file is only replaced when `overwrite = true` (which also drops stale conversion caches).
Writes are atomic: data is written to a temporary file that is then renamed into place.
Extra `metadata` entries are stored as top-level keys of the file.
"""
function save_sim_data(sim_data::AbstractSimData; overwrite::Bool = false, verbose::Bool = true,
                       metadata::AbstractDict = Dict{String, Any}())
    for k in keys(metadata)
        key = string(k)
        (key in _MANAGED_KEYS || startswith(key, "conv_")) && throw(ArgumentError(
            "metadata key '$key' is reserved for the data file layout; choose another name"))
    end
    normalized = _normalize_params(sim_data.params)
    empty!(sim_data.params)
    merge!(sim_data.params, normalized)

    hash_val = calculate_hash(sim_data.params)
    existing = _files_with_hash(hash_val)
    if isempty(existing)
        mkpath(_data_dir())
        path = joinpath(_data_dir(), "$(_timestamp())_$(hash_val).jld2")
        _write_sim_file(path, sim_data, metadata, nothing)
        verbose && @info "Saved simulation data to $(basename(path))"
        return path
    end
    path = first(existing)
    if overwrite || !_has_key(path, "raw")
        _write_sim_file(path, sim_data, metadata, path)
        verbose && @info "Saved simulation data to $(basename(path)) (replaced; conversion caches cleared)"
    else
        @debug "'raw' already exists in $(basename(path)); skipping save."
    end
    return path
end
save_sim_data(::NoSimData; kwargs...) = nothing

"""
    load_sim_data(params)
    load_sim_data(params, Val(:raw))
    load_sim_data(params, Val(:eulerian), res)
    load_sim_data(params, Val(:lagrangian))
    load_sim_data(params, kind::Symbol, args...)

Loads the dataset matching `params`. `:raw` returns the native data; `:eulerian` returns a grid at the
spacetime resolution `res` (resampled or scattered from particles if needed); `:lagrangian` returns
particle data (grid nodes become stationary particles). Conversions are computed on the fly and
cached in the file when [`enable_cache!`](@ref) is on.
"""
load_sim_data(params::AbstractDict) = load_sim_data(params, Val(:raw))
load_sim_data(params::AbstractDict, kind::Symbol, args...) = load_sim_data(params, Val(kind), args...)
load_sim_data(params::AbstractDict, ::Val{:raw}) = _load_key(get_file_name(params), "raw")

function _native_and_shape(path::AbstractString)
    return jldopen(path, "r") do file
        (haskey(file, "native") && haskey(file, "shape")) || throw(ArgumentError(
            "$(basename(path)) was written by an older PDEStudioCore version (no \"native\"/\"shape\" metadata). " *
            "Run its configuration again (`run_all_simulations` recomputes outdated files) or delete it."))
        return file["native"], Tuple(file["shape"])
    end
end

# Whether `data` has the representation stored as "raw" in `path` (and is not, e.g., a conversion of it).
function _is_native_representation(path::AbstractString, data::AbstractSimData)
    native, shape = _native_and_shape(path)
    native === _data_kind(data) || return false
    return shape == _data_shape(data)
end

function load_sim_data(params::AbstractDict, ::Val{:eulerian}, res::Tuple)
    path = get_file_name(params)
    native, shape = _native_and_shape(path)
    native === :eulerian && shape == Tuple(res) && return _load_key(path, "raw")
    key = get_conv_key(res)
    _has_key(path, key) && return _load_key(path, key)

    raw = _load_key(path, "raw")
    @info "Eulerian representation $key not cached; generating it on the fly..."
    conv = raw isa LSimData ? convert_to_eulerian(raw, res) : resample_eulerian(raw, res)
    _ENABLE_CONV_CACHE[] && _append_key!(path, key, conv)
    return conv
end

function load_sim_data(params::AbstractDict, ::Val{:lagrangian})
    path = get_file_name(params)
    native, _ = _native_and_shape(path)
    native === :lagrangian && return _load_key(path, "raw")
    _has_key(path, "conv_L") && return _load_key(path, "conv_L")

    @info "Lagrangian representation not cached; generating it on the fly..."
    conv = convert_to_lagrangian(_load_key(path, "raw"))
    _ENABLE_CONV_CACHE[] && _append_key!(path, "conv_L", conv)
    return conv
end

"""
    load_sim_data(hash_prefix::AbstractString; index = 1, key = nothing)

Loads a file by (a prefix of) its parameter hash or by its file name, newest first. Returns the
`Dict` of all stored entries, or only the entry `key` (e.g. `"raw"`) when given.
"""
function load_sim_data(hash_prefix::AbstractString; index::Int = 1, key::Union{Nothing, AbstractString} = nothing)
    isdir(_data_dir()) || throw(SimFileNotFoundError("Data directory $(_data_dir()) does not exist."))
    files = _files_matching(hash_prefix)
    isempty(files) && throw(SimFileNotFoundError("No files found matching the hash prefix: $hash_prefix"))
    1 <= index <= length(files) || throw(ArgumentError(
        "requested index $index, but only $(length(files)) file(s) match '$hash_prefix'"))
    path = files[index]
    @info "Manual load: $(length(files)) matching file(s); loading index $index: $(basename(path))"
    return key === nothing ? JLD2.load(path) : _load_key(path, String(key))
end

"""
    list_available_conversions(params::AbstractDict)

Returns the conversion keys cached in the matching file (e.g. `["conv_L", "conv_E_100x50"]`).
"""
function list_available_conversions(params::AbstractDict)
    return jldopen(file -> filter(k -> startswith(k, "conv_"), collect(keys(file))), get_file_name(params), "r")
end

"""
    does_sim_data_exist(params)
    does_sim_data_exist(params, Val(:raw))
    does_sim_data_exist(params, Val(:eulerian), res)
    does_sim_data_exist(params, Val(:lagrangian))

Checks (without loading the data) whether the native dataset, or a representation that can be
loaded without recomputation, exists on disk.
"""
does_sim_data_exist(params::AbstractDict) = does_sim_data_exist(params, Val(:raw))
does_sim_data_exist(params::AbstractDict, kind::Symbol, args...) = does_sim_data_exist(params, Val(kind), args...)

# A file of this version: native data plus the metadata loading relies on. Files written by earlier versions
# count as missing, so the pipeline recomputes (and replaces) them instead of failing on them.
function _is_current_file(path::AbstractString)
    ok, has_raw = jldopen(f -> (haskey(f, "raw") && haskey(f, "native") && haskey(f, "shape"), haskey(f, "raw")), path, "r")
    has_raw && !ok && @info "$(basename(path)) was written by an older PDEStudioCore version; it is recomputed."
    return ok
end

function does_sim_data_exist(params::AbstractDict, ::Val{:raw})
    path = _find_file(params)
    return path !== nothing && _is_current_file(path)
end

function does_sim_data_exist(params::AbstractDict, ::Val{:eulerian}, res::Tuple)
    path = _find_file(params)
    (path === nothing || !_is_current_file(path)) && return false
    _has_key(path, get_conv_key(res)) && return true
    native, shape = _native_and_shape(path)
    native === :eulerian || return false
    return shape == Tuple(res)
end

function does_sim_data_exist(params::AbstractDict, ::Val{:lagrangian})
    path = _find_file(params)
    (path === nothing || !_is_current_file(path)) && return false
    native, _ = _native_and_shape(path)
    return _has_key(path, native === :lagrangian ? "raw" : "conv_L")
end
