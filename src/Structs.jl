# ==============================================================================
# --- Type aliases & global state ---
# ==============================================================================

"""
    ParamDict

Alias for `Dict{Symbol, Any}`: the flat, resolved parameters of a single simulation run.
Its normalised content is hashed to locate the run on disk (see [`calculate_hash`](@ref)).
"""
const ParamDict = Dict{Symbol, Any}

"""
    MethodDict

Alias for `Dict{Symbol, ParamDict}`: maps method names (e.g. `:upwind`) to their parameter overrides.
"""
const MethodDict = Dict{Symbol, ParamDict}

"""
    VariedDict

Alias for `Dict{Symbol, Vector}`: the parameter sweep. The pipeline runs the Cartesian product of
all vectors. Keys of the form `:Ns__x`/`:Ns__1` sweep a single entry of the tuple parameter `:Ns`.
"""
const VariedDict = Dict{Symbol, Vector}

"""
    StatRegistry

Alias for `Dict{Symbol, Union{Symbol, Vector{Symbol}}}`: maps a statistic name to the dimensions
it keeps (`:all`, `:space`, `:time`, a single dimension name, or a vector of dimension names).
"""
const StatRegistry = Dict{Symbol, Union{Symbol, Vector{Symbol}}}

# Initialised in `__init__` so that the precompile-time values are never baked into the image.
const _SAVE_ROOT_PATH = Ref{String}("")
const _TARGET_MODULE = Ref{Module}(Main)
const _ACTIVE_STAT_REGISTRY = Ref{StatRegistry}(StatRegistry())

"""
    set_target_module!(target_module::Module)

Sets the module in which solver, reference and post-processing functions are looked up by name.
"""
set_target_module!(target_module::Module) = (_TARGET_MODULE[] = target_module)

"""
    get_target_module()

Returns the module in which functions referenced by name are resolved (default: `Main`).
"""
get_target_module() = _TARGET_MODULE[]

# ==============================================================================
# --- Dictionary creators ---
# ==============================================================================

_to_paramdict(d) = ParamDict(Symbol(k) => v for (k, v) in d)

"""
    create_param_dict(kv...)

Builds a [`ParamDict`](@ref) from pairs or any key-value iterable; `String` keys become `Symbol`s.
"""
create_param_dict(kv::Pair...) = _to_paramdict(kv)
create_param_dict(kv) = _to_paramdict(kv)
create_param_dict() = ParamDict()

"""
    create_method_dict(kv...)

Builds a [`MethodDict`](@ref); nested dictionaries are converted to [`ParamDict`](@ref)s.
"""
create_method_dict(kv::Pair...) = MethodDict(Symbol(k) => _to_paramdict(v) for (k, v) in kv)
create_method_dict(kv) = MethodDict(Symbol(k) => _to_paramdict(v) for (k, v) in kv)
create_method_dict() = MethodDict()

"""
    create_varied_dict(kv...)

Builds a [`VariedDict`](@ref) for a parameter sweep. Ranges are collected into vectors.
"""
create_varied_dict(kv::Pair...) = _to_varieddict(kv)
create_varied_dict(kv) = _to_varieddict(kv)
create_varied_dict() = VariedDict()

function _to_varieddict(kv)
    out = VariedDict()
    for (k, v) in kv
        v isa AbstractVector || throw(ArgumentError(
            "the swept values of :$k must be a vector (or range), got $(typeof(v))"))
        out[Symbol(k)] = collect(v)
    end
    return out
end

# ==============================================================================
# --- Global statistic registry ---
# ==============================================================================

_registry_value(v::Symbol) = v
_registry_value(v::AbstractString) = Symbol(v)
_registry_value(v) = Symbol[Symbol(x) for x in v]

"""
    get_stat_registry()

Returns the active global statistic registry (a [`StatRegistry`](@ref)). Datasets copy it on creation
and [`calculate_all_stats!`](@ref) brings them back in sync with it.
"""
get_stat_registry() = _ACTIVE_STAT_REGISTRY[]

"""
    set_stat_registry!(new_registry::AbstractDict)

Replaces the global statistic registry with a copy of `new_registry`.
"""
function set_stat_registry!(new_registry::AbstractDict)
    _ACTIVE_STAT_REGISTRY[] = StatRegistry(Symbol(k) => _registry_value(v) for (k, v) in new_registry)
    @info "Global statistic registry has been overwritten."
    return get_stat_registry()
end

"""
    reset_stat_registry!()

Clears the global statistic registry.
"""
function reset_stat_registry!()
    empty!(_ACTIVE_STAT_REGISTRY[])
    @info "Global statistic registry has been cleared."
    return get_stat_registry()
end

"""
    register_stat!(name::Symbol, kept_dims)

Registers a statistic evaluated through a [`calc_stat`](@ref) overload. `kept_dims` lists the dimensions
that are *retained*; all others are integrated out. Valid values: `:all`, `:space`, `:time`, a single
dimension name (e.g. `:x`) or a vector of dimension names (e.g. `[:x, :t]`; `Symbol[]` gives a scalar).

`:Solution` is always registered (keeping all dimensions); registering it is a no-op.

```julia-repl
julia> register_stat!(:l2error, :time)
```
"""
function register_stat!(name::Symbol, kept_dims::Union{Symbol, AbstractString, AbstractVector})
    name === :Solution && return :all
    value = _registry_value(kept_dims)
    _ACTIVE_STAT_REGISTRY[][name] = value
    @info "Registered statistic :$name keeping dimensions: $value"
    return value
end

"""
    delete_stat!(name::Symbol)
    delete_stat!(data::AbstractSimData, name::Symbol)

Removes a statistic from the global registry, or removes a statistic (value and registry entry)
from a single dataset. `:Solution` cannot be removed.
"""
function delete_stat!(name::Symbol)
    delete!(_ACTIVE_STAT_REGISTRY[], name)
    return get_stat_registry()
end

# ==============================================================================
# --- Domain metadata ---
# ==============================================================================

"""
    DomainInfo{D, T <: Real}

Metadata describing the bounding box and dimension layout of a dataset.

# Fields
- `dim_keys::NTuple{D, Symbol}`: ordered axis identifiers, e.g. `(:x, :y, :t)`.
- `mins`, `maxs::SVector{D, T}`: bounding box of the domain.
- `spacing::SVector{D, T}`: mean node spacing per axis (used as integration measure).
- `time_dim::Union{Nothing, Symbol}`: the time axis (always the last one), or `nothing` for static data.
- `stat_registry::StatRegistry`: the dimensions kept by each statistic of this dataset.

The constructor validates the layout, copies the registry (datasets never share registries) and
always registers `:Solution => :all`.
"""
struct DomainInfo{D, T <: Real}
    dim_keys::NTuple{D, Symbol}
    mins::SVector{D, T}
    maxs::SVector{D, T}
    spacing::SVector{D, T}
    time_dim::Union{Nothing, Symbol}
    stat_registry::StatRegistry

    function DomainInfo{D, T}(dim_keys, mins, maxs, spacing, time_dim::Union{Nothing, Symbol},
                              stat_registry::AbstractDict = get_stat_registry()) where {D, T <: Real}
        length(dim_keys) == D || throw(ArgumentError("expected $D dimension keys, got $(length(dim_keys))"))
        dkeys = ntuple(i -> Symbol(dim_keys[i]), Val(D))
        allunique(dkeys) || throw(ArgumentError("dimension keys must be unique, got $dkeys"))
        if time_dim !== nothing && (D == 0 || dkeys[D] !== time_dim)
            throw(ArgumentError("the time dimension :$time_dim must be the last dimension key, got $dkeys"))
        end
        registry = StatRegistry(Symbol(k) => _registry_value(v) for (k, v) in stat_registry)
        registry[:Solution] = :all
        return new{D, T}(dkeys, SVector{D, T}(mins), SVector{D, T}(maxs), SVector{D, T}(spacing),
                         time_dim, registry)
    end
end

function DomainInfo(dim_keys, mins, maxs, spacing, time_dim::Union{Nothing, Symbol},
                    stat_registry::AbstractDict = get_stat_registry())
    T = float(promote_type(eltype(collect(mins)), eltype(collect(maxs)), eltype(collect(spacing))))
    return DomainInfo{length(dim_keys), T}(dim_keys, mins, maxs, spacing, time_dim, stat_registry)
end

"""
    has_time(domain::DomainInfo)
    has_time(data::AbstractSimData)

`true` if the dataset is transient (has a time axis).
"""
has_time(domain::DomainInfo) = domain.time_dim !== nothing

"""
    get_time_dim(domain::DomainInfo)

Index of the time axis within `domain.dim_keys`, or `nothing` for static data.
"""
get_time_dim(domain::DomainInfo) = domain.time_dim === nothing ? nothing :
                                   findfirst(==(domain.time_dim), domain.dim_keys)

# Number of spatial dimensions of a domain.
_nspace(domain::DomainInfo{D}) where {D} = has_time(domain) ? D - 1 : D
