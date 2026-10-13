# ==============================================================================
# --- Simulation data containers ---
# ==============================================================================

"""
    AbstractSimData{D, DS, M, T <: Real}

Root type of all simulation datasets.
- `D`: total (spacetime) dimensions, `DS`: spatial dimensions (`D == DS` for static data),
- `M`: number of field components, `T`: floating point precision.
"""
abstract type AbstractSimData{D, DS, M, T <: Real} end

"""
    NoSimData()

Placeholder returned by solvers or the pipeline when no data was produced (or the run was skipped).
"""
struct NoSimData <: AbstractSimData{0, 0, 0, Real} end

const AbstractStatTensor{M, T} = Union{
    SVector{M, T},
    AbstractArray{SVector{M, T}},
    Vector{Vector{SVector{M, T}}}
}

"""
    StatDict{M, T}

Alias for `Dict{Symbol, AbstractStatTensor{M, T}}`, the storage type of derived statistics.
"""
const StatDict{M, T} = Dict{Symbol, AbstractStatTensor{M, T}}

function _check_layout(D::Int, DS::Int, domain::DomainInfo)
    expected = has_time(domain) ? D - 1 : D
    DS == expected || throw(ArgumentError(
        "DS = $DS spatial dimensions is inconsistent with D = $D and time_dim = $(repr(domain.time_dim))"))
    return nothing
end

# Derived statistics are stored without `:Solution`, which always lives in the `u` field.
function _derived_stats(::Type{SD}, stats::AbstractDict) where {SD <: AbstractDict}
    out = SD()
    for (k, v) in stats
        Symbol(k) === :Solution && continue
        out[Symbol(k)] = v
    end
    return out
end

"""
    ESimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}

Eulerian data: fields on a structured tensor grid (time, if present, is the last axis).

# Fields
- `params::ParamDict`: the parameters that produced the data (copied on construction).
- `domain::DomainInfo{D, T}`: bounding box, dimension names and statistic registry.
- `axes::NTuple{D, Vector{T}}`: coordinate vector of every axis.
- `u::Array{SVector{M, T}, D}`: the solution tensor.
- `stats`: a [`StatsView`](@ref) of all statistics. `:Solution` is always present and *is* `u`;
  derived statistics are stored alongside it.

Prefer [`create_sim_data`](@ref) for construction; the inner constructor validates that the axes
match the solution and that `DS` matches the domain.
"""
mutable struct ESimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}
    params::ParamDict
    domain::DomainInfo{D, T}
    axes::NTuple{D, Vector{T}}
    u::Array{SVector{M, T}, D}
    stats::StatDict{M, T}

    function ESimData{D, DS, M, T}(params::AbstractDict, domain::DomainInfo{D, T}, axes, u::AbstractArray,
                                   stats::AbstractDict = StatDict{M, T}()) where {D, DS, M, T}
        _check_layout(D, DS, domain)
        length(axes) == D || throw(DimensionMismatch("expected $D coordinate axes, got $(length(axes))"))
        ax = ntuple(d -> convert(Vector{T}, axes[d]), Val(D))
        sol = convert(Array{SVector{M, T}, D}, u)
        size(sol) == map(length, ax) || throw(DimensionMismatch(
            "the solution has size $(size(sol)) but the axes $(domain.dim_keys) have lengths $(map(length, ax))"))
        return new{D, DS, M, T}(_to_paramdict(params), domain, ax, sol, _derived_stats(StatDict{M, T}, stats))
    end
end

function ESimData(params::AbstractDict, domain::DomainInfo{D, T}, axes, u::AbstractArray{SVector{M, S}, D},
                  stats::AbstractDict = StatDict{M, T}()) where {D, T, M, S}
    return ESimData{D, _nspace(domain), M, T}(params, domain, axes, u, stats)
end

"""
    LSimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}

Lagrangian data: particle positions and states recorded at discrete times.

# Fields
- `params::ParamDict`: the parameters that produced the data (copied on construction).
- `domain::DomainInfo{D, T}`: bounding box, dimension names and statistic registry.
- `t::Vector{T}`: recording times (a single `0` for static data).
- `x::Vector{Vector{SVector{DS, T}}}`: particle positions per time step.
- `u::Vector{Vector{SVector{M, T}}}`: particle states per time step.
- `stats`: a [`StatsView`](@ref); `:Solution` is always present and *is* `u`.
"""
mutable struct LSimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}
    params::ParamDict
    domain::DomainInfo{D, T}
    t::Vector{T}
    x::Vector{Vector{SVector{DS, T}}}
    u::Vector{Vector{SVector{M, T}}}
    stats::StatDict{M, T}

    function LSimData{D, DS, M, T}(params::AbstractDict, domain::DomainInfo{D, T}, t::AbstractVector,
                                   x::AbstractVector, u::AbstractVector,
                                   stats::AbstractDict = StatDict{M, T}()) where {D, DS, M, T}
        _check_layout(D, DS, domain)
        tt = convert(Vector{T}, t)
        xx = convert(Vector{Vector{SVector{DS, T}}}, x)
        uu = convert(Vector{Vector{SVector{M, T}}}, u)
        nt = length(tt)
        (length(xx) == nt && length(uu) == nt) || throw(DimensionMismatch(
            "t, x and u need one entry per time step (got $nt, $(length(xx)) and $(length(uu)))"))
        if D == DS && nt != 1
            throw(DimensionMismatch("static (time-independent) data must contain exactly one snapshot, got $nt"))
        end
        for i in 1:nt
            length(xx[i]) == length(uu[i]) || throw(DimensionMismatch(
                "time step $i has $(length(xx[i])) particle positions but $(length(uu[i])) particle states"))
        end
        return new{D, DS, M, T}(_to_paramdict(params), domain, tt, xx, uu, _derived_stats(StatDict{M, T}, stats))
    end
end

function LSimData(params::AbstractDict, domain::DomainInfo{D, T}, t::AbstractVector,
                  x::AbstractVector{<:AbstractVector{SVector{DS, S1}}},
                  u::AbstractVector{<:AbstractVector{SVector{M, S2}}},
                  stats::AbstractDict = StatDict{M, T}()) where {D, T, DS, M, S1, S2}
    return LSimData{D, DS, M, T}(params, domain, t, x, u, stats)
end

has_time(data::AbstractSimData) = has_time(data.domain)

# ==============================================================================
# --- Struct-level coupling of `:Solution` ---
# ==============================================================================

"""
    StatsView{M, T, S} <: AbstractDict{Symbol, AbstractStatTensor{M, T}}

Dictionary view returned by `data.stats`. It behaves like a `Dict`, but `:Solution` is always present
and always refers to `data.u` (reading, writing and iterating it go through the solution field), so
the solution can never be missing from or out of sync with the statistics. All other entries are
the dataset's derived statistics. `copy(data.stats)` returns a plain `Dict` snapshot.
"""
struct StatsView{M, T, S <: AbstractSimData} <: AbstractDict{Symbol, AbstractStatTensor{M, T}}
    data::S
end
StatsView(data::AbstractSimData{D, DS, M, T}) where {D, DS, M, T} = StatsView{M, T, typeof(data)}(data)

@inline _raw_stats(data::AbstractSimData) = getfield(data, :stats)
@inline _solution(data::AbstractSimData) = getfield(data, :u)

Base.length(v::StatsView) = 1 + length(_raw_stats(v.data))
Base.haskey(v::StatsView, key) = key === :Solution || haskey(_raw_stats(v.data), key)
Base.get(v::StatsView, key, default) = key === :Solution ? _solution(v.data) : get(_raw_stats(v.data), key, default)
Base.get(f::Base.Callable, v::StatsView, key) = key === :Solution ? _solution(v.data) : get(f, _raw_stats(v.data), key)
Base.getindex(v::StatsView, key) = key === :Solution ? _solution(v.data) : _raw_stats(v.data)[key]

Base.iterate(v::StatsView) = ((:Solution => _solution(v.data)), nothing)
function Base.iterate(v::StatsView, state)
    raw = _raw_stats(v.data)
    return state === nothing ? iterate(raw) : iterate(raw, state)
end

# Derived statistics are stored as given (numbers are converted to SVectors); use `add_stat!`
# to also record the dimensions a statistic keeps.
function Base.setindex!(v::StatsView{M, T}, value, key::Symbol) where {M, T}
    if key === :Solution
        _set_solution!(v.data, value)
    else
        _raw_stats(v.data)[key] = _stat_value(value, Val(M), T)
    end
    return v
end

function Base.delete!(v::StatsView, key)
    key === :Solution && throw(ArgumentError(":Solution is the dataset's `u` field and cannot be removed."))
    delete!(_raw_stats(v.data), key)
    return v
end

Base.pop!(v::StatsView, key) = (value = v[key]; delete!(v, key); value)
Base.pop!(v::StatsView, key, default) = haskey(v, key) ? pop!(v, key) : default
Base.empty!(v::StatsView) = (empty!(_raw_stats(v.data)); v)

# The checks are skipped while an object is still being reconstructed field by field
# (deserialisers create uninitialised objects and then assign the fields).
function _set_solution!(data::ESimData{D, DS, M, T}, value) where {D, DS, M, T}
    sol = convert(Array{SVector{M, T}, D}, value)
    if isdefined(data, :u) && size(sol) != size(getfield(data, :u))
        throw(DimensionMismatch("the new solution has size $(size(sol)), expected $(size(getfield(data, :u)))"))
    end
    setfield!(data, :u, sol)
    return data
end

function _set_solution!(data::LSimData{D, DS, M, T}, value) where {D, DS, M, T}
    sol = convert(Vector{Vector{SVector{M, T}}}, value)
    if isdefined(data, :x) && map(length, sol) != map(length, getfield(data, :x))
        throw(DimensionMismatch("the particle counts of the new solution do not match the particle positions"))
    end
    setfield!(data, :u, sol)
    return data
end

const _SimDataTypes = Union{ESimData, LSimData}

@inline function Base.getproperty(data::_SimDataTypes, name::Symbol)
    name === :stats && return StatsView(data)
    return getfield(data, name)
end

function Base.setproperty!(data::_SimDataTypes, name::Symbol, value)
    if name === :stats
        # Convert everything first, so a failing entry leaves the dataset unchanged.
        raw = fieldtype(typeof(data), :stats)()
        solution = nothing
        for (k, v) in value
            Symbol(k) === :Solution ? (solution = v) : (raw[Symbol(k)] = v)
        end
        solution === nothing || _set_solution!(data, solution)
        setfield!(data, :stats, raw)
    elseif name === :u
        _set_solution!(data, value)
    else
        setfield!(data, name, convert(fieldtype(typeof(data), name), value))
    end
    return value
end
