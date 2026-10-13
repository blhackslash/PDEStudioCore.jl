# ==============================================================================
# --- Statistics: registry resolution, custom stats & the evaluation pipeline ---
# ==============================================================================

"""
    set_stat_preset!(name::AbstractString)

Loads a predefined set of statistics (registrations and `calc_stat` overloads) from `presets/<name>.jl`.

```julia-repl
julia> set_stat_preset!("hyperbolic")
```
"""
function set_stat_preset!(name::AbstractString)
    preset_path = joinpath(@__DIR__, "..", "presets", "$name.jl")
    if !isfile(preset_path)
        @error "Preset file not found: $preset_path"
        return false
    end
    Base.include(@__MODULE__, preset_path)
    @info "Successfully loaded stat preset: $name"
    return true
end

function _kept_dims(reg_val, domain::DomainInfo)
    if reg_val === :all
        return collect(domain.dim_keys)
    elseif reg_val === :space
        return Symbol[d for d in domain.dim_keys if d !== domain.time_dim]
    elseif reg_val === :time
        return domain.time_dim === nothing ? Symbol[] : Symbol[domain.time_dim]
    elseif reg_val isa Symbol  # a single dimension name
        return reg_val in domain.dim_keys ? Symbol[reg_val] : Symbol[]
    elseif reg_val isa AbstractVector
        return Symbol[d for d in reg_val if d in domain.dim_keys]
    end
    return Symbol[]
end

_kept_indices(kept::AbstractVector{Symbol}, domain::DomainInfo) =
    sort!(Int[i for (i, d) in enumerate(domain.dim_keys) if d in kept])

"""
    get_kept_dims(stat::Symbol, domain::DomainInfo)

Resolves the registry entry of `stat` (aliases `:all`, `:space`, `:time`, a dimension name or a
vector of names) into the dimension names it keeps. `:Solution` always keeps all dimensions.
"""
function get_kept_dims(stat::Symbol, domain::DomainInfo)
    stat === :Solution && return collect(domain.dim_keys)
    return _kept_dims(get(domain.stat_registry, stat, Symbol[]), domain)
end

"""
    get_kept_indices(stat::Symbol, domain::DomainInfo)

Sorted indices (into `domain.dim_keys`) of the dimensions kept by `stat`.
"""
get_kept_indices(stat::Symbol, domain::DomainInfo) = _kept_indices(get_kept_dims(stat, domain), domain)

"""
    get_integration_measure(stat::Symbol, domain::DomainInfo)

Product of the spacings of all dimensions that `stat` integrates out (e.g. `dx * dy`).
"""
function get_integration_measure(stat::Symbol, domain::DomainInfo{D, T}) where {D, T}
    kept = get_kept_dims(stat, domain)
    measure = one(T)
    for d in 1:D
        domain.dim_keys[d] in kept || (measure *= domain.spacing[d])
    end
    return measure
end

# --- add_stat! ---------------------------------------------------------------

_stat_value(v::Real, ::Val{M}, ::Type{T}) where {M, T} = SVector{M, T}(ntuple(_ -> T(v), Val(M)))
_stat_value(v::SVector{M, <:Real}, ::Val{M}, ::Type{T}) where {M, T} = SVector{M, T}(v)
_stat_value(v::AbstractArray{<:Real}, m::Val{M}, ::Type{T}) where {M, T} = map(x -> _stat_value(x, m, T), v)
# Per-particle scalar fields (one `Vector` per time step); SVector elements are left untouched.
_stat_value(v::AbstractVector{<:Vector{<:Real}}, m::Val{M}, ::Type{T}) where {M, T} =
    Vector{SVector{M, T}}[_stat_value(step, m, T) for step in v]
_stat_value(v, ::Val{M}, ::Type{T}) where {M, T} = v

function _check_stat_shape(data::ESimData, name::Symbol, value, kept::Vector{Symbol})
    (value isa AbstractArray && !(value isa SVector)) || return nothing
    idx = _kept_indices(kept, data.domain)
    expected = Tuple(length(data.axes[i]) for i in idx)
    size(value) == expected || throw(DimensionMismatch(
        "statistic :$name has size $(size(value)), but the dimensions it keeps $(Tuple(kept)) have lengths $expected"))
    return nothing
end
# Per-particle fields need one entry per particle and time step; time series one entry per step.
_is_particle_field(value, data::LSimData) = value isa AbstractVector{<:AbstractVector} &&
    length(value) == length(data.x) && all(n -> length(value[n]) == length(data.x[n]), eachindex(value))

function _check_stat_shape(data::LSimData, name::Symbol, value, kept::Vector{Symbol})
    if length(kept) == length(data.domain.dim_keys)
        _is_particle_field(value, data) || throw(DimensionMismatch(
            "statistic :$name keeps all dimensions, so it needs one vector per time step with one entry per particle"))
    elseif has_time(data) && kept == [data.domain.time_dim] && value isa AbstractVector && !(value isa SVector)
        length(value) == length(data.t) || throw(DimensionMismatch(
            "time series :$name has $(length(value)) entries, but the data has $(length(data.t)) time steps"))
    end
    return nothing
end
_check_stat_shape(::AbstractSimData, ::Symbol, value, kept) = nothing

"""
    add_stat!(data::AbstractSimData, name::Symbol, value, kept_dims)

Attaches a pre-computed statistic and records the dimensions it keeps (`:all`, `:space`, `:time`,
dimension names, or `Symbol[]` for a scalar) so that conversions and frontends can slice it.
Numbers and arrays of numbers are converted to `SVector`s with the dataset's component count, and
shapes are checked against the kept axes (Eulerian) or time steps and particles (Lagrangian).

[`calculate_all_stats!`](@ref) leaves such statistics alone, unless the same name is registered
globally (then it is recomputed with `force_overwrite = true` or when the kept dimensions differ).
"""
function add_stat!(sim_data::AbstractSimData{D, DS, M, T}, name::Symbol, value, kept_dims) where {D, DS, M, T}
    name === :Solution && throw(ArgumentError("use `data.u = ...` (or `data.stats[:Solution] = ...`) to replace the solution"))
    reg_val = _registry_value(kept_dims)
    stored = _stat_value(value, Val(M), T)
    _check_stat_shape(sim_data, name, stored, _kept_dims(reg_val, sim_data.domain))
    global_dims = get(get_stat_registry(), name, nothing)
    if global_dims !== nothing && global_dims != reg_val
        @warn "Statistic :$name is registered globally with kept dimensions $global_dims; " *
              "`calculate_all_stats!` will replace the value added here."
    end
    sim_data.domain.stat_registry[name] = reg_val
    _raw_stats(sim_data)[name] = stored
    @info "Added custom stat '$name' keeping dimensions: $reg_val"
    return stored
end

function delete_stat!(data::AbstractSimData, name::Symbol)
    name === :Solution && throw(ArgumentError(":Solution is the dataset's `u` field and cannot be removed."))
    delete!(_raw_stats(data), name)
    delete!(data.domain.stat_registry, name)
    return data
end

"""
    calc_stat(::Val{name}, fixed_coords, u, ana, domain)

Extension point for statistics registered with [`register_stat!`](@ref). Called once per retained
slice: `fixed_coords` holds the coordinates of the kept dimensions, `u`/`ana` iterate the numerical
and reference values of the slice (`ana` is `NaN` without a reference function). Must return an
`SVector{M}`. The fallback returns `NaN`s, which removes the statistic.
"""
calc_stat(stat, fixed_coords, u, ana, domain) = zero(eltype(u)) .* NaN

"""
    is_all_nan(val)

`true` if every entry of a statistic (scalar `SVector`, array or per-particle series) contains a NaN.
"""
function is_all_nan(val)
    if val isa Vector{<:Vector}
        return all(step -> all(svec -> any(isnan, svec), step), val)
    elseif val isa SVector
        return any(isnan, val)
    elseif val isa AbstractArray
        return all(svec -> any(isnan, svec), val)
    end
    return false
end

"""
    remove_nan_stats!(stats::AbstractDict)

Removes every statistic whose values are all `NaN` (`:Solution` is never removed).
"""
function remove_nan_stats!(stats::AbstractDict)
    for name in [k for (k, v) in stats if k !== :Solution && is_all_nan(v)]
        @info "Removing statistic :$name because all values are NaN (no reference data)."
        delete!(stats, name)
    end
    return stats
end

# Brings the dataset registry in line with the global one. Values computed for different kept
# dimensions are dropped as stale. Returns the (sorted) global statistic names and whether anything changed.
function _sync_registry!(sim_data::AbstractSimData)
    reg = sim_data.domain.stat_registry
    raw = _raw_stats(sim_data)
    changed = false
    for (name, dims) in get_stat_registry()
        name === :Solution && continue
        old = get(reg, name, nothing)
        if old === nothing || old != dims
            old === nothing || delete!(raw, name)
            reg[name] = dims isa Vector ? copy(dims) : dims
            changed = true
        end
    end
    names = sort!(Symbol[k for k in keys(get_stat_registry()) if k !== :Solution])
    return names, changed
end

"""
    calculate_all_stats!(sim_data::AbstractSimData, ref_func = nothing; force_overwrite = false, save = true)

Evaluates every statistic of the global registry (see [`register_stat!`](@ref)) that is missing
from the dataset (or all of them with `force_overwrite = true`). `ref_func(x)` is the exact solution
at the spacetime point `x::SVector{D}`; it is evaluated once per node/particle. Statistics that come
out all-`NaN` (e.g. errors without a reference) are dropped. Statistics added with
[`add_stat!`](@ref) are never touched. Saves the dataset when `save = true`. Returns `sim_data`.
"""
calculate_all_stats!(sim_data::NoSimData, ref_func = nothing; kwargs...) = sim_data

function calculate_all_stats!(sim_data::AbstractSimData, ref_func = nothing; force_overwrite::Bool = false, save::Bool = true)
    names, changed = _sync_registry!(sim_data)
    raw = _raw_stats(sim_data)
    pending = force_overwrite ? names : filter(n -> !haskey(raw, n), names)
    if !isempty(pending)
        # `invokelatest` makes `calc_stat` methods and reference functions that were defined after the
        # caller started (e.g. by `set_stat_preset!` inside a function) visible.
        u_ana = ref_func === nothing ? generate_pointwise_nan(sim_data) :
                Base.invokelatest(generate_pointwise_reference, sim_data, ref_func)
        for name in pending
            result = Base.invokelatest(_calc_stat!, sim_data, u_ana, name)
            result === nothing && continue
            if is_all_nan(result)
                haskey(raw, name) && (delete!(raw, name); changed = true)
            else
                raw[name] = result
                changed = true
            end
        end
    end
    if save
        path = _find_file(sim_data.params)
        if path !== nothing && !_is_native_representation(path, sim_data)
            # A conversion or reference dataset shares the parameters (and hash) of the native run;
            # saving it would replace the native data on disk.
            @warn "Statistics were not saved: the file for these parameters holds a different representation " *
                  "(this looks like a converted or reference dataset). Pass `save = false` for derived datasets."
        else
            path = save_sim_data(sim_data; overwrite = changed, verbose = false)
            # Remember what was evaluated, so the pipeline does not reload data for stats that came out NaN.
            _set_key!(path, "evaluated_stats", names)
        end
    end
    return sim_data
end

# ==============================================================================
# --- Eulerian reductions ---
# ==============================================================================

@inline function _extract_coords(axes, kept_idx, I, ::Val{L}, ::Type{T}) where {L, T}
    return SVector{L, T}(ntuple(d -> axes[kept_idx[d]][I[d]], Val(L)))
end

function _batch_stat_calc!(res, u_slices, ana_slices, axes, kept_idx, domain, stat_name, ::Val{L}, ::Type{T}) where {L, T}
    @batch for i in eachindex(u_slices)
        I = CartesianIndices(u_slices)[i]
        fixed_coords = _extract_coords(axes, kept_idx, I, Val(L), T)
        res[i] = calc_stat(Val(stat_name), fixed_coords, vec(u_slices[i]), vec(ana_slices[i]), domain)
    end
    return res
end

"""
    _calc_stat!(sim_data, u_ana, stat_name)

Internal: slices the data along the kept dimensions of `stat_name` and evaluates `calc_stat` on
every slice (multithreaded). Returns `nothing` if the kept dimensions are not supported.
"""
function _calc_stat!(sim_data::ESimData{D, DS, M, T}, u_ana, stat_name::Symbol) where {D, DS, M, T}
    kept_idx = get_kept_indices(stat_name, sim_data.domain)
    if isempty(kept_idx)  # scalar (converted, like the array results, to SVector{M, T})
        value = calc_stat(Val(stat_name), SVector{0, T}(), vec(sim_data.u), vec(u_ana), sim_data.domain)
        return _as_svector(value, Val(M), T)
    end
    L = length(kept_idx)
    res = Array{SVector{M, T}, L}(undef, ntuple(d -> length(sim_data.axes[kept_idx[d]]), L)...)
    u_slices = eachslice(sim_data.u, dims = Tuple(kept_idx))
    ana_slices = eachslice(u_ana, dims = Tuple(kept_idx))
    _batch_stat_calc!(res, u_slices, ana_slices, sim_data.axes, kept_idx, sim_data.domain, stat_name, Val(L), T)
    return res
end

# ==============================================================================
# --- Lagrangian reductions ---
# ==============================================================================

function _calc_stat!(sim_data::LSimData{D, DS, M, T}, u_ana, stat_name::Symbol) where {D, DS, M, T}
    domain = sim_data.domain
    kept = get_kept_dims(stat_name, domain)
    u = sim_data.u
    t = sim_data.t
    Nt = length(t)

    if isempty(kept)  # scalar over all particles (and time steps)
        value = calc_stat(Val(stat_name), SVector{0, T}(), reduce(vcat, u), reduce(vcat, u_ana), domain)
        return _as_svector(value, Val(M), T)
    elseif D > DS && kept == [domain.time_dim]  # time series
        series = Vector{SVector{M, T}}(undef, Nt)
        @batch for n in 1:Nt
            series[n] = calc_stat(Val(stat_name), SVector{1, T}(t[n]), u[n], u_ana[n], domain)
        end
        return series
    elseif length(kept) == D  # per-particle field
        field = Vector{Vector{SVector{M, T}}}(undef, Nt)
        x = sim_data.x
        @batch for n in 1:Nt
            x_n = x[n]
            values = Vector{SVector{M, T}}(undef, length(x_n))
            for p in eachindex(x_n)
                fixed = D > DS ? SVector{D, T}(x_n[p]..., t[n]) : SVector{D, T}(x_n[p]...)
                values[p] = calc_stat(Val(stat_name), fixed, (u[n][p],), (u_ana[n][p],), domain)
            end
            field[n] = values
        end
        return field
    end
    @debug "Statistic :$stat_name keeps $kept, which is not supported for Lagrangian data; skipping."
    return nothing
end

# ==============================================================================
# --- Pointwise reference caches ---
# ==============================================================================

_as_svector(v::Real, ::Val{M}, ::Type{T}) where {M, T} = SVector{M, T}(ntuple(_ -> T(v), Val(M)))
_as_svector(v, ::Val{M}, ::Type{T}) where {M, T} = SVector{M, T}(v)

"""
    generate_pointwise_nan(data)

`NaN`-filled replica of `data.u`, used as the reference when no reference function is available.
"""
generate_pointwise_nan(data::LSimData{D, DS, M, T}) where {D, DS, M, T} =
    [fill(SVector{M, T}(ntuple(_ -> T(NaN), Val(M))), length(x)) for x in data.x]

generate_pointwise_nan(data::ESimData{D, DS, M, T}) where {D, DS, M, T} =
    fill(SVector{M, T}(ntuple(_ -> T(NaN), Val(M))), size(data.u))

"""
    generate_pointwise_reference(data, ref_func)

Evaluates `ref_func(x)` at the spacetime coordinates `x::SVector{D}` of every grid node or particle
(multithreaded). `ref_func` may return an `SVector`, a tuple/vector or a number.
"""
function generate_pointwise_reference(ldata::LSimData{D, DS, M, T}, ref_func) where {D, DS, M, T}
    Nt = length(ldata.t)
    u_ana = Vector{Vector{SVector{M, T}}}(undef, Nt)
    t = ldata.t
    x = ldata.x
    @batch for n in 1:Nt
        t_n = t[n]
        u_ana[n] = D > DS ? [_as_svector(ref_func(SVector{D, T}(p..., t_n)), Val(M), T) for p in x[n]] :
                            [_as_svector(ref_func(SVector{D, T}(p...)), Val(M), T) for p in x[n]]
    end
    return u_ana
end

function generate_pointwise_reference(edata::ESimData{D, DS, M, T}, ref_func) where {D, DS, M, T}
    u = edata.u
    u_ana = similar(u)
    axes = edata.axes
    @batch for i in eachindex(u)
        I = CartesianIndices(u)[i]
        u_ana[i] = _as_svector(ref_func(SVector{D, T}(ntuple(d -> axes[d][I[d]], Val(D)))), Val(M), T)
    end
    return u_ana
end
