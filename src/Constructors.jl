# ==============================================================================
# --- create_sim_data: one N-dimensional front end ---
# ==============================================================================

_default_space_keys(DS::Int) = DS <= 3 ? (:x, :y, :z)[1:DS] : ntuple(i -> Symbol(:x, i), DS)

function _space_keys(DS::Int, dims)
    names = dims === nothing ? Symbol[_default_space_keys(DS)...] :
            dims isa Union{Symbol, AbstractString} ? [Symbol(dims)] : Symbol[Symbol(d) for d in dims]
    length(names) == DS || throw(ArgumentError("`dims` must name $DS spatial dimension(s), got $(length(names))"))
    allunique(names) || throw(ArgumentError("`dims` must not repeat a name, got $(Tuple(names))"))
    return Tuple(names)
end

# --- Spatial coordinates: a vector, a tuple of axis vectors, or a tuple of ndgrid arrays ---
_space_axes(x::AbstractVector{<:Real}) = (x,)
_space_axes(::Tuple{}) = ()
function _space_axes(space::Tuple)
    all(a -> a isa AbstractVector{<:Real}, space) && return space
    if all(a -> a isa AbstractArray{<:Real} && ndims(a) == length(space), space)
        sz = size(first(space))
        all(a -> size(a) == sz, space) || throw(DimensionMismatch("all coordinate grids must have the same size"))
        return ntuple(d -> _axis_from_grid(space[d], d), length(space))
    end
    throw(ArgumentError("spatial coordinates must be a vector, a tuple of axis vectors or a tuple of ndgrid-style coordinate arrays"))
end

function _axis_from_grid(grid::AbstractArray{<:Real}, d::Int)
    index = ntuple(i -> i == d ? Colon() : firstindex(grid, i), ndims(grid))
    axis = vec(collect(grid[index...]))
    if length(axis) > 1 && allequal(axis) && !allequal(grid)
        throw(ArgumentError("coordinate grid $d does not vary along dimension $d. Grids must be in ndgrid order " *
                            "(grid[i, j] == x[i]); transpose MATLAB/meshgrid-style arrays."))
    end
    return axis
end

# --- Solution packing: SVector arrays, scalar arrays, components-last arrays, arrays of vectors ---
function _pack_solution(u::AbstractArray, D::Int, lens::Tuple)
    E = eltype(u)
    if E <: SVector && ndims(u) == D
        return convert(Array{SVector{length(E), float(eltype(E))}, D}, u)
    elseif E <: Real && ndims(u) == D
        return _map_svec(u, Val(1), float(E))
    elseif E <: Real && ndims(u) == D + 1 && size(u)[1:D] == lens
        return _pack_components(u, Val(size(u, D + 1)), float(E), Val(D))
    elseif E <: AbstractVector && ndims(u) == D && !isempty(u)
        return _map_svec(u, Val(length(first(u))), float(eltype(E)))
    end
    throw(DimensionMismatch("cannot interpret a solution of type $(typeof(u)) and size $(size(u)) on axes of lengths $lens. " *
        "Expected an array of SVectors (or of numbers) with one dimension per axis, or numbers with an extra trailing component dimension."))
end

_map_svec(u::AbstractArray, ::Val{M}, ::Type{T}) where {M, T} = map(v -> SVector{M, T}(v), u)

function _pack_components(u::AbstractArray{<:Real}, ::Val{M}, ::Type{T}, ::Val{D}) where {M, T, D}
    out = Array{SVector{M, T}, D}(undef, ntuple(d -> size(u, d), Val(D)))
    for I in CartesianIndices(out)
        out[I] = SVector{M, T}(ntuple(c -> u[Tuple(I)..., c], Val(M)))
    end
    return out
end

# A bound is `nothing` (use the data extent), a number (all spatial dimensions) or one entry per dimension.
_bound(b, d::Int) = b isa Number ? b : b[d]

function _bounds_tuple(DS::Int, transient::Bool, xbounds, tbound)
    if xbounds !== nothing && !(xbounds isa Number) && length(xbounds) != DS
        throw(ArgumentError("spatial bounds must have $DS entries, got $(length(xbounds))"))
    end
    s = ntuple(d -> xbounds === nothing ? nothing : _bound(xbounds, d), DS)
    return transient ? (s..., tbound) : s
end

function _create_eulerian(axes_in::Tuple, u::AbstractArray, params::AbstractDict, dim_keys::NTuple{D, Symbol},
                          time_dim::Union{Nothing, Symbol}, mins_in::Tuple, maxs_in::Tuple) where {D}
    length(axes_in) == D || throw(DimensionMismatch("expected $D coordinate axes, got $(length(axes_in))"))
    lens = ntuple(d -> length(axes_in[d]), D)
    all(>(0), lens) || throw(ArgumentError("all axes need at least one point, got lengths $lens"))
    return _finish_eulerian(axes_in, _pack_solution(u, D, lens), params, dim_keys, time_dim, mins_in, maxs_in)
end

function _finish_eulerian(axes_in::Tuple, sol::Array{SVector{M, T}, D}, params::AbstractDict, dim_keys::NTuple{D, Symbol},
                          time_dim::Union{Nothing, Symbol}, mins_in::Tuple, maxs_in::Tuple) where {M, T, D}
    axes = ntuple(d -> collect(T, axes_in[d]), Val(D))
    for d in 1:D
        issorted(axes[d]) || throw(ArgumentError("axis :$(dim_keys[d]) must be sorted in increasing order"))
    end
    mins = SVector{D, T}(ntuple(d -> mins_in[d] === nothing ? first(axes[d]) : T(mins_in[d]), Val(D)))
    maxs = SVector{D, T}(ntuple(d -> maxs_in[d] === nothing ? last(axes[d]) : T(maxs_in[d]), Val(D)))
    # The spacing is the mean node distance of each axis (independent of overridden bounds).
    spacing = SVector{D, T}(ntuple(d -> length(axes[d]) > 1 ?
        (last(axes[d]) - first(axes[d])) / T(length(axes[d]) - 1) : one(T), Val(D)))
    domain = DomainInfo{D, T}(dim_keys, mins, maxs, spacing, time_dim, get_stat_registry())
    DS = time_dim === nothing ? D : D - 1
    return ESimData{D, DS, M, T}(params, domain, axes, sol)
end

"""
    create_sim_data(space, u, t, params; dims, time_dim = :t, xmins, xmaxs, tmin, tmax)   # transient Eulerian
    create_sim_data(space, u, params; dims, xmins, xmaxs)                                 # static Eulerian
    create_sim_data(x, u, t, params; dims, time_dim = :t, xmins, xmaxs, tmin, tmax)       # transient Lagrangian
    create_sim_data(x, u, params; dims, xmins, xmaxs)                                     # static Lagrangian

Builds an [`ESimData`](@ref) or [`LSimData`](@ref) of any spatial dimension. The global statistic
registry is copied into the dataset and `:Solution` is coupled to `u` automatically.

# Eulerian input
- `space`: a coordinate vector (1D), a tuple of axis vectors `(x, y, z, ...)`, or a tuple of
  ndgrid-style coordinate arrays (`x_grid[i, j] == x[i]`).
- `u`: an array with one dimension per spatial axis (plus a trailing time dimension when `t` is
  given) whose elements are `SVector`s, numbers (a scalar field) or vectors. A numeric array with one
  *extra* trailing dimension is read as components-last (`u[i, j, n, c]`).
- `t`: the time vector (time is always the last axis).

# Lagrangian input
- `x`: particle positions, `Vector{Vector{SVector{DS}}}` (one vector per time step) or a single
  `Vector{SVector{DS}}` for static data.
- `u`: particle states with the same layout (`SVector`s or numbers).

# Keywords
- `dims`: names of the spatial axes (default `(:x, :y, :z)`, or `(:x1, :x2, ...)` beyond 3D).
- `time_dim`: name of the time axis (default `:t`).
- `xmins`, `xmaxs`, `tmin`, `tmax`: override the bounding box (defaults: extent of the data).
  The integration measure (`domain.spacing`) is always derived from the axes themselves.

The parameters are copied, so later changes to `params` do not affect the dataset.

# Examples
```julia
create_sim_data(x, u, t, params)                       # 1D + time, u[i, n]::SVector{M}
create_sim_data((x, y), u, t, params)                  # 2D + time, u[i, j, n]
create_sim_data((x, y, z), ρ, params; dims = (:r, :θ, :φ))   # static 3D scalar field
create_sim_data((x_grid, y_grid), u, t, params)        # ndgrid coordinate arrays
create_sim_data(x_traj, u_traj, t, params)             # particles over time
```
"""
function create_sim_data(space::Union{AbstractVector{<:Real}, Tuple}, u::AbstractArray, t::AbstractVector{<:Real},
                         params::AbstractDict; dims = nothing, time_dim::Union{Nothing, Symbol} = :t,
                         xmins = nothing, xmaxs = nothing, tmin = nothing, tmax = nothing)
    time_dim === nothing && throw(ArgumentError(
        "a time vector was given, so `time_dim` must be a Symbol; use `create_sim_data(space, u, params)` for static data"))
    s_axes = _space_axes(space)
    DS = length(s_axes)
    dim_keys = (_space_keys(DS, dims)..., time_dim)
    time_dim in dim_keys[1:DS] && throw(ArgumentError("the time dimension :$time_dim is also used as a spatial dimension name"))
    return _create_eulerian((s_axes..., t), u, params, dim_keys, time_dim,
                            _bounds_tuple(DS, true, xmins, tmin), _bounds_tuple(DS, true, xmaxs, tmax))
end

function create_sim_data(space::Union{AbstractVector{<:Real}, Tuple}, u::AbstractArray, params::AbstractDict;
                         dims = nothing, xmins = nothing, xmaxs = nothing)
    s_axes = _space_axes(space)
    DS = length(s_axes)
    DS >= 1 || throw(ArgumentError("static data needs at least one spatial axis"))
    return _create_eulerian(s_axes, u, params, _space_keys(DS, dims), nothing,
                            _bounds_tuple(DS, false, xmins, nothing), _bounds_tuple(DS, false, xmaxs, nothing))
end

# ------------------------------------------------------------------------------
# --- Lagrangian ---
# ------------------------------------------------------------------------------

function _particle_bounds(x::Vector{Vector{SVector{DS, T}}}) where {DS, T}
    lo = SVector{DS, T}(ntuple(_ -> typemax(T), Val(DS)))
    hi = SVector{DS, T}(ntuple(_ -> typemin(T), Val(DS)))
    found = false
    for step in x, p in step
        lo = min.(lo, p)
        hi = max.(hi, p)
        found = true
    end
    return found ? (lo, hi) : (zero(SVector{DS, T}), zero(SVector{DS, T}))
end

_particle_scalar(::Type{E}) where {E <: SVector} = eltype(E)
_particle_scalar(::Type{E}) where {E <: Real} = E
_particle_scalar(::Type{E}) where {E} = throw(ArgumentError(
    "particle states must be SVectors or numbers, got elements of type $E"))

_pack_particles(u::AbstractVector, ::Type{E}, ::Type{T}) where {E <: SVector, T} =
    Vector{SVector{length(E), T}}[convert(Vector{SVector{length(E), T}}, step) for step in u]
_pack_particles(u::AbstractVector, ::Type{E}, ::Type{T}) where {E <: Real, T} =
    Vector{SVector{1, T}}[SVector{1, T}.(step) for step in u]

function _create_lagrangian(x::AbstractVector, u::AbstractVector, t::AbstractVector{<:Real}, params::AbstractDict,
                            space_keys::Tuple, time_dim::Union{Nothing, Symbol}, xmins, xmaxs, tmin, tmax)
    length(x) == length(u) || throw(DimensionMismatch("x and u must have the same number of time steps"))
    Ex = eltype(eltype(x))
    Eu = eltype(eltype(u))
    T = float(promote_type(eltype(Ex), _particle_scalar(Eu), eltype(t)))
    DS = length(Ex)
    xx = Vector{SVector{DS, T}}[convert(Vector{SVector{DS, T}}, step) for step in x]
    uu = _pack_particles(u, Eu, T)
    tt = collect(T, t)
    issorted(tt) || throw(ArgumentError("the time vector must be sorted in increasing order"))
    dim_keys = time_dim === nothing ? space_keys : (space_keys..., time_dim)
    return _build_lsimdata(xx, uu, tt, params, dim_keys, time_dim, xmins, xmaxs, tmin, tmax)
end

function _build_lsimdata(x::Vector{Vector{SVector{DS, T}}}, u::Vector{Vector{SVector{M, T}}}, t::Vector{T},
                         params::AbstractDict, dim_keys::NTuple{D, Symbol}, time_dim::Union{Nothing, Symbol},
                         xmins, xmaxs, tmin, tmax) where {DS, M, T, D}
    _bounds_tuple(DS, false, xmins, nothing)  # validates the number of bounds
    _bounds_tuple(DS, false, xmaxs, nothing)
    lo, hi = _particle_bounds(x)
    smins = ntuple(d -> xmins === nothing ? lo[d] : T(_bound(xmins, d)), Val(DS))
    smaxs = ntuple(d -> xmaxs === nothing ? hi[d] : T(_bound(xmaxs, d)), Val(DS))
    # Equivalent particle spacing: (volume / particles)^(1/DS).
    np = max(1, maximum(length, x; init = 0))
    vol = prod(smaxs .- smins)
    dx = vol > 0 ? T((vol / np)^(1 / DS)) : one(T)
    s_spacing = ntuple(_ -> dx, Val(DS))
    registry = get_stat_registry()
    if time_dim === nothing
        domain = DomainInfo{D, T}(dim_keys, SVector{D, T}(smins), SVector{D, T}(smaxs), SVector{D, T}(s_spacing),
                                  nothing, registry)
    else
        t0 = tmin === nothing ? first(t) : T(tmin)
        t1 = tmax === nothing ? last(t) : T(tmax)
        dt = length(t) > 1 ? (last(t) - first(t)) / T(length(t) - 1) : one(T)
        domain = DomainInfo{D, T}(dim_keys, SVector{D, T}((smins..., t0)), SVector{D, T}((smaxs..., t1)),
                                  SVector{D, T}((s_spacing..., dt)), time_dim, registry)
    end
    return LSimData{D, DS, M, T}(params, domain, t, x, u)
end

function create_sim_data(x::AbstractVector{<:AbstractVector{<:SVector{DS}}}, u::AbstractVector{<:AbstractVector},
                         t::AbstractVector{<:Real}, params::AbstractDict;
                         dims = nothing, time_dim::Union{Nothing, Symbol} = :t,
                         xmins = nothing, xmaxs = nothing, tmin = nothing, tmax = nothing) where {DS}
    time_dim === nothing && throw(ArgumentError(
        "a time vector was given, so `time_dim` must be a Symbol; use `create_sim_data(x, u, params)` for static particles"))
    isempty(t) && throw(ArgumentError("the time vector must not be empty"))
    return _create_lagrangian(x, u, t, params, _space_keys(DS, dims), time_dim, xmins, xmaxs, tmin, tmax)
end

function create_sim_data(x::AbstractVector{<:SVector{DS}}, u::AbstractVector, params::AbstractDict;
                         dims = nothing, xmins = nothing, xmaxs = nothing) where {DS}
    T0 = float(promote_type(eltype(eltype(x)), _particle_scalar(eltype(u))))  # keeps e.g. Float32 data
    return _create_lagrangian([x], [u], T0[0], params, _space_keys(DS, dims), nothing, xmins, xmaxs, nothing, nothing)
end
