# ==============================================================================
# --- Conversions between Eulerian and Lagrangian representations ---
# ==============================================================================

function _check_res(res::Tuple, D::Int)
    length(res) == D || throw(ArgumentError("expected a spacetime resolution with $D entries, got $res"))
    all(r -> r isa Integer && r >= 1, res) || throw(ArgumentError("resolutions must be positive integers, got $res"))
    return ntuple(d -> Int(res[d]), D)
end

_linspace(a::T, b::T, n::Int) where {T} = n == 1 ? T[a] : collect(range(a, b, length = n))

function _copy_domain(domain::DomainInfo{D, T}; spacing = domain.spacing) where {D, T}
    return DomainInfo{D, T}(domain.dim_keys, domain.mins, domain.maxs, spacing, domain.time_dim, domain.stat_registry)
end

function _nearest_index(t::AbstractVector, tv)
    i = searchsortedfirst(t, tv)
    i <= 1 && return 1
    i > length(t) && return length(t)
    return (tv - t[i - 1]) <= (t[i] - tv) ? i - 1 : i
end

# ------------------------------------------------------------------------------
# Scatter algorithm (Lagrangian -> Eulerian)
# ------------------------------------------------------------------------------

"""
    interpolate_to_grid!(::Val{:scatter}, u_out, w_out, field_outs, ldata, field_vals, steps, s_mins, s_maxs, s_dx, grid_shape)

Internal, multithreaded inverse-distance scatter of the particles (and particle field statistics)
of the native time steps `steps` onto a structured grid. Output slice `k` receives step `steps[k]`.
Cells that receive no particle within the smoothing radius are set to `NaN`.
"""
function interpolate_to_grid!(::Val{:scatter}, u_out::AbstractArray, w_out::AbstractArray, field_outs::Tuple,
                              ldata::LSimData{D, DS, M, T}, field_vals::Tuple, steps::AbstractVector{Int},
                              s_mins::SVector{DS, T}, s_maxs::SVector{DS, T}, s_dx::SVector{DS, T},
                              grid_shape::NTuple{DS, Int}) where {D, DS, M, T}
    nan_vec = SVector{M, T}(ntuple(_ -> T(NaN), Val(M)))
    s_inv_dx = one(T) ./ s_dx

    # Smoothing radius: 1.5x the larger of the grid spacing and the mean particle spacing.
    np0 = max(1, length(ldata.x[first(steps)]))
    pts_per_dim = max(1.0, np0^(1 / DS) - 1.0)
    particle_spacing = (s_maxs .- s_mins) ./ T(pts_per_dim)
    effective = max.(s_dx, particle_spacing)
    radius2 = (norm(effective) * T(1.5))^2
    rad_idx = sqrt(radius2) .* s_inv_dx

    Threads.@threads for k in eachindex(steps)
        n = steps[k]
        x_step = ldata.x[n]
        u_step = ldata.u[n]
        @inbounds for p in eachindex(x_step)
            pos = x_step[p]
            all(isfinite, pos) || continue
            idx_float = (pos .- s_mins) .* s_inv_dx .+ one(T)
            # Clamp in floating point first, so particles far outside the grid cannot overflow `Int`.
            lo = ntuple(d -> floor(Int, clamp(idx_float[d] - rad_idx[d], one(T), T(grid_shape[d] + 1))), Val(DS))
            hi = ntuple(d -> ceil(Int, clamp(idx_float[d] + rad_idx[d], zero(T), T(grid_shape[d]))), Val(DS))
            for cell in CartesianIndices(ntuple(d -> lo[d]:hi[d], Val(DS)))
                cell_pos = s_mins .+ s_dx .* (SVector{DS, T}(Tuple(cell)) .- one(T))
                dist2 = sum(abs2, cell_pos .- pos)
                dist2 <= radius2 || continue
                w = one(T) / max(dist2, T(1e-12))
                I = D > DS ? CartesianIndex(Tuple(cell)..., k) : cell
                w_out[I] += w
                u_out[I] += u_step[p] * w
                for i in eachindex(field_outs)
                    field_outs[i][I] += field_vals[i][n][p] * w
                end
            end
        end
        # Normalise this time slice (empty cells become NaN).
        @inbounds for cell in CartesianIndices(grid_shape)
            I = D > DS ? CartesianIndex(Tuple(cell)..., k) : cell
            wsum = w_out[I]
            if wsum > 0
                u_out[I] /= wsum
                for i in eachindex(field_outs)
                    field_outs[i][I] /= wsum
                end
            else
                u_out[I] = nan_vec
                for i in eachindex(field_outs)
                    field_outs[i][I] = nan_vec
                end
            end
        end
    end
    return u_out
end

interpolate_to_grid!(::Val{S}, args...) where {S} = throw(ArgumentError("unknown spatial interpolation algorithm :$S"))

"""
    convert_to_eulerian(ldata::LSimData, res::Tuple; spatial_interp = :scatter)

Scatters Lagrangian particles onto a grid with the spacetime resolution `res` (spatial entries,
then the number of time slices). Particle field statistics are scattered as well, time series are
resampled to the output times and other statistics are copied. Output times are evenly spaced
between the first and last recording time; each takes the nearest recorded particle snapshot (the
native times are kept when the number of slices matches).
"""
function convert_to_eulerian(ldata::LSimData{D, DS, M, T}, res::Tuple; spatial_interp::Symbol = :scatter) where {D, DS, M, T}
    res_ = _check_res(res, D)
    domain = ldata.domain
    s_res = ntuple(d -> res_[d], Val(DS))
    s_mins = SVector{DS, T}(ntuple(d -> domain.mins[d], Val(DS)))
    s_maxs = SVector{DS, T}(ntuple(d -> domain.maxs[d], Val(DS)))
    for d in 1:DS
        if s_res[d] > 1 && !(s_maxs[d] > s_mins[d])
            throw(ArgumentError("the particles have no extent along :$(domain.dim_keys[d]); use a resolution of 1 for that dimension"))
        end
    end
    s_dx = SVector{DS, T}(ntuple(d -> s_res[d] > 1 ? (s_maxs[d] - s_mins[d]) / T(s_res[d] - 1) : one(T), Val(DS)))
    grid_axes = ntuple(d -> _linspace(s_mins[d], s_maxs[d], s_res[d]), Val(DS))

    if D > DS
        nt = res_[D]
        if nt == length(ldata.t)
            steps, t_new = collect(1:nt), copy(ldata.t)
        else
            t_new = _linspace(first(ldata.t), last(ldata.t), nt)
            steps = [_nearest_index(ldata.t, tv) for tv in t_new]
        end
    else
        steps, t_new = [1], T[]
    end
    e_shape = D > DS ? (s_res..., length(steps)) : s_res

    zero_vec = zero(SVector{M, T})
    u_out = fill(zero_vec, e_shape)
    w_out = zeros(T, e_shape)

    e_stats = StatDict{M, T}()
    field_keys = Symbol[]
    field_outs = Array{SVector{M, T}, D}[]
    field_vals = Vector{Vector{SVector{M, T}}}[]
    for (key, value) in _raw_stats(ldata)
        key === :Solution && continue
        kept = get_kept_dims(key, domain)
        if length(kept) == D && value isa Vector{Vector{SVector{M, T}}} && _is_particle_field(value, ldata)
            push!(field_keys, key)
            push!(field_outs, fill(zero_vec, e_shape))
            push!(field_vals, value)
        elseif D > DS && kept == [domain.time_dim] && value isa AbstractVector && length(value) == length(ldata.t)
            e_stats[key] = value[steps]
        else
            e_stats[key] = deepcopy(value)
        end
    end

    interpolate_to_grid!(Val(spatial_interp), u_out, w_out, Tuple(field_outs), ldata, Tuple(field_vals), steps,
                         s_mins, s_maxs, s_dx, s_res)
    for (i, key) in enumerate(field_keys)
        e_stats[key] = field_outs[i]
    end

    if D > DS
        dt = length(t_new) > 1 ? (last(t_new) - first(t_new)) / T(length(t_new) - 1) : domain.spacing[D]
        e_axes = (grid_axes..., t_new)
        e_spacing = SVector{D, T}((s_dx..., dt))
    else
        e_axes = grid_axes
        e_spacing = SVector{D, T}(Tuple(s_dx))
    end
    return ESimData{D, DS, M, T}(ldata.params, _copy_domain(domain; spacing = e_spacing), e_axes, u_out, e_stats)
end

# ------------------------------------------------------------------------------
# Eulerian resampling
# ------------------------------------------------------------------------------

_interp_kind(domain::DomainInfo, d::Int) = domain.dim_keys[d] === domain.time_dim ? Gridded(Constant()) : Gridded(Linear())

function _fill_resampled!(out::AbstractArray{E, L}, itp, new_axes::Tuple) where {E, L}
    cart = CartesianIndices(out)
    Threads.@threads for i in eachindex(out)
        I = cart[i]
        out[i] = itp(ntuple(k -> new_axes[k][I[k]], Val(L))...)
    end
    return out
end

# Interpolates `values` (whose dimensions are the domain dimensions `dims`) from `old_axes` to `new_axes`:
# linear in space, nearest-neighbour (constant) in time, flat extrapolation.
function _resample_array(values::AbstractArray, old_axes::Tuple, new_axes::Tuple, domain::DomainInfo, dims::Tuple)
    vals, axs = values, old_axes
    for i in 1:length(axs)
        if length(axs[i]) == 1  # Gridded interpolation needs two knots: duplicate singleton axes
            vals = cat(vals, vals; dims = i)
            axs = Base.setindex(axs, [axs[i][1], axs[i][1] + one(eltype(axs[i]))], i)
        end
    end
    kinds = ntuple(i -> _interp_kind(domain, dims[i]), length(dims))
    itp = extrapolate(interpolate(axs, vals, kinds), Flat())
    out = similar(values, map(length, new_axes))
    return _fill_resampled!(out, itp, new_axes)
end

function _resample_stat(key::Symbol, value, data::ESimData, new_axes::Tuple)
    (value isa AbstractArray && !(value isa SVector)) || return deepcopy(value)
    kept = get_kept_indices(key, data.domain)
    (isempty(kept) || ndims(value) != length(kept)) && return deepcopy(value)
    old_axes = ntuple(i -> data.axes[kept[i]], length(kept))
    target = ntuple(i -> new_axes[kept[i]], length(kept))
    if size(value) != map(length, old_axes)
        @warn "Statistic :$key does not match the axes it keeps; copying it unchanged."
        return deepcopy(value)
    end
    all(i -> old_axes[i] == target[i], eachindex(old_axes)) && return deepcopy(value)
    try
        return _resample_array(value, old_axes, target, data.domain, Tuple(kept))
    catch e
        @warn "Could not resample statistic :$key; copying it unchanged." exception = e
        return deepcopy(value)
    end
end

"""
    resample_eulerian(data::ESimData, res::Tuple)

Resamples the solution and all statistics that keep spatial/temporal dimensions to the spacetime
resolution `res`. Space is interpolated linearly, time with nearest-neighbour (constant)
interpolation to avoid cross-fading between snapshots. Resampled axes are uniform and span the old
axes' node range; axes whose resolution does not change are kept as they are.
Returns `data` itself if it already has the requested resolution.
"""
function resample_eulerian(data::ESimData{D, DS, M, T}, res::Tuple) where {D, DS, M, T}
    res_ = _check_res(res, D)
    size(data.u) == res_ && return data
    @info "Resampling Eulerian data from $(size(data.u)) to $res_ (linear in space, nearest in time)..."

    # Axes whose resolution does not change are kept as they are (e.g. adaptive time steps).
    new_axes = ntuple(d -> res_[d] == length(data.axes[d]) ? copy(data.axes[d]) :
                           _linspace(first(data.axes[d]), last(data.axes[d]), res_[d]), Val(D))
    new_u = _resample_array(data.u, data.axes, new_axes, data.domain, ntuple(identity, Val(D)))
    new_stats = StatDict{M, T}()
    for (key, value) in _raw_stats(data)
        key === :Solution && continue
        new_stats[key] = _resample_stat(key, value, data, new_axes)
    end
    new_spacing = SVector{D, T}(ntuple(d -> res_[d] > 1 ?
        (last(new_axes[d]) - first(new_axes[d])) / T(res_[d] - 1) : one(T), Val(D)))
    return ESimData{D, DS, M, T}(data.params, _copy_domain(data.domain; spacing = new_spacing), new_axes, new_u, new_stats)
end

# ------------------------------------------------------------------------------
# Eulerian -> Lagrangian
# ------------------------------------------------------------------------------

"""
    convert_to_lagrangian(data::ESimData)

Turns grid nodes into stationary particles (all time steps share one position vector). Full
fields become per-particle statistics, purely spatial fields are repeated for every time step and
all other statistics are copied.
"""
function convert_to_lagrangian(data::ESimData{D, DS, M, T}) where {D, DS, M, T}
    timed = has_time(data)
    nt = timed ? size(data.u, D) : 1
    t_vec = timed ? copy(data.axes[D]) : T[0]
    s_shape = ntuple(d -> length(data.axes[d]), Val(DS))
    points = vec([SVector{DS, T}(ntuple(d -> data.axes[d][I[d]], Val(DS))) for I in CartesianIndices(s_shape)])
    slice(a, n) = timed ? vec(collect(selectdim(a, D, n))) : vec(collect(a))

    new_x = fill(points, nt)
    new_u = Vector{SVector{M, T}}[slice(data.u, n) for n in 1:nt]

    space_keys = Set(data.domain.dim_keys[1:DS])
    l_domain = _copy_domain(data.domain)
    new_stats = StatDict{M, T}()
    for (key, value) in _raw_stats(data)
        key === :Solution && continue
        kept = get_kept_dims(key, data.domain)
        if value isa AbstractArray{SVector{M, T}} && length(kept) == D && size(value) == size(data.u)
            new_stats[key] = Vector{SVector{M, T}}[slice(value, n) for n in 1:nt]
        elseif timed && value isa AbstractArray{SVector{M, T}} && Set(kept) == space_keys && size(value) == s_shape
            # A purely spatial field becomes a per-particle field at every time step.
            new_stats[key] = fill(vec(collect(value)), nt)
            l_domain.stat_registry[key] = :all
        else
            new_stats[key] = deepcopy(value)
        end
    end
    return LSimData{D, DS, M, T}(data.params, l_domain, t_vec, new_x, new_u, new_stats)
end
