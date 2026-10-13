# ==============================================================================
# --- Internal helpers kept for code written against earlier versions ---
# ==============================================================================
# These were internal (not exported) but may be used by frontends such as PDEStudio.jl.

# Normalise a scalar resolution or a tuple into a spatial resolution tuple.
function _normalize_spatial_res(N_grid::Union{Int, Tuple}, DS::Int)
    if N_grid isa Int
        return ntuple(_ -> N_grid, DS)
    end
    length(N_grid) == DS || error("Spatial grid tuple length $(length(N_grid)) does not match DS = $DS")
    return N_grid
end

# Bounding box of all particle positions as `(mins, maxs)` tuples.
function _get_lsim_bounds(x::Vector{Vector{SVector{DS, T}}}) where {DS, T}
    lo, hi = _particle_bounds(x)
    return Tuple(lo), Tuple(hi)
end

# Native representation (:eulerian or :lagrangian) of a data file, read from its metadata.
_get_native_type(file_name::AbstractString) = first(_native_and_shape(file_name))
