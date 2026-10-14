# 1. Register the metrics into the active registry
register_stat!(:mass, :time)
register_stat!(:l1norm, :time)
register_stat!(:l2norm, :time)
register_stat!(:wave_height, :time)
register_stat!(:l1error, :time)
register_stat!(:l2error, :time)
register_stat!(:relative_l2error, :time)
register_stat!(:relative_l1error, :time)
register_stat!(:relative_mass, :time)
register_stat!(:spacetime_relative_mass, Symbol[])
register_stat!(:mass_error, :time)
register_stat!(:mass_signed_error, :time)

# ==============================================================================
# --- STANDARD METRICS ---
# ==============================================================================

function calc_stat(::Val{:mass}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:mass, domain)
    return sum(u .* measure)
end

function calc_stat(::Val{:l1norm}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:l1norm, domain)
    return sum(map(v -> abs.(v), u) .* measure)
end

function calc_stat(::Val{:l2norm}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:l2norm, domain)
    return sqrt.(sum(map(v -> abs2.(v), u) .* measure))
end

function calc_stat(::Val{:wave_height}, fixed_coords, u, ana, domain::DomainInfo)
    return reduce((a, b) -> max.(a, b), u)
end


# ==============================================================================
# --- ERROR METRICS (Naturally propagates NaNs if analytical data is missing) ---
# ==============================================================================
function calc_stat(::Val{:l1error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:l1error, domain)
    return sum(map((v, a) -> abs.(v - a), u, ana) .* measure)
end

function calc_stat(::Val{:relative_l1error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:relative_l1error, domain)
    error_norm = sum(map((v, a) -> abs.(v - a), u, ana) .* measure)
    ana_norm = sum(map(a -> abs.(a), ana) .* measure)
    return error_norm ./ ana_norm
end

function calc_stat(::Val{:l2error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:l2error, domain)
    return sqrt.(sum(map((v, a) -> abs2.(v - a), u, ana) .* measure))
end

function calc_stat(::Val{:relative_l2error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:relative_l2error, domain)
    error_norm = sqrt.(sum(map((v, a) -> abs2.(v - a), u, ana) .* measure))
    ana_norm = sqrt.(sum(map(a -> abs2.(a), ana) .* measure))
    return error_norm ./ ana_norm
end

function calc_stat(::Val{:relative_mass}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:relative_mass, domain)
    sum_u = sum(u .* measure)
    sum_ana = sum(ana .* measure)
    m_ana = abs.(sum_ana) 

    M = length(eltype(u))
    T = eltype(eltype(u)) # Dynamically get T
    return SVector{M, T}(ntuple(M) do c
        m_ana[c] < 1e-9 ? T(NaN) : T(sum_u[c] / sum_ana[c])
    end)
end

function calc_stat(::Val{:mass_error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:mass_error, domain)
    sum_u = sum(u .* measure)
    sum_ana = sum(ana .* measure)
    m_err = abs.(sum_u - sum_ana)
    M = length(eltype(u))
    T = eltype(eltype(u)) # Dynamically get T
    return SVector{M, T}(ntuple(M) do c
        m_err[c] < 1e-9 ? T(1e-9) : T(m_err[c])
    end)
end
function calc_stat(::Val{:mass_signed_error}, fixed_coords, u, ana, domain::DomainInfo)
    measure = get_integration_measure(:mass_signed_error, domain)
    sum_u = sum(u .* measure)
    sum_ana = sum(ana .* measure)
    return sum_u - sum_ana
end

# 2. Add the calc_stat overload
function calc_stat(::Val{:spacetime_relative_mass}, fixed_coords, u, ana, domain::DomainInfo)
    # Gets the combined integration measure for dx * dy * dz * dt
    measure = get_integration_measure(:spacetime_relative_mass, domain)
    
    sum_u = sum(u .* measure)
    sum_ana = sum(ana .* measure)
    m_ana = abs.(sum_ana) 

    M = length(eltype(u))
    T = eltype(eltype(u)) 
    
    return SVector{M, T}(ntuple(M) do c
        m_ana[c] < 1e-9 ? T(NaN) : T(sum_u[c] / sum_ana[c])
    end)
end