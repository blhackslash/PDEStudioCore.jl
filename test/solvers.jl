# ==============================================================================
# Test solvers, reference solutions and custom statistics
# (defined in the target module so the pipeline can resolve them by name)
# ==============================================================================

# --- 1D linear advection (upwind & Lax-Friedrichs), periodic ---
function advection_solver_1d(params::ParamDict)
    N = params[:N]
    scheme = get(params, :scheme, "upwind")
    cfl = get(params, :cfl, 0.5)
    L, c, T_end = 2π, 1.0, 2.0

    dx = L / N              # periodic grid without the redundant endpoint x = L
    dt = cfl * dx / c
    Nt = ceil(Int, T_end / dt) + 1
    x = collect(range(0.0, step = dx, length = N))
    t = collect(range(0.0, step = dt, length = Nt))

    u = fill(SVector{1, Float64}(0.0), N, Nt)
    for i in 1:N
        u[i, 1] = SVector(sin(x[i]) + 2.0)
    end
    for n in 1:(Nt - 1), i in 1:N
        im1 = i == 1 ? N : i - 1
        ip1 = i == N ? 1 : i + 1
        val = if scheme == "upwind"
            u[i, n][1] - cfl * (u[i, n][1] - u[im1, n][1])
        elseif scheme == "lax_friedrichs"
            0.5 * (u[ip1, n][1] + u[im1, n][1]) - 0.5 * cfl * (u[ip1, n][1] - u[im1, n][1])
        else
            error("Unknown scheme: $scheme")
        end
        u[i, n + 1] = SVector(val)
    end
    return create_sim_data(x, u, t, params; time_dim = :t, dims = :x)
end

exact_advection(st::SVector{2, Float64}) = SVector(sin(st[1] - st[2]) + 2.0)
exact_advection_factory(params::ParamDict) = (x -> exact_advection(x))

# --- 1D Lagrangian particles moving with constant velocity ---
function particle_solver_1d(params::ParamDict)
    N = params[:N]
    v = get(params, :v, 1.0)
    x0 = collect(range(0.0, 1.0, length = N))
    t = [0.0, 0.5, 1.0]
    x_traj = [[SVector(p + v * tn) for p in x0] for tn in t]
    u_traj = [[SVector(1.0) for _ in x0] for _ in t]
    return create_sim_data(x_traj, u_traj, t, params; time_dim = :t)
end

exact_particle(st::SVector{2, Float64}) = SVector(1.0)

# --- Tuple-parameter dummy solver ---
function dummy_tuple_solver(params::ParamDict)
    Nx, Ny = params[:Ns]
    x = collect(range(0.0, 1.0, length = Nx))
    y = collect(range(0.0, 1.0, length = Ny))
    t = [0.0, 0.1]
    u = fill(SVector(1.0), Nx, Ny, length(t))
    return create_sim_data((x, y), u, t, params)
end

# --- Pipeline robustness solvers ---
failing_solver(params::ParamDict) = error("intentional failure")
empty_solver(params::ParamDict) = NoSimData()

function mutating_solver(params::ParamDict)
    params[:injected] = "by the solver"     # must not change where the result is stored
    x = collect(range(0.0, 1.0, length = 5))
    return create_sim_data(x, fill(SVector(1.0), 5, 2), [0.0, 1.0], params)
end

function static_field_solver(params::ParamDict)
    n = params[:n]
    x = range(0.0, 1.0, length = n)
    y = range(0.0, 2.0, length = n + 1)
    return create_sim_data((x, y), [xi + yi for xi in x, yi in y], params)  # static scalar field
end

# Counts its calls, so tests can tell a cached run from a rerun.
SOLVER_CALLS = Ref(0)
function counting_solver(params::ParamDict)
    SOLVER_CALLS[] += 1
    return advection_solver_1d(params)
end

# --- Custom statistics used throughout the tests ---
PDEStudioCore.register_stat!(:pointwise_diff, :all)
function PDEStudioCore.calc_stat(::Val{:pointwise_diff}, fixed_coords, u, ana, domain::DomainInfo)
    return u[1] - ana[1]   # one-element slices (Eulerian) or 1-tuples (Lagrangian)
end

# Registered/unregistered on demand inside the statistics tests.
function PDEStudioCore.calc_stat(::Val{:total}, fixed_coords, u, ana, domain::DomainInfo)
    return sum(u) * get_integration_measure(:total, domain)
end

# A statistic returning a plain number (instead of an SVector) for every time slice.
PDEStudioCore.register_stat!(:slice_count, :time)
PDEStudioCore.calc_stat(::Val{:slice_count}, fixed_coords, u, ana, domain::DomainInfo) = length(u)
