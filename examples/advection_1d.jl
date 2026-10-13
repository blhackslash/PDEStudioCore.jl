# examples/advection_1d.jl
# 1D linear advection with two schemes, swept over the CFL number.
using PDEStudioCore
using StaticArrays

# 1. The solver: one run for one set of parameters
function advection_1d(params::ParamDict)
    N      = params[:N]
    c      = params[:c]
    T      = params[:T]
    cfl    = params[:cfl]
    scheme = params[:scheme]

    # Periodic grid on [0, 1) and the time step from the CFL condition
    x  = collect(range(0.0, 1.0, length = N + 1))[1:N]
    dx = x[2] - x[1]
    dt = cfl * dx / abs(c)
    t  = collect(0.0:dt:T)

    # Space-time tensor u[i, n]::SVector{1} with a sine wave as initial condition
    u = zeros(SVector{1, Float64}, N, length(t))
    for i in 1:N
        u[i, 1] = SVector(sin(2π * x[i]))
    end

    for n in 1:(length(t) - 1), i in 1:N
        u_prev, u_curr, u_next = u[mod1(i - 1, N), n][1], u[i, n][1], u[mod1(i + 1, N), n][1]
        val = if scheme == "upwind"            # first order upwind (c > 0)
            u_curr - cfl * (u_curr - u_prev)
        elseif scheme == "lax_friedrichs"
            0.5 * (u_next + u_prev) - 0.5 * cfl * (u_next - u_prev)
        else
            error("Unknown scheme: $scheme")
        end
        u[i, n + 1] = SVector(val)
    end

    return create_sim_data(x, u, t, params)   # dims (:x, :t)
end

# 2. The exact solution, as a reference factory: params -> ((x, t) -> u)
exact_advection(params::ParamDict) = xt -> SVector(sin(2π * (xt[1] - params[:c] * xt[2])))

# 3. Parameters
shared = create_param_dict(
    :N => 100,
    :c => 1.0,
    :T => 2.0,
    :cfl => 0.5,                      # default, overwritten by the sweep
    :sim_func_name => :advection_1d,  # the function every run calls
)

methods = create_method_dict(         # each method injects its :scheme
    :upwind => create_param_dict(:scheme => "upwind"),
    :lax_friedrichs => create_param_dict(:scheme => "lax_friedrichs"),
)

varied = create_varied_dict(:cfl => [0.3, 0.5, 0.8, 1.0])

# 4. The configuration; the reference enables the error statistics (e.g. of the "hyperbolic" preset)
config = SimulationConfig(shared, methods, [:upwind, :lax_friedrichs];
                          varied = varied, reference = :exact_advection)

# 5. Run the pipeline (2 schemes × 4 CFL numbers = 8 runs)
set_stat_preset!("hyperbolic")
set_save_path!(joinpath(@__DIR__, "results"))
run_all_simulations(config; parallel = true, calculate_stats = true)
