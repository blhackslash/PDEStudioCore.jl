# examples/dummy.jl
# A minimal pipeline: a mock "solver", two methods and a sweep over the time step.
using PDEStudioCore
using StaticArrays

# 1. A user simulation function: takes the resolved parameters of one run, returns a dataset
function my_wave_sim(params::ParamDict)
    N  = params[:N]
    dt = params[:dt]
    c  = params[:wave_speed]

    # 1D space + time grid
    x = collect(range(0.0, 1.0, length = N))
    t = collect(range(0.0, 2.0, step = dt))

    # Mock data: u(x, t) = sin(x - c t), stored as u[i, n]::SVector{1}
    u = [SVector(sin(xi - c * tn)) for xi in x, tn in t]

    return create_sim_data(x, u, t, params)   # dims (:x, :t)
end

# 2. Parameters: shared by every run, per-method overrides, and the sweep
shared = create_param_dict(
    :N => 100,
    :wave_speed => 1.5,
    :dt => 0.1,                     # default, overwritten by the sweep
    :sim_func_name => :my_wave_sim, # the function every run calls
)

methods = create_method_dict(
    :upwind => create_param_dict(:solver => "upwind"),
    :lax_wendroff => create_param_dict(:solver => "lax_wendroff"),
)

varied = create_varied_dict(:dt => [0.1, 0.05])

# 3. The configuration: shared, methods, active methods (`defaults`), sweep
config = SimulationConfig(shared, methods, [:upwind, :lax_wendroff]; varied = varied)

# 4. Run the pipeline (2 methods × 2 time steps = 4 runs, cached on disk by parameter hash)
set_save_path!(joinpath(@__DIR__, "results"))
run_all_simulations(config; parallel = true, calculate_stats = true)
