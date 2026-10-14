POST_CALLS = Ref(0)  # non-const so the suite can be re-run in one session
count_post(data) = (POST_CALLS[] += 1; nothing)  # returns nothing: nothing is saved

@testset "SimulationConfig" begin
    # Varied keys must also appear in the shared (or method) parameters.
    shared = Dict("cfl" => 0.5, :N => 5, :sim_func_name => :advection_solver_1d)
    methods = Dict(:upwind => Dict(:scheme => "upwind"), "lax_friedrichs" => Dict(:scheme => "lax_friedrichs"))
    cfg = SimulationConfig(shared, methods; varied = Dict(:N => [10, 20]), reference = "exact_advection_factory")
    @test cfg.shared isa ParamDict && cfg.shared[:cfl] == 0.5
    @test cfg.methods isa MethodDict && cfg.defaults == [:lax_friedrichs, :upwind]  # default: all methods
    @test cfg.varied isa VariedDict && cfg.varied[:N] == [10, 20]
    @test cfg.reference === :exact_advection_factory && cfg.postprocess === nothing && isempty(cfg.sources)
    @test length(generate_tasks(cfg)) == 4
    @test occursin("SimulationConfig", repr("text/plain", cfg))

    # Only the current keyword and field names exist
    full = SimulationConfig(shared, methods, [:upwind]; varied = Dict(:N => [10]), reference = :exact_advection_factory,
                            postprocess = :count_post, sources = "not_loaded_here.jl")
    @test full.varied[:N] == [10] && full.postprocess === :count_post && full.sources == ["not_loaded_here.jl"]
    @test_throws MethodError SimulationConfig(shared, methods; varied_params = Dict(:N => [10]))
    @test_throws Exception full.active_methods  # ErrorException (FieldError on Julia ≥ 1.12)
    @test_throws ArgumentError full.varied_params = Dict(:N => [30])

    # Assigning fields accepts the same loose input as the constructor
    full.reference = "exact_advection_factory"
    full.sources = "a.jl"
    full.defaults = :lax_friedrichs
    full.shared = Dict("N" => 3)
    full.varied = Dict("N" => 1:3)
    @test full.reference === :exact_advection_factory && full.sources == ["a.jl"]
    @test full.defaults == [:lax_friedrichs] && full.shared == ParamDict(:N => 3) && full.varied[:N] == [1, 2, 3]

    # Without methods, the shared parameters run once
    @test length(generate_tasks(SimulationConfig(Dict(:sim_func_name => :advection_solver_1d, :N => 8)))) == 1

    # A varied key used by only one method does not duplicate the runs of the others
    m2 = Dict(:a => Dict(:scheme => "upwind", :theta => 0.5), :b => Dict(:scheme => "lax_friedrichs"))
    cfg2 = SimulationConfig(Dict(:sim_func_name => :advection_solver_1d, :N => 8), m2; varied = Dict(:theta => [0.1, 0.2, 0.3]))
    @test length(generate_tasks(cfg2)) == 4  # 3 for :a, 1 for :b

    # Reference methods are skipped and unused varied keys are reported
    cfg3 = SimulationConfig(Dict(:sim_func_name => :advection_solver_1d, :N => 8),
                            Dict(:upwind => Dict(:scheme => "upwind")), [:upwind, :analytic];
                            varied = Dict(:typo => [1, 2]))
    tasks3 = @test_logs (:warn, r"typo") match_mode = :any generate_tasks(cfg3)
    @test length(tasks3) == 1
    @test is_reference_method(:analytic) && is_reference_method(:exact_solution)
    @test !is_reference_method(:inexact_newton) && !is_reference_method(:construe)
end

@testset "Pipeline: sweep, statistics & loading" begin
    shared = create_param_dict(:cfl => 0.5, :N => 75, :sim_func_name => :advection_solver_1d)
    methods = create_method_dict(:upwind => Dict(:scheme => "upwind"), :lax_friedrichs => Dict(:scheme => "lax_friedrichs"))
    config = SimulationConfig(shared, methods, [:upwind, :lax_friedrichs];
                              varied = Dict(:N => [50, 100]), reference = :exact_advection_factory)
    run_all_simulations(config; force_overwrite = true, calculate_stats = true)

    for scheme in ["upwind", "lax_friedrichs"], N in [50, 100]
        p = create_param_dict(:scheme => scheme, :N => N, :cfl => 0.5, :sim_func_name => :advection_solver_1d)
        @test does_sim_data_exist(p)
        sim = load_sim_data(p)
        @test sim isa ESimData && sim.stats[:Solution] === sim.u
        @test haskey(sim.stats, :l1error) && haskey(sim.stats, :mass)
        initial_mass = sim.stats[:mass][1][1]
        @test all(m -> isapprox(m[1], initial_mass; rtol = 1e-12), sim.stats[:mass])
    end

    results = [load_sim_data(p) for p in generate_tasks(config)]
    @test length(results) == 4 && all(r -> r isa ESimData, results)

    lf(N) = load_sim_data(create_param_dict(:scheme => "lax_friedrichs", :N => N, :cfl => 0.5, :sim_func_name => :advection_solver_1d))
    @test lf(100).stats[:l1error][end][1] < lf(50).stats[:l1error][end][1]

    # A second run finds everything on disk and touches nothing
    file = get_file_name(first(generate_tasks(config)))
    stamp = mtime(file)
    sleep(0.05)
    run_all_simulations(config; calculate_stats = true)
    @test mtime(file) == stamp
end

@testset "Pipeline: tuple sweeps" begin
    base_params = create_param_dict(:Ns => (10, 10), :scheme => "upwind", :cfl => 0.5)
    tasks, grid_indices = generate_method_tasks(base_params, [:Ns__x, :Ns__2], [[20, 40], [15, 30]])
    @test length(tasks) == 4 && length(grid_indices) == 4
    @test [t[:Ns] for t in tasks] == [(20, 15), (40, 15), (20, 30), (40, 30)]
    @test all(t[:Ns] isa Tuple{Int, Int} for t in tasks)
    # `__0` is not a tuple entry: the key is an ordinary (here absent) parameter
    tasks0, _ = generate_method_tasks(base_params, [:Ns__0], [[1, 2]])
    @test all(t -> t == base_params, tasks0)

    # `:ignore` drops shared keys but is not itself a parameter (so it is not hashed)
    ign = assemble_params(create_param_dict(:a => 1, :b => 2), create_method_dict(:m => Dict(:c => 3, :ignore => [:b])), :m)
    @test ign == create_param_dict(:a => 1, :c => 3)

    shared = create_param_dict(:Ns => (10, 10), :cfl => 0.5, :sim_func_name => :dummy_tuple_solver)
    methods = create_method_dict(:upwind => create_param_dict(:scheme => "upwind"))
    config = SimulationConfig(shared, methods, [:upwind]; varied = create_varied_dict(:Ns__1 => [20, 30], :Ns__y => [5, 15]))
    run_all_simulations(config; force_overwrite = true)
    for Nx in [20, 30], Ny in [5, 15]
        p = create_param_dict(:Ns => (Nx, Ny), :scheme => "upwind", :cfl => 0.5, :sim_func_name => :dummy_tuple_solver)
        @test does_sim_data_exist(p)
        sim = load_sim_data(p)
        @test sim.params[:Ns] == (Nx, Ny) && size(sim.u) == (Nx, Ny, 2)
    end
end

@testset "Pipeline robustness" begin
    POST_CALLS[] = 0
    shared = Dict(:N => 6, :sim_func_name => :advection_solver_1d)
    methods = Dict(:fail => Dict(:sim_func_name => :failing_solver),
                   :empty => Dict(:sim_func_name => :empty_solver),
                   :mutate => Dict(:sim_func_name => :mutating_solver),
                   :ok => Dict(:scheme => "upwind"))
    cfg = SimulationConfig(shared, methods; postprocess = :count_post)
    @test_logs (:error, r"Simulation failed") match_mode = :any run_all_simulations(cfg; calculate_stats = true)

    tasks = generate_tasks(cfg)
    @test length(tasks) == 4
    exists = Dict(t[:sim_func_name] => does_sim_data_exist(t) for t in tasks)
    @test !exists[:failing_solver] && !exists[:empty_solver]
    @test exists[:mutating_solver] && exists[:advection_solver_1d]
    # Results are stored under the task parameters even if the solver modified its parameters
    mutated = only(filter(t -> t[:sim_func_name] === :mutating_solver, tasks))
    @test !haskey(load_sim_data(mutated).params, :injected)
    @test POST_CALLS[] == 2  # post-processing ran for the two datasets that exist
end

@testset "Pipeline: statistics are retried once a reference exists" begin
    SOLVER_CALLS[] = 0
    shared = create_param_dict(:N => 12, :scheme => "upwind", :cfl => 0.5, :sim_func_name => :counting_solver)
    cfg_noref = SimulationConfig(shared)
    run_all_simulations(cfg_noref; force_overwrite = true, calculate_stats = true)
    p = only(generate_tasks(cfg_noref))
    @test SOLVER_CALLS[] == 1
    @test haskey(load_sim_data(p).stats, :mass) && !haskey(load_sim_data(p).stats, :l1error)
    entries = load_sim_data(calculate_hash(get_clean_params(p))[1:12])
    @test entries["evaluated_with_reference"] == false

    # Without a reference nothing is recomputed ...
    file = get_file_name(p)
    stamp = mtime(file)
    sleep(0.05)
    run_all_simulations(cfg_noref; calculate_stats = true)
    @test mtime(file) == stamp && SOLVER_CALLS[] == 1

    # ... but adding a reference evaluates the dropped error statistics without rerunning the solver
    cfg_ref = SimulationConfig(shared; reference = :exact_advection_factory)
    run_all_simulations(cfg_ref; calculate_stats = true)
    @test SOLVER_CALLS[] == 1
    @test haskey(load_sim_data(p).stats, :l1error)

    # force_stats recomputes statistics but still does not rerun the solver
    stamp = mtime(get_file_name(p))
    sleep(0.05)
    run_all_simulations(cfg_ref; calculate_stats = true, force_stats = true)
    @test SOLVER_CALLS[] == 1 && mtime(get_file_name(p)) != stamp

    # A sweep without values produces no tasks (and says so)
    @test isempty(@test_logs (:warn, r"no values") match_mode = :any generate_tasks(SimulationConfig(shared; varied = Dict(:N => Int[]))))
end

@testset "Pipeline: source files" begin
    src = joinpath(mktempdir(), "sourced_solver.jl")
    write(src, """
    function sourced_solver(params)
        x = collect(range(0.0, 1.0, length = params[:n]))
        return create_sim_data(x, ones(length(x)), params)
    end
    """)
    cfg = SimulationConfig(Dict(:sim_func_name => :sourced_solver, :n => 7); sources = src)
    run_all_simulations(cfg)
    p = only(generate_tasks(cfg))
    @test does_sim_data_exist(p)
    sim = load_sim_data(p)
    @test sim isa ESimData{1, 1, 1, Float64} && !has_time(sim)
    entries = load_sim_data(calculate_hash(get_clean_params(p))[1:12])
    @test haskey(entries, "sources") && haskey(entries["sources"], abspath(src))
end
