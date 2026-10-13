@testset "Manual solver execution (upwind convergence)" begin
    params_50 = create_param_dict(:N => 50, :scheme => "upwind", :cfl => 0.5)
    params_100 = create_param_dict(:N => 100, :scheme => "upwind", :cfl => 0.5)
    sim_50 = advection_solver_1d(params_50)
    sim_100 = advection_solver_1d(params_100)
    calculate_all_stats!(sim_50, exact_advection)
    calculate_all_stats!(sim_100, exact_advection)

    t_len = length(sim_50.axes[get_time_dim(sim_50.domain)])
    @test haskey(sim_50.stats, :l1error)
    @test length(sim_50.stats[:mass]) == t_len
    initial_mass = sim_50.stats[:mass][1][1]
    @test all(m -> isapprox(m[1], initial_mass; rtol = 1e-12), sim_50.stats[:mass])

    err_50 = sim_50.stats[:l1error][end][1]
    err_100 = sim_100.stats[:l1error][end][1]
    @test err_100 < err_50
    @test 0.45 < (err_100 / err_50) < 0.55
    @test does_sim_data_exist(params_50)  # calculate_all_stats! saves by default
end

@testset "Statistics registry & custom statistics" begin
    params = create_param_dict(:N => 32, :scheme => "upwind", :cfl => 0.5, :test => "registry")
    sim = advection_solver_1d(params)
    nt = size(sim.u, 2)

    register_stat!(:total, :time)
    try
        # A statistic registered *after* the dataset was created is still evaluated
        calculate_all_stats!(sim, exact_advection; save = false)
        @test haskey(sim.stats, :total) && length(sim.stats[:total]) == nt
        @test get_kept_dims(:total, sim.domain) == [:t]

        # Re-registering with other kept dimensions makes the stored values stale
        register_stat!(:total, :all)
        calculate_all_stats!(sim, exact_advection; save = false)
        @test size(sim.stats[:total]) == size(sim.u)

        # A single dimension name is a valid kept dimension
        register_stat!(:total, :x)
        calculate_all_stats!(sim, exact_advection; save = false)
        @test size(sim.stats[:total]) == (size(sim.u, 1),)
        @test get_integration_measure(:total, sim.domain) ≈ sim.domain.spacing[2]

        # Kept dimensions are reported in domain order, whatever order the registry lists them in
        register_stat!(:total, [:t, :x])
        calculate_all_stats!(sim, exact_advection; save = false)
        @test get_kept_dims(:total, sim.domain) == [:x, :t]
        @test size(sim.stats[:total]) == size(sim.u)
        @test get_kept_indices(:total, sim.domain) == [1, 2]
    finally
        delete_stat!(:total)
    end
    @test !haskey(get_stat_registry(), :total)

    # add_stat! converts numbers, validates shapes and survives force_overwrite
    add_stat!(sim, :runtime, 1.5, Symbol[])
    add_stat!(sim, :series, collect(1.0:nt), :time)
    @test sim.stats[:runtime] == SVector(1.5) && sim.stats[:series][2] == SVector(2.0)
    @test_throws DimensionMismatch add_stat!(sim, :bad, collect(1.0:(nt + 1)), :time)
    calculate_all_stats!(sim, exact_advection; force_overwrite = true, save = false)
    @test sim.stats[:runtime] == SVector(1.5) && haskey(sim.stats, :series)

    # delete_stat! on a dataset removes value and registry entry
    delete_stat!(sim, :runtime)
    @test !haskey(sim.stats, :runtime) && !haskey(sim.domain.stat_registry, :runtime)

    # Reference-dependent statistics are dropped when no reference is given
    sim_noref = advection_solver_1d(create_param_dict(:N => 16, :test => "noref"))
    calculate_all_stats!(sim_noref; save = false)
    @test haskey(sim_noref.stats, :mass) && !haskey(sim_noref.stats, :l1error)
end

@testset "Lagrangian statistics & references" begin
    params_L = create_param_dict(:N => 10, :v => 2.0)
    sim_L = particle_solver_1d(params_L)
    @test sim_L isa LSimData{2, 1, 1, Float64}
    @test length(sim_L.x) == 3

    ref_L = generate_reference_simdata(exact_particle, params_L, sim_L.domain, (10, 3), Val(:lagrangian))
    @test ref_L isa LSimData && length(ref_L.u) == 3 && length(ref_L.u[1]) == 10
    ref_E = generate_reference_simdata(exact_particle, params_L, sim_L.domain, (20, 3), :eulerian)
    @test ref_E isa ESimData{2, 1, 1, Float64} && size(ref_E.u) == (20, 3) && all(==(SVector(1.0)), ref_E.u)

    add_stat!(sim_L, :execution_time, 0.042, Symbol[])
    @test haskey(sim_L.stats, :execution_time)
    @test get_kept_dims(:execution_time, sim_L.domain) == Symbol[]
    @test sim_L.stats[:execution_time] == SVector{1, Float64}(0.042)

    calculate_all_stats!(sim_L, exact_particle)
    # Time series
    @test haskey(sim_L.stats, :l1error)
    @test all(err -> isapprox(err[1], 0.0, atol = 1e-12), sim_L.stats[:l1error])
    # Per-particle field
    @test haskey(sim_L.stats, :pointwise_diff)
    @test length(sim_L.stats[:pointwise_diff]) == 3
    @test length(sim_L.stats[:pointwise_diff][1]) == 10
    @test all(diff -> isapprox(diff[1], 0.0, atol = 1e-12), sim_L.stats[:pointwise_diff][1])
    @test sim_L.stats[:execution_time] == SVector(0.042)

    # Static particles get per-particle fields too (one snapshot)
    static = create_sim_data([SVector(0.1 * i) for i in 1:5], ones(5), create_param_dict(:case => "static-L"))
    u_ana = PDEStudioCore.generate_pointwise_reference(static, st -> 1.0)
    field = PDEStudioCore._calc_stat!(static, u_ana, :pointwise_diff)
    @test length(field) == 1 && length(field[1]) == 5 && all(==(SVector(0.0)), field[1])
end
