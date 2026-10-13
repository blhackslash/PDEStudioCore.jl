@testset "Dictionary creators" begin
    d = create_param_dict("a" => 1, :b => 2.0)
    @test d isa ParamDict && d[:a] == 1 && d[:b] == 2.0
    @test create_param_dict(Dict("x" => 10, "y" => 20))[:y] == 20
    @test isempty(create_param_dict())

    m = create_method_dict("upwind" => Dict("cfl" => 0.5), :lax => Dict(:cfl => 0.8))
    @test m isa MethodDict && m[:upwind] isa ParamDict
    @test m[:upwind][:cfl] == 0.5 && m[:lax][:cfl] == 0.8
    @test create_method_dict(Dict("method_a" => Dict("order" => 2)))[:method_a][:order] == 2
    @test isempty(create_method_dict())

    v = create_varied_dict("N" => [50, 100], :cfl => 0.1:0.2:0.5)
    @test v isa VariedDict && v[:N] == [50, 100] && v[:cfl] ≈ [0.1, 0.3, 0.5]
    @test create_varied_dict(Dict("dx" => [0.1, 0.01]))[:dx] == [0.1, 0.01]
    @test isempty(create_varied_dict())
    @test_throws ArgumentError create_varied_dict(:N => 5)
end

@testset "DomainInfo validation" begin
    reg = Dict(:mass => :time)
    dom = DomainInfo{2, Float64}((:x, :t), (0.0, 0.0), (1.0, 2.0), (0.1, 0.5), :t, reg)
    @test dom.stat_registry[:Solution] == :all
    @test dom.stat_registry[:mass] == :time
    @test !haskey(reg, :Solution)  # the registry is copied, never aliased
    @test has_time(dom) && get_time_dim(dom) == 2
    @test_throws ArgumentError DomainInfo{2, Float64}((:t, :x), (0.0, 0.0), (1.0, 1.0), (0.1, 0.1), :t, reg)
    @test_throws ArgumentError DomainInfo{2, Float64}((:x, :x), (0.0, 0.0), (1.0, 1.0), (0.1, 0.1), nothing, reg)
    @test_throws ArgumentError DomainInfo{2, Float64}((:x,), (0.0, 0.0), (1.0, 1.0), (0.1, 0.1), nothing, reg)
    @test DomainInfo((:x, :y), (0, 0), (1, 2), (0.5, 0.5), nothing) isa DomainInfo{2, Float64}
end

@testset "Struct-level :Solution coupling" begin
    x = collect(range(0.0, 1.0, length = 4))
    t = [0.0, 0.5, 1.0]
    u = [SVector(xi + ti) for xi in x, ti in t]
    domain = DomainInfo{2, Float64}((:x, :t), (0.0, 0.0), (1.0, 1.0), (1 / 3, 0.5), :t, Dict{Symbol, Symbol}())

    # Direct construction, no create_sim_data involved
    sim = ESimData{2, 1, 1, Float64}(Dict(:a => 1), domain, (x, t), u)
    @test sim.stats isa StatsView
    @test haskey(sim.stats, :Solution) && :Solution in keys(sim.stats)
    @test sim.stats[:Solution] === sim.u
    @test length(sim.stats) == 1
    @test get_kept_dims(:Solution, sim.domain) == [:x, :t]

    # The solution is never stored twice, even when passed explicitly
    sim2 = ESimData(Dict(:a => 1), domain, (x, t), u, Dict(:Solution => u, :extra => u))
    @test sim2 isa ESimData{2, 1, 1, Float64}
    @test sort(collect(keys(sim2.stats))) == [:Solution, :extra]

    # Replacing the solution keeps :Solution in sync, in both directions
    sim.u = 2 .* u
    @test sim.stats[:Solution] === sim.u && sim.u[2, 2] == 2 .* u[2, 2]
    sim.stats[:Solution] = u
    @test sim.u == u
    @test_throws DimensionMismatch (sim.u = u[1:2, :])
    @test_throws ArgumentError delete!(sim.stats, :Solution)
    @test_throws ArgumentError add_stat!(sim, :Solution, u, :all)

    # Assigning the whole stats dictionary converts values like `setindex!` does
    sim3 = ESimData(Dict(:a => 1), domain, (x, t), u)
    sim3.stats[:one] = 1.0
    sim3.stats = Dict(:Solution => u, :two => 2.0)
    @test sim3.stats[:two] === SVector(2.0) && !haskey(sim3.stats, :one) && sim3.u == u

    # Derived statistics live next to the solution (numbers are stored as SVectors)
    sim.stats[:copy] = u
    @test sort(collect(keys(sim.stats))) == [:Solution, :copy]
    sim.stats[:number] = 2.0
    @test sim.stats[:number] === SVector(2.0)
    delete!(sim.stats, :number)
    # Assigning a whole dictionary is all-or-nothing
    @test_throws MethodError (sim.stats = Dict(:Solution => 3 .* u, :bad => "not a statistic"))
    @test sim.u == u && haskey(sim.stats, :copy)
    snapshot = copy(sim.stats)
    @test snapshot isa Dict && snapshot[:Solution] === sim.u
    delete!(sim.stats, :copy)
    @test collect(keys(sim.stats)) == [:Solution]
    @test occursin(":Solution", repr("text/plain", sim.stats))

    # A diverged (all-NaN) solution is never removed as an all-NaN statistic
    sim.u = fill(SVector(NaN), size(u))
    PDEStudioCore.remove_nan_stats!(sim.stats)
    @test haskey(sim.stats, :Solution)

    # Parameters are copied and the layout is validated
    p = Dict(:a => 1)
    sim3 = ESimData(p, domain, (x, t), u)
    p[:a] = 2
    @test sim3.params[:a] == 1
    @test_throws DimensionMismatch ESimData(p, domain, (x, t), u[1:3, :])
    @test_throws ArgumentError ESimData{2, 2, 1, Float64}(p, domain, (x, t), u)
end

@testset "LSimData validation" begin
    dom = DomainInfo{2, Float64}((:x, :t), (0.0, 0.0), (1.5, 1.0), (0.5, 1.0), :t, Dict{Symbol, Symbol}())
    x = [[SVector(0.0), SVector(1.0)], [SVector(0.5), SVector(1.5)]]
    u = [[SVector(1.0), SVector(2.0)], [SVector(1.0), SVector(2.0)]]
    sim = LSimData(Dict(), dom, [0.0, 1.0], x, u)
    @test sim isa LSimData{2, 1, 1, Float64}
    @test sim.stats[:Solution] === sim.u
    @test_throws DimensionMismatch LSimData(Dict(), dom, [0.0], x, u)
    @test_throws DimensionMismatch LSimData(Dict(), dom, [0.0, 1.0], x, [u[1], u[2][1:1]])
    @test_throws DimensionMismatch (sim.u = [u[1]])
end
