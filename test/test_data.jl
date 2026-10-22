PARAMS_CONV = create_param_dict(:N => 40, :scheme => "upwind", :cfl => 0.5, :test_mode => "conversion")

@testset "Conversions & caching" begin
    sim_base = advection_solver_1d(PARAMS_CONV)
    add_stat!(sim_base, :u_copy, deepcopy(sim_base.u), :all)
    add_stat!(sim_base, :profile, sim_base.u[:, end], :x)
    save_sim_data(sim_base; overwrite = true)
    @test does_sim_data_exist(PARAMS_CONV)
    nt = size(sim_base.u, 2)

    @testset "On-the-fly conversion (cache disabled)" begin
        enable_cache!(false)
        sim_L = load_sim_data(PARAMS_CONV, Val(:lagrangian))
        @test sim_L isa LSimData{2, 1, 1, Float64}
        @test !does_sim_data_exist(PARAMS_CONV, Val(:lagrangian))
        @test sim_L.stats[:Solution] === sim_L.u
        @test sim_L.stats[:u_copy] == sim_L.u        # full fields become particle fields
        @test length(sim_L.stats[:profile]) == nt    # spatial fields are repeated per step
        @test sim_L.x[1] === sim_L.x[end]            # stationary particles share their positions

        sim_E = load_sim_data(PARAMS_CONV, :eulerian, (20, 30))
        @test sim_E isa ESimData && size(sim_E.u) == (20, 30)
        @test size(sim_E.stats[:u_copy]) == (20, 30) && size(sim_E.stats[:profile]) == (20,)
        @test !does_sim_data_exist(PARAMS_CONV, Val(:eulerian), (20, 30))
        @test does_sim_data_exist(PARAMS_CONV, Val(:eulerian), size(sim_base.u))  # native resolution
    end

    @testset "Persistent conversion (cache enabled)" begin
        enable_cache!(true)
        try
            load_sim_data(PARAMS_CONV, Val(:lagrangian))
            @test does_sim_data_exist(PARAMS_CONV, Val(:lagrangian))
            @test "conv_L" in list_available_conversions(PARAMS_CONV)
            load_sim_data(PARAMS_CONV, Val(:eulerian), (25, 30))
            @test does_sim_data_exist(PARAMS_CONV, Val(:eulerian), (25, 30))
            @test "conv_E_25x30" in list_available_conversions(PARAMS_CONV)
            # Rewriting the raw data drops the now stale caches
            save_sim_data(sim_base; overwrite = true)
            @test isempty(list_available_conversions(PARAMS_CONV))
        finally
            enable_cache!(false)
        end
    end

    @testset "Scatter (Lagrangian -> Eulerian)" begin
        sim_L = load_sim_data(PARAMS_CONV, Val(:lagrangian))
        sim_E = convert_to_eulerian(sim_L, (30, nt))
        @test size(sim_E.u) == (30, nt) && sim_E.axes[2] == sim_L.t
        @test size(sim_E.stats[:u_copy]) == (30, nt)
        @test !any(v -> any(isnan, v), sim_E.u)  # every cell is reached
        # Scattering grid nodes back onto the same grid reproduces the data
        sim_same = convert_to_eulerian(sim_L, size(sim_base.u))
        @test all(isapprox.(first.(sim_same.u), first.(sim_base.u); atol = 1e-8))
        # Fewer time slices pick the nearest recorded snapshots
        sim_few = convert_to_eulerian(sim_L, (30, 5))
        @test size(sim_few.u) == (30, 5)
        @test sim_few.axes[2][1] == sim_L.t[1] && sim_few.axes[2][end] ≈ sim_L.t[end]
        @test_throws ArgumentError convert_to_eulerian(sim_L, (30,))
    end

    @testset "Derived datasets never replace the native data" begin
        sim_L = load_sim_data(PARAMS_CONV, Val(:lagrangian))
        @test_logs (:warn, r"not saved") match_mode = :any calculate_all_stats!(sim_L, nothing)
        @test load_sim_data(PARAMS_CONV) isa ESimData

        ref = generate_reference_simdata(exact_advection, PARAMS_CONV, sim_base.domain, (10, 5), :eulerian)
        @test_logs (:warn, r"not saved") match_mode = :any calculate_all_stats!(ref, exact_advection)
        @test size(load_sim_data(PARAMS_CONV).u) == size(sim_base.u)
        @test haskey(ref.stats, :mass)  # the statistics are still computed in memory
    end

    @testset "Resampling keeps non-uniform axes it does not resample" begin
        x = collect(range(0.0, 1.0, length = 11))
        t = [0.0, 0.1, 0.3, 0.7, 1.5]                      # adaptive time steps
        s = create_sim_data(x, [SVector(xi + ti) for xi in x, ti in t], t, create_param_dict(:case => "nonuniform"))
        add_stat!(s, :series, collect(1.0:5.0), :time)
        r = resample_eulerian(s, (21, 5))
        @test r.axes[2] == t && r.stats[:series] == s.stats[:series]
        @test r.u[end, 3] ≈ s.u[end, 3]
        r2 = resample_eulerian(s, (11, 3))                 # time is resampled: uniform, nearest snapshot
        @test r2.axes[2] ≈ [0.0, 0.75, 1.5] && length(r2.stats[:series]) == 3
    end

    @testset "Static data" begin
        x = collect(range(0.0, 1.0, length = 6))
        y = collect(range(0.0, 2.0, length = 5))
        s = create_sim_data((x, y), [SVector(xi + yi) for xi in x, yi in y], create_param_dict(:case => "static-conv"))
        L = convert_to_lagrangian(s)
        @test L isa LSimData{2, 2, 1, Float64} && length(L.t) == 1 && length(L.x[1]) == 30
        E = convert_to_eulerian(L, (6, 5))
        @test all(isapprox.(first.(E.u), first.(s.u); atol = 1e-8))
        R = resample_eulerian(s, (11, 9))
        @test size(R.u) == (11, 9) && R.u[1, 1] ≈ s.u[1, 1] && R.u[end, end] ≈ s.u[end, end]
        @test resample_eulerian(s, (6, 5)) === s
    end
end

@testset "Data health checking" begin
    sim_E = load_sim_data(PARAMS_CONV)
    df_E = check_data(sim_E)
    @test df_E isa DataFrame && size(df_E, 1) == 1
    @test names(df_E) == ["Time", "C1_Min", "C1_Max", "C1_Mean", "C1_NaNs"]
    @test df_E.C1_NaNs[1] == 0.0
    df_steps = check_data(sim_E; per_step = true)
    @test size(df_steps, 1) == size(sim_E.u, 2) && df_steps.Time == sim_E.axes[2]

    sim_L = load_sim_data(PARAMS_CONV, Val(:lagrangian))
    df_L = check_data(sim_L)
    @test df_L isa DataFrame && size(df_L, 1) == length(sim_L.t)
end

@testset "Data management: deletion" begin
    delete_params = create_param_dict(:N => 9, :scheme => "upwind", :cfl => 0.5, :test_flag => "delete_me")
    keep_params = create_param_dict(:N => 9, :scheme => "upwind", :cfl => 0.5, :test_flag => "keep_me")
    save_sim_data(advection_solver_1d(delete_params); overwrite = true)
    save_sim_data(advection_solver_1d(keep_params); overwrite = true)
    @test does_sim_data_exist(delete_params) && does_sim_data_exist(keep_params)

    # find_sim_data: any subset of the parameters, string or symbol keys, newest first
    found = find_sim_data(Dict("test_flag" => "delete_me"); verbose = false)
    @test found == [get_file_name(delete_params)]
    found_n = find_sim_data(Dict(:N => 9, :scheme => "upwind"); verbose = false)
    @test Set(found_n) ⊇ Set([get_file_name(delete_params), get_file_name(keep_params)])
    @test issorted(mtime.(found_n); rev = true)
    @test isempty(find_sim_data(Dict(:N => 9, :test_flag => "no such flag"); verbose = false))
    @test length(find_sim_data(Dict(); verbose = false)) == length(PDEStudioCore._data_files())
    io = IOBuffer()
    @test find_sim_data(Dict(:test_flag => "delete_me"); io = io) == found
    listing = String(take!(io))
    @test startswith(listing, "1 data file with test_flag = delete_me:")
    @test occursin(basename(found[1]), listing) && occursin("N = 9", listing) && !occursin("test_flag", split(listing, '\n')[2])
    io = IOBuffer(); find_sim_data(Dict(:N => -1); io = io)
    @test startswith(String(take!(io)), "0 data files with N = -1.")

    @test_throws ArgumentError delete_sim_data(Dict())  # would match every file
    @test length(delete_sim_data(Dict(:test_flag => "delete_me"); dry_run = true)) == 1
    @test does_sim_data_exist(delete_params)

    deleted = delete_sim_data(Dict(:N => 9, :test_flag => "delete_me"))
    @test length(deleted) == 1
    @test !does_sim_data_exist(delete_params) && does_sim_data_exist(keep_params)
end

@testset "Data management: rehash" begin
    params_target = create_param_dict(:N => 10, :scheme => "upwind", :cfl => 0.5, :target => true, :obsolete => "delete_me")
    params_ignore = create_param_dict(:N => 11, :scheme => "upwind", :cfl => 0.5, :target => false, :obsolete => "keep_me")
    save_sim_data(advection_solver_1d(params_target); overwrite = true)
    save_sim_data(advection_solver_1d(params_ignore); overwrite = true)
    file_target_old = get_file_name(params_target)
    file_ignore_old = get_file_name(params_ignore)

    renamed = rehash_sim_data(joinpath(get_save_path(), "data"); delete_old = true,
                              filter_pairs = Dict(:target => true), remove_keys = [:obsolete])
    @test length(renamed) == 1 && first(renamed[1]) == file_target_old
    @test isfile(file_ignore_old) && !isfile(file_target_old)

    params_new = create_param_dict(:N => 10, :scheme => "upwind", :cfl => 0.5, :target => true)
    @test does_sim_data_exist(params_new)
    sim_rehashed = load_sim_data(params_new)
    @test !haskey(sim_rehashed.params, :obsolete) && sim_rehashed.params[:target] == true
end

@testset "Output formatting & manual loading" begin
    params_E = create_param_dict(:N => 25, :scheme => "upwind", :cfl => 0.5, :test_id => "format_E")
    sim_E = advection_solver_1d(params_E)
    save_sim_data(sim_E; overwrite = true)
    sim_L = particle_solver_1d(create_param_dict(:N => 10, :v => 1.5, :test_id => "format_L"))

    @testset "Clean parameter processing & printing" begin
        clean_from_dict = get_clean_params(params_E)
        @test clean_from_dict isa Dict{Symbol, Any} && clean_from_dict == get_clean_params(sim_E)
        @test clean_from_dict[:test_id] == "format_E"
        pipe = Pipe()
        printed = redirect_stdout(pipe) do
            print_clean_params(sim_E)
        end
        close(pipe.in)
        out = read(pipe, String)
        @test printed == clean_from_dict
        @test occursin("Cleaned Parameters", out) && occursin("test_id => format_E", out) && occursin("scheme => upwind", out)
    end

    @testset "REPL display" begin
        out_E = repr("text/plain", sim_E)
        for label in ("ESimData", "Time (T)", "Space", "Grid Size", "Parameters", "Statistics", "Solution")
            @test occursin(label, out_E)
        end
        out_L = repr("text/plain", sim_L)
        for label in ("LSimData", "Time (T)", "Space", "Particles", "Parameters", "Statistics")
            @test occursin(label, out_L)
        end
        @test startswith(repr(sim_E), "ESimData{2, 1, 1, Float64}(25×")
        @test startswith(repr(sim_L), "LSimData{2, 1, 1, Float64}(3 steps")
    end

    @testset "Manual loading by hash prefix" begin
        prefix = calculate_hash(params_E)[1:8]
        entries = load_sim_data(prefix)
        @test entries["raw"] isa ESimData && entries["raw"].params[:test_id] == "format_E"
        @test load_sim_data(prefix; key = "raw") isa ESimData
        # Extra metadata cannot clobber the file layout
        @test_throws ArgumentError save_sim_data(sim_E; overwrite = true, metadata = Dict("raw" => 1))
        @test_throws ArgumentError save_sim_data(sim_E; overwrite = true, metadata = Dict(:conv_L => 1))
        save_sim_data(sim_E; overwrite = true, metadata = Dict(:note => "kept"))
        @test load_sim_data(prefix; key = "note") == "kept"
        @test_throws SimFileNotFoundError load_sim_data("00000000000000000")
        @test_throws SimFileNotFoundError load_sim_data("2026-10")  # timestamps are not matched
    end
end
