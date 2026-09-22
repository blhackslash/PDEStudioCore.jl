using Test
using PDEStudioCore
using StaticArrays
using DataFrames

# 1. Register the namespace so PDEStudioCore can dynamically resolve these functions
PDEStudioCore.set_target_module!(@__MODULE__)

# ==============================================================================
# --- EXPERIMENT: 1D Linear Advection (Upwind & Lax-Friedrichs) ---
# ==============================================================================

function advection_solver_1d(params::ParamDict)
    N = params[:N]
    scheme = get(params, :scheme, "upwind")
    cfl = get(params, :cfl, 0.5)
    
    L = 2π
    c = 1.0     
    
    # Strictly periodic grid (dropping the redundant endpoint at x=L)
    dx = L / N  
    dt = cfl * dx / c
    T_end = 2.0
    Nt = ceil(Int, T_end / dt) + 1
    
    x = collect(range(0.0, step=dx, length=N))
    t = collect(range(0.0, step=dt, length=Nt))
    
    # Preallocate the spacetime tensor
    u_num = fill(SVector{1, Float64}(0.0), N, Nt)
    
    # Periodic initial condition: u(x,0) = sin(x) + 2.0
    for i in 1:N
        u_num[i, 1] = SVector{1, Float64}(sin(x[i]) + 2.0)
    end
    
    # Time Marching
    for n in 1:(Nt-1)
        for i in 1:N
            im1 = i == 1 ? N : i - 1 # Wrap around cleanly
            ip1 = i == N ? 1 : i + 1 
            
            if scheme == "upwind"
                val = u_num[i, n][1] - cfl * (u_num[i, n][1] - u_num[im1, n][1])
            elseif scheme == "lax_friedrichs"
                val = 0.5 * (u_num[ip1, n][1] + u_num[im1, n][1]) - 0.5 * cfl * (u_num[ip1, n][1] - u_num[im1, n][1])
            else
                error("Unknown scheme: $scheme")
            end
            
            u_num[i, n+1] = SVector{1, Float64}(val)
        end
    end
    
    return create_sim_data(x, u_num, t, params; time_dim=:t, x_dim=:x)
end

function exact_advection(st::SVector{2, Float64})
    x, t = st[1], st[2]
    c = 1.0
    return SVector{1, Float64}(sin(x - c*t) + 2.0)
end

function exact_advection_factory(params::ParamDict)
    return (x -> exact_advection(x))
end

# ==============================================================================
# --- EXPERIMENT: 1D Lagrangian Particle Tracking ---
# ==============================================================================

function particle_solver_1d(params::ParamDict)
    N = params[:N]
    v = get(params, :v, 1.0)
    
    x0 = collect(range(0.0, 1.0, length=N))
    t = [0.0, 0.5, 1.0]
    Nt = length(t)
    
    x_traj = Vector{Vector{SVector{1, Float64}}}(undef, Nt)
    u_traj = Vector{Vector{SVector{1, Float64}}}(undef, Nt)
    
    for i in 1:Nt
        x_traj[i] = [SVector{1, Float64}(pos + v * t[i]) for pos in x0]
        # Dummy property scalar (e.g., mass or concentration = 1.0)
        u_traj[i] = [SVector{1, Float64}(1.0) for _ in x0] 
    end
    
    # Trigger the Lagrangian constructor
    return create_sim_data(x_traj, u_traj, t, params; time_dim=:t)
end

function exact_particle(st::SVector{2, Float64})
    # st = [x, t]. The exact scalar property is just 1.0.
    return SVector{1, Float64}(1.0)
end

# Define a custom pointwise field statistic for coverage
PDEStudioCore.register_stat!(:pointwise_diff, :all)
function PDEStudioCore.calc_stat(::Val{:pointwise_diff}, fixed_coords, u, ana, domain::DomainInfo)
    # The Lagrangian field evaluator passes 1-element tuples for specific particles
    return u[1] - ana[1]
end


function dummy_tuple_solver(params::ParamDict)
    Nx, Ny = params[:Ns]
    x = collect(range(0.0, 1.0, length=Nx))
    t = [0.0, 0.1]
    u = fill(SVector{1, Float64}(1.0), Nx, length(t))
    return create_sim_data(x, u, t, params; time_dim=:t)
end
# ==============================================================================
# --- TEST SUITE ---
# ==============================================================================

@testset "PDEStudioCore.jl Physical Experiments" begin
    set_stat_preset!("hyperbolic")
    set_save_path!(mktempdir())
    @test NoSimData() isa NoSimData

    @testset "Dictionary Creation Utilities" begin
        @testset "ParamDict Creators" begin
            # 1. Empty creator
            d_empty = create_param_dict()
            @test d_empty isa ParamDict
            @test isempty(d_empty)

            # 2. Pair... arguments (mixed String and Symbol keys)
            d_pairs = create_param_dict("a" => 1, :b => 2.0)
            @test d_pairs isa ParamDict
            @test d_pairs[:a] == 1
            @test d_pairs[:b] == 2.0

            # 3. Generic collection / Dict input
            d_iter = create_param_dict(Dict("x" => 10, "y" => 20))
            @test d_iter isa ParamDict
            @test d_iter[:x] == 10
            @test d_iter[:y] == 20
        end

        @testset "MethodDict Creators" begin
            # 1. Empty creator
            m_empty = create_method_dict()
            @test m_empty isa MethodDict
            @test isempty(m_empty)

            # 2. Pair... arguments with nested String/Symbol dictionaries
            m_pairs = create_method_dict(
                "upwind" => Dict("cfl" => 0.5),
                :lax => Dict(:cfl => 0.8)
            )
            @test m_pairs isa MethodDict
            @test m_pairs[:upwind] isa ParamDict
            @test m_pairs[:upwind][:cfl] == 0.5
            @test m_pairs[:lax][:cfl] == 0.8

            # 3. Generic collection input
            raw_methods = Dict("method_a" => Dict("order" => 2))
            m_iter = create_method_dict(raw_methods)
            @test m_iter isa MethodDict
            @test m_iter[:method_a] isa ParamDict
            @test m_iter[:method_a][:order] == 2
        end

        @testset "VariedDict Creators" begin
            # 1. Empty creator
            v_empty = create_varied_dict()
            @test v_empty isa VariedDict
            @test isempty(v_empty)

            # 2. Pair... arguments
            v_pairs = create_varied_dict("N" => [50, 100], :cfl => [0.1, 0.5])
            @test v_pairs isa VariedDict
            @test v_pairs[:N] == [50, 100]
            @test v_pairs[:cfl] == [0.1, 0.5]

            # 3. Generic collection input
            raw_varied = Dict("dx" => [0.1, 0.01])
            v_iter = create_varied_dict(raw_varied)
            @test v_iter isa VariedDict
            @test v_iter[:dx] == [0.1, 0.01]
        end
    end
    
    @testset "Manual Solver Execution (Upwind)" begin
        params_50 = create_param_dict(:N => 50, :scheme => "upwind", :cfl => 0.5)
        params_100 = create_param_dict(:N => 100, :scheme => "upwind", :cfl => 0.5)
        
        sim_50 = advection_solver_1d(params_50)
        sim_100 = advection_solver_1d(params_100)
        
        calculate_all_stats!(sim_50, exact_advection)
        calculate_all_stats!(sim_100, exact_advection)
        
        time_idx = get_time_dim(sim_50.domain)
        t_len = length(sim_50.axes[time_idx])
        
        @test haskey(sim_50.stats, :l1error)
        @test length(sim_50.stats[:mass]) == t_len
        
        initial_mass = sim_50.stats[:mass][1][1]
        @test all(m -> isapprox(m[1], initial_mass; rtol=1e-12), sim_50.stats[:mass])
        
        err_50 = sim_50.stats[:l1error][end][1]
        err_100 = sim_100.stats[:l1error][end][1]
        @test err_100 < err_50
        @test 0.45 < (err_100 / err_50) < 0.55
    end

    @testset "SimulationConfig Pipeline (Sweep & Orchestration)" begin
        # 1. Define the orchestration layers
        shared = Dict(:cfl => 0.5, :N => 75)
        methods = Dict(
            :upwind => Dict(:scheme => "upwind"),
            :lax_friedrichs => Dict(:scheme => "lax_friedrichs")
        )
        varied = Dict(:N => [50, 100])
        
        # 2. Build the Config
        config = SimulationConfig(
            "advection_solver_1d", 
            shared, 
            methods, 
            [:upwind, :lax_friedrichs]; 
            varied_params = varied,
            ref_func_name = "exact_advection_factory"
        )
        
        # 3. Execute the full Cartesian sweep (4 simulations total)
        run_all_simulations(config; force_overwrite=true, calculate_stats=true)
        
        # 4. Verify disk state and automated stats generation
        for scheme in ["upwind", "lax_friedrichs"]
            for N in [50, 100]
                p = create_param_dict(:scheme => scheme, :N => N, :cfl => 0.5)
                
                # Check that caching and hashing works
                @test does_sim_data_exist(p)
                
                sim = load_sim_data(p)
                @test sim isa ESimData
                @test haskey(sim.stats, :l1error)
                @test haskey(sim.stats, :mass)
                
                # Lax-Friedrichs and Upwind both conserve mass on a periodic grid
                initial_mass = sim.stats[:mass][1][1]
                @test all(m -> isapprox(m[1], initial_mass; rtol=1e-12), sim.stats[:mass])
            end
        end
        
        # 5. Verify Lax-Friedrichs grid convergence from disk
        sim_lf_50 = load_sim_data(create_param_dict(:scheme => "lax_friedrichs", :N => 50, :cfl => 0.5))
        sim_lf_100 = load_sim_data(create_param_dict(:scheme => "lax_friedrichs", :N => 100, :cfl => 0.5))
        
        err_50 = sim_lf_50.stats[:l1error][end][1]
        err_100 = sim_lf_100.stats[:l1error][end][1]
        @test err_100 < err_50
    end
    @testset "Sub-Tuple Varied Sweeps & Parameter Assembly" begin
        # Verify generate_method_tasks directly parses and reconstructs tuples
        base_params = create_param_dict(
            :Ns => (10, 10),
            :scheme => "upwind",
            :cfl => 0.5
        )

        # Test both alphanumeric and numeric axis suffixes (__x, __y and __1, __2)
        active_keys = [:Ns__x, :Ns__2]
        active_values = [[20, 40], [15, 30]]

        tasks, grid_indices = generate_method_tasks(base_params, active_keys, active_values)

        @test length(tasks) == 4
        @test length(grid_indices) == 4

        # Check all Cartesian product combinations were reconstructed into Tuples
        expected_tuples = [(20, 15),  (40, 15), (20, 30), (40, 30)]
        reconstructed_tuples = [t[:Ns] for t in tasks]
        @test reconstructed_tuples == expected_tuples
        @test all(t[:Ns] isa Tuple{Int, Int} for t in tasks)

        shared = create_param_dict(:Ns => (10, 10), :cfl => 0.5)
        methods = create_method_dict(:upwind => create_param_dict(:scheme => "upwind"))
        varied = create_varied_dict(:Ns__1 => [20, 30], :Ns__y => [5, 15])

        config = SimulationConfig(
            :dummy_tuple_solver,
            shared,
            methods,
            [:upwind];
            varied_params = varied
        )

        run_all_simulations(config; force_overwrite=true, calculate_stats=false)

        for Nx in [20, 30]
            for Ny in [5, 15]
                p = create_param_dict(
                    :Ns => (Nx, Ny),
                    :scheme => "upwind",
                    :cfl => 0.5
                )

                @test does_sim_data_exist(p)
                sim = load_sim_data(p)
                @test sim.params[:Ns] == (Nx, Ny)
                @test size(sim.u, 1) == Nx
            end
        end
    end
    @testset "Data Conversion & Caching Pipeline" begin
        # 1. Generate and save a baseline Eulerian dataset to disk
        params_conv = create_param_dict(:N => 40, :scheme => "upwind", :cfl => 0.5, :test_mode => "conversion")
        sim_base = advection_solver_1d(params_conv)
        
        add_stat!(sim_base,:u_copy,deepcopy(sim_base.u),:all)

        save_sim_data(sim_base; overwrite=true)
        
        # Verify the raw data exists
        @test does_sim_data_exist(params_conv)
        
        @testset "On-the-Fly Conversion (Cache Disabled)" begin
            enable_cache!(false) 
            
            # Eulerian -> Lagrangian
            sim_L = load_sim_data(params_conv, Val(:lagrangian)) 
            @test sim_L isa LSimData
            # Since caching is disabled, the disk check for the conversion should be false
            @test !does_sim_data_exist(params_conv, Val(:lagrangian)) 
            
            # Eulerian -> Resampled Eulerian
            target_res = (20,30)
            sim_E_resampled = load_sim_data(params_conv, Val(:eulerian), target_res) 
            @test sim_E_resampled isa ESimData
            @test size(sim_E_resampled.u) == target_res
            @test !does_sim_data_exist(params_conv, Val(:eulerian), target_res) 
        end

        @testset "Persistent Conversion (Cache Enabled)" begin
            enable_cache!(true) 
            
            # Eulerian -> Lagrangian
            sim_L = load_sim_data(params_conv, Val(:lagrangian)) 
            @test does_sim_data_exist(params_conv, Val(:lagrangian)) 
            
            # Verify the explicit key was saved to the JLD2 file
            conversions = list_available_conversions(params_conv) 
            @test "conv_L" in conversions
            
            # Eulerian -> Resampled Eulerian
            target_res = (25,30)
            sim_E_resampled = load_sim_data(params_conv, Val(:eulerian), target_res) 
            @test does_sim_data_exist(params_conv, Val(:eulerian), target_res) 
            
            conversions = list_available_conversions(params_conv) 
            @test "conv_E_25x30" in conversions
            
            # Reset cache state to avoid side effects on other tests
            enable_cache!(false) 
        end
        
        @testset "Algorithmic Integrity (Scatter L -> E)" begin
            # Test that we can dynamically reconstruct Eulerian data from the Lagrangian particles
            sim_L = load_sim_data(params_conv, Val(:lagrangian)) 
            t_length = length(sim_L.t)
            
            # Convert back to Eulerian using the scatter algorithm
            sim_E_reconstructed = convert_to_eulerian(sim_L, (30,t_length)) 
            @test sim_E_reconstructed isa ESimData
            @test size(sim_E_reconstructed.u) == (30,t_length)
            
            # Verify basic dimensionality integrity 
            time_idx = get_time_dim(sim_E_reconstructed.domain) 
            @test length(sim_E_reconstructed.axes[time_idx]) == t_length
        end
    end
    @testset "Data Health Checking (check_data)" begin
        # 1. Test Eulerian DataFrame Generation
        params_conv = create_param_dict(:N => 40, :scheme => "upwind", :cfl => 0.5, :test_mode => "conversion")
        sim_E = load_sim_data(params_conv)
        
        df_E = check_data(sim_E) 
        @test df_E isa DataFrame
        @test size(df_E, 1) == 1 # Eulerian data must aggregate into a single row
        @test "C1_NaNs" in names(df_E)
        @test df_E.C1_NaNs[1] == 0.0 # Our advection solver should not produce NaNs
        
        # 2. Test Lagrangian DataFrame Generation
        sim_L = load_sim_data(params_conv, Val(:lagrangian)) 
        
        df_L = check_data(sim_L) 
        @test df_L isa DataFrame
        @test size(df_L, 1) == length(sim_L.t) # Lagrangian data must have one row per timestep
    end

    @testset "Data Management Deletion" begin
        # 1. Generate a temporary dummy file to safely delete
        delete_params = create_param_dict(:N => 99, :scheme => "upwind", :cfl => 0.5, :test_flag => "delete_me")
        keep_params = create_param_dict(:N => 99, :scheme => "upwind", :cfl => 0.5, :test_flag => "keep_me")
        delete_dummy = advection_solver_1d(delete_params)
        keep_dummy = advection_solver_1d(keep_params)
        save_sim_data(delete_dummy; overwrite=true)
        save_sim_data(keep_dummy; overwrite=true)
        
        @test does_sim_data_exist(delete_params)
        @test does_sim_data_exist(keep_params)
        
        # 2. Test Deletion based on exact key-value matches via Dict
        delete_sim_data(Dict(:N => 99, :test_flag => "delete_me"))
        @test !does_sim_data_exist(delete_params)
        @test does_sim_data_exist(keep_params)
    end

    @testset "Data Management (Rehash)" begin
        # 1. Setup: Create two distinct dummy files
        # One will be targeted for rehashing, the other should be ignored by the filter
        params_target = create_param_dict(:N => 10, :scheme => "upwind", :cfl => 0.5, :target => true, :obsolete => "delete_me")
        params_ignore = create_param_dict(:N => 11, :scheme => "upwind", :cfl => 0.5, :target => false, :obsolete => "keep_me")
        
        sim_target = advection_solver_1d(params_target)
        sim_ignore = advection_solver_1d(params_ignore)
        
        save_sim_data(sim_target; overwrite=true)
        save_sim_data(sim_ignore; overwrite=true)
        
        file_target_old = get_file_name(params_target)
        file_ignore_old = get_file_name(params_ignore)
        
        # 2. Perform the rehash on the entire data directory
        data_dir = joinpath(get_save_path(), "data")
        rehash_sim_data(
            data_dir; 
            delete_old=true, 
            filter_pairs=Dict(:target => true), 
            remove_keys=[:obsolete]
        )
        
        # 3. Verification: Ignored file
        # The file with :target => false should be completely untouched
        @test isfile(file_ignore_old)
        
        # 4. Verification: Target file
        # The old file should have been safely deleted
        @test !isfile(file_target_old)
        
        # We manually construct the expected new parameter state to locate the rehashed file
        params_new = create_param_dict(:N => 10, :scheme => "upwind", :cfl => 0.5, :target => true)
        
        # Verify the new file exists with the updated cryptographic hash
        @test does_sim_data_exist(params_new)
        
        # Load the rehashed data and ensure the internal dictionary was scrubbed correctly
        sim_rehashed = load_sim_data(params_new)
        @test !haskey(sim_rehashed.params, :obsolete)
        @test sim_rehashed.params[:target] == true
    end
    @testset "Lagrangian Pipeline & Custom Stats" begin
        # 1. Create Lagrangian Data
        params_L = create_param_dict(:N => 10, :v => 2.0)
        sim_L = particle_solver_1d(params_L)
        
        @test sim_L isa LSimData
        @test length(sim_L.x) == 3 # 3 time steps simulated
        
        # 2. Test generate_reference_simdata for Lagrangian grids
        # res requires (spatial_res, temporal_res) for a 2D spacetime domain
        ref_L = generate_reference_simdata(exact_particle, params_L, sim_L.domain, (10, 3), Val(:lagrangian))
        @test ref_L isa LSimData
        @test length(ref_L.u) == 3
        
        # 3. Test Custom Stat Injection (add_stat!)
        # Inject a custom execution time metric (scalar, 0 dimensions kept)
        add_stat!(sim_L, :execution_time, 0.042, Symbol[])
        @test haskey(sim_L.stats, :execution_time)
        @test get_kept_dims(:execution_time, sim_L.domain) == Symbol[]
        @test sim_L.stats[:execution_time] == SVector{1, Float64}(0.042)
        
        # 4. Test calculate_all_stats! for Lagrangian series and fields
        # This will evaluate standard series (like :l1error) and our custom :pointwise_diff field
        calculate_all_stats!(sim_L, exact_particle)
        
        # Verify Series Evaluation (Time-retained metrics)
        @test haskey(sim_L.stats, :l1error)
        @test all(err -> isapprox(err[1], 0.0, atol=1e-12), sim_L.stats[:l1error])
        
        # Verify Field Evaluation (All-dimensions-retained metrics)
        @test haskey(sim_L.stats, :pointwise_diff)
        @test length(sim_L.stats[:pointwise_diff]) == 3 # 3 time steps
        @test length(sim_L.stats[:pointwise_diff][1]) == 10 # 10 particles per step
        @test all(diff -> isapprox(diff[1], 0.0, atol=1e-12), sim_L.stats[:pointwise_diff][1])
    end

    @testset "Serialization Primitives (val2str & str2val)" begin
        # 1. Base primitives & edge cases
        @test val2str("") == "<empty>"
        @test str2val("<empty>") == ""
        @test str2val("") == ""

        @test val2str(42) == "42"
        @test str2val("42") === 42

        @test val2str(3.14) == "3.14"
        @test str2val("3.14") === 3.14

        @test val2str(:sample_sym) == ":sample_sym"
        @test str2val(":sample_sym") === :sample_sym

        @test val2str(Float64) == "Float64"
        @test str2val("Float64") === Float64

        # 2. String representation handling (top_level vs nested)
        @test val2str("plain_str"; top_level=true) == "plain_str"
        @test val2str("nested_str"; top_level=false) == "\"nested_str\""
        @test str2val("\"nested_str\"") == "nested_str"

        # 3. Containers (Vectors, Tuples, Dicts)
        v = [1, 2, 3]
        @test str2val(val2str(v)) == v

        tup = (10, "nested", :val)
        @test str2val(val2str(tup)) == tup

        single_tup = (5,)
        @test val2str(single_tup) == "(5,)"
        @test str2val(val2str(single_tup)) == single_tup

        d = Dict(:a => 1, :b => [10, 20])
        parsed_d = str2val(val2str(d))
        @test parsed_d isa Dict
        @test parsed_d[:a] == 1
        @test parsed_d[:b] == [10, 20]

        # 4. Closures & custom fallbacks
        @test val2str(x -> x + 1) == "Closure"
        @test str2val("raw_unparsed_string") == "raw_unparsed_string"
    end

    @testset "Output Formatting & Manual Loading" begin
        # 1. Setup specific datasets for output testing
        params_E = create_param_dict(:N => 25, :scheme => "upwind", :cfl => 0.5, :test_id => "format_E")
        sim_E = advection_solver_1d(params_E)
        save_sim_data(sim_E; overwrite=true)

        params_L = create_param_dict(:N => 10, :v => 1.5, :test_id => "format_L")
        sim_L = particle_solver_1d(params_L)
        
        @testset "Clean Parameter Processing & Printing" begin
            # Test get_clean_params from Dict and AbstractSimData
            clean_from_dict = PDEStudioCore.get_clean_params(params_E)
            clean_from_sim = PDEStudioCore.get_clean_params(sim_E)
            @test clean_from_dict isa Dict{Symbol, Any}
            @test clean_from_dict == clean_from_sim
            @test clean_from_dict[:test_id] == "format_E"

            # Capture stdout using a Pipe
            pipe = Pipe()
            printed_dict = redirect_stdout(pipe) do
                PDEStudioCore.print_clean_params(sim_E)
            end
            close(pipe.in)
            out_str = read(pipe, String)

            @test printed_dict == clean_from_dict
            @test occursin("Cleaned Parameters", out_str)
            @test occursin("test_id => format_E", out_str)
            @test occursin("scheme => upwind", out_str)
        end

        @testset "REPL Display Output (Base.show)" begin
            # Test ESimData text/plain formatting
            out_E = repr("text/plain", sim_E)
            @test occursin("ESimData", out_E)
            @test occursin("Time (T)", out_E)
            @test occursin("Space", out_E)
            @test occursin("Grid Size", out_E)
            @test occursin("Parameters", out_E)
            @test occursin("Statistics", out_E)

            # Test LSimData text/plain formatting
            out_L = repr("text/plain", sim_L)
            @test occursin("LSimData", out_L)
            @test occursin("Time (T)", out_L)
            @test occursin("Space", out_L)
            @test occursin("Particles", out_L)
            @test occursin("Parameters", out_L)
            @test occursin("Statistics", out_L)
        end

        @testset "Manual SimData Loading by Hash" begin
            # Extract a partial hash signature
            full_hash = calculate_hash(params_E)
            partial_hash = full_hash[1:8]

            # Attempt to load using just the string prefix
            manual_sim = load_sim_data(partial_hash)["raw"]
            @test manual_sim isa ESimData
            @test manual_sim.params[:test_id] == "format_E"

            # Verify that providing a bad hash properly throws the custom exception
            @test_throws PDEStudioCore.SimFileNotFoundError load_sim_data("00000000000000000")
        end
    end
    @testset "2+1D Eulerian SimData Creation" begin
        # 1. Setup spatial grids and time vector
        Nx, Ny, Nt = 12, 8, 4
        x_vec = collect(range(0.0, 2.0, length=Nx))
        y_vec = collect(range(-1.0, 1.0, length=Ny))
        t_vec = collect(range(0.0, 0.5, length=Nt))

        # Generate 2D meshgrids
        x_grid = [x for x in x_vec, _ in y_vec]
        y_grid = [y for _ in x_vec, y in y_vec]

        # 2. Allocate 3D spacetime array of SVectors (M = 1 component)
        u_tensor = [
            SVector{1, Float64}(x * y + t) 
            for x in x_vec, y in y_vec, t in t_vec
        ]

        params_2d = create_param_dict(:Nx => Nx, :Ny => Ny, :Nt => Nt, :dim => "2+1D")

        # 3. Trigger the 2D space + 1D time constructor
        sim_2d = create_sim_data(x_grid, y_grid, u_tensor, t_vec, params_2d; time_dim=:t, x_dim=:x, y_dim=:y)

        # 4. Verify types and dimensional invariants (D = 3, DS = 2, M = 1)
        @test sim_2d isa ESimData{3, 2, 1, Float64}
        @test sim_2d.domain.dim_keys == (:x, :y, :t)
        @test sim_2d.domain.time_dim == :t
        @test size(sim_2d.u) == (Nx, Ny, Nt)

        # Verify axes mapping
        @test length(sim_2d.axes) == 3
        @test sim_2d.axes[1] ≈ x_vec
        @test sim_2d.axes[2] ≈ y_vec
        @test sim_2d.axes[3] ≈ t_vec

        # Verify initial stats registry and solution assignment
        @test haskey(sim_2d.stats, :Solution)
        @test sim_2d.stats[:Solution] === sim_2d.u
    end
    @testset "README Example" begin
        @test include("examples/advection_1d.jl")
    end
end
