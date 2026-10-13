@testset "create_sim_data: N-dimensional Eulerian" begin
    p = create_param_dict(:case => "nd")
    x = collect(range(0.0, 1.0, length = 5))
    y = collect(range(-1.0, 1.0, length = 4))
    z = [0.0, 0.5, 1.0]
    t = collect(range(0.0, 0.3, length = 3))

    # 1D + time, SVector field
    s1 = create_sim_data(x, [SVector(xi * ti, xi + ti) for xi in x, ti in t], t, p)
    @test s1 isa ESimData{2, 1, 2, Float64}
    @test s1.domain.dim_keys == (:x, :t) && s1.stats[:Solution] === s1.u

    # 2D + time from axis vectors, scalar (numeric) field
    u2 = [xi + yi + ti for xi in x, yi in y, ti in t]
    s2 = create_sim_data((x, y), u2, t, p)
    @test s2 isa ESimData{3, 2, 1, Float64}
    @test s2.u[2, 3, 1] == SVector(x[2] + y[3] + t[1])
    @test s2.domain.spacing ≈ SVector(0.25, 2 / 3, 0.15)

    # 3D static field with custom dimension names
    s3 = create_sim_data((x, y, z), [SVector(xi, yi, zi) for xi in x, yi in y, zi in z], p; dims = (:r, :θ, :φ))
    @test s3 isa ESimData{3, 3, 3, Float64}
    @test s3.domain.dim_keys == (:r, :θ, :φ) && !has_time(s3)
    @test get_kept_dims(:Solution, s3.domain) == [:r, :θ, :φ]

    # 4D static (generic default names) and 3D + time
    @test create_sim_data((x, y, z, [0.0, 1.0]), zeros(5, 4, 3, 2), p).domain.dim_keys == (:x1, :x2, :x3, :x4)
    s5 = create_sim_data((x, y, z), zeros(5, 4, 3, 3), t, p)
    @test s5 isa ESimData{4, 3, 1, Float64} && s5.domain.dim_keys == (:x, :y, :z, :t)

    # Components-last numeric arrays, vector elements, integer and Float32 data
    @test create_sim_data(x, zeros(5, 3, 2), t, p) isa ESimData{2, 1, 2, Float64}
    @test create_sim_data(x, [[1.0, 2.0, 3.0] for _ in x, _ in t], t, p) isa ESimData{2, 1, 3, Float64}
    @test create_sim_data(x, ones(Int, 5, 3), t, p) isa ESimData{2, 1, 1, Float64}
    @test create_sim_data(Float32.(x), ones(Float32, 5, 3), Float32.(t), p) isa ESimData{2, 1, 1, Float32}

    # ndgrid coordinate arrays (tuple form)
    xg = [xi for xi in x, _ in y]
    yg = [yi for _ in x, yi in y]
    s10 = create_sim_data((xg, yg), u2, t, p; dims = (:a, :b))
    @test s10.domain.dim_keys == (:a, :b, :t) && s10.axes[1] ≈ x && s10.axes[2] ≈ y
    # Dimension names must be unique (also against the time dimension)
    @test_throws ArgumentError create_sim_data((xg, yg), u2, t, p; dims = (:a, :a))
    @test_throws ArgumentError create_sim_data((xg, yg), u2, t, p; dims = (:a, :t))
    @test create_sim_data((xg, yg), u2, t, p).axes[2] ≈ y
    # MATLAB/meshgrid-ordered arrays are rejected instead of silently producing wrong axes
    @test_throws ArgumentError create_sim_data((permutedims(xg), permutedims(yg)), u2, t, p)

    # Overridden bounds do not change the integration measure (cell-centred grid)
    N = 10
    xc = [(i - 0.5) / N for i in 1:N]
    s11 = create_sim_data(xc, zeros(N), p; xmins = 0.0, xmaxs = 1.0)
    @test s11.domain.mins[1] == 0.0 && s11.domain.maxs[1] == 1.0
    @test s11.domain.spacing[1] ≈ 1 / N

    # Parameters are copied
    q = create_param_dict(:N => 1)
    s12 = create_sim_data(x, zeros(5, 3), t, q)
    q[:N] = 2
    @test s12.params[:N] == 1

    # Input validation
    @test_throws DimensionMismatch create_sim_data(x, zeros(4, 3), t, p)
    @test_throws ArgumentError create_sim_data(reverse(x), zeros(5, 3), t, p)
    @test_throws ArgumentError create_sim_data(x, zeros(5, 3), t, p; time_dim = nothing)
    @test_throws ArgumentError create_sim_data(x, zeros(5, 3), t, p; dims = (:a, :b))
end

@testset "create_sim_data: Lagrangian" begin
    p = create_param_dict(:case => "particles")
    t = [0.0, 1.0]
    x = [[SVector(0.0, 0.0), SVector(1.0, 2.0)], [SVector(0.5, 0.5), SVector(1.5, 2.5), SVector(1.0, 1.0)]]
    u = [[SVector(1.0), SVector(2.0)], [SVector(1.0), SVector(2.0), SVector(3.0)]]

    s = create_sim_data(x, u, t, p; dims = (:a, :b))
    @test s isa LSimData{3, 2, 1, Float64}
    @test s.domain.dim_keys == (:a, :b, :t)
    @test s.domain.mins ≈ SVector(0.0, 0.0, 0.0) && s.domain.maxs ≈ SVector(1.5, 2.5, 1.0)
    @test s.stats[:Solution] === s.u

    # Scalar particle states
    s2 = create_sim_data(x, [[1.0, 2.0], [1.0, 2.0, 3.0]], t, p)
    @test s2 isa LSimData{3, 2, 1, Float64} && s2.u[2][3] == SVector(3.0)

    # Static particles (no time axis)
    s3 = create_sim_data([SVector(0.0), SVector(1.0), SVector(3.0)], [1.0, 2.0, 3.0], p)
    @test s3 isa LSimData{1, 1, 1, Float64}
    @test !has_time(s3) && s3.t == [0.0] && s3.domain.dim_keys == (:x,)
    @test create_sim_data([SVector(0.0f0), SVector(1.0f0)], Float32[1, 2], p) isa LSimData{1, 1, 1, Float32}
    @test_throws ArgumentError create_sim_data(x, u, t, p; time_dim = nothing)

    # Scalar bounds apply to every spatial dimension
    s4 = create_sim_data(x, u, t, p; xmins = -1.0, xmaxs = 3.0)
    @test s4.domain.mins[1:2] == [-1.0, -1.0] && s4.domain.maxs[1:2] == [3.0, 3.0]

    # Per-particle statistics are checked against the particle layout
    @test_throws DimensionMismatch add_stat!(s, :bad_field, [[1.0]], :all)
    @test_throws DimensionMismatch add_stat!(s, :bad_series, [1.0, 2.0, 3.0], :time)
    add_stat!(s, :tag, [[1.0, 2.0], [1.0, 2.0, 3.0]], :all)
    @test s.stats[:tag][2][3] == SVector(3.0)
end

@testset "2+1D Eulerian SimData creation" begin
    Nx, Ny, Nt = 12, 8, 4
    x_vec = collect(range(0.0, 2.0, length = Nx))
    y_vec = collect(range(-1.0, 1.0, length = Ny))
    t_vec = collect(range(0.0, 0.5, length = Nt))
    x_grid = [x for x in x_vec, _ in y_vec]
    y_grid = [y for _ in x_vec, y in y_vec]
    u_tensor = [SVector{1, Float64}(x * y + t) for x in x_vec, y in y_vec, t in t_vec]
    params_2d = create_param_dict(:Nx => Nx, :Ny => Ny, :Nt => Nt, :dim => "2+1D")

    sim_2d = create_sim_data((x_grid, y_grid), u_tensor, t_vec, params_2d; time_dim = :t, dims = (:x, :y))
    @test sim_2d isa ESimData{3, 2, 1, Float64}
    @test sim_2d.domain.dim_keys == (:x, :y, :t)
    @test sim_2d.domain.time_dim == :t
    @test size(sim_2d.u) == (Nx, Ny, Nt)
    @test length(sim_2d.axes) == 3
    @test sim_2d.axes[1] ≈ x_vec && sim_2d.axes[2] ≈ y_vec && sim_2d.axes[3] ≈ t_vec
    @test haskey(sim_2d.stats, :Solution)
    @test sim_2d.stats[:Solution] === sim_2d.u
end
