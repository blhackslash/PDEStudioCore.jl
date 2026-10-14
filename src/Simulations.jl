# ==============================================================================
# --- Task generation & the batch pipeline ---
# ==============================================================================

const _REFERENCE_KEYWORDS = ("analytic", "reference", "exact", "baseline", "true")

"""
    is_reference_method(m_name::Symbol)

`true` if a word of the method name (split at `_`, `-` and non-alphanumerics) starts with
"analytic", "reference", "exact", "baseline" or "true" (e.g. `:analytic`, `:exact_solution`).
Reference methods are skipped by [`run_all_simulations`](@ref).
"""
function is_reference_method(m_name::Union{Symbol, AbstractString})
    words = split(lowercase(String(m_name)), r"[^a-z0-9]+"; keepempty = false)
    return any(w -> any(k -> startswith(w, k), _REFERENCE_KEYWORDS), words)
end

"""
    get_ignore_keys(methods::AbstractDict, method_name::Symbol)

The shared keys that `method_name` excludes via its `:ignore` entry.
"""
function get_ignore_keys(method_collection::AbstractDict, method_name::Symbol)
    raw_ignore = get(get(method_collection, method_name, ParamDict()), :ignore, Symbol[])
    if raw_ignore isa AbstractVector || raw_ignore isa Tuple
        return Symbol[Symbol(k) for k in raw_ignore]
    elseif raw_ignore isa Union{AbstractString, Symbol}
        return [Symbol(raw_ignore)]
    end
    return Symbol[]
end

"""
    assemble_params(shared::AbstractDict, methods::AbstractDict, method_name::Symbol)

Shared parameters (minus the method's ignored keys) overridden by the method's own parameters.
The `:ignore` entry itself only controls the assembly and is not part of the result (or the hash).
"""
function assemble_params(shared_params::AbstractDict, method_collection::AbstractDict, method_name::Symbol)
    ignore_keys = get_ignore_keys(method_collection, method_name)
    params = ParamDict()
    for (key, val) in shared_params
        Symbol(key) in ignore_keys || (params[Symbol(key)] = val)
    end
    for (key, val) in get(method_collection, method_name, ParamDict())
        Symbol(key) === :ignore && continue
        params[Symbol(key)] = val
    end
    return params
end

function _tuple_index(s::AbstractString)
    s in ("x", "1") && return 1
    s in ("y", "2") && return 2
    s in ("z", "3") && return 3
    idx = tryparse(Int, s)
    return idx !== nothing && idx >= 1 ? idx : nothing
end

# Splits a varied key into (base key, tuple index); the index is `nothing` for ordinary keys.
function _split_varied_key(k::Symbol)
    m = match(r"^(.+)__([a-zA-Z0-9]+)$", String(k))
    m === nothing && return (k, nothing)
    idx = _tuple_index(m.captures[2])
    return idx === nothing ? (k, nothing) : (Symbol(m.captures[1]), idx)
end

"""
    generate_method_tasks(base_params, active_keys, active_values; ignore_keys = Symbol[])

Expands one method's parameters over the Cartesian product of `active_values`. Keys of the form
`:Ns__x`/`:Ns__1` (`__y`/`__2`, `__z`/`__3`, `__<n>`) set one entry of the tuple parameter `:Ns`.
Varied keys that are absent from `base_params` (or ignored) are not applied to this method.
Returns `(tasks, grid_indices)`.
"""
function generate_method_tasks(base_params::AbstractDict, active_keys::AbstractVector, active_values::AbstractVector;
                               ignore_keys::AbstractVector = Symbol[])
    length(active_keys) == length(active_values) || throw(ArgumentError("active_keys and active_values must have the same length"))
    base = _to_paramdict(base_params)
    ignore = Symbol[Symbol(k) for k in ignore_keys]

    varied_std = Tuple{Int, Symbol}[]       # (index into the value tuple, key)
    varied_tup = Tuple{Int, Symbol, Int}[]  # (index into the value tuple, base key, tuple entry)
    for (i, k) in enumerate(Symbol.(active_keys))
        k in ignore && continue
        base_key, idx = _split_varied_key(k)
        if idx === nothing
            push!(varied_std, (i, k))
        elseif !(base_key in ignore)
            push!(varied_tup, (i, base_key, idx))
        end
    end

    grid = isempty(active_values) ? [()] : collect(Iterators.product(active_values...))
    tasks, grid_indices = ParamDict[], Tuple[]
    for (linear_idx, p_vals) in enumerate(grid)
        task = copy(base)
        for (i, k) in varied_std
            haskey(task, k) && (task[k] = p_vals[i])
        end
        if !isempty(varied_tup)
            updates = Dict{Symbol, Vector{Any}}()
            for (i, base_key, idx) in varied_tup
                haskey(task, base_key) || continue
                arr = get!(() -> collect(Any, task[base_key]), updates, base_key)
                while length(arr) < idx
                    push!(arr, 0.0)
                end
                arr[idx] = p_vals[i]
            end
            for (base_key, arr) in updates
                task[base_key] = all(x -> x isa Integer, arr) ? Tuple(Int.(arr)) : Tuple(arr)
            end
        end
        push!(tasks, task)
        push!(grid_indices, isempty(active_values) ? () : Tuple(CartesianIndices(grid)[linear_idx]))
    end
    return tasks, grid_indices
end

"""
    generate_tasks(config::SimulationConfig)

All parameter sets [`run_all_simulations`](@ref) would execute (reference methods skipped,
duplicates removed). Handy to load the results of a sweep:
`[load_sim_data(p) for p in generate_tasks(config)]`.
"""
function generate_tasks(config::SimulationConfig)
    vkeys = sort!(collect(keys(config.varied)))
    vvals = Vector[config.varied[k] for k in vkeys]
    for (k, v) in zip(vkeys, vvals)
        isempty(v) && @warn "Varied parameter :$k has no values: the Cartesian product is empty, so no task is generated."
    end
    methods = isempty(config.methods) && isempty(config.defaults) ? Union{Nothing, Symbol}[nothing] :
              Union{Nothing, Symbol}[m for m in config.defaults if !is_reference_method(m)]

    tasks = ParamDict[]
    seen = Set{String}()
    used = Set{Symbol}()
    for method in methods
        base = method === nothing ? copy(config.shared) : assemble_params(config.shared, config.methods, method)
        ignore = method === nothing ? Symbol[] : get_ignore_keys(config.methods, method)
        for k in vkeys
            base_key, _ = _split_varied_key(k)
            (haskey(base, base_key) && !(k in ignore) && !(base_key in ignore)) && push!(used, k)
        end
        method_tasks, _ = generate_method_tasks(base, vkeys, vvals; ignore_keys = ignore)
        for task in method_tasks
            h = _param_hash(task)
            h in seen && continue
            push!(seen, h)
            push!(tasks, task)
        end
    end
    for k in vkeys
        k in used || @warn "Varied parameter :$k does not appear in the parameters of any active method and is ignored."
    end
    return tasks
end

function _resolve_solver(params::AbstractDict)
    haskey(params, :sim_func_name) || throw(ArgumentError(
        "the parameters contain no :sim_func_name (the name of the solver function)"))
    name = params[:sim_func_name]
    name isa Function && throw(ArgumentError(
        ":sim_func_name must be the *name* of the solver (a Symbol); functions cannot be hashed"))
    f = resolve_dynamic_function(name)
    f === nothing && throw(ArgumentError("could not resolve the solver function `$name`"))
    return f
end

"""
    run_smart_simulation(sim_func, params; force_overwrite = false, metadata = Dict())

Runs `sim_func(params)` unless a dataset for `params` already exists (then returns `NoSimData()`).
The solver receives a copy of the parameters; the result is always stored under the task
parameters so it can be found again. Returns the produced data (or `NoSimData()`).
"""
function run_smart_simulation(sim_func, params::AbstractDict; force_overwrite::Bool = false,
                              metadata::AbstractDict = Dict{String, Any}())
    !force_overwrite && does_sim_data_exist(params) && return NoSimData()
    task_params = _to_paramdict(params)
    # invokelatest: the solver may have been defined after the caller started (e.g. via `sources`).
    sim_data = Base.invokelatest(sim_func, copy(task_params))
    (sim_data === nothing || sim_data isa NoSimData) && return NoSimData()
    sim_data isa AbstractSimData || throw(ArgumentError(
        "the solver must return an AbstractSimData (see `create_sim_data`), got $(typeof(sim_data))"))
    if _param_hash(sim_data.params) != _param_hash(task_params)
        @warn "The solver returned data whose parameters differ from the task parameters; " *
              "storing it under the task parameters so that it can be found again."
        empty!(sim_data.params)
        merge!(sim_data.params, task_params)
    end
    save_sim_data(sim_data; overwrite = true, verbose = false, metadata = metadata)
    return sim_data
end

function _run_task(params::ParamDict, force_overwrite::Bool, metadata::AbstractDict)
    try
        !force_overwrite && does_sim_data_exist(params) && return :cached
        result = run_smart_simulation(_resolve_solver(params), params; force_overwrite = true, metadata = metadata)
        return result isa NoSimData ? :empty : :ran
    catch e
        @error "Simulation failed for parameters $params" exception = (e, catch_backtrace())
        return :failed
    end
end

function _load_sources(sources::AbstractVector{<:AbstractString})
    for file in sources
        path = abspath(file)
        isfile(path) || throw(ArgumentError("source file not found: $path"))
        Base.include(get_target_module(), path)
    end
    return nothing
end

# Always written (possibly empty), so a rerun without sources replaces digests from an earlier run.
function _source_metadata(sources::AbstractVector{<:AbstractString})
    digests = Dict{String, String}(abspath(f) => bytes2hex(open(sha256, abspath(f))) for f in sources)
    return Dict{String, Any}("sources" => digests)
end

function _stats_task(params::ParamDict, ref_factory, required::Vector{Symbol}, force_overwrite::Bool)
    try
        path = _find_file(params)
        path === nothing && return :missing
        has_reference = ref_factory !== nothing
        (force_overwrite || _needs_stats(path, required, has_reference)) || return :cached
        sim_data = _load_key(path, "raw")
        ref_func = has_reference ? ref_factory(params) : nothing
        calculate_all_stats!(sim_data, ref_func; force_overwrite = force_overwrite)
        return :ran
    catch e
        @error "Statistics failed for parameters $params" exception = (e, catch_backtrace())
        return :failed
    end
end

function _post_task(params::ParamDict, post_func)
    try
        path = _find_file(params)
        path === nothing && return :missing
        sim_data = _load_key(path, "raw")
        post_func(sim_data) === true && save_sim_data(sim_data; overwrite = true, verbose = false)
        return :ran
    catch e
        @error "Post-processing failed for parameters $params" exception = (e, catch_backtrace())
        return :failed
    end
end

# Whether a file is missing (or has stale) entries for any of the required statistics. A statistic that
# was evaluated without a reference and dropped (all NaN) is due again once a reference is available.
function _needs_stats(path::AbstractString, required::Vector{Symbol}, has_reference::Bool = false)
    return jldopen(path, "r") do file
        saved = haskey(file, "stat_keys") ? file["stat_keys"] : Symbol[]
        evaluated = haskey(file, "evaluated_stats") ? file["evaluated_stats"] : Symbol[]
        evaluated_with_reference = haskey(file, "evaluated_with_reference") ? file["evaluated_with_reference"] : false
        file_registry = haskey(file, "registry") ? file["registry"] : nothing
        registry = get_stat_registry()
        for k in required
            if k in saved
                # present: stale only if the kept dimensions changed
            elseif k in evaluated
                has_reference && !evaluated_with_reference && return true
            else
                return true
            end
            file_registry === nothing || get(file_registry, k, nothing) == registry[k] || return true
        end
        return false
    end
end

"""
    run_all_simulations(config::SimulationConfig; force_overwrite = false, calculate_stats = false,
                        force_stats = false, post_process = true, parallel = false)

Runs the complete pipeline for a [`SimulationConfig`](@ref):
0. includes `config.sources` into the target module,
1. runs every task of [`generate_tasks`](@ref) whose data is not on disk yet (all of them with
   `force_overwrite`), in parallel over Julia threads with `parallel = true`,
2. with `calculate_stats = true`, evaluates missing statistics using the reference factory
   `config.reference` (`params -> (x -> u_exact)`); statistics that were dropped earlier for lack of
   a reference are evaluated again once a reference is configured. `force_stats = true` recomputes
   every registered statistic without rerunning the simulations,
3. with `post_process = true`, calls `config.postprocess(data)` for every dataset and saves the
   data when it returns `true`.
Failures of individual tasks are logged and do not abort the batch.
"""
function run_all_simulations(config::SimulationConfig; force_overwrite::Bool = false, calculate_stats::Bool = false,
                             force_stats::Bool = false, post_process::Bool = true, parallel::Bool = false)
    @info "Started simulation pipeline."
    _load_sources(config.sources)
    tasks = generate_tasks(config)
    n = length(tasks)
    if n == 0
        @info "No numerical simulations to run."
        return nothing
    end

    # Every per-task call goes through `invokelatest`, so functions defined after this call started
    # (e.g. in `config.sources`, or a stat preset loaded inside a function) are visible.

    # --- Pass 1: simulations ---
    @info "Pass 1: executing $n simulation task(s) (parallel: $parallel)..."
    metadata = _source_metadata(config.sources)
    status = fill(:pending, n)
    progress = Progress(n; desc = "Running simulations...")
    if parallel
        Threads.@threads for i in 1:n
            status[i] = Base.invokelatest(_run_task, tasks[i], force_overwrite, metadata)
            next!(progress)
        end
    else
        for i in 1:n
            status[i] = Base.invokelatest(_run_task, tasks[i], force_overwrite, metadata)
            next!(progress)
        end
    end
    @info "Pass 1 finished: $(count(==(:ran), status)) run, $(count(==(:cached), status)) cached, " *
          "$(count(==(:empty), status)) without output, $(count(==(:failed), status)) failed."

    # --- Pass 2: statistics ---
    if calculate_stats
        @info "Pass 2: calculating statistics..."
        ref_factory = resolve_dynamic_function(config.reference)
        required = sort!(Symbol[k for k in keys(get_stat_registry()) if k !== :Solution])
        progress = Progress(n; desc = "Calculating stats...")
        for params in tasks
            Base.invokelatest(_stats_task, params, ref_factory, required, force_overwrite || force_stats)
            next!(progress)
        end
    end

    # --- Pass 3: post-processing ---
    post_func = resolve_dynamic_function(config.postprocess)
    if post_process && post_func !== nothing
        @info "Pass 3: custom post-processing..."
        progress = Progress(n; desc = "Post-processing...")
        for params in tasks
            Base.invokelatest(_post_task, params, post_func)
            next!(progress)
        end
    end

    @info "Batch simulation run complete!"
    return nothing
end

# ==============================================================================
# --- Reference generators ---
# ==============================================================================

function _grid_spacing(axes::Tuple, ::Type{T}) where {T}
    return SVector{length(axes), T}(ntuple(d -> length(axes[d]) > 1 ?
        (last(axes[d]) - first(axes[d])) / T(length(axes[d]) - 1) : one(T), length(axes)))
end

"""
    generate_reference_simdata(ref_func, params, template_domain, res, Val(:eulerian))
    generate_reference_simdata(ref_func, params, template_domain, res, Val(:lagrangian))
    generate_reference_simdata(ref_func, params, template_domain, res, kind::Symbol)

Evaluates the exact solution `ref_func(x::SVector{D})` on a uniform grid spanning the bounding box
of `template_domain` with spacetime resolution `res`, as Eulerian data or as stationary particles.
"""
generate_reference_simdata(ref_func, params::AbstractDict, template_domain::DomainInfo, res::Tuple, kind::Symbol) =
    generate_reference_simdata(ref_func, params, template_domain, res, Val(kind))

function generate_reference_simdata(ref_func, params::AbstractDict, template_domain::DomainInfo{D, T}, res::Tuple,
                                    ::Val{:eulerian}) where {D, T}
    res_ = _check_res(res, D)
    axes = ntuple(d -> _linspace(template_domain.mins[d], template_domain.maxs[d], res_[d]), Val(D))
    M = length(ref_func(SVector{D, T}(ntuple(d -> axes[d][1], Val(D)))))
    return _reference_eulerian(ref_func, params, template_domain, axes, Val(M))
end

function _reference_eulerian(ref_func, params, domain::DomainInfo{D, T}, axes::NTuple{D, Vector{T}}, ::Val{M}) where {D, T, M}
    u = Array{SVector{M, T}, D}(undef, map(length, axes))
    Threads.@threads for I in CartesianIndices(u)
        u[I] = _as_svector(ref_func(SVector{D, T}(ntuple(d -> axes[d][I[d]], Val(D)))), Val(M), T)
    end
    ref_domain = _copy_domain(domain; spacing = _grid_spacing(axes, T))
    return ESimData{D, _nspace(domain), M, T}(params, ref_domain, axes, u)
end

function generate_reference_simdata(ref_func, params::AbstractDict, template_domain::DomainInfo{D, T}, res::Tuple,
                                    ::Val{:lagrangian}) where {D, T}
    return _reference_lagrangian(ref_func, params, template_domain, _check_res(res, D), Val(_nspace(template_domain)))
end

function _reference_lagrangian(ref_func, params, domain::DomainInfo{D, T}, res::NTuple{D, Int}, ::Val{DS}) where {D, T, DS}
    s_axes = ntuple(d -> _linspace(domain.mins[d], domain.maxs[d], res[d]), Val(DS))
    points = vec([SVector{DS, T}(ntuple(d -> s_axes[d][I[d]], Val(DS))) for I in CartesianIndices(ntuple(d -> res[d], Val(DS)))])
    t_vec = D > DS ? _linspace(domain.mins[D], domain.maxs[D], res[D]) : T[0]
    first_point = D > DS ? SVector{D, T}(points[1]..., t_vec[1]) : SVector{D, T}(points[1]...)
    M = length(ref_func(first_point))
    spacing = D > DS ? _grid_spacing((s_axes..., t_vec), T) : _grid_spacing(s_axes, T)
    return _reference_particles(ref_func, params, _copy_domain(domain; spacing = spacing), points, t_vec, Val(M))
end

function _reference_particles(ref_func, params, domain::DomainInfo{D, T}, points::Vector{SVector{DS, T}},
                              t_vec::Vector{T}, ::Val{M}) where {D, T, DS, M}
    nt = length(t_vec)
    u = Vector{Vector{SVector{M, T}}}(undef, nt)
    Threads.@threads for n in 1:nt
        tv = t_vec[n]
        u[n] = [_as_svector(ref_func(D > DS ? SVector{D, T}(p..., tv) : SVector{D, T}(p...)), Val(M), T) for p in points]
    end
    return LSimData{D, DS, M, T}(params, domain, t_vec, fill(points, nt), u)
end
