# ==============================================================================
# --- REPL display ---
# ==============================================================================

_fmt(x) = round(x, digits = 3)

function _space_summary(domain::DomainInfo, DS::Int)
    DS == 0 && return "(none)"
    parts = ["$(domain.dim_keys[d]): $(_fmt(domain.mins[d])) ➔ $(_fmt(domain.maxs[d]))" for d in 1:DS]
    return "[" * join(parts, "] × [") * "]"
end

_stat_names(data::AbstractSimData) = join(sort!([string(k) for k in keys(data.stats)]), ", ")

"""
    Base.show(io::IO, ::MIME"text/plain", data::LSimData)

Summary of a Lagrangian dataset: time range, spatial bounding box, particle counts, parameters and statistics.
"""
function Base.show(io::IO, ::MIME"text/plain", data::LSimData{D, DS, M, T}) where {D, DS, M, T}
    println(io, "🟢 LSimData{$D, $DS, $M, $T} (Lagrangian Simulation Data)")
    println(io, "==================================================")
    if D > DS && !isempty(data.t)
        println(io, "  Time (T)   : $(length(data.t)) steps [$(_fmt(first(data.t))) ➔ $(_fmt(last(data.t)))]")
    else
        println(io, "  Time (T)   : Static (1 step)")
    end
    println(io, "  Space      : ", _space_summary(data.domain, DS))
    counts = length.(data.x)
    lo, hi = isempty(counts) ? (0, 0) : extrema(counts)
    println(io, "  Particles  : ", lo == hi ? "$lo" : "$lo to $hi (variable)")
    println(io, "  Parameters : $(length(data.params)) keys")
    print(io, "  Statistics : ", _stat_names(data))
end

"""
    Base.show(io::IO, ::MIME"text/plain", data::ESimData)

Summary of an Eulerian dataset: time range, spatial bounding box, grid size, parameters and statistics.
"""
function Base.show(io::IO, ::MIME"text/plain", data::ESimData{D, DS, M, T}) where {D, DS, M, T}
    println(io, "🟦 ESimData{$D, $DS, $M, $T} (Eulerian Grid Data)")
    println(io, "==================================================")
    if D > DS
        t = data.axes[D]
        println(io, "  Time (T)   : $(length(t)) steps [$(_fmt(first(t))) ➔ $(_fmt(last(t)))]")
    else
        println(io, "  Time (T)   : Static (1 step)")
    end
    println(io, "  Space      : ", _space_summary(data.domain, DS))
    grid = ntuple(d -> length(data.axes[d]), DS)
    println(io, "  Grid Size  : $(join(grid, " × ")) ($(prod(grid)) spatial points)")
    println(io, "  Parameters : $(length(data.params)) keys")
    print(io, "  Statistics : ", _stat_names(data))
end

# Compact one-line forms (used inside containers, e.g. a Vector of datasets).
function Base.show(io::IO, data::ESimData{D, DS, M, T}) where {D, DS, M, T}
    print(io, "ESimData{$D, $DS, $M, $T}(", join(size(data.u), "×"), " on ", join(data.domain.dim_keys, "×"),
          ", ", length(data.stats), " stats)")
end

function Base.show(io::IO, data::LSimData{D, DS, M, T}) where {D, DS, M, T}
    print(io, "LSimData{$D, $DS, $M, $T}(", length(data.t), " steps, ", maximum(length, data.x; init = 0),
          " particles, ", length(data.stats), " stats)")
end

_value_summary(x::SVector) = string(x)
_value_summary(x::AbstractArray) = summary(x)
_value_summary(x) = string(typeof(x))

function Base.show(io::IO, ::MIME"text/plain", view::StatsView)
    n = length(view)
    print(io, "StatsView with $n entr", n == 1 ? "y" : "ies", ":")
    for k in sort!(collect(keys(view)), by = k -> (k !== :Solution, string(k)))
        dims = get_kept_dims(k, view.data.domain)
        print(io, "\n  :", k, " => ", _value_summary(view[k]), "  [", join(dims, ", "), "]")
    end
end

Base.show(io::IO, view::StatsView) = print(io, "StatsView(", join((":" * string(k) for k in keys(view)), ", "), ")")

function Base.show(io::IO, ::MIME"text/plain", c::SimulationConfig)
    solver = get(c.shared, :sim_func_name, nothing)
    println(io, "SimulationConfig")
    println(io, "  shared      : ", length(c.shared), " parameter(s)", solver === nothing ? "" : " (solver: $solver)")
    println(io, "  methods     : ", isempty(c.methods) ? "(none)" : join(sort!(collect(keys(c.methods))), ", "))
    println(io, "  defaults    : ", isempty(c.defaults) ? "(none)" : join(c.defaults, ", "))
    println(io, "  varied      : ", isempty(c.varied) ? "(none)" :
            join(["$k ($(length(v)) values)" for (k, v) in sort!(collect(c.varied), by = first)], ", "))
    println(io, "  reference   : ", something(c.reference, "(none)"))
    println(io, "  postprocess : ", something(c.postprocess, "(none)"))
    print(io, "  sources     : ", isempty(c.sources) ? "(none)" : join(c.sources, ", "))
end
