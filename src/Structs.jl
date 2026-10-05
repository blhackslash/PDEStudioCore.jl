# --- 1. Type Aliases ---
"""
    ParamDict

A type alias for `Dict{Symbol, Any}`. It stores a flat, resolved list of simulation parameters for a single pipeline execution.
"""
const ParamDict = Dict{Symbol, Any}

"""
    MethodDict

A type alias for `Dict{Symbol, ParamDict}`. It maps specific numerical methods or solver names (e.g., `:upwind`) to their localized parameter overrides.
"""
const MethodDict = Dict{Symbol, ParamDict}

"""
    VariedDict

A type alias for `Dict{Symbol, Vector}`. It defines the parameter space for grid sweeps; the backend will automatically execute the Cartesian product of all vectors provided here.
"""
const VariedDict = Dict{Symbol, Vector}

const _SAVE_ROOT_PATH = Ref{String}(pwd())
const _TARGET_MODULE = Ref{Module}(Main)

"""
    set_target_module!(target_module::Module)

Sets the global reference module where the backend will search when dynamically resolving simulation and analytical reference functions by their `Symbol` names.
"""
set_target_module!(target_module::Module) = (_TARGET_MODULE[] = target_module)
get_target_module() = _TARGET_MODULE[]

# --- 2. Explicit Creator Functions ---
# Converts any generic iterator or mixed string/symbol inputs into strictly typed Symbol-keyed dictionaries

# ParamDict Creators
"""
    create_param_dict(kv...)

Converts generic iterators, pairs, or keyword arguments into a strictly typed `ParamDict` (`Dict{Symbol, Any}`). 
This guarantees type stability across the simulation backend.
"""
create_param_dict(kv::Pair...) = ParamDict(Symbol(k) => v for (k, v) in kv)
create_param_dict(kv) = ParamDict(Symbol(k) => v for (k, v) in kv) 
create_param_dict() = ParamDict()

# MethodDict Creators
"""
    create_method_dict(kv...)

Converts generic inputs into a strictly typed `MethodDict`, automatically parsing nested dictionaries into `ParamDict`s.
"""
create_method_dict(kv::Pair...) = MethodDict(Symbol(k) => ParamDict(Symbol(ki) => vi for (ki, vi) in v) for (k, v) in kv)
create_method_dict(kv) = MethodDict(Symbol(k) => ParamDict(Symbol(ki) => vi for (ki, vi) in v) for (k, v) in kv)
create_method_dict() = MethodDict()

# VariedDict Creators
"""
    create_varied_dict(kv...)

Converts generic inputs into a strictly typed `VariedDict` for defining parameter sweeps.
"""
create_varied_dict(kv::Pair...) = VariedDict(Symbol(k) => v for (k, v) in kv)
create_varied_dict(kv) = VariedDict(Symbol(k) => v for (k, v) in kv)
create_varied_dict() = VariedDict()


# ==============================================================================
# --- 1. Abstract Hierarchy & Metadata ---
# ==============================================================================

# 1. Abstract Hierarchy (Now with T)
"""
    AbstractSimData{D, DS, M, T <: Real}

The root abstract type for all simulation data structures in the backend. 
- `D`: Total spacetime dimensions.
- `DS`: Spatial dimensions.
- `M`: Number of field components.
- `T`: The numeric precision (e.g., `Float64`).
"""
abstract type AbstractSimData{D, DS, M, T <: Real} end

struct NoSimData <: AbstractSimData{0, 0, 0, Real} end

const AbstractStatTensor{M, T} = Union{
    SVector{M, T},
    AbstractArray{SVector{M, T}},
    Vector{Vector{SVector{M, T}}} 
}
const StatDict{M, T} = Dict{Symbol, AbstractStatTensor{M, T}} 

# 2. Modernized DomainInfo (Using SVector and T)
"""
    DomainInfo{D, T <: Real}

Stores metadata defining the mathematical bounding box and dimension identifiers of the simulation.

# Fields
- `dim_keys::Tuple{Vararg{Symbol, D}}`: The strict ordered identifiers for the axes (e.g., `(:x, :y, :t)`).
- `mins::SVector{D, T}`: The spatial/temporal minimum boundaries of the domain.
- `maxs::SVector{D, T}`: The spatial/temporal maximum boundaries of the domain.
- `spacing::SVector{D, T}`: The uniform spacing (dx, dy, dt) along each axis.
- `time_dim::Union{Nothing, Symbol}`: Explicitly identifies which dimension acts as the time vector.
- `stat_registry::Dict`: A local registry tracking which custom statistical dimensions are retained for this specific dataset.
"""
struct DomainInfo{D, T <: Real}
    dim_keys::Tuple{Vararg{Symbol, D}}
    mins::SVector{D, T}
    maxs::SVector{D, T}
    spacing::SVector{D, T}
    time_dim::Union{Nothing,Symbol}
    stat_registry::Dict{Symbol, Union{Symbol, Vector{Symbol}}} 
end

# ==============================================================================
# --- 2. D-Dimensional Data Structures ---
# ==============================================================================

"""
    ESimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}

Represents Eulerian grid data. The mathematical fields are defined on a static, uniform mesh.

# Fields
- `params::ParamDict`: The strictly typed parameters used to generate this data.
- `domain::DomainInfo{D, T}`: The domain metadata bounding the grid.
- `axes::NTuple{D, Vector{T}}`: The exact coordinate vectors for every dimension.
- `u::Array{SVector{M, T}, D}`: The dense, multidimensional grid of simulation state vectors.
- `stats::StatDict{M, T}`: The dictionary containing calculated statistical arrays (like error metrics or mass).
"""
mutable struct ESimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}
    params::ParamDict
    domain::DomainInfo{D, T}
    axes::NTuple{D, Vector{T}}
    u::Array{SVector{M, T}, D}
    stats::StatDict{M, T}
end

"""
    LSimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}

Represents Lagrangian particle data. The fields are defined on scattered points moving freely through space over time.

# Fields
- `params::ParamDict`: The strictly typed parameters used to generate this data.
- `domain::DomainInfo{D, T}`: The domain metadata bounding the global space.
- `t::Vector{T}`: The discrete time steps at which particles were recorded.
- `x::Vector{Vector{SVector{DS, T}}}`: The spatial coordinates of the particles at each time step.
- `u::Vector{Vector{SVector{M, T}}}`: The state vectors of the particles at each time step.
- `stats::StatDict{M, T}`: The dictionary containing calculated statistical arrays.
"""
mutable struct LSimData{D, DS, M, T} <: AbstractSimData{D, DS, M, T}
    params::ParamDict
    domain::DomainInfo{D, T}
    t::Vector{T}
    x::Vector{Vector{SVector{DS, T}}}
    u::Vector{Vector{SVector{M, T}}}
    stats::StatDict{M, T} 
end

"""
    SimulationConfig{F, A, P}

The central orchestration structure defining a complete simulation pipeline run. 
It encapsulates the core simulation function alongside the baseline parameters, method-specific overrides, and multi-dimensional parameter sweeps.

# Fields
- `reference_name::Union{Symbol, Nothing}`: The registered name of the reference function.
- `post_process_name::Union{Symbol, Nothing}`: The registered name of the post-processing function.
- `shared_params::ParamDict`: The baseline parameters shared across all pipeline runs.
- `methods_dict::MethodDict`: Method-specific parameter overrides.
- `active_methods::Vector{Symbol}`: A list of the specific methods from the `methods_dict` to execute.
- `varied_params::VariedDict`: The parameter grid to sweep over (executes the Cartesian product).
- `source_files::Vector{String}`: Optional source files to track for reproducibility.
"""
mutable struct SimulationConfig
    reference_name::Union{Symbol, Nothing}
    post_process_name::Union{Symbol, Nothing}
    shared_params::ParamDict
    methods_dict::MethodDict
    active_methods::Vector{Symbol}
    varied_params::VariedDict
    source_files::Vector{String}
end

"""
    SimulationConfig(shared, methods, defaults; kwargs...)

Constructs a strictly typed `SimulationConfig`. The core simulation function should be specified via the `:sim_func_name` key directly within the `shared` or `methods` dictionaries.
Automatically converts all loosely typed dictionary inputs into `ParamDict`s, `MethodDict`s, and `VariedDict`s.
"""
function SimulationConfig(
    shared::Dict, 
    methods::Dict, 
    defaults::Vector;
    varied_params::Dict = create_varied_dict(),
    ref_func_name::Union{String, Symbol, Nothing} = nothing,
    post_process_name::Union{String, Symbol, Nothing} = nothing,
    source_files::Union{<:AbstractString, Vector{String}} = String[]
)
    # Convert generic dictionaries to enforced Symbol-keyed dictionaries
    shared_sym   = ParamDict(Symbol(k) => v for (k, v) in shared)
    methods_sym  = MethodDict(Symbol(k) => ParamDict(Symbol(ki) => vi for (ki, vi) in v) for (k, v) in methods)
    varied_sym   = VariedDict(Symbol(k) => v for (k, v) in varied_params)
    defaults_sym = Symbol.(defaults)
    
    ref_name_sym = isnothing(ref_func_name) ? nothing : Symbol(ref_func_name)
    post_name_sym = isnothing(post_process_name) ? nothing : Symbol(post_process_name)
    src_files = source_files isa AbstractString ? [String(source_files)] : String.(source_files)

    return SimulationConfig(
        ref_name_sym, post_name_sym, 
        shared_sym, methods_sym, defaults_sym, varied_sym, src_files
    )
end

"""
    resolve_dynamic_function(func_name::Union{Symbol, Nothing}, provided_func::Union{Function, Nothing} = nothing)

A pure lookup mechanism that fetches a compiled function object directly from the registered target module namespace using its `Symbol` name.

# Arguments
- `func_name`: The symbolic name of the function to fetch.
- `provided_func`: An optional fallback; if a compiled function is passed here, it bypasses the lookup and returns it directly.

# Returns
- `Function`: The resolved function object, or `nothing` if the lookup fails.
"""
function resolve_dynamic_function(
    func_name::Union{Symbol, Nothing}, 
    provided_func::Union{Function, Nothing} = nothing
)
    target_module = _TARGET_MODULE[]
    
    if !isnothing(provided_func)
        return provided_func
    end
    
    if isnothing(func_name) || func_name === :none
        return nothing 
    end
    
    try
        # Native symbol lookup directly from memory
        return getglobal(target_module, func_name)
    catch e
        @warn "Failed to resolve function '$func_name' from $target_module."
        return nothing
    end
end