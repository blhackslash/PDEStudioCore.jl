"""
    PDEStudioCore

A headless-safe backend for running, storing and analysing PDE simulations.

Provides strictly typed Eulerian ([`ESimData`](@ref)) and Lagrangian ([`LSimData`](@ref)) datasets of
any dimension, deterministic parameter hashing with atomic on-disk caching, parameter sweeps through
[`SimulationConfig`](@ref), and a multithreaded statistics pipeline.
"""
module PDEStudioCore

using LinearAlgebra, StaticArrays, ProgressMeter, Polyester, Interpolations
using JLD2, FileIO, SHA, Dates, DataFrames

# --- Core types ---
export AbstractSimData, ESimData, LSimData, NoSimData, DomainInfo, StatsView, SimulationConfig
export ParamDict, MethodDict, VariedDict, StatDict, StatRegistry
export create_param_dict, create_method_dict, create_varied_dict, create_sim_data

# --- Globals & settings ---
export set_save_path!, get_save_path, set_target_module!, get_target_module, enable_cache!
export register_stat!, delete_stat!, add_stat!, calc_stat, get_kept_dims, get_kept_indices, get_integration_measure
export get_stat_registry, set_stat_registry!, reset_stat_registry!, set_stat_preset!

# --- Simulation & data pipeline ---
export run_all_simulations, generate_tasks, generate_method_tasks, run_smart_simulation
export load_sim_data, save_sim_data, get_file_name, does_sim_data_exist, list_available_conversions
export find_sim_data, delete_sim_data, rehash_sim_data, check_data, SimFileNotFoundError
export calculate_hash, val2str, str2val, get_clean_params, print_clean_params
export calculate_all_stats!, generate_reference_simdata, get_time_dim, has_time
export convert_to_eulerian, convert_to_lagrangian, resample_eulerian

# --- Utilities ---
export get_ignore_keys, assemble_params, is_reference_method, resolve_dynamic_function

include("Structs.jl")
include("SimData.jl")
include("Config.jl")
include("Serialization.jl")
include("IOUtils.jl")
include("DataManagement.jl")
include("Constructors.jl")
include("ConversionUtils.jl")
include("StatCalculation.jl")
include("Simulations.jl")
include("Display.jl")

function __init__()
    _SAVE_ROOT_PATH[] = pwd()
    _TARGET_MODULE[] = Main
    return nothing
end

end
