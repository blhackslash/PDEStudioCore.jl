# ==============================================================================
# --- Parameter serialisation, safe parsing & hashing ---
# ==============================================================================

"""
    val2str(v; top_level::Bool = true)

Converts a value into a Julia-literal-like string (the basis of the parameter hash).
Top-level strings are written unquoted, strings inside containers are quoted, functions become
`"Closure"` and custom structs are written as `TypeName(field=value, ...)`.
"""
function val2str(v; top_level::Bool = true)
    if v isa AbstractString && isempty(v)
        return "<empty>"
    elseif isa(v, Function)
        return "Closure"
    elseif isa(v, Val)
        return string(v)
    elseif isa(v, Symbol)
        return repr(v)
    elseif isa(v, AbstractString)
        return top_level ? String(v) : repr(v)
    elseif isa(v, Number) || isa(v, Type)
        return string(v)
    elseif isa(v, AbstractArray)
        T = eltype(v)
        prefix = T === Any ? "Any" : string(T)
        elements = join([val2str(x; top_level = false) for x in v], ", ")
        return "$(prefix)[$elements]"
    elseif isa(v, Tuple)
        elements = join([val2str(x; top_level = false) for x in v], ", ")
        return length(v) == 1 ? "($elements,)" : "($elements)"
    elseif isa(v, Dict)
        K, V = keytype(v), valtype(v)
        sorted_keys = sort(collect(keys(v)), by = string)
        elements = join(["$(val2str(k; top_level = false)) => $(val2str(v[k]; top_level = false))" for k in sorted_keys], ", ")
        return "Dict{$K, $V}($elements)"
    else
        type_name = string(typeof(v).name.name)
        fields = propertynames(v)
        isempty(fields) && return type_name
        field_strs = ["$f=$(val2str(getproperty(v, f); top_level = false))" for f in fields]
        return "$type_name(" * join(field_strs, ", ") * ")"
    end
end

# ------------------------------------------------------------------------------
# Safe literal evaluation (str2val never executes arbitrary code)
# ------------------------------------------------------------------------------

struct _NotALiteral <: Exception end

# (`true`/`false` are parsed as Bool literals already and need no entry.)
const _LITERAL_CONSTANTS = Dict{Symbol, Any}(
    :nothing => nothing, :missing => missing,
    :Inf => Inf, :NaN => NaN, :Inf16 => Inf16, :Inf32 => Inf32, :Inf64 => Inf64,
    :NaN16 => NaN16, :NaN32 => NaN32, :NaN64 => NaN64, :pi => π, :π => π, :ℯ => ℯ, :im => im,
)

# Pure functions that may be applied to numeric literals (e.g. "2π", "1/3", "-Inf", "1:10").
const _LITERAL_FUNCTIONS = Dict{Symbol, Function}(
    :+ => +, :- => -, :* => *, :/ => /, :// => //, :^ => ^, :% => %, :(:) => (:),
    :sqrt => sqrt, :exp => exp, :log => log, :log2 => log2, :log10 => log10,
    :sin => sin, :cos => cos, :tan => tan, :abs => abs,
)

function _literal_type(sym::Symbol)
    for mod in (Core, Base, @__MODULE__)
        if isdefined(mod, sym)
            T = getglobal(mod, sym)
            T isa Type && return T
        end
    end
    throw(_NotALiteral())
end

_literal_type(ex::Expr) = (T = _literal(ex); T isa Type ? T : throw(_NotALiteral()))
_literal_type(::Any) = throw(_NotALiteral())

_literal(x::Union{Number, AbstractString, AbstractChar}) = x
_literal(q::QuoteNode) = q.value isa Symbol ? q.value : throw(_NotALiteral())
_literal(sym::Symbol) = haskey(_LITERAL_CONSTANTS, sym) ? _LITERAL_CONSTANTS[sym] : _literal_type(sym)
_literal(::Any) = throw(_NotALiteral())

function _literal(ex::Expr)
    head, args = ex.head, ex.args
    if head === :vect
        return Base.vect(map(_literal, args)...)
    elseif head === :tuple
        any(a -> a isa Expr && a.head in (:(=), :kw, :parameters), args) && throw(_NotALiteral())
        return tuple(map(_literal, args)...)
    elseif head === :ref
        return getindex(_literal_type(args[1]), map(_literal, args[2:end])...)
    elseif head === :curly
        return Core.apply_type(_literal_type(args[1]), map(_literal, args[2:end])...)
    elseif head === :call
        return _literal_call(args[1], map(_literal, args[2:end]))
    end
    throw(_NotALiteral())
end

function _literal_call(f, values::Vector)
    if f === :(=>) && length(values) == 2
        return values[1] => values[2]
    elseif f isa Symbol && haskey(_LITERAL_FUNCTIONS, f) && all(v -> v isa Number, values)
        return _LITERAL_FUNCTIONS[f](values...)
    end
    T = _literal_type(f)
    if T <: AbstractDict && all(v -> v isa Pair, values)
        return T(values...)
    elseif T <: Val && isempty(values)
        return T()
    elseif T === Symbol && length(values) == 1 && values[1] isa AbstractString
        return Symbol(values[1])
    elseif T <: Number && length(values) == 1 && values[1] isa Number
        return T(values[1])
    end
    throw(_NotALiteral())
end

"""
    str2val(val_str::AbstractString)

Inverse of [`val2str`](@ref): parses literals (numbers, strings, symbols, types, tuples, typed or
untyped vectors, `Dict{K, V}(...)`, `Val{...}()`, and arithmetic on numbers such as `2π` or `1/3`).
Anything else is returned as a (cleaned) string. Arbitrary code is never evaluated.
"""
function str2val(val_str::AbstractString)
    s = strip(val_str)
    (s == "<empty>" || isempty(s)) && return ""
    try
        return _literal(Meta.parse(s))
    catch
        return replace(s, r"^\"|\"$" => "")
    end
end

# Normalisation used before saving and hashing: strings are kept verbatim (they are never
# re-interpreted as code), closures become "Closure", other values round-trip through val2str.
_normalize_param(v::AbstractString) = String(v)
_normalize_param(v::Union{Nothing, Missing}) = v  # would otherwise become the types `Nothing`/`Missing`
_normalize_param(v) = str2val(val2str(v))
_normalize_params(params) = ParamDict(Symbol(k) => _normalize_param(v) for (k, v) in params)

"""
    calculate_hash(params::AbstractDict)

SHA-256 hash (hex string) of the parameters, computed from the sorted `key => val2str(value)` pairs.
"""
function calculate_hash(params::AbstractDict)
    sorted_keys = sort!(collect(keys(params)), by = string)
    to_hash = join(["$k => $(val2str(params[k]))" for k in sorted_keys], ", ")
    return bytes2hex(sha256(to_hash))
end

"""
    get_clean_params(params::AbstractDict)
    get_clean_params(data::AbstractSimData)

Returns the normalised copy of the parameters, exactly as it is stored on disk and hashed.
"""
get_clean_params(params::AbstractDict) = _normalize_params(params)
get_clean_params(sim_data::AbstractSimData) = get_clean_params(sim_data.params)

"""
    print_clean_params(data)

Prints the normalised parameters (of a dictionary or a dataset) in alphabetical order and returns them.
"""
function print_clean_params(data)
    clean_dict = get_clean_params(data)
    println("\n--- Cleaned Parameters ---")
    for (k, v) in sort(collect(clean_dict), by = x -> string(x[1]))
        println("  $k => $v")
    end
    println("--------------------------\n")
    return clean_dict
end
