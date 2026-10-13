@testset "Serialization primitives (val2str & str2val)" begin
    # Base primitives & edge cases
    @test val2str("") == "<empty>"
    @test str2val("<empty>") == "" && str2val("") == ""
    @test val2str(42) == "42" && str2val("42") === 42
    @test val2str(3.14) == "3.14" && str2val("3.14") === 3.14
    @test val2str(:sample_sym) == ":sample_sym" && str2val(":sample_sym") === :sample_sym
    @test val2str(Float64) == "Float64" && str2val("Float64") === Float64
    @test val2str(missing) == "Missing"

    # Strings: unquoted at top level, quoted inside containers
    @test val2str("plain_str"; top_level = true) == "plain_str"
    @test val2str("nested_str"; top_level = false) == "\"nested_str\""
    @test str2val("\"nested_str\"") == "nested_str"

    # Containers
    @test str2val(val2str([1, 2, 3])) == [1, 2, 3]
    @test str2val(val2str((10, "nested", :val))) == (10, "nested", :val)
    @test val2str((5,)) == "(5,)" && str2val(val2str((5,))) == (5,)
    parsed_d = str2val(val2str(Dict(:a => 1, :b => [10, 20])))
    @test parsed_d isa Dict && parsed_d[:a] == 1 && parsed_d[:b] == [10, 20]
    @test str2val("Dict{Symbol, Int64}()") == Dict{Symbol, Int64}()
    @test isequal(str2val("Float64[1.0, NaN]"), [1.0, NaN])
    @test str2val("Val{:x}()") === Val(:x)
    @test str2val("Symbol(\"a b\")") === Symbol("a b")

    # Numeric expressions typed into a UI
    @test str2val("2π") ≈ 2π
    @test str2val("-Inf") == -Inf
    @test str2val("1/4") == 0.25
    @test str2val("1:3") == 1:3
    @test str2val("nothing") === nothing

    # Closures & fallbacks
    @test val2str(x -> x + 1) == "Closure"
    @test str2val("raw_unparsed_string") == "raw_unparsed_string"
    @test str2val("sin") == "sin"  # function names stay strings

    # Arbitrary code is never executed
    @test str2val("exit()") == "exit()"
    @test str2val("rm(\"x\")") == "rm(\"x\")"
    @test str2val("run(`echo hacked`)") isa String
end

@testset "Parameter normalisation & hashing" begin
    p = create_param_dict(:f => "sin", :c => "pi", :n => "1", :x => 0.1, :tup => (1, 2), :v => [1, 2],
                          :s => :sym, :h => 0.5f0, :e => "", :nothing_val => nothing)
    clean = get_clean_params(p)
    # Strings are never re-interpreted (they used to turn into functions/numbers and break lookups)
    @test clean[:f] == "sin" && clean[:c] == "pi" && clean[:n] == "1" && clean[:e] == ""
    # Float32 parameters print without the `f0` suffix and are stored as Float64 (as before);
    # `nothing` stays `nothing` instead of becoming the type `Nothing`.
    @test clean[:tup] == (1, 2) && clean[:v] == [1, 2] && clean[:s] === :sym && clean[:h] === 0.5
    @test clean[:nothing_val] === nothing
    @test calculate_hash(clean) == calculate_hash(p)
    @test calculate_hash(get_clean_params(clean)) == calculate_hash(p)

    # The hash format is unchanged, so files written by earlier versions are still found
    @test calculate_hash(create_param_dict(:N => 50, :cfl => 0.5, :scheme => "upwind")) ==
          "504a0a25d87114ea4c1c3fa7fe16e28a5329892b6da664b1fbd5804d2bf6d720"
    @test calculate_hash(create_param_dict(:Ns => (20, 15), :sim_func_name => :advection_solver_1d,
                                           :tag => "", :values => [0.1, 0.2])) ==
          "c9f0432adf92ccfec508e15b196b623d2593d3e860ee967cf394686235934985"
end
