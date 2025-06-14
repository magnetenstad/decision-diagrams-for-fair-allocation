using Test
using Allocations
using JuMP
using Gurobi

include("test_util.jl")
include("ac_bnb.jl")
include("bc_bins.jl")
include("bc_bnb.jl")
include("bc_flow.jl")

P = [
    (2, [1 9 2 5 1 5 2 1], 13),
    (3, [93 41 34 63 80], 93),
    (3, [44 75 45 6 61 38 11 63 63 20 4 51 18 43 13 49 5 35], 214),
    (4, [56 15 78 64 40 67], 67),
    (4, [47 78 56 29 10], 39),
    (4, [69 49 1 92 83], 50),
    (4, [77 13 45 48 45 2 28 40 56 53 96 91 46 65 19 28 30 27 36], 211),
    (5, [92 57 18 67 77 46 25 57 68 55 9 83 91 74 98 27 8 91], 208),
    (8, [56 81 4 4 4 46 90 65 52 46 96 56 34 41 56 96 58 55], 108),
    (11, [30 94 40 13 53 73 1 40 22 56 35 17], 14),
    (12, [89 31 27 83 47 37 18 73 70 66 76 6 88 37], 37),
    (12, [41 58 72 64 5 88 4 21 28 12 11 44 51 85 17 3 17], 28),
    (16, [36 97 74 94 66 11 32 13 24 6 37 13 2 72 18 77 15 27 31 52 28 67 25 5 43 5 59], 52),
    (19, [29 78 86 49 54 72 77 23 99 74 60 60 2 30 36 61 73 87 18 15 43 80 92 90 31 78 47 25], 74),
    (29, [21 39 89 39 5 94 50 55 23 73 37 22 78 89 73 85 50 62 23 93 33 12 61 43 20 32 68 71 13 40 2 83 24], 23),
    (38, [98 72 11 5 81 4 70 46 76 52 2 38 50 10 40 29 75 6 67 39 13 63 83 78 26 5 69 87 78 3 15 28 11 21 40 62 95 30 6 24], 5)
]

methods_for_mms_with_allocations = [
    ("BC-BinS", bc_bins),
    ("BC-Flow", bc_flow),
]

methods_for_mms = [
    methods_for_mms_with_allocations...,
    ("Allocations.jl", (V, i; kwds...) -> Allocations.mms(V, i)),
    ("AC-BnB", ac_bnb),
    ("BC-BnB", bc_bnb),
]

function test_unconstrained()

    Allocations.conf.MIP_SOLVER = optimizer_with_attributes(
        Gurobi.Optimizer,
        "log_to_console" => false,
    )

    @testset "mms values" begin
        for (n, V_i, _mms) in P
            V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))

            for (method_name, method) in methods_for_mms
                error = []
                result = method(V, 1; error=error)
                fail = !(@test(
                    haskey(result, :mms) && isapprox(result.mms, _mms)
                ) isa Test.Pass)

                if fail
                    println("$method_name returned $result, expected (mms=$_mms,) for V=$V_i")
                    continue
                end

                fail = !(@test(
                    length(error) == 0
                ) isa Test.Pass)
                if fail
                    println("$method_name: error was $error for V=$V_i")
                end
            end
        end
    end

    @testset "valid allocations" begin
        for (n, V_i, _mms) in P
            V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))

            for (method_name, method) in methods_for_mms_with_allocations
                result = method(V, 1)

                fail = !(@test(
                    haskey(result, :X) && (result.X === nothing || result.X isa Matrix)
                ) isa Test.Pass)

                if fail
                    println("$method_name returned $result, expected (X=Matrix,) for V=$V_i")
                    continue
                end

                if result.X === nothing
                    println("warning: $method_name returned X=nothing for V=$V_i")
                    continue
                end

                test_valid_allocation(V, result.X, _mms)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    test_unconstrained()
end
