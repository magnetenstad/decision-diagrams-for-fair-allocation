using Test
using Allocations
using Graphs
using JuMP
using Gurobi

include("test_util.jl")
include("bc_bins.jl")
include("bc_flow.jl")

methods_no_charity = [
    ("BC-BinS", (V, i, C) -> bc_bins(V, i, C; charity=false)),
    # ("BC-Flow", (V, i, C) -> bc_flow(V, i, C; charity=false)),
]

methods_charity = [
    ("BC-BinS", (V, i, C) -> bc_bins(V, i, C; charity=true)),
    # ("BC-Flow", (V, i, C) -> bc_flow(V, i, C; charity=true)),
]

function test_constrained()

    Allocations.conf.MIP_SOLVER = optimizer_with_attributes(
        Gurobi.Optimizer,
        "log_to_console" => false,
    )

    @testset "connectivity with charity" begin
        n = 3
        i = 1
        V_i = [45 73 3 40 8 32 20 33 96]
        V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
        graph = SimpleGraph{Int64}(8, [Int64[], [3, 7], [2, 4, 6, 7], [3, 6, 9], Int64[], [3, 4], [2, 3, 9], Int64[], [4, 7]])
        C = Constraints(Connections(graph))
        _mms = round(Int, mms(V, i, C, min_owners=0).mms)

        for (_, method) in methods_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)

            for i in agents(V)
                bundle = Set(g for (g, v) in enumerate(result.X[i, :]) if v > 0)
                @assert(length(bundle) > 0)
                @test is_connectable(bundle, [], graph)
            end
        end
    end

    # @testset "connectivity b with charity" begin
    #     # TODO: is the Allocations.jl impl or the BBD impl incorrect?
    #     n = 2
    #     i = 1
    #     V_i = [60 81 12 7 90 60 86 70 70]
    #     V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
    #     graph = SimpleGraph{Int64}(3, [[7, 8], Int64[], Int64[], Int64[], Int64[], Int64[], [1, 9], [1], [7]])
    #     C = Constraints(Connections(graph))
    #     _mms = round(Int, mms(V, i, C, min_owners=0).mms)

    #     for (_, method) in methods_charity
    #         result = method(V, i, C)
    #         @test isapprox(result.mms, _mms)
    #         # Both: 130 != 232
    #         # test_valid_allocation(V, result.X, _mms)

    #         for i in agents(V)
    #             bundle = Set(g for (g, v) in enumerate(result.X[i, :]) if v > 0)
    #             @assert(length(bundle) > 0)
    #             @test is_connectable(bundle, [], graph)
    #         end
    #     end
    # end

    @testset "connectivity without charity" begin
        n = 3
        i = 1
        V_i = [45 73 3 40 8 32 20 33 96]
        V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
        graph = SimpleGraph{Int64}(8, [Int64[], [3, 7], [2, 4, 6, 7], [3, 6, 9], Int64[], [3, 4], [2, 3, 9], Int64[], [4, 7]])
        C = Constraints(Connections(graph))
        @test_throws "INFEASIBLE" _mms = round(Int, mms(V, i, C, min_owners=1).mms)
        _mms = 0

        for (name, method) in methods_no_charity
            result = method(V, i, C)
            fail = !isapprox(result.mms, _mms)
            @test !fail
            if fail
                println("$name: $(result.mms) != $_mms")
            end
        end
    end

    @testset "connectivity b without charity" begin
        n = 2
        i = 1
        V_i = [60 81 12 7 90 60 86 70 70]
        V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
        graph = SimpleGraph{Int64}(3, [[7, 8], Int64[], Int64[], Int64[], Int64[], Int64[], [1, 9], [1], [7]])
        C = Constraints(Connections(graph))
        @test_throws "INFEASIBLE" _mms = round(Int, mms(V, i, C, min_owners=1).mms)
        _mms = 0

        for (name, method) in methods_no_charity
            result = method(V, i, C)
            fail = !isapprox(result.mms, _mms)
            @test !fail
            if fail
                println("$name: $(result.mms) != $_mms")
            end
        end
    end

    @testset "cardinality" begin
        n = 7
        V_i = [99 60 9 1 25 3 3 5 37 68 55]
        V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
        cardinality = Counts([Allocations.Category(Set([10]), 1), Allocations.Category(Set([4, 7, 2, 11]), 4), Allocations.Category(Set([8, 3]), 1), Allocations.Category(Set([5, 6, 9]), 1), Allocations.Category(Set([1]), 1)])
        C = Constraints(cardinality)
        i = 1

        # V = Additive([1 9 2 5 1 5 2 1; 9 4 8 5 1 7 2 8])
        # cardinality = Counts(
        #     [1, 2, 3] => 2,
        #     [4, 5, 6] => 2,
        #     [6, 7] => 1,
        #     [7, 8] => 1,
        # )
        # C = Constraints(cardinality)
        _mms = round(Int, mms(V, i, C).mms)

        for (_, method) in methods_no_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)
            test_valid_allocation(V, result.X, _mms)

            for i in agents(V)
                for category in cardinality
                    @test sum(result.X[i, g]
                              for g in category.members) <= category.threshold
                end
            end
        end
    end

    @testset "conflicts" begin
        V = Additive([1 9 2 5 1 5 2 1; 9 4 8 5 1 7 2 8])
        i = 1
        edges = [(1, 2), (1, 5), (1, 8), (2, 4), (4, 6)]
        graph = SimpleGraph(Edge.(edges))
        C = Constraints(Conflicts(graph))
        _mms = round(Int, mms(V, i, C).mms)

        for (_, method) in methods_no_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)
            test_valid_allocation(V, result.X, _mms)

            for i in agents(V)
                for (a, b) in edges
                    @test !(result.X[i, a] > 0 && result.X[i, b] > 0)
                end
            end
        end
    end

    @testset "conflicts b" begin
        V = Additive{Matrix{Int64}}([48 78 35 73; 48 78 35 73; 48 78 35 73])
        graph = SimpleGraph{Int64}(5, [[2, 3, 4], [1, 3], [1, 2, 4], [1, 3]])
        C = Conflicts(graph)

        i = 1
        _mms = 35

        for (_, method) in methods_no_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)
            test_valid_allocation(V, result.X, _mms)

            A = Allocations.Allocation(na(V), ni(V), (i => [g for g in items(V) if result.X[i, g] > 0] for i in agents(V)))

            @test Allocations.check(V, A, C)

            result_a = mms(V, i, C)
            @test isapprox(result_a.mms, _mms)
        end
    end

    @testset "conflicts c" begin
        V = Additive([56 94 5 33 83; 56 94 5 33 83; 56 94 5 33 83])
        i = 1
        C = Conflicts(SimpleGraph{Int64}(4, [[5], [3, 4], [2, 5], [2], [1, 3]]))
        _mms = 83.0

        @test isapprox(mms(V, i, C).mms, _mms)

        for (_, method) in methods_no_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)
            test_valid_allocation(V, result.X, _mms)

            A = Allocations.Allocation(na(V), ni(V), (i => [g for g in items(V) if result.X[i, g] > 0] for i in agents(V)))

            @test Allocations.check(V, A, C)
        end
    end

    @testset "connectivity" begin
        V = Additive([10 1 1 1 1 1 1 1 1 1 1 1 10; 10 1 1 1 1 1 1 1 1 1 1 1 10])
        i = 1
        edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 10), (9, 12), (10, 12), (11, 12), (12, 13)]
        graph = SimpleGraph(Edge.(edges))
        C = Constraints(Connections(graph))
        _mms = round(Int, mms(V, i, C).mms)

        for (_, method) in methods_no_charity
            result = method(V, i, C)
            @test isapprox(result.mms, _mms)
            # test_valid_allocation(V, result.X, _mms)

            for i in agents(V)
                bundle = Set(g for (g, v) in enumerate(result.X[i, :]) if v > 0)
                @assert(length(bundle) > 0)
                @test is_connectable(bundle, [], graph)
            end
        end
    end

    @testset "matroids" begin
        P = [
            (2, [14 10 65 39 28 58 49 73 85 37 45 44 47 30 90 75]),
            # (4, [77 13 45 48 45 2 28 40 56 53 96 91 46 65 19 28 30 27 36]),
        ]

        for (n, V_i) in P
            V = Additive{Matrix{Int}}(vcat([V_i for i in 1:n]...))
            i = 1

            matroid = rand_matroid_er59(V)
            C = MatroidConstraint(matroid)
            _mms = round(Int, mms(V, i, C).mms)

            for (_, method) in methods_no_charity
                result = method(V, i, C)
                @test isapprox(result.mms, _mms)
                test_valid_allocation(V, result.X, _mms)

                A = Allocations.Allocation(na(V), ni(V), (i => [g for g in items(V) if result.X[i, g] > 0] for i in agents(V)))

                @test Allocations.check(V, A, C)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    test_constrained()
end
