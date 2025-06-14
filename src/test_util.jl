using Test
using Allocations

function test_valid_allocation(V, X, _mms)
    @test size(X, 1) == na(V)
    @test size(X, 2) == ni(V)
    for g in items(V)
        @test sum(X[:, g]) <= 1
    end
    for _i in agents(V)
        @test sum(Allocations.value(V, 1, g) for (g, ok) in enumerate(X[_i, :]) if ok > 0) >= _mms
    end
end
