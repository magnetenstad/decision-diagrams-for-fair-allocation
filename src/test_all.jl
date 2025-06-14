include("test_unconstrained.jl")
include("test_constrained.jl")

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    test_unconstrained()
    test_constrained()
end
