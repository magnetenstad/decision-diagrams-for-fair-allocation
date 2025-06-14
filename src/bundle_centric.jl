using Allocations

include("decision_diagram.jl")
include("constraints.jl")

function construct_bc_dd(
    V::Additive, M::Array{Int}, i::Int, C::Union{Constraint,Nothing}=nothing;
    width=nothing, kwds...)

    start_state = BitArray(undef, length(M))
    start_state .= 0

    test_feature = get(kwds, :test_feature, false)
    get_satisfied = (unconstrained(C) && test_feature) ? nothing : function (state::BitArray)
        bundle = [_g for (_g, ok) in enumerate(state) if ok]
        return constraint_satisfied(C, V, i, bundle)
    end

    get_actions = function (state::BitArray, g_idx::Int)
        bundle = [_g for (_g, ok) in enumerate(state) if ok]
        M_r = M[g_idx:end]
        g = M[g_idx]
        return filter(a -> constraint_allow(C, V, i, M_r, bundle, g, a), [0, 1])
    end

    get_transition = function (state::BitArray, g_idx::Int, a::Int)
        state = copy(state)
        g = M[g_idx]
        state[g] = a
        return state
    end

    get_hash = function (state::BitArray, g_idx::Int)
        bundle = [g for (g, ok) in enumerate(state) if ok]
        M_r = M[g_idx:end]
        return tuple(constraint_hash(C, V, i, M_r, bundle)...)
    end

    return construct_dd(
        M, start_state, get_actions, get_transition, get_hash, get_satisfied; width=width, kwds...)
end
