using Graphs
using Allocations
using JuMP
using Gurobi
using MathOptInterface
using Graphs

include("approximations.jl")

function is_connectable(bundle, subgraph, graph)
    if length(bundle) <= 1
        return true
    end
    union!(subgraph, bundle)
    item = rand(bundle)
    visited = Set(item)
    component = Set(item)
    Q = Set(item)

    while length(Q) > 0
        neighbors = intersect(Graphs.neighbors(graph, pop!(Q)), subgraph)
        union!(Q, setdiff(neighbors, visited))
        union!(visited, neighbors)
        union!(component, intersect(neighbors, bundle))
        if length(component) == length(bundle)
            return true
        end
    end
    return false
end

# General

constraint_allow(_::Nothing, args...) = true
constraint_satisfied(_::Nothing, args...) = true
constraint_hash(_::Nothing, args...) = []

function constraint_allow(C::Constraints,
    V::Additive, i::Int, M_r::Array{Int}, bundle::Array{Int}, g::Int, a::Int)
    return all(constraint_allow(c, V, i, M_r, bundle, g, a)
               for c in C.parts)
end

function constraint_satisfied(C::Constraints,
    V::Additive, i::Int, bundle::Array{Int})
    return all(constraint_satisfied(c, V, i, bundle) for c in C.parts)
end

function constraint_hash(C::Constraints,
    V::Additive, i::Int, M_r::Array{Int}, bundle::Array{Int})
    hash = []
    for c in C.parts
        append!(hash, constraint_hash(c, V, i, M_r, bundle))
        push!(hash, -1)
    end
    return hash
end

# combine_constraints

combine_constraints(first::Constraint, _::Nothing) = first
combine_constraints(_::Nothing, second::Constraint) = second
combine_constraints(first::Constraints, second::Constraint) =
    Constraints(first.parts..., second)
combine_constraints(first::Constraint, second::Constraints) =
    Constraints(first, second.parts...)
combine_constraints(first::Constraints, second::Constraints) =
    Constraints(first.parts..., second.parts...)
combine_constraints(first::Constraint, second::Constraint) =
    Constraints(first, second)

unconstrained(C::Constraints) = all(unconstrained(c) for c in C.parts)
unconstrained(_::Nothing) = true

# Minimum Utility

struct MinimumUtility{T} <: Constraint
    u_min::T
end

unconstrained(_::MinimumUtility) = true

function constraint_allow(minimum_utility::MinimumUtility,
    V::Additive, i::Int, M_r::Array{Int}, bundle::Array{Int}, _::Int, a::Int)
    if a == 0
        u = Allocations.value(V, i, bundle)
        u_r = Allocations.value(V, i, M_r)
        return u + u_r >= minimum_utility.u_min
    end
    if a == 1
        return true
    end
    return false
end

function constraint_satisfied(minimum_utility::MinimumUtility,
    V::Additive, i::Int, bundle::Array{Int})
    u = Allocations.value(V, i, bundle)
    return u >= minimum_utility.u_min
end

function constraint_hash(_::MinimumUtility,
    V::Additive, i::Int, _::Array{Int}, bundle::Array{Int})
    u = Allocations.value(V, i, bundle)
    return [u]
end

# Maximum Utility

struct MaximumUtility{T} <: Constraint
    u_max::T
end

unconstrained(_::MaximumUtility) = true

function constraint_allow(maximum_utility::MaximumUtility,
    V::Additive, i::Int, _::Array{Int}, bundle::Array{Int}, _::Int, a::Int)
    u = Allocations.value(V, i, bundle)
    # for both a == 0 and a == 1:
    return u < maximum_utility.u_max
end

function constraint_satisfied(_::MaximumUtility, args...)
    return true
end

function constraint_hash(_::MaximumUtility, args...)
    return []
end

# Cardinality

unconstrained(_::Counts) = false

function constraint_allow(counts::Counts,
    _::Additive, _::Int, _::Array{Int}, bundle::Array{Int}, g::Int, a::Int)
    if a == 0
        return true
    end
    if a == 1
        category_idx = findfirst(category -> g in category.members,
            counts.categories)
        @assert category_idx !== nothing
        category = counts[category_idx]
        count = length(intersect(category.members, union(bundle, g)))
        return count <= category.threshold
    end
    return false
end

function constraint_satisfied(_::Counts, args...)
    return true
end

function constraint_hash(counts::Counts,
    _::Additive, _::Int, M_r::Array{Int}, bundle::Array{Int})
    return [min(
        category.threshold - length(intersect(bundle, category.members)),
        length(intersect(M_r, category.members))
    ) for category in counts]
end

# Conflicts

unconstrained(_::Conflicts) = false

function constraint_allow(conflicts::Conflicts,
    _::Additive, _::Int, _::Array{Int}, bundle::Array{Int}, g::Int, a::Int)
    if a == 0
        return true
    end
    if a == 1
        for _g in bundle
            if has_edge(conflicts.graph, g, _g)
                return false
            end
        end
        return true
    end
    return false
end

function constraint_satisfied(_::Conflicts, args...)
    return true
end

function constraint_hash(conflicts::Conflicts,
    _::Additive, i::Int, M_r::Array{Int}, bundle::Array{Int})
    return sort(filter(g_r ->
            any(has_edge(conflicts.graph, (g, g_r)) for g in bundle), M_r))
end

# Connections

struct Connections{T<:AbstractGraph} <: Constraint
    graph::T
end

unconstrained(_::Connections) = false

function constraint_allow(connections::Connections,
    _::Additive, _::Int, M_r::Array{Int}, bundle::Array{Int}, g::Int, a::Int)
    if a == 0
        return is_connectable(bundle, M_r, connections.graph)
    end
    if a == 1
        return is_connectable(union(bundle, g), M_r, connections.graph)
    end
    return false
end

function constraint_satisfied(connections::Connections,
    _::Additive, _::Int, bundle::Array{Int})
    return is_connectable(bundle, [], connections.graph)
end

function constraint_hash(connections::Connections,
    _::Additive, _::Int, M_r::Array{Int}, bundle::Array{Int})
    return sort(setdiff(union([],
            (neighbors(connections.graph, g) for g in bundle)...),
        bundle))
end

# TODO: this assumes use of Gurobi.
# Enforce connectivity constraints on the JuMP model.
# https://jump.dev/JuMP.jl/stable/manual/callbacks/
Allocations.enforce(connections::Connections) = function (ctx)

    V, A, model = ctx.profile, ctx.alloc_var, ctx.model
    graph = connections.graph

    set_attribute(model, "LazyConstraints", 1)

    set_attribute(model, Gurobi.CallbackFunction(),
        function (cb_data, cb_where::Cint)

            if cb_where != Gurobi.GRB_CB_MIPNODE && cb_where != GRB_CB_MIPSOL
                return
            end
            try
                Gurobi.load_callback_variable_primal(cb_data, cb_where)
            catch
                return
            end
            for i in agents(V)
                bundle = Int[]
                for g in items(V)
                    if callback_value(cb_data, A[i, g]) > 0.5
                        # 0.5 is important! (prevents floating point errors)
                        push!(bundle, g)
                    end
                end

                if !isempty(bundle)
                    if !is_connectable(bundle, [], graph)
                        neighbors = Set{Int}()
                        for g in bundle
                            for h in Graphs.neighbors(graph, g)
                                if !(h in bundle)
                                    push!(neighbors, h)
                                end
                            end
                        end
                        constraint = @build_constraint(
                            sum(A[i, g] for g in bundle) <=
                            (length(bundle) - 1) + sum(A[i, g] for g in neighbors))
                        MathOptInterface.submit(model,
                            MathOptInterface.LazyConstraint(cb_data), constraint)
                    end
                end
            end
        end)

    return ctx
end

# Matroids

unconstrained(_::MatroidConstraint) = false

function constraint_allow(matroid::MatroidConstraint,
    _::Additive, _::Int, M_r::Array{Int}, bundle::Array{Int}, g::Int, a::Int)
    if a == 0
        return true
    end
    if a == 1
        return is_indep(matroid.matroid, union(bundle, g))
    end
end

function constraint_satisfied(_::MatroidConstraint, args...)
    return true
end

function constraint_hash(matroid::MatroidConstraint,
    _::Additive, _::Int, M_r::Array{Int}, bundle::Array{Int})
    return [g for g in M_r if is_indep(matroid.matroid, union(bundle, g))]
end

# 

function add_utility_constraints(
    V::Additive, i::Int, C::Union{Constraint,Nothing}; u_min=0, kwds...)

    prop = floor(Int, sum(V.values[i, :]) / na(V))
    u_max = get(kwds, :u_max, prop)

    allow_max = true
    if C isa Connections ||
       C isa Constraints && any(c isa Connections for c in C.parts)
        allow_max = false
    end

    if unconstrained(C)
        if u_min == 0
            u_min = greedy_allocation(V)[2]
        end
        C = combine_constraints(C, MinimumUtility(u_min))
        C = combine_constraints(C, MaximumUtility(u_max))
    else
        C = combine_constraints(C, MinimumUtility(u_min))
        charity = get(kwds, :charity, false)
        if charity && allow_max
            C = combine_constraints(C, MaximumUtility(u_max))
        end
    end

    return C
end

function symmetric(V::Additive, i, C::Union{Constraint,Nothing}=nothing)
    V_i = Additive([Allocations.value(V, i, g)
                    for _ in agents(V), g in items(V)])
    C_i = C # TODO: Allocations.SymmetrizedConstraint(C, i)
    return (V_i, C_i)
end
