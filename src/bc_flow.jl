using JuMP
using Dates
using MathOptInterface
using Allocations
using Gurobi

include("items.jl")
include("bundle_centric.jl")

function bc_flow(
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing; kwds...)

    # if unconstrained, charity is always allowed
    charity = get(kwds, :charity, unconstrained(C))

    V, C = symmetric(V, i, C)
    C = add_utility_constraints(V, i, C)

    (mms, X) = bc_flow_internal(V, i, C; charity=charity, kwds...)

    return (mms=mms, X=X)
end

function bc_flow_internal(
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing;
    solver=Allocations.conf.MIP_SOLVER, kwds...)

    flow = na(V)
    _, M_sorted, _ = order_items(V, i; kwds...)

    (G, ok) = construct_bc_dd(V, M_sorted, i, C; kwds...)

    if !ok
        return (z=0, X=nothing)
    end

    model = Model(solver)
    set_time_limit_sec(model, get(kwds, :time_limit, 60))

    @variable(model, MM, integer = true)
    @objective(model, Max, MM)

    z = Dict()
    G._t = add_vertex!(G, -1)
    for vertex_idx in vertex_idxs(G)
        if vertex_idx == G._t
            continue
        end
        vertex = G.vertices[vertex_idx]
        bundle = [g for (g, ok) in enumerate(vertex.state) if ok]
        u = Allocations.value(V, i, bundle)
        if !constraint_satisfied(C, V, i, bundle)
            continue
        end
        edge_idx = add_edge!(G, -1, vertex.idx, G._t, 0, u=u)
        z[edge_idx] = @variable(model, binary = true)
    end

    y = add_flow_constraints!(model, G, flow)
    add_layer_constraints!(model, G, y; kwds...)

    M_1 = na(V)
    M_2 = Allocations.value(V, i, items(V))
    for edge in in_edges(G, G._t)
        @constraint(model, M_1 * z[edge.idx] >= y[edge.idx])
        @constraint(model, MM <= edge.u + M_2 * (1 - z[edge.idx]))
    end

    optimize!(model)

    if !is_solved_and_feasible(model; allow_local=false)
        return (z=0, X=nothing)
    end

    y_values = Dict(edge_idx => JuMP.value(y[edge_idx])
                    for edge_idx in edge_idxs(G))
    z_opti = JuMP.value(MM)
    X = extract_assignment_matrix_from_bc_diagram(y_values, G, flow, M_sorted)
    assert_valid_allocation(V, X)

    return (z=z_opti, X=X)
end
