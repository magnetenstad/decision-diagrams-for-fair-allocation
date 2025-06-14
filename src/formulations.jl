using JuMP
using Allocations
using Gurobi
using HiGHS

include("decision_diagram.jl")

function add_flow_constraints!(
    model, G::DecisionDiagram, flow::Int)

    # The flow along each edge is an integer value
    # indicates how many agents have taken that path

    if flow == 1
        y = Dict(edge_idx => @variable(model, binary = true)
                 for edge_idx in edge_idxs(G))
    else
        y = Dict(edge_idx => @variable(model, integer = true)
                 for edge_idx in edge_idxs(G))

        for _y in values(y)
            @constraint(model, _y >= 0)
        end
    end

    # sum of flow out = sum of flow in, for each node
    for v in vertex_idxs(G)
        in_edges = G.vertices[v].edges_in
        out_edges = G.vertices[v].edges_out

        # skip if source or leaf
        if length(in_edges) == 0 || length(out_edges) == 0
            continue
        end

        @constraint(model,
            sum(y[e_idx] for e_idx in in_edges) ==
            sum(y[e_idx] for e_idx in out_edges))
    end

    # number of paths equal to number of agents
    # -> sum of flow out of source == num_agents
    @constraint(model,
        flow ==
        sum(y[e_idx] for e_idx in G.vertices[G._s].edges_out))

    return y
end

function add_layer_constraints!(model, G::DecisionDiagram, y; kwds...)
    layers = [map(x -> G.edges[x], edges) for edges in G.edge_layers]
    charity = get(kwds, :charity, false)
    if charity
        for layer in layers
            @constraint(model, sum(e.value * y[e.idx] for e in layer) <= 1)
        end
    else
        for layer in layers
            @constraint(model, sum(e.value * y[e.idx] for e in layer) == 1)
        end
    end
end


function assert_valid_allocation(V, X)
    @assert size(X, 1) == na(V)
    @assert size(X, 2) == ni(V)
    for g in items(V)
        @assert sum(X[:, g]) <= 1
    end
end

function extract_assignment_matrix_from_bc_diagram(
    y, G::DecisionDiagram, num_agents::Int, M_sorted)

    X = zeros(num_agents, length(M_sorted))

    for i in 1:num_agents
        outs = out_edges(G, G._s)
        g_idx = 1
        while length(outs) > 0
            outs_next = []
            for e in outs
                if y[e.idx] > 0.5
                    y[e.idx] -= 1
                    if e.value > 0
                        X[i, M_sorted[g_idx]] = e.value
                    end
                    outs_next = out_edges(G, e.dst)
                    break
                end
            end
            outs = outs_next
            g_idx += 1
        end
    end

    return X
end
