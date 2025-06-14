using Distributions
using DataStructures
using Graphs
using Dates
using Base.Iterators

include("time.jl")

mutable struct DecisionDiagramEdge
    idx::Int
    src::Int
    dst::Int
    value::Int
    u::Int
end

mutable struct DecisionDiagramVertex
    idx::Int
    edges_in::Vector{Int}
    edges_out::Vector{Int}
    dead::Bool
    state::Any
    layer_idx::Int
    is_exact::Bool
    loc_b::Int
    left::Int
    right::Int
    u::Int
end

mutable struct DecisionDiagram
    variables::Any
    vertices::Vector{DecisionDiagramVertex}
    edges::Vector{DecisionDiagramEdge}

    vertex_layers::Vector{Vector{Int}}
    edge_layers::Vector{Vector{Int}}
    _s::Int
    _t::Int

    is_exact::Bool
    width::Int
end

DecisionDiagram(variables::Any) = DecisionDiagram(
    variables,
    [],
    [],
    [[] for _ in 1:(length(variables)+1)],
    [[] for _ in 1:length(variables)],
    0,
    0,
    true,
    0,
)

_vertex_count(G::DecisionDiagram) = length(G.vertices)
_edge_count(G::DecisionDiagram) = length(G.edges)
decision_count(G::DecisionDiagram) = length(G.variables)
vertex_idxs(G::DecisionDiagram) = map(v -> v.idx,
    filter(v -> !v.dead, G.vertices))
edge_idxs(G::DecisionDiagram) = map(e -> e.idx,
    filter(edge ->
            !G.vertices[edge.src].dead &&
                !G.vertices[edge.dst].dead,
        G.edges))

out_edges(G::DecisionDiagram, v::DecisionDiagramVertex) = map(
    x -> G.edges[x], v.edges_out)
in_edges(G::DecisionDiagram, v::DecisionDiagramVertex) = map(
    x -> G.edges[x], v.edges_in)
out_edges(G::DecisionDiagram, v::Int) = out_edges(G, G.vertices[v])
in_edges(G::DecisionDiagram, v::Int) = in_edges(G, G.vertices[v])

function print_decision_diagram(G::DecisionDiagram)
    println("DECISION DIAGRAM")
    println("variables: $(G.variables)")
    println("---VERTICES")
    for layer in G.vertex_layers
        for vertex_idx in layer
            print(vertex_idx)
            print(",")
        end
        println()
    end
    println("---EDGES")
    for edges in G.edge_layers
        for edge_idx in edges
            edge = G.edges[edge_idx]
            print("$(edge.src)-$(edge.dst)($(edge.value)),")
        end
        println()
    end
    println("---PATHS")
    print_paths_from(G, G._s, [])
end

function print_paths_from(G::DecisionDiagram, vertex_idx::Int, prev)
    edges = out_edges(G, vertex_idx)
    if length(edges) == 0
        values = [G.variables[_g] for (_g, ok) in enumerate(prev) if ok > 0]
        println("$prev ($values)")
    else
        for edge in edges
            next = [prev..., edge.value]
            print_paths_from(G, edge.dst, next)
        end
    end
end

function print_paths_to(G::DecisionDiagram, vertex_idx::Int, prev)
    edges = in_edges(G, vertex_idx)
    if length(edges) == 0
        values = [G.variables[_g] for (_g, ok) in enumerate(prev) if ok > 0]
        println("$prev ($values)")
    else
        for edge in edges
            next = [edge.value, prev...]
            print_paths_to(G, edge.src, next)
        end
    end
end

function add_vertex!(G::DecisionDiagram, layer::Int; state::Any=nothing, is_exact::Bool=true)
    idx = _vertex_count(G) + 1
    push!(G.vertices, DecisionDiagramVertex(idx, [], [], false, state, layer, is_exact, 0, -1, -1, 0))

    if layer > 0
        push!(G.vertex_layers[layer], idx)
    end

    return idx
end

function add_edge!(G::DecisionDiagram,
    layer::Int, src::Int, dst::Int, value::Int; u::Int=0)

    idx = _edge_count(G) + 1
    push!(G.edges, DecisionDiagramEdge(idx, src, dst, value, u))
    push!(G.vertices[src].edges_out, idx)
    push!(G.vertices[dst].edges_in, idx)

    if layer > 0
        push!(G.edge_layers[layer], idx)
    end

    return idx
end

"""
Constructs a decision diagram from:
- decision_variables: List of decision variables used to build vertices and edges
- get_actions: Function that gets the possible values for a decision variable at a given layer
- get_transition: Get the state of a node in the next layer given the current state, value and next value
- initial_state: The initial state of the decision diagram
"""
function construct_dd(
    decision_variables, initial_state, get_actions, get_transition, get_hash, get_satisfied; width=nothing, kwds...)

    G = DecisionDiagram(decision_variables)
    G._s = add_vertex!(G, 1; state=initial_state)

    layer = Dict{Any,Int}(get_hash(initial_state, 1) => G._s)
    height = decision_count(G)
    states = Dict{Int,Any}(G._s => initial_state)
    G.is_exact = true

    relax = get(kwds, :relax, false)
    select_relax = get(kwds, :select_relax,
        (layer, count) -> collect(take(keys(layer), count)))
    select_restrict = get(kwds, :select_restrict,
        (layer, count) -> collect(take(keys(layer), count)))

    for d_idx in 1:height
        throw_if_timeout(; kwds...)

        d_value = decision_variables[d_idx]
        layer_next = Dict{Any,Int}()

        if width !== nothing
            while length(layer) > width
                count = ceil(Int, length(layer) - width + 1)

                G.is_exact = false
                if relax
                    hashes = select_relax(layer, count)
                    node_idx = add_vertex!(G, d_idx; is_exact=false)
                    hash = argmax(x -> x[1], hashes)
                    states[node_idx] = states[layer[hash]]
                    edges_in = []
                    for hash in hashes
                        vertex = G.vertices[layer[hash]]
                        delete!(layer, hash)
                        vertex.dead = true
                        for edge in in_edges(G, vertex)
                            edge.dst = node_idx
                            push!(edges_in, edge.idx)
                        end
                    end
                    layer[hash] = node_idx
                    G.vertices[node_idx].edges_in = edges_in
                else
                    hashes = select_restrict(layer, count)
                    for hash in hashes
                        node_idx = layer[hash]
                        node = G.vertices[node_idx]
                        node.dead = true
                        delete!(layer, hash)
                    end
                end
            end
        end

        G.width = max(G.width, length(layer))

        for (_, src) in layer
            state = states[src]

            for value in get_actions(state, d_idx)
                state_next = get_transition(state, d_idx, value)
                hash_next = get_hash(state_next, d_idx)
                if !haskey(layer_next, hash_next)
                    layer_next[hash_next] = add_vertex!(
                        G, d_idx + 1, state=state_next)
                end
                dst = layer_next[hash_next]
                states[dst] = state_next
                add_edge!(G, d_idx, src, dst, value)
            end
        end
        layer = layer_next
    end

    # Validate-DD:

    for layer in reverse(G.vertex_layers)
        for vertex_idx in layer
            vertex = G.vertices[vertex_idx]
            state = states[vertex.idx]
            if length(vertex.edges_out) == 0
                vertex.dead = !get_satisfied(state)
            elseif all(G.vertices[G.edges[edge_idx].dst].dead
                       for edge_idx in vertex.edges_out)
                vertex.dead = true
            end
        end
    end

    if G.vertices[G._s].dead
        return (G=G, ok=false)
    end

    for edge_layer in G.edge_layers
        filter!(edge_idx ->
                !G.vertices[G.edges[edge_idx].src].dead &&
                    !G.vertices[G.edges[edge_idx].dst].dead,
            edge_layer)
    end
    for vertex_layer in G.vertex_layers
        filter!(vertex_idx -> !G.vertices[vertex_idx].dead,
            vertex_layer)
    end
    for vertex in G.vertices
        filter!(edge_idx ->
                !G.vertices[G.edges[edge_idx].src].dead &&
                    !G.vertices[G.edges[edge_idx].dst].dead,
            vertex.edges_in)
        filter!(edge_idx ->
                !G.vertices[G.edges[edge_idx].src].dead &&
                    !G.vertices[G.edges[edge_idx].dst].dead,
            vertex.edges_out)
    end

    return (G=G, ok=true)
end
