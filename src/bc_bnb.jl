using DataStructures

include("bundle_centric.jl")
include("time.jl")

struct BCNode
    vertices::Vector{Int}
    b::Int
    depth::Int
end

function bc_bnb(
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing; kwds...)

    V, C = symmetric(V, i, C)
    N = agents(V)
    m = ni(V)

    prop = floor(Int, sum(V.values[i, :]) / na(V))
    candidates = unique_sums(V.values[i, :], 0, prop)
    u_max = candidates[end]
    C = combine_constraints(C, MaximumUtility(u_max))
    _, lower_bound = greedy_allocation(V)
    C = combine_constraints(C, MinimumUtility(lower_bound))

    item_values, M_sorted, remainders = order_items(V, i; kwds...)

    (G, ok) = construct_bc_dd(V, M_sorted, i, C; kwds...)

    if !ok
        return (mms=0,)
    end

    loc_b = get(kwds, :loc_b, true)
    for vertex_layer in reverse(G.vertex_layers)
        for vertex_idx in vertex_layer
            vertex = G.vertices[vertex_idx]
            vertex.u = Allocations.value(V, i,
                [_g for (_g, ok) in enumerate(vertex.state) if ok])

            if loc_b
                vertex.loc_b = max(vertex.loc_b, vertex.u)

                for edge in in_edges(G, vertex)
                    parent = G.vertices[edge.src]
                    parent.loc_b = max(parent.loc_b, vertex.loc_b)
                end
            end

            @assert length(out_edges(G, vertex)) <= 2
            for edge in out_edges(G, vertex)
                if edge.value > 0
                    vertex.right = edge.dst
                else
                    vertex.left = edge.dst
                    @assert G.vertices[vertex.left].u == vertex.u
                end
            end
        end
    end

    start = BCNode([G._s for _ in N], prop, 1)

    node_idx = 1
    get_h = get(kwds, :get_h,
        (node, G) -> begin
            U = [node.b, [G.vertices[v].u for v in node.vertices]...]
            return maximum(U) - minimum(U)
        end)
    Q = BinaryMinHeap{Tuple{Float32,Int,BCNode}}()

    push!(Q, (get_h(start, G), node_idx, start))
    node_idx += 1

    z = 0

    while !isempty(Q)
        throw_if_timeout(; kwds...)

        (_, _, node) = pop!(Q)

        if node.b <= z
            continue
        end

        vertex_idxs = node.vertices
        vertices = [G.vertices[v_idx] for v_idx in vertex_idxs]

        if loc_b && any(v.loc_b <= z for v in vertices)
            continue
        end

        bundle = sort([v.u for v in vertices])

        _z = min(node.b, bundle[1])
        if _z > z
            z = _z
            if z == prop
                break
            end
        end

        if node.b <= z
            continue
        end

        if node.depth > m
            continue
        end

        d_idx = node.depth
        r = remainders[d_idx]

        rub = rough_upper_bound(d_idx, bundle, item_values, r; kwds...)
        if rub <= z
            continue
        end

        n = length(vertex_idxs)

        for i in eachindex(vertex_idxs)
            if i < n && vertex_idxs[i] == vertex_idxs[i+1]
                continue
            end

            v_idx = vertex_idxs[i]

            v = G.vertices[v_idx]
            if v.right == -1
                continue
            end

            lefts = []
            upp = node.b

            for j in eachindex(vertex_idxs)
                if i == j
                    continue
                end
                _v_idx = vertex_idxs[j]
                _v = G.vertices[_v_idx]
                if length(_v.edges_out) == 0
                    upp = min(upp, _v.u)
                else
                    push!(lefts, _v.left)
                end
            end

            if -1 in lefts
                continue
            end

            if upp <= z
                continue
            end

            next_vertex_idxs = sort([v.right, lefts...])
            next = BCNode(next_vertex_idxs, upp, node.depth + 1)
            push!(Q, (get_h(next, G), node_idx, next))
            node_idx += 1
        end
    end

    return (mms=z,)
end
