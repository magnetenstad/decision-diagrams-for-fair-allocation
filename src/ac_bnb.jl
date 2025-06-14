using DataStructures
using Allocations
using Base.Iterators

include("approximations.jl")
include("time.jl")

struct ACNode
    id::Int
    d_idx::Int
    state::Vector{Int}
end

function ac_bnb(V::Additive, i::Int; kwds...)
    get_width = get(kwds, :get_width,
        (d_idx) -> (ni(V) + 1 - d_idx))
    reuse_exact_cut = get(kwds, :reuse_exact_cut, true)
    get_h = get(kwds, :get_h,
        (node) -> node.state[end] - node.state[1])

    Q = BinaryMinHeap{Tuple{Float32,Int,ACNode}}()

    node_id = 1
    initial_node = ACNode(node_id, 1, [0 for _ in agents(V)])
    node_id += 1

    push!(Q, (get_h(initial_node), initial_node.id, initial_node))

    z_opt = 0
    item_values, _, remainders = order_items(V, i; kwds...)
    prop = floor(Int, sum(V.values[i, :]) / na(V))
    t = 0

    while length(Q) > 0 && z_opt != prop
        t += 1
        _, _, state = pop!(Q)

        width_max = get_width(state.d_idx)

        bdd_restricted = bdd_lel(
            item_values, na(V), [state], z_opt, prop, width_max, (layer, count) -> handle_width_restricted!(layer, count; kwds...), remainders, node_id; kwds...)
        node_id = bdd_restricted.node_id
        z_opt = max(z_opt, bdd_restricted.z)

        throw_if_timeout(; kwds...)

        if !bdd_restricted.is_exact
            layer = reuse_exact_cut ? bdd_restricted.exact_cut : [state]
            bdd_relaxed = bdd_lel(
                item_values, na(V), layer, z_opt, prop, width_max, (layer, count) -> handle_width_relaxed_lel!(layer, count; kwds...), remainders, node_id; kwds...)
            node_id = bdd_relaxed.node_id
            if bdd_relaxed.is_exact
                z_opt = max(z_opt, bdd_relaxed.z)
            elseif bdd_relaxed.z > z_opt
                for node in bdd_relaxed.exact_cut
                    push!(Q, (get_h(node), node.id, node))
                end
            end
        end

        throw_if_timeout(; kwds...)
    end

    return (mms=z_opt,)
end

function bdd_lel(item_values::Vector{Int}, n::Int,
    initial_layer::Array{ACNode}, z_opt::Int, upper_bound::Int, width_max::Int, handle_width!, remainders, node_id; kwds...)

    m = length(item_values)
    layer = Dict{Any,ACNode}(
        node.state => node for node in initial_layer)
    layer_prev = Dict()
    d_idx_initial = initial_layer[1].d_idx

    exact_cut = Set()
    is_exact = true

    for d_idx in d_idx_initial:m

        while length(layer) > width_max && d_idx > d_idx_initial + 1
            handle_width!(layer, (length(layer) - width_max + 1))
            if is_exact
                exact_cut = collect(values(layer_prev))
                is_exact = false
            end
        end

        layer_next = Dict{Any,ACNode}()
        v_g = item_values[d_idx]
        r = remainders[d_idx]

        for node in values(layer)
            z_opt = max(z_opt, node.state[1])

            if rough_upper_bound(d_idx, node.state, item_values, r; kwds...) <= z_opt
                continue
            end

            for (j, u) in enumerate(node.state)

                if j < n && node.state[j] == node.state[j+1]
                    continue
                end

                if u >= upper_bound
                    break
                end

                state_next = copy(node.state)
                state_next[j] = u + v_g

                sort!(state_next)

                if !haskey(layer_next, state_next)
                    layer_next[state_next] = ACNode(
                        node_id, node.d_idx + 1, state_next)
                    node_id += 1
                    z_opt = max(z_opt, state_next[1])
                end
            end
        end

        layer_prev = layer
        layer = layer_next
    end

    return (z=z_opt, exact_cut=exact_cut, is_exact=is_exact, node_id=node_id)
end

function handle_width_relaxed_lel!(layer, count; kwds...)
    # default = random
    select_relax = get(kwds, :select_relax,
        (layer, count) -> collect(take(values(layer), count)))

    nodes_to_merge = select_relax(layer, count)
    node_merged = merge_nodes(nodes_to_merge)

    for node in nodes_to_merge
        delete!(layer, node.state)
    end

    layer[node_merged.state] = node_merged
end

function handle_width_restricted!(layer, count; kwds...)
    # default = random
    select_restrict = get(kwds, :select_restrict,
        (layer, count) -> collect(take(values(layer), count)))

    nodes_to_restrict = select_restrict(layer, count)
    for node in nodes_to_restrict
        delete!(layer, node.state)
    end
end

function merge_nodes(nodes)
    state_next = [maximum(node.state[i] for node in nodes)
                  for i in eachindex(nodes[1].state)]
    return ACNode(nodes[1].id, nodes[1].d_idx, state_next)
end
