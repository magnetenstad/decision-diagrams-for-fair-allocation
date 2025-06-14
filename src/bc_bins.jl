using Allocations

include("items.jl")
include("formulations.jl")
include("approximations.jl")
include("bundle_centric.jl")

function bc_bins(
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing;
    method::Function=bc_bins_internal, kwds...)

    width = get(kwds, :width, ni(V))
    # if unconstrained, charity is always allowed
    charity = get(kwds, :charity, unconstrained(C))

    V, C = symmetric(V, i, C)
    (value, result) = binary_search_unique_sums(method, V, i, C; width=width, charity=charity, kwds...)

    if result !== nothing
        return (mms=value, result...)
    end
    return (mms=value, X=nothing)
end

function unique_sums(V, lower, upper)
    sums = Set{Int}(0)

    for g in V
        for r in collect(sums)
            x = r + g
            if x <= upper
                push!(sums, x)
            end
        end
    end

    return sort(collect(x for x in sums if lower <= x))
end

function get_candidates(V::Additive, i, C::Union{Constraint,Nothing}=nothing)
    lower_bound = 0
    if unconstrained(C)
        _, lower_bound = greedy_allocation(V)
    end
    upper_bound = floor(Int, sum(V.values[i, :]) / na(V))
    return unique_sums(V.values[i, :], lower_bound, upper_bound)
end

function binary_search_unique_sums(method::Function,
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing; kwds...)

    candidates = get_candidates(V, i, C)
    final = nothing

    index = binary_search(candidates,
        function (u_min, u_max, width, relax)

            _C = add_utility_constraints(
                V, i, C; kwds..., u_min=u_min, u_max=u_max)

            result = method(V, i, _C; kwds..., relax=relax, width=width)
            if result.z > 0
                final = result
            end
            return result
        end; kwds...)

    return (value=candidates[index], result=final)
end

function binary_search(candidates::Array, callback::Function; kwds...)
    width = get(kwds, :width, nothing)
    growth_factor = get(kwds, :growth_factor, 2)
    relaxation_enabled = get(kwds, :relaxation_enabled, false)
    errors = get(kwds, :errors, [])

    lo = 1
    hi = length(candidates)

    while lo != hi
        index = ceil(Int, (lo + hi) / 2)
        result = callback(candidates[index], candidates[hi], width, false)
        if result.ok
            if result.error !== nothing
                push!(errors, string(result.error))
            end

            if result.z > 0
                lo = findfirst(x -> x == result.z, candidates)
                @assert lo !== nothing
                @assert lo >= index
            else
                lo = index
            end
        else
            if result.is_exact
                hi = index - 1
            elseif relaxation_enabled
                result_relax = callback(
                    candidates[index], candidates[hi], width, true)

                if !result_relax.ok
                    hi = index - 1
                elseif result_relax.is_exact
                    lo = index
                elseif width !== nothing
                    width_prev = width
                    width *= growth_factor
                end
            elseif width !== nothing
                width_prev = width
                width *= growth_factor
            end
        end
    end

    return lo
end

function bc_bins_internal(
    V::Additive, i::Int, C::Union{Constraint,Nothing}=nothing; solver=Allocations.conf.MIP_SOLVER, kwds...)

    flow = na(V)

    _, M_sorted, _ = order_items(V, i; kwds...)

    (G, ok) = construct_bc_dd(V, M_sorted, i, C;
        width=get(kwds, :width, nothing), kwds...)

    if !ok
        return (ok=false, is_exact=G.is_exact, z=0, X=nothing)
    end

    model = Model(solver) # TODO
    set_time_limit_sec(model, get(kwds, :time_limit, 60))

    y = add_flow_constraints!(model, G, flow)
    add_layer_constraints!(model, G, y; kwds...)

    optimize!(model)

    if !is_solved_and_feasible(model; allow_local=false)
        return (ok=false, is_exact=G.is_exact, X=nothing, z=0, error="infeasible")
    end

    y_values = Dict(edge_idx => JuMP.value(y[edge_idx])
                    for edge_idx in edge_idxs(G))
    X = extract_assignment_matrix_from_bc_diagram(y_values, G, flow, M_sorted)
    assert_valid_allocation(V, X)

    bundle_values = [sum(X[_i, :] .* V.values[i, :]) for _i in agents(V)]

    return (ok=true, is_exact=G.is_exact, X=X, z=minimum(bundle_values), error=nothing)
end
