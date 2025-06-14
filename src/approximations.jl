using Allocations

function greedy_allocation(V::Additive)
    N = agents(V)
    M = items(V)

    A = falses(na(V), ni(V))
    bags = zeros(Int, na(V))

    for _ in M
        i_min = 1
        for i in N
            if bags[i] < bags[i_min]
                i_min = i
            end
        end

        g_best = nothing
        for g in M
            if sum(A[:, g]) != 0
                continue
            end
            if g_best === nothing
                g_best = g
                break
            end
            if Allocations.value(V, i_min, g) >
               Allocations.value(V, i_min, g_best)
                g_best = g
            end
        end

        if g_best !== nothing
            A[i_min, g_best] = true
            bags[i_min] += Allocations.value(V, i_min, g_best)
        end
    end

    return A, minimum(bags)
end

function rough_upper_bound(d_idx, bundle, item_values, r; kwds...)
    if length(bundle) == 1
        return bundle[1] + r
    end

    rub_mvi = get(kwds, :rub_mvi, true)
    rub_di = get(kwds, :rub_di, true)

    rub = Inf

    if rub_mvi
        # Relaxation: remaining items are valued equally to the most valued remaining item
        items_r = item_values[d_idx:end]
        _max = maximum(items_r)
        _bundle = copy(bundle)
        for _ in items_r
            i = argmin(_bundle)
            _bundle[i] += _max
        end
        rub = minimum(_bundle)
    end

    if rub_di
        # Relaxation: remaining items are divisible
        value = bundle[1]
        rest = r
        for i in 2:length(bundle)
            need = (i - 1) * (bundle[i] - value)
            if need >= rest
                value += rest / (i - 1)
                rest = 0
                break
            else
                rest -= need
                value = bundle[i]
            end
        end
        value += rest / length(bundle)
        rub = min(value, rub)
    end

    return rub
end
