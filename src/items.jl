using Allocations

function order_items(V, i; kwds...)
    order = get(kwds, :order, "descending")

    item_key_values = [(value=V.values[i, g], key=g) for g in items(V)]

    if order == "ascending"
        sort!(item_key_values, by=x -> x.value, rev=true)
    elseif order == "random"
        shuffle!(item_key_values)
    elseif order == "alternating"
        result = []
        pick_min = true
        while length(item_key_values) > 0
            index = 1
            if pick_min
                index = argmin(i -> item_key_values[i].value,
                    eachindex(item_key_values))
            else
                index = argmax(i -> item_key_values[i].value,
                    eachindex(item_key_values))
            end
            push!(result, item_key_values[index])
            deleteat!(item_key_values, index)
            pick_min = !pick_min
        end
        item_key_values = result
    elseif order == "descending"
        sort!(item_key_values, by=x -> x.value, rev=true)
    end

    M_sorted = map(x -> x.key, item_key_values)
    item_values = map(x -> x.value, item_key_values)
    remainders = Dict([(d_idx, sum(item_values[d_idx:end]))
                       for d_idx in items(V)])

    return item_values, M_sorted, remainders
end
