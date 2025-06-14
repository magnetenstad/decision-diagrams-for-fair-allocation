
function throw_if_timeout(; kwds...)
    t0 = get(kwds, :t0, nothing)
    time_limit = get(kwds, :time_limit, nothing)

    if t0 !== nothing &&
       time_limit !== nothing &&
       time() - t0 > time_limit

        throw("time limit reached")
    end
end
