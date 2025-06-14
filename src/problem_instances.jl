using Allocations
using Distributions
using Random

include("constraints.jl")

struct ProblemInfo
    id::Int
    name::String
    dist_v_range::Union{UnitRange{Int},Nothing}
    cardinality_model::String
    conflicts_model::String
    connections_model::String
    matroid_model::String
end

mutable struct ProblemInstance
    V::Additive
    C::Union{Constraint,Nothing}
    info::ProblemInfo
end

function random_partition_nonempty(values::Vector, n::Int) # help from chatgpt
    if n > length(values)
        throw(ArgumentError("Cannot partition a list into more parts than its length"))
    end

    shuffled = shuffle(values)
    partitions = [[shuffled[i]] for i in 1:n]
    for i in (n+1):length(shuffled)
        push!(partitions[rand(1:n)], shuffled[i])
    end

    return partitions
end

function problem_instance(
    num_agents::Int,
    num_items::Int,
    name::String;
    dist_v::String="uniform",
    dist_v_mean::Union{Int,Nothing}=50,
    dist_v_std::Union{Int,Nothing}=10,
    dist_v_range::Union{UnitRange,Nothing}=1:99,
    cardinality_model::String="",
    conflicts_model::String="",
    connections_model::String="",
    matroid_model::String=""
)::ProblemInstance

    @assert(dist_v == "uniform" || dist_v == "normal")

    # Valuations
    values = Array{Int}(undef, num_agents, 0)
    for _ in 1:num_items
        if dist_v == "uniform"
            col = rand(dist_v_range, num_agents)
        end

        if dist_v == "normal"
            col = Normal(dist_v_mean, dist_v_std) |>
                  x -> rand(x, num_agents) |>
                       x -> round.(Int, x) |>
                            x -> clamp.(x, 1, dist_v_mean * 2 - 1)
        end

        values = hcat(values, col)
    end

    V = Additive(values)

    @assert(na(V) == num_agents)
    @assert(num_agents <= num_items)

    constraints = []

    # Cardinality
    @assert(cardinality_model == "" || cardinality_model == "uniform_number_of_categories")

    if cardinality_model == "uniform_number_of_categories"
        category_count = rand(1:floor(Int, ni(V) / 2))
        partition = random_partition_nonempty(collect(items(V)), category_count)

        push!(constraints, Counts([set => rand(1:length(set)) for set in partition]...))
    end

    # Conflicts
    @assert(conflicts_model == "" || conflicts_model == "er59" || conflicts_model == "ba02" || conflicts_model == "ws98")

    if conflicts_model == "ba02"
        push!(constraints, rand_conflicts_ba02(V))
    elseif conflicts_model == "er59"
        push!(constraints, rand_conflicts_er59(V))
    elseif conflicts_model == "ws98"
        push!(constraints, rand_conflicts_ws98(V))
    end

    # Connections
    @assert(connections_model == "" || connections_model == "er59" || connections_model == "ba02" || connections_model == "ws98")

    if connections_model == "ba02"
        push!(constraints, Connections(rand_conflicts_ba02(V).graph))
    elseif connections_model == "er59"
        push!(constraints, Connections(rand_conflicts_er59(V).graph))
    elseif connections_model == "ws98"
        push!(constraints, Connections(rand_conflicts_ws98(V).graph))
    end

    # Matroids
    @assert(matroid_model == "" || matroid_model == "er59")
    if matroid_model == "er59"
        push!(constraints, MatroidConstraint(rand_matroid_er59(V)))
    end

    C = length(constraints) > 0 ? Constraints(constraints...) : nothing

    return ProblemInstance(
        V,
        C,
        ProblemInfo(
            0,
            name,
            dist_v_range,
            cardinality_model,
            conflicts_model,
            connections_model,
            matroid_model,
        )
    )
end
