# project: partitioning
# module PartitioningData.jl
#
# General functions for handling data associated with the partitioning
# problem

module PartitioningData

using Graphs
using SimpleWeightedGraphs

export
    set_generator,
    set_generator_Ver5,
    get_instance,
    delta_gradient

const IVector = Vector{Int}

function set_generator(gen_parameters ::Dict{String, Any})
    # placeholder for dispatching between generators
    return set_generator_Ver5(gen_parameters["set_size"],
                              gen_parameters["interval_max"])
end


# TODO: add types and docstring
function set_generator_Ver5(L, boundry)
    # Generating the floating point set
    pre_set = randn(L)
    avg = sum(pre_set)/L
    pre_set .-= avg
    pre_set .*= boundry;

    # Sorting into positive and negative sets
    pre_set_p = pre_set[findall(x -> x > 0, pre_set)]
    pre_set_n = pre_set[findall(x -> x < 0, pre_set)]

    # Rounding
    set_p = ceil.(pre_set_p)
    set_n = floor.(pre_set_n);

    # Greedy Search Multiple Iterations, this will take the bulk of the computational time
    min_miss = false
    new_miss = sum(set_p) + sum(set_n)
    new_miss
    while !min_miss 
        old_miss = new_miss
        if old_miss > 0 
        # check if such values exist
            candidates = findall(x -> x <= div(old_miss,2), set_p)
            if length(candidates) != 0
                cand_index = maximum(candidates)
                over = set_p[cand_index]
                deleteat!(set_p, cand_index)
                push!(set_n, -1*over)
            end
        elseif old_miss < 0 
            # check if such values exist
             candidates = findall(x -> x >= div(old_miss,2), set_n)
            if length(candidates) != 0
                cand_index = maximum(candidates)
                over = set_n[cand_index]
                deleteat!(set_n, cand_index)
                push!(set_p, -1*over)
            end
        end
        # Check if error improved or not
        new_miss = sum(set_p) + sum(set_n)
        if !(abs(new_miss) < abs(old_miss))
            min_miss = true
        end
    end

    # Distributing remaining error

    miss_rem = sum(set_p) + sum(set_n)
    #println(miss_rem)
    while miss_rem != 0 
        if miss_rem > 0
            modify_index = rand(eachindex(set_n))
            #println(modify_index)
            set_n[modify_index] -= 1
        elseif miss_rem < 0 
            modify_index = rand(eachindex(set_p))
             #println(modify_index)
            set_n[modify_index] += 1
        end
        miss_rem = sum(set_p) + sum(set_n)
    end

    return (round.(Int,set_p), round.(Int,set_n))
end

function get_instance(parameters ::Dict{String, Any}) ::Tuple{IVector, IVector}
    return get_instance(gen_parameters["set_size"],
                        gen_parameters["interval_max"],
                        gen_parameters["logging"])
end

function get_instance(len ::Int, max ::Int, log = 0) ::Tuple{IVector, IVector}
    raw_set = set_generator_Ver5(len, max)
    full_set = [raw_set[1]; abs.(raw_set[2])]
    log > 0 && println("Generated instance: $full_set")
    solution = sign.([raw_set[1]; raw_set[2]])
    log > 0 && println("Solution: $solution")
    log > 0 && println("Consistency: $(sum(solution .* full_set))")

    return (full_set, solution)
end

function delta_gradient(graph)
    w = Vector{Float64}(undef, nv(graph))
    for node in 1:nv(graph)  # nv(g) = number of vertices
        for neighbor in neighbors(graph, node)
            w[node] += get_weight(graph, node, neighbor)
        end
    end
    dt = 0.2 ./ w
    return dt
end

end # module ends here
