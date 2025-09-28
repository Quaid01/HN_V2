# project: partitioning
# module PartitioningProcess.jl
#
# General functions for processing the partitionin problem

module PartitioningProcess

using Graphs
using SimpleWeightedGraphs
using Dice 

export
    Agitator,
    get_partitioning_graph

const IVector = Vector{Int}

# A typical form of the main data structure
# parameters = Dict{String, Any}(
#     "set" => full_set_1,
#     "sim_time" => 15.1,
#     "steps" => 760,
#     "iterations" => 200,
#     "num_agitations" => 20,
#     "scaling" => 4.5 * (maximum(abs.(full_set_1)))
# );

function get_partitioning_graph(part_data ::IVector,
                                scale ::Float64) ::SimpleWeightedGraph
    graph_set = SimpleWeightedGraph(length(part_data))
    
    for i in range(1,length(part_data) - 1)
        for j in range(i+1,length(part_data))
            add_edge!(graph_set, i, j, part_data[i] * part_data[j] / scale)
        end
    end
    return graph_set
end

# Creating Agitator which uses agitations to improve convergence rates. The
# index tells you the agitation count (1 being the first attempt, so no
# reshuffles yet) The second index means one run was done and we reshuffle
# the Continuous portion once. Everytime the iteration is run, a count is
# added to the index of that agitation count. For example if an attempt
# converges to the correct solution on the 3rd agitation, the index 3 is
# incremented by 1. One thing to note is that convergence could still
# happen on the final attempt, so the final value in the counts_list array
# is NOT an accurate representation of the number that diverged.
function Agitator(parameters::Dict{String, Any}) 
    scaling = parameters["scaling"]
    S = parameters["set"]
    time_total= parameters["sim_time"]
    steps = parameters["steps"]

    # Make Graph
    graph_set = get_partitioning_graph(S, scaling)
    
    # Making model
    # Stop time in model units
    total_time = time_total
    # Number of timesteps
    num_steps = steps
    # Step size
    delta_t = total_time/num_steps
    # Makes model
    model = Dice.Model(graph_set, Dice.model_2_hybrid_coupling, delta_t)

    # Making randomized initial state
    num_vertices = Graphs.nv(model.graph)
    
    converged = 0
    diverged = 0
    counts_list = zeros(parameters["num_agitations"])

    for _ in 1:parameters["iterations"]
        state::Dice.Hybrid = Dice.get_random_hybrid(num_vertices, 2.0)
        agnum = 0
        for _ in 1:parameters["num_agitations"]
            agnum += 1
            state = Dice.propagate(model.graph, num_steps, parameters["delta"], model.coupling, state)
            if abs(sum(S .* state[1])) < 1
            #if sum(S .* state[1]) == 0
                converged += 1
                break
            end
            state = (state[1], Dice.get_random_cube(num_vertices, 2.0))
        end
        counts_list[agnum] += 1
    end
    return [counts_list, (converged/parameters["iterations"])]
end

end # module ends here
