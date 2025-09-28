module HNProcess

using Graphs
using SimpleWeightedGraphs
using Dice 

export
    Agitator,
    get_HN_graph

function get_HN_graph(images ::Vector{Matrix{Int}},
                                scale ::Float64) ::SimpleWeightedGraph
    part_data = vec(images[1]')
    graph_set = SimpleWeightedGraph(length(part_data))
        
    for og in range(1,length(part_data) - 1)
        for term in range(og+1,length(part_data))
            # EDIT HERE, i is origin, j is end. We need to do this multiplication for every image
            w = 0
            for i in images
                w += i[og] * i[term] / 1
            end
            add_edge!(graph_set, og, term, w)
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
    S = parameters["images"]
    time_total= parameters["sim_time"]
    steps = parameters["steps"]

    # Make Graph
    graph_set = get_HN_graph(S, scaling)
    
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
        pinned::Vector{Tuple{Int64, Int8}} = []
        state::Dice.Hybrid = Dice.get_random_hybrid(num_vertices, 2.0)
        println(state[1])
        for stim in parameters["initial_stimuli"]
            pos = (stim[1]-1) * size(parameters["images"][1],1) + stim[2]
            state[1][pos] = stim[3]
            push!(pinned, (pos, stim[3]))
        end
        println(state[1])
        agnum = 0
        for _ in 1:parameters["num_agitations"]
            agnum += 1
            #println(agnum)
            state = Dice.propagate_pinned(model.graph, num_steps, parameters["delta"], model.coupling, state, pinned)
            if (reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
            -1 .*reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                converged += 1
                break
            end
            state = (state[1], Dice.get_random_cube(num_vertices, 2.0))
        end
        counts_list[agnum] += 1
    end
    return [
end

end