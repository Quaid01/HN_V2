module HNProcess

using Graphs
using SimpleWeightedGraphs
using Dice 
using PrettyTables
using Plots

export
    HN_Solver,
    get_HN_graph, 
    HN_Solver_Traj,
    iterative_rotater_state,
    iterative_rotater_list,
    HN_cut_plotter,
    HN_og,
    sol_finder


function get_HN_graph(images ::Vector{Matrix{Int}}, scale ::Float64) ::SimpleWeightedGraph
    part_data = vec(images[1]')
    graph_set = SimpleWeightedGraph(length(part_data))
        
    for og in range(1,length(part_data) - 1)
        for term in range(og+1,length(part_data))
            # EDIT HERE, i is origin, j is end. We need to do this multiplication for every image
            w = 0
            for i in images
                w += i[og] * i[term] / scale
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
function HN_Solver(parameters::Dict{String, Any}, debug::Bool = false)
    # Just gets raw result, no visualizer

    scaling = parameters["scaling"] #Scaling coeff
    S = parameters["images"] # Images
    time_total= parameters["sim_time"] # how long the sim is
    num_steps = parameters["steps"] # number of steps in the sim
    dt_sim = parameters["delta"] # dt
    delta_t = time_total/num_steps

    # Make Graph
    graph_set = get_HN_graph(S, scaling)
    
    # Making model
    model = Dice.Model(graph_set, Dice.model_2_hybrid_coupling, delta_t)

    # Making randomized initial state
    num_vertices = Graphs.nv(model.graph)
    
    converged = 0
    diverged = 0


    pinned::Vector{Tuple{Int64, Int8}} = []
    state::Dice.Hybrid = Dice.get_random_hybrid(num_vertices, 2.0)
    #println(reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
    for stim in parameters["initial_stimuli"]
        # Every n entries is a column, thus taking # of columns and subtracting 1 brings you to where the column begins
        # Adding 1 will bring you to the first entry in the column and so on for +k
        pos = (stim[2]-1) * size(parameters["images"][1],1) + stim[1]
        state[1][pos] = stim[3]
        push!(pinned, (pos, stim[3]))
    end
    #println(state[1]
    state = Dice.propagate_pinned(model.graph, num_steps, dt_sim, model.coupling, state, pinned)
    # Comment out the other part of the or conditional to ignore negative images
 #=   if (reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
        -1 .*reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
        converged += 1
        global sol = state[1]
        break
    end =#

    
    if debug
        pretty_table(reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
    end
    return(state)
end

function HN_Solver_Traj(parameters::Dict{String, Any}, debug::Bool = false)
    # Just gets raw result, no visualizer

    scaling = parameters["scaling"] #Scaling coeff
    S = parameters["images"] # Images
    time_total= parameters["sim_time"] # how long the sim is
    num_steps = parameters["steps"] # number of steps in the sim
    dt_sim = parameters["delta"] # dt
    delta_t = time_total/num_steps
    traj_collection::Vector{Vector{Dice.Hybrid}} = []

    # Make Graph
    graph_set = get_HN_graph(S, scaling)
    
    # Making model
    model = Dice.Model(graph_set, Dice.model_2_hybrid_coupling, delta_t)

    # Making randomized initial state
    num_vertices = Graphs.nv(model.graph)
    
    converged = 0
    diverged = 0


    pinned::Vector{Tuple{Int64, Int8}} = []
    state::Dice.Hybrid = Dice.get_random_hybrid(num_vertices, 2.0)
    #println(reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
    for stim in parameters["initial_stimuli"]
        # Every n entries is a column, thus taking # of columns and subtracting 1 brings you to where the column begins
        # Adding 1 will bring you to the first entry in the column and so on for +k
        pos = (stim[2]-1) * size(parameters["images"][1],1) + stim[1]
        state[1][pos] = stim[3]
        push!(pinned, (pos, stim[3]))
    end
    #println(state[1]
    traj = Dice.trajectories_pinned(model.graph, num_steps, dt_sim, model.coupling, state, pinned)
    push!(traj_collection, traj)
    # Comment out the other part of the or conditional to ignore negative images
 #=   if (reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
        -1 .*reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
        converged += 1
        global sol = state[1]
        break
    end =#

    
    if debug
        pretty_table(reshape(traj[end][1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
    end
    return(state,traj_collection)
end    

function iterative_rotater_state(state, params, debug = false)
    rotations = []
    for i in state[2]
        rotated = Dice.realign_hybrid(state, 1+i)
        if debug
            println("rotated by $i")
            pretty_table(reshape(rotated[1],size(params["images"][1],1),size(params["images"][1],1)))
        end
        push!(rotations, rotated)
    end
    return rotations
end

function iterative_rotater_list(state, list, debug = false)
    rotations = []
    for i in list
        rotated = Dice.realign_hybrid(state, 1+i)
        if debug
            println("rotated by $i")
            pretty_table(reshape(rotated[1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
        end
        push!(rotations, rotated)
    end
    return rotations
end

function HN_cut_plotter(params, state)
    rot = iterative_rotater_state(state,params)
    g = get_HN_graph(params["images"],params["scaling"])
    binary = [i[1] for i in rot]
    x = state[2]
    y = [Dice.cut(g,s) for s in (rot[k][1] for k in 1:length(rot))]
    p = scatter(x,y)
    return p
end

function HN_og(params)
    scaling = params["scaling"] #Scaling coeff
    S = params["images"] # Images
    time_total= params["sim_time"] # how long the sim is
    num_steps = params["steps"] # number of steps in the sim
    dt_sim = params["delta"] # dt
    delta_t = -1*time_total/num_steps

    # Make Graph
    graph_set = get_HN_graph(S, -1*scaling)
    
    # Making model
    model = Dice.Model(graph_set, Dice.model_2_hybrid_coupling, delta_t)
    # Making randomized initial state
    num_vertices = Graphs.nv(model.graph)
    
    converged = 0
    diverged = 0


    pinned::Vector{Tuple{Int64, Int8}} = []
    spins = Dice.get_random_configuration(num_vertices)
    #println(reshape(state[1],size(parameters["images"][1],1),size(parameters["images"][1],1)))
    for stim in params["initial_stimuli"]
        # Every n entries is a column, thus taking # of columns and subtracting 1 brings you to where the column begins
        # Adding 1 will bring you to the first entry in the column and so on for +k
        pos = (stim[2]-1) * size(params["images"][1],1) + stim[1]
        spins[pos] = stim[3]
        push!(pinned, (pos, stim[3]))
    end

    for i in 1:1000
        res = Dice.local_search_pinned(model.graph, spins, pinned)
        spins = res
    end
    return spins
end

function sol_finder(state, params)
    rots = iterative_rotater_state(state, params)
    sol = []
    for st in rots
        if (reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)) in params["images"] ||
            -1 .*reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)) in params["images"])
            println("FOUND")
            pretty_table(reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)))
            push!(sol, st)
        end
    end
    return sol
end

end