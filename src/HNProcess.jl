module HNProcess

using Graphs
using SimpleWeightedGraphs
using Dice 
using PrettyTables
using Plots
using Random

export
    HN_Solver,
    get_HN_graph, 
    HN_Solver_Traj,
    iterative_rotater_state,
    iterative_rotater_list,
    HN_cut_plotter,
    HN_og,
    sol_finder,
    hadamard_gen,
    orthogonal_image_generator,
    three_random_orthogonal_image,
    lowest_cut_states,
    objective_func_G,
    lambda_gen,
    unique_random_binary_images



function get_HN_graph(images ::Vector{Matrix{Int}}, scale ::Float64) ::SimpleWeightedGraph
    part_data = vec(images[1]')
    graph_set = SimpleWeightedGraph(length(part_data))
        
    for og in range(1,length(part_data) - 1)
        for term in range(og+1,length(part_data))
            # EDIT HERE, i is origin, j is end. We need to do this multiplication for every image
            w = 0
            for i in images
                w += i[og] * i[term] * scale
            end
            add_edge!(graph_set, og, term, w)
        end
    end
    return graph_set
end

function get_HN_graph(images ::Vector{Matrix{Int}}, scale ::Vector) ::SimpleWeightedGraph
    part_data = vec(images[1]')
    graph_set = SimpleWeightedGraph(length(part_data))
        
    for og in range(1,length(part_data) - 1)
        for term in range(og+1,length(part_data))
            # EDIT HERE, i is origin, j is end. We need to do this multiplication for every image
            w = 0
            for i in 1:length(images)
                w += images[i][og] * images[i][term] * scale[i]
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

function iterative_rotater_list(state, list, s, debug = false)
    rotations = []
    for i in list
        rotated = Dice.realign_hybrid(state, 1+i)
        if debug
            println("rotated by $i")
            pretty_table(reshape(rotated[1],s,s))
        end
        push!(rotations, rotated)
    end
    return rotations
end

function HN_cut_plotter(params, state, debug = false)
    rot = iterative_rotater_state(state,params)
    g = get_HN_graph(params["images"],params["scaling"])
    binary = [i[1] for i in rot]
    x = state[2]
    y = [Dice.cut(g,s) for s in (rot[k][1] for k in 1:length(rot))]
    p = Plots.scatter(x,y)
    if debug
        x_p = p[1][1][:x]
        y_p = p[1][1][:y]
        o = sortperm(y_p)
        x_sorted = x_p[o]
        y_sorted = y_p[o]
        for i in 1:length(x_p)
            println("($(x_sorted[i]),$(y_sorted[i]))")
        end
    end
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

function sol_finder(state, params, d = false)
    rots = iterative_rotater_state(state, params)
    sol = []
    for st in rots
        if (reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)) in params["images"] ||
            -1 .*reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)) in params["images"])
            println("FOUND")
            if d
                pretty_table(reshape(st[1],size(params["images"][1],1),size(params["images"][1],2)))
            end
            push!(sol, st)
        end
    end
    return sol
end

# Note, n must be an even power of 2 (divisible by 4)
# n represents the size of the image (total pixels)
# Creates a hadamard matrix of size n, which contains n images
# Each image has size sqrt(n) by sqrt(n)
function hadamard_gen(n::Int)
    if !(n % 4 == 0 || n == 1 || n == 2)
        throw("argument must be divisible by 4 (or equal to 1 or 2)")
    end
    k = Integer(log(n)/log(2))
    global H = [1]
    for i in 1:k
        # Each time this runs, the resulting size doubles. 
        r1 = hcat(H,H)
        r2 = hcat(H,-1 .* H)
        H_n = vcat(r1,r2)
        H = H_n
    end
    
    return H
end

# Creates a list of images using the hadamard_gen
function orthogonal_image_generator(n::Int)::Vector{Matrix{Int64}}
    had = hadamard_gen(n)
    s =  Integer(sqrt(size(had,1)))
    ims = []
    for r in eachrow(had)
        i = reshape(r,s,s)
        push!(ims, i)
    end
    return ims
end

function orthogonal_image_generator(hada::Matrix)::Vector{Matrix{Int64}}
    had = hada
    s =  Integer(sqrt(size(had,1)))
    ims = []
    for r in eachrow(had)
        i = reshape(r,s,s)
        push!(ims, i)
    end
    return ims
end

# Encodes objective function to maximize
function objective_func_G(state, images)
    s = 0
    for i in images
        for m in 1:length(state)
            for n in 1:length(state)
                s += state[m] * i[m] * state[n] * i[n]
            end
        end
    end
    return (s * 0.25)
end

function three_random_orthogonal_image(N::Int) ::Vector{Matrix{Int64}}
    s = Int(sqrt(N))
    
    # All ones
    i1 = reshape(ones(N), s, s)
    
    # Creates half and half
    i2 = copy(i1)
    i2[1 : Int(N/2)] .= 1
    i2[Int(N/2)+1 : N] .= -1
    
    # Random Orthogonal
    i3 = copy(i2)
    
    # Randomly chooses positions to flip, r1 for first half, r2 for second
    r1 = shuffle(1 : Int(N/2))[1:Int(N/4)]
    r2 = shuffle(Int(N/2)+1 : N)[1:Int(N/4)]
    
    i3[r1] .= -1
    i3[r2] .= 1
    return [i1, i2, i3]
end

function lowest_cut_states(state, params; disp = false)
    graph = get_HN_graph(params["images"],1.0)
    r = iterative_rotater_state(state, params)
    s = []
    c_now = 999
    for i in r
        c_i = cut(graph, i)
        if c_i < c_now
            c_now = c_i
        end
    end
    
    for i in r
        c = cut(graph, i)
        if c == c_now
            push!(s,i)
        end
    end

    if disp
        for i in s 
            pretty_table(reshape(i[1],8,8))
        end
    end
    return s
end

# Generates lambdas which guarentee all images to be equal in size
function lambda_gen(images, scale = 1, debug = false)
    A = zeros(length(images),length(images))
    for i in 1:(length(images))
        for k in i:(length(images))
            D_ij = (vec(images[i])' * vec(images[k]))^2
            A[i,k] = D_ij
            A[k,i] = D_ij
        end
    end
    res = scale*ones(length(images)) # change coeff if precision is not ideal
    lambdas = A \ res
    return lambdas
end

function unique_random_binary_images(num, cardinality)
    images = []
    i = 0
    while i < num
        v = rand((-1,1),cardinality)
        if !(v in images || -1*v in images)
            push!(images, v)
            i += 1
        end
    end
    return reshape.(images, Int(sqrt(cardinality)), Int(sqrt(cardinality)))
end

end