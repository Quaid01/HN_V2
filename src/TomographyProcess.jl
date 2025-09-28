# project: image_project
# module TomographyProcess.jl
#
# General functions for processing discrete tomography data

module TomographyProcess

using Graphs
using SimpleWeightedGraphs
using DataFilesFunctions

const WEAK_TOL = 1e-5

export
    # Graph functions
    get_tomography_graph,
    get_tomography_size,

    # Processing functions
    get_total_charge,
    get_state_charges,
    get_nodes_charges,
    get_displacements_field,

    # Reporting functions
    report_charges,
    show_image,
    show_restored_image

const IVector = Vector{Int}
const FVector = Vector{Float64}
const SVector = Vector{String}

### Graph functions

function get_tomography_size(tom_data::DHVector)
    max_number = 0
    for ray in tom_data
        max_number = max(max_number, maximum(ray.nodes))
    end
    return max_number
end

"""
    get_tomography_graph(tom_data::DHVector)::SimpleWeightedGraph

Construct weighted spin graph for the discrete tomography problem 
given by the collection of rays (sets of nodes) with projections 
supplied in `tom_data`. The graph formulation of the tomography
problems is described in the tomography paper.
"""
function get_tomography_graph(tom_data::DHVector)::SimpleWeightedGraph
    #println("Construct tomography graph")
    num_vertices = get_tomography_size(tom_data)
    aux_index = num_vertices + 1

    graph = SimpleWeightedGraph(aux_index)
    for ray in tom_data
        aux_charge = length(ray.nodes) - 2 * ray.projection
        for (src_index, src_node) in enumerate(ray.nodes)
            for dst_node in ray.nodes[(src_index+1):end]
                add_edge!(graph, src_node, dst_node, 1)
            end
            new_aux_weight = weights(graph)[src_node, aux_index] + aux_charge
            if abs(new_aux_weight) > WEAK_TOL
                add_edge!(
                    graph,
                    src_node,
                    aux_index,
                    new_aux_weight,
                )
            else
                # We need to remove the edge explicitly instead
                # of setting its weight to zero
                rem_edge!(graph, src_node, aux_index)
            end
        end
    end
    return graph
end

### Processing functions

"""
    get_nodes_charges(rays::DHVector)::Vector{IVector}

Construct the bare charge vector for each node in the tomography
problem described in `rays`. All nodes are processed, including
the auxiliary spin. The actual current charge is obtained by 
multiplying the bare charge vector by the spin.
"""
function get_nodes_charges(rays::DHVector)::Vector{IVector}
    num_charges = length(rays)
    num_particles = get_tomography_size(rays) + 1
    q_list::Vector{IVector} = []

    for i = 1:num_particles
        push!(q_list, zeros(Int, num_charges))
    end
    for ray in rays
        er = zeros(Int, num_charges)
        # NOTE: we assume strict ray numbering here
        er[ray.id] = 1
        for pixel in ray.nodes
            q_list[pixel] += er
        end
        q_list[end] += (length(ray.nodes) - 2 * ray.projection) .* er
    end
    return q_list
end

function get_state_charges(rays::DHVector, state::Vector{Int8})::Vector{IVector}
    q_list = get_nodes_charges(rays)

    for (index, spin) in enumerate(state)
        q_list[index] .*= spin
    end
    return q_list
end

function get_total_charge(q_list::Vector{IVector})
    Q_tot_charge = 0 .* q_list[1]
    for q in q_list
        Q_tot_charge += q
    end
    return Q_tot_charge
end

function get_displacements_field(rays::DHVector, state::Vector{Int8})
    q_list = get_state_charges(rays, state)
    Q_tot_charge = get_total_charge(q_list)
    #println("Total charge: $(Q_tot_charge)")
    devs = zeros(Int, length(state))
    for (part, sigma) in enumerate(state)
        qQ_part = transpose(q_list[part]) * Q_tot_charge
        q2_part = transpose(q_list[part]) * q_list[part]
        #        println("Node $part : $(qQ_part - q2_part)")
        devs[part] = qQ_part - q2_part
    end
    return devs
end

function report_charges(rays::DHVector, state::Vector{Int8})
    q_list = get_state_charges(rays, state)
    Q_tot_charge = get_total_charge(q_list)

    println("Configuration: $(state)")
    println("The total vector charge: $Q_tot_charge")
    println("with the magnitude: $(transpose(Q_tot_charge) * Q_tot_charge)")

    num_charges = length(rays)
    R_charges = zeros(Int, num_charges)
    for ray in rays
        er = zeros(Int, num_charges)
        er[ray.id] = 1
        r_charge = zeros(Int, num_charges)
        for node in ray.nodes
            r_charge += q_list[node]
        end
        #        println("Ray $(ray.id) vector charge: $(r_charge)")
        R_charges[ray.id] = transpose(er) * (r_charge + q_list[end])
        println("Ray $(ray.id) projection charge: $(R_charges[ray.id])")
    end

    println("Displacements outcome")

    for (part, sigma) in enumerate(state)
        qQ_part = transpose(q_list[part]) * Q_tot_charge
        q2_part = transpose(q_list[part]) * q_list[part]
        println("Node $part : $(qQ_part - q2_part)")
    end
end

function show_image(image::Matrix{Int})
    im_shape = size(image)
    for i = 1:im_shape[1]
        println()
        out_str = ""
        for j = 1:im_shape[2]
            out_str *= image[i, j] == 0 ? "\u00b7 " : "\u2b24 "
        end
        print(out_str)
    end
    println()
end

function show_restored_image(width::Int, state::Vector{Int8})
    if length(state) % width != 0
        println("Showing non-rectangular images is not supported")
        return
    end
    num_lines = length(state) ÷ width
    for i = 1:num_lines
        println()
        out_str = ""
        for j = 1:width
            spin = state[j+(i-1)*width]
            out_str *= spin == -1 ? "\u00b7 " : "\u2b24 "
        end
        print(out_str)
    end
    println()
end

end # module ends here
