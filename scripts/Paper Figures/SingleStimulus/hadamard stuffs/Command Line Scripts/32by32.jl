include(raw"C:\Documents\Research Professor\Mikhail Erementchouk\HN_V2\scripts\intro.jl")
using Graphs
using SimpleWeightedGraphs
using Dice 
using PrettyTables
using LinearAlgebra

using GraphPlot
using Plots
using StatsPlots
using HNProcess
using DataRefine
using PartitioningData
using IterTools
using Random

sim_time = 4
steps = 3000
N = 1024 #image size squared

parameters = Dict{String, Any}(
    "images" => three_random_orthogonal_image(N),
    "sim_time" => sim_time,
    "steps" => steps,
    "iterations" => 100, #Num of times attempted
    "num_agitations" => 1,
    "scaling" => 1.0,
    "delta" => -1* sim_time/steps,
    # This is the region that we will provide and the machine must keep this region constant
    # Will be a vector of tuples which are (row, col, state)
    "initial_stimuli" => [(1,1,1)]
)
println(length(parameters["images"]))

image_set = orthogonal_image_generator(N)
convergences_256 = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => [],
    "multiple_16" => []
)

times_done = 100
image_max = 20

println("V2 Time!")

elp = @elapsed begin
    for p in 1:image_max
        println(p)
        conv_16 = 0
        twos_16 = 0 
        multi_16 = 0
        for i in 1:times_done
            parameters["images"] = image_set[shuffle(1:N)[1:p]]
            r = HN_Solver(parameters)
            sol_count = 0 
            rots = iterative_rotater_state(r,parameters)
            for st in rots
                if (reshape(st[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
                    -1 .*reshape(st[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                    sol_count +=1 
                end
            end
            if sol_count > 0
                conv_16 += 1
            end
            if sol_count >= 2
                multi_16 += 1 
            end
        end
        push!(convergences_256["image_count"], p)
        push!(convergences_256["conv_16"], conv_16/times_done)
        push!(convergences_256["multiple_16"], multi_16/times_done)
    end
end

println("took $elp seconds")

record_data(convergences_256, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_RawData_V2_detail_$(times_done)")

println("Hopfield Time!")

convergences_256_hn = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => []
)
elp = @elapsed begin
    for p in 1:image_max
        println(p)
        hn_conv_16 =0 
        for i in 1:times_done
            parameters["images"] = image_set[shuffle(1:N)[1:p]]
            r = HN_og(parameters)
            if (reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
                -1 .*reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                hn_conv_16 +=1 
            end
        end
        push!(convergences_256_hn["image_count"], p)
        push!(convergences_256_hn["conv_16"], hn_conv_16/times_done)
    end
end
println("took $elp seconds")

record_data(convergences_256_hn, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_RawData_HN_detail_$(times_done)")


p = plot(convergences_256["image_count"], 
    convergences_256["conv_16"], 
    xlabel="Number of Stored Images (p)",
    ylabel="Probability of Convergence",
    ylims=(-0.05, 1.05),
    dpi=400,
    marker=:circle, 
    markersize=3, 
    markercolor=:blue,
    legendtitle="Images Found",
    label="V2: At least one (1+)"
)

plot!(
    convergences_256["image_count"],
    convergences_256["multiple_16"],
    marker=:square, 
    markersize=3, 
    markercolor=:orange,
    label="V2: At least two (2+)"
)

plot!(
    convergences_256_hn["image_count"],
    convergences_256_hn["conv_16"],
    marker=:diamond, 
    markersize=3, 
    markercolor=:green,
    label="Traditional Hopfield Network"
)

savefig(p, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_detail_$(times_done).png")