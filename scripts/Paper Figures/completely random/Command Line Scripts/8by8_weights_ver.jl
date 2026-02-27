include("/Users/quaidzahid/Documents/Research Professor/Mikhail Erementchouk/HN_V2/scripts/intro.jl")
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
parameters = Dict{String, Any}(
    "images" => three_random_orthogonal_image(64),
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

println("\n\n V2: 8 by 8!\n\n")

convergences_64 = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => [],
    "multiple_16" => []
)
times_done = 10
s= 3750
max_images = 3
elp = @elapsed begin
    for p in 1:max_images
        println(p)
        conv_16 = 0
        twos_16 = 0 
        multi_16 = 0
        for i in 1:times_done
            parameters["images"] = unique_random_binary_images(p,64)
            k = lambda_gen(parameters["images"],s)
            parameters["scaling"] = k
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
        push!(convergences_64["image_count"], p)
        push!(convergences_64["conv_16"], conv_16/times_done)
        push!(convergences_64["multiple_16"], multi_16/times_done)
    end
end

println("took $elp seconds")

record_data(convergences_64, parameters, "8by8_rand_RawData_V2_Weighted_val_$(s)_detail_$(times_done)")

#HN

println("\n\n2 HN: 8 by 8!\n\n")

convergences_64_hn = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => []
)
elp = @elapsed begin
    for p in 1:max_images
        println(p)
        hn_conv_16 =0 
        for i in 1:times_done
            parameters["images"] = unique_random_binary_images(p,64)
            r = HN_og(parameters)
            if (reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
                -1 .*reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                hn_conv_16 +=1 
            end
        end
        push!(convergences_64_hn["image_count"], p)
        push!(convergences_64_hn["conv_16"], hn_conv_16/times_done)
    end
end
println("took $elp seconds")

record_data(convergences_64_hn, parameters, "8by8_rand_RawData_HN_Weighted_val_$(s)_detail_$(times_done)")


#plot(convergences_64_hn["image_count"], [convergences_64_hn["conv_16"]])

p = plot(convergences_64["image_count"], 
    convergences_64["conv_16"], 
    xlabel="Number of Stored Images (p)",
    ylabel="Probability of Convergence",
    ylims=(-0.05, 1.05),
    dpi=400,
    marker=:circle, 
    markersize=3, 
    markercolor=:blue,
    legendtitle="Images Found",
    label="At least one (1+)"
)

plot!(
    convergences_64["image_count"],
    convergences_64["multiple_16"],
    marker=:square, 
    markersize=3, 
    markercolor=:orange,
    label="At least two (2+)"
)

plot!(
    convergences_64_hn["image_count"],
    convergences_64_hn["conv_16"],
    marker=:diamond, 
    markersize=3, 
    markercolor=:green,
    label="Traditional Hopfield Network"
)

savefig(p, "8by8_rand_Weighted_val_$(s)_detail_$(times_done).png")