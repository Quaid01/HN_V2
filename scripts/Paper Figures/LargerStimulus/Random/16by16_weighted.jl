include(raw"C:\Users\qz202\Downloads\Share to Windows\Share to Windows\Research Professor\Mikhail Erementchouk\HN_V2\scripts\intro.jl")
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



N = 256 #image size squared
times_done = 250
max_images = 20
marksize = 5
stimsize = N/8


sim_time = 4
steps = 3000
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

println("\n\n V2: 4 by 4!\n\n")

convergences_64 = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => [],
    "twos_16" => [],
    "multis_16" => []
)
s= 3750
elp = @elapsed begin
    for p in 1:max_images
        println(p)
        conv_16 = 0
        twos_16 = 0 
        multi_16 = 0
        for i in 1:times_done
            parameters["images"] = unique_random_binary_images(p,N)
            k = lambda_gen(parameters["images"],s)
            parameters["scaling"] = k

            parameters["initial_stimuli"] = get_random_stimulus(parameters, stim_size)
            
            r = HN_Solver(parameters)
            sol_count = 0 
            rots = iterative_rotater_state(r,parameters)
            for st in rots
                if (reshape(st[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
                    -1 .*reshape(st[1],size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                    sol_count +=1 
                end
            end
            if sol_count >= 1
                conv_16 += 1
            end
            if sol_count >= 2
                twos_16 += 1 
            end
            if sol_count >= 3
                multi_16 += 1 
            end
        end
        push!(convergences_64["image_count"], p)
        push!(convergences_64["conv_16"], conv_16/times_done)
        push!(convergences_64["twos_16"], twos_16/times_done)
        push!(convergences_64["multis_16"], multi_16/times_done)
    end
end

println("took $elp seconds")

record_data(convergences_64, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_rand_RawData_V2_Weighted_val_$(s)_detail_$(times_done)_StimSize_$(stim_size)")

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
            parameters["images"] = unique_random_binary_images(p,N)

            parameters["initial_stimuli"] = get_random_stimulus(parameters, stim_size)
            
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

record_data(convergences_64_hn, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_rand_RawData_HN_Weighted_val_$(s)_detail_$(times_done)_StimSize_$(stim_size)")


#plot(convergences_64_hn["image_count"], [convergences_64_hn["conv_16"]])

p = plot(convergences_64["image_count"], 
    convergences_64["conv_16"], 
    xlabel="Number of Stored Images (R)",
    ylabel="Probability of Convergence",
    ylims=(-0.05, 1.05),
    dpi=400,
    marker=:circle, 
    markersize=marksize, 
    markercolor=:blue,
    legendtitle="Images Found",
    label="At least one (1+)"
)

plot!(
    convergences_64["image_count"],
    convergences_64["twos_16"],
    marker=:square, 
    markersize=marksize, 
    markercolor=:orange,
    label="At least two (2+)"
)

plot!(
    convergences_64["image_count"],
    convergences_64["multis_16"],
    marker=:star5, 
    markersize=marksize, 
    markercolor=:red,
    label="At least three (3+)"
)

plot!(
    convergences_64_hn["image_count"],
    convergences_64_hn["conv_16"],
    marker=:diamond, 
    markersize=marksize, 
    markercolor=:green,
    label="Traditional Hopfield Network"
)

savefig(p, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_and_Weighted_val_$(s)_detail_$(times_done)_StimSize_$(stim_size).png")
plot(p)