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


N = 64 #image size squared
times_done = 2000
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
println(length(parameters["images"]))

println("V2 Time!")

image_set = orthogonal_image_generator(N)
convergences_64 = Dict{String, Any}(
    "image_count" => [],
    "conv_16" => [],
    "multiple_16" => []
)

elp = @elapsed begin
    for p in 1:image_max
        println(p)
        conv_16 = 0
        twos_16 = 0 
        multi_16 = 0
        for i in 1:times_done
            parameters["images"] = image_set[shuffle(1:N)[1:p]]
            
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

record_data(convergences_64, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_RawData_V2_detail_$(times_done)_StimSize_$(stim_size)")

println("Hopfield Time!")

convergences_64_hn = Dict{String, Any}(
    "image_count" => [],
    "conv_64" => []
)
elp = @elapsed begin
    for p in 1:image_max
        println(p)
        hn_conv_16 =0 
        for i in 1:times_done
            parameters["images"] = image_set[shuffle(1:N)[1:p]]

            parameters["initial_stimuli"] = get_random_stimulus(parameters, stim_size)
            
            r = HN_og(parameters)
            if (reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"] ||
                -1 .*reshape(r,size(parameters["images"][1],1),size(parameters["images"][1],1)) in parameters["images"])
                hn_conv_16 +=1 
            end
        end
        push!(convergences_64_hn["image_count"], p)
        push!(convergences_64_hn["conv_64"], hn_conv_16/times_done)
    end
end
println("took $elp seconds")

record_data(convergences_64_hn, parameters, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_RawData_HN_detail_$(times_done)_StimSize_$(stim_size)")

p = plot(convergences_64["image_count"], 
    convergences_64["conv_16"], 
    xlabel="Number of Stored Images (p)",
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
    convergences_64["multiple_16"],
    marker=:square, 
    markersize=marksize, 
    markercolor=:orange,
    label="Two"
)

plot!(
    convergences_64_hn["image_count"],
    convergences_64_hn["conv_64"],
    marker=:diamond, 
    markersize=marksize, 
    markercolor=:green,
    label="Traditional Hopfield Network"
)

savefig(p, "$(Int(sqrt(N)))by$(Int(sqrt(N)))_hada_val_$(s)_detail_$(times_done)_StimSize_$(stim_size).png")