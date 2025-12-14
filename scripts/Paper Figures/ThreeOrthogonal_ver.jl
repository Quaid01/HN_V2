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


function experiment_64_v2(params, detail)
    conv_64 = 0
    twos_64 = 0 
    others_64 = 0
    elp = @elapsed begin
        for i in 1:detail
            params["images"] = three_random_orthogonal_image(64)
            r = HN_Solver(params)
            sol_count = 0 
            rots = iterative_rotater_state(r,params)
            for st in rots
                if (reshape(st[1],size(params["images"][1],1),size(params["images"][1],1)) in params["images"] ||
                    -1 .*reshape(st[1],size(params["images"][1],1),size(params["images"][1],1)) in params["images"])
                    sol_count +=1 
                end
            end
            if sol_count > 0
                conv_64 += 1
            end
            if sol_count == 2
                twos_64 += 1 
            end
            if sol_count > 2
                others_64 += 1 
            end
        end
    end
    println("Number of convs: $(conv_64)")
    println("Number of twos: $(twos_64)")
    println("Number of more than two sols: $(others_64)")
    println("took $elp seconds")
    
    d_64 =  Dict{String, Any}(
        "conv_64" => conv_64,
        "twos_64" => twos_64, 
        "others_64" => others_64
        )
    
    record_data(d_64, params, "ThreeOrthogonal_V2_8x8_Raw_Data_detail_$(detail)")
    return [conv_64, twos_64, others_64]
end 

function experiment_256_v2(params, detail)
    conv_256 = 0
    twos_256 = 0 
    others_256 = 0
    elp = @elapsed begin
        for i in 1:detail
            params["images"] = three_random_orthogonal_image(256)
            r = HN_Solver(params)
            sol_count = 0 
            rots = iterative_rotater_state(r,params)
            for st in rots
                if (reshape(st[1],size(params["images"][1],1),size(params["images"][1],1)) in params["images"] ||
                    -1 .*reshape(st[1],size(params["images"][1],1),size(params["images"][1],1)) in params["images"])
                    sol_count +=1 
                end
            end
            if sol_count > 0
                conv_256 += 1
            end
            if sol_count == 2
                twos_256 += 1 
            end
            if sol_count > 2
                others_256 += 1 
            end
        end
    end
    println("Number of convs: $(conv_256)")
    println("Number of twos: $(twos_256)")
    println("Number of more than two sols: $(others_256)")
    println("took $elp seconds")
    
    d_256 =  Dict{String, Any}(
        "conv_256" => conv_256,
        "twos_256" => twos_256, 
        "others_256" => others_256
        )
    
    record_data(d_256, params, "ThreeOrthogonal_V2_16x16_Raw_Data_detail_$(detail)")
    return [conv_256, twos_256, others_256]
end 

function experiment_64_hn(params, detail)
    hn_conv_64 = 0
    elp = @elapsed begin
        for i in 1:detail
            params["images"] = three_random_orthogonal_image(64)
            r = HN_og(parameters)
            if (reshape(r,size(params["images"][1],1),size(params["images"][1],1)) in params["images"] ||
                -1 .*reshape(r,size(params["images"][1],1),size(params["images"][1],1)) in params["images"])
                hn_conv_64 +=1 
            end
        end
    end
    println("Conv: $(hn_conv_64)")
    println("took $elp seconds")
    d_64 =  Dict{String, Any}(
        "conv_64" => hn_conv_64,
        )
    record_data(d_64, params, "ThreeOrthogonal_HN_8x8_Raw_Data_detail_$(detail)")
    return hn_conv_64
end

function experiment_256_hn(params, detail)
    hn_conv_256 = 0
    elp = @elapsed begin
        for i in 1:detail
            params["images"] = three_random_orthogonal_image(256)
            r = HN_og(parameters)
            if (reshape(r,size(params["images"][1],1),size(params["images"][1],1)) in params["images"] ||
                -1 .*reshape(r,size(params["images"][1],1),size(params["images"][1],1)) in params["images"])
                hn_conv_256 +=1 
            end
        end
    end
    println("Conv: $(hn_conv_256)")
    println("took $elp seconds")

    d_256 =  Dict{String, Any}(
        "conv_256" => hn_conv_256,
        )
    record_data(d_256, params, "ThreeOrthogonal_HN_16x16_Raw_Data_detail_$(detail)")

    return hn_conv_256
end

sim_time = 4
steps = 3000
detail = 5
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
println(length(parameters["images"]))

println("\n V2: 8 by 8 time! \n")

exp_8x8_v2 = experiment_64_v2(parameters, detail)

println("\n V2: 16 by 16 time! \n")

exp_16x16_v2 = experiment_256_v2(parameters, detail)

println("\n HN: 8 by 8 time! \n")

exp_8x8_hn = experiment_64_hn(parameters, detail)

println("\n HN: 16 by 16 time! \n")

exp_16x16_hn = experiment_256_hn(parameters, detail)

println("\n Graph time! \n")


sizes = ["64", "256"]
convs = [exp_8x8_v2[1]/detail, exp_16x16_v2[1]/detail]
twos = [exp_8x8_v2[2]/detail, exp_16x16_v2[2]/detail]
hns = [exp_8x8_hn/detail, exp_16x16_hn/detail]

# Note, others is impossible because of thm 6 proven. 

p = groupedbar(
    sizes, 
    [convs twos hns],
    #bar_position = :dodge,
    labels = ["V2: At least one (1+)" "V2: At least two (2+)" "Traditional Hopfield Networks"],
    xlabel="Pixel Count (N)",
    ylabel="Probability of Convergence",
    legendtitle="Images Found",
    legend = :outerright,
    ylims = (0, 1.05),
    dpi = 400,
   
)
@show p
savefig(p, "ThreeOrthogonalImages_V2_detail_$(detail).png")