module DataRefine

using Statistics

export record_data, file_parser, dist_mean

# Things that start with # are ignored when parsing
function record_data(data::Dict{String, Any}, params::Dict{String, Any}, out_name)
    open(out_name, "a") do outf
        println(outf, "#PARAMETERS")
        for param in keys(params)
            println(outf, "# $param = $(params[param])")
        end
        header = join(collect(keys(data)), ", ")
        println(outf, "\n#Note, format for DATA is [$header]")
        println(outf, "\n#DATA")

        println(outf, "\n[")
        #Writing data to file
        entry = collect(keys(data))[1]
        for iteration in 1:length(data[entry])
            row = [data[obs][iteration] for obs in keys(data)]
            println(outf, "\t [", join(row, ", "), "],")
        end
        
        println(outf, "]\n")
    end
    println("The results are saved to $(out_name)")

end

# Parses a file made by record data. Each array in data is a single column.
function file_parser(file_path)
    data = []
    open(file_path, "r") do file
        data_num = 0
        lines = collect(eachline(file))
        for line in lines
            line = strip(line)
            if startswith(line, "#") || isempty(line)
                continue
            end
            if startswith(line, "[") && endswith(line,",")
                line = strip(line, ['[',',',']'])
                data_num = length(split(line,","))
            end
        end
        
        for column in 1:data_num
            temp = []
            for line in lines
                line = strip(line)
                if startswith(line, "#") || isempty(line)
                    continue
                end
                if startswith(line, "[") && endswith(line,",")
                    line = strip(line, ['[',',',']'])
                    push!(temp, parse(Float64, split(line,",")[column]))
                end
            end
            push!(data,temp)
        end
    end
    return data
end

#Given a bunch of vectors of the same length, takes the average for each element and outputs that as a vector

function dist_mean(data)
    results_processed = []
    for entry in data
        s = Float64.(entry)
        push!(results_processed, mean(s));
    end
    return results_processed;
end

end