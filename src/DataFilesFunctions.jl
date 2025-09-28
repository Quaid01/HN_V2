# module DataFilesFunctions
#
# functions for parsing data files describing discrete tomogrpaphy problems
# Implemented formats:
#    1. Rays and projections
#       Example file: ./battle - 6x6 unpacked.dat
#       Annotated Example: ./problem-1.dat

module DataFilesFunctions

export
    DataHolder,
    DHVector,
    processFile,
    print_ray,
    save_unpacked_rays,
    save_trajectories

const IVector = Vector{Int}
const FVector = Vector{Float64}

# Dice's types
const SpinConf = Vector{Int8}
const Hybrid = Tuple{SpinConf,FVector}


### Data types

"The data structure holding a tomographic ray"
mutable struct DataHolder
    id::Int
    projection::Int
    nodes::IVector
end

"The vector of rays (tomographic data)"
const DHVector = Vector{DataHolder}

DataHolder() = DataHolder(-1, -1, [])
DataHolder(projection::Int, nodes::IVector) = DataHolder(-1, projection, nodes)

### String processing

"""
    string_split(str::AbstractString, sep = ' ') ::Tuple{String, String}

Split the string at the separator and return `(car::String, cdr::String)` pair
of the parts of the string strictly before and strictly after the separator.
The returned parts are stripped of the leading and trailing whitespaces. If the
separator is at the beginning or end of the non-white part of `str`, the
returned parts are empty strings. The default separator is space.
"""
function string_split(str::AbstractString, sep=' ')::Tuple{String,String}
    pos_sep = findfirst(sep, str)
    return pos_sep isa Nothing ?
           (str, "") :
           (strip(str[1:(pos_sep-1)]), strip(str[(pos_sep+1):end]))
end

"""
    split_string_at(str::AbstractString, samples::Vector{Char})

Split the string `str` at the first found separator from the list provided
in `samples` and return tuple `(before_separator, found_separator,
after_separator)` with `before_separator` and `after_separator` stripped.
If non of the separators are found, return `(str, "", "")`.
"""
function string_split_at(str::AbstractString,
    samples::Vector{Char})::Tuple{String,String,String}
    l_str = length(str)
    pos = l_str + 1
    for s in samples
        p = findfirst(s, str)
        p isa Nothing && continue

        pos = min(pos, p[1])
        break
    end
    if pos > l_str
        return (strip(str), "", "")
    end
    return (strip(str[1:(pos-1)]), str[pos:pos], strip(str[(pos+1):end]))
end

### FSM

abstract type State end

struct InvalidState <: State
    error_msg::String
end

struct Default <: State end

struct Ray <: State end
struct finalizeRay <: State end
struct processId <: State end
struct processProjection <: State end
struct processNodes <: State end
struct finalizeNodes <: State end
struct readNodes <: State end

struct processControl <: State end
struct processBatch <: State end

struct Command <: State end
struct cmd_STOP <: State end

getElement(::DHVector, ::State, ::AbstractString) =
    error("Undefined file parsing method")

function getElement(acc::DHVector, state::Default, line::AbstractString)
    # assume a nonempty string without leading whitespace as input
    # and perform basic dispatching
    if line[1] == '#'
        return (Command(), line)
    end
    if line[1:3] == "ray"  # currently equivalent to "ray : {"
        push!(acc, DataHolder())
        return (Ray(), "")
    end

    return (InvalidState("Expected '#' or 'ray', got '$(line[1:10])' in '$line'"), "")
end

function getElement(acc::DHVector, state::Command, line::AbstractString)
    # process line starting with '#'
    # currently only END is implemented
    # TODO: parse the PARAMETERS block
    _, command = string_split(line, '#')
    command == "END" && return (cmd_STOP(), "")

    command == "PARAMETERS" && return (Default(), "")

    return (Default(), "")
end

function getElement(acc::DHVector, state::Ray, line::AbstractString)
    # Expects
    #     1. "}" -> finalizeNode
    #     2. "nodes : [" -> processNodes
    #     3. "projection : integer" -> processProjection
    #     4. "id : integer" -> processId
    #     5. Invalid input
    line[1] == '}' && return (finalizeRay(), line)

    data_type, rest = string_split(line, ':')

    data_type == "id" && return (processId(), rest)

    data_type == "projection" && return (processProjection(), rest)

    data_type == "nodes" && return (processNodes(), rest)

    return (InvalidState("Invalid element '$data_type' in line '$line' in the ray description"), "")
end

function getElement(acc::DHVector, state::finalizeRay, line::AbstractString)
    if acc[end].id == -1 || acc[end].projection == -1 || length(acc[end].nodes) == 0
        return (InvalidState("Incomplete ray description"), "")
    end
    return (Default(), "")
end

function getElement(acc::DHVector, state::processId, line::AbstractString)
    # check the id validity and register it
    # Each ray must have a unique id
    if acc[end].id != -1
        return (InvalidState("Multiple id's in the ray description"), "")
    end
    id = parse(Int, line)
    if id < 0
        return (InvalidState("Invalid negative id is provided"), "")
    end
    id_set = Set([r.id for r in acc[1:(end-1)]])
    if in(id, id_set)
        return (InvalidState("Duplicated id, multiple rays with the same id"), "")
    end
    acc[end].id = id
    return (Ray(), "")
end

function getElement(acc::DHVector, state::processProjection, line::AbstractString)
    if acc[end].projection != -1
        return (InvalidState("Duplicated value of ray's projection"), "")
    end

    projection = parse(Int, line)
    if projection < 0
        return (InvalidState("Invalid negative projection is provided"), "")
    end
    acc[end].projection = projection
    return (Ray(), "")
end

function getElement(acc::DHVector, state::processNodes, line::AbstractString)
    if length(acc[end].nodes) > 0
        return (InvalidState("Duplicated list of ray's nodes"), "")
    end

    _, line = string_split(strip(line), '[')
    return (readNodes(), line)
end

function getElement(acc::DHVector, state::finalizeNodes, line::AbstractString)
    return (Ray(), "")
end

function getElement(acc::DHVector, state::readNodes, line::AbstractString)
    value, separator, rest = string_split_at(strip(line), [',', ']'])
    if separator == "," # value contains a number
        push!(acc[end].nodes, parse(Int, value))
        return (readNodes(), rest)
    end

    if separator == "]" # there may be value containing a number
        if length(value) > 0
            push!(acc[end].nodes, parse(Int, value))
        end
        return (Ray(), "")
    end

    return (InvalidState("Invalid format of ray's projection"), "")
end

### API functions

function processFile(file_name::String)::DHVector
    data_coll::DHVector = []
    open(file_name, "r") do res_file
        pars_state = Default()
        for (linecount, line) in enumerate(eachline(res_file) .|> strip)
            while !isempty(line)
                (pars_state, line) = getElement(data_coll, pars_state, line)
                if pars_state isa InvalidState
                    @error "$file_name:$linecount: $(pars_state.error_msg)"
                end
            end
            pars_state isa cmd_STOP && break
        end
    end
    return data_coll
end

# number of nodes printed in the same line
const BATCH_LENGTH = 15

function print_ray(ray::DataHolder, out::Union{IO,IOStream}=stdout)
    println(out, "ray : {")
    println(out, "\tid : $(ray.id)")
    println(out, "\tprojection : $(ray.projection)")
    print(out, "\tnodes : [")
    for (ind, node) in enumerate(ray.nodes)
        print(out, node)
        if ind < length(ray.nodes)
            print(out, ", ")
        end
        mod(ind, BATCH_LENGTH) == 0 && println(out, "\t\t")
    end
    println(out, "]")
    println(out, "}")
end

function save_unpacked_rays(file_name::String,
    rays::DHVector,
    header::Vector{String})
    open(file_name, "w") do out
        println(out, "# Unpacked tomographic data\n")
        println(out, "#PARAMETERS")
        for hd_line in header
            println(out, "# $hd_line")
        end
        println(out, "")

        for ray in rays
            print_ray(ray, out)
        end
    end
end


function save_trajectories(traj_collection::Vector{Vector{Hybrid}},
                           prefix::String,
                           suffix::String)
    # Save the trajectories in the Spin Reader format
    # traj_collection is a collection of trajectories collected after
    # agitations. We save each agitation in a separate file.
    for (ind, traj) in enumerate(traj_collection)
        out_file_name = prefix * "_$(ind)_" * suffix * ".dat"
        open(out_file_name, "w") do outf
            for (t, state) in enumerate(traj)
                print(outf, "$t")
                for (s, x) in zip(state[1], state[2])
                    print(outf, " $s $x")
                end
                println(outf, "")
            end
        end
        println("File $out_file_name is generated")
    end
    
end

end # module ends here
