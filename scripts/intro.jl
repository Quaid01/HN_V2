using DrWatson
@quickactivate "HN_V2"

# Here you may include files from the source directory
include(srcdir("dummy_src_file.jl"))

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())

Have fun with your new project!

You can help us improve DrWatson by opening
issues on GitHub, submitting feature requests,
or even opening your own Pull Requests!
"""
)
sim_output_dir = mkpath(datadir("sims/"));
data_input_dir = datadir("data_sets/");

modules_path = projectdir() * "/src"
push!(LOAD_PATH, modules_path)
