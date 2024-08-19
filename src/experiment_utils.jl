using Statistics
using Glob

write_baseline_result_to_file(m, final_belief, final_state, means, stds; n_fly, reward, seed, r_massive, grid, which_map=:base) = write_result_to_file(m, final_belief, final_state, means, stds; n_fly=n_fly, reward=reward, seed=seed, r_massive=r_massive, baseline=true, grid=grid, which_map=which_map)
write_intelligent_result_to_file(m, final_belief, final_state, means, stds; n_fly, reward, seed, r_massive, which_map=:smooth) = write_result_to_file(m, final_belief, final_state, means, stds; n_fly=n_fly, reward=reward, seed=seed, r_massive=r_massive, baseline=false, grid=nothing, which_map=which_map)
function write_result_to_file(m, final_belief, final_state, means=nothing, stds=nothing; n_fly, reward, seed, r_massive, baseline, grid, which_map=:base)
    dir = get_results_dir(baseline=baseline)
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)

    # vertical lines setup
    coords = [final_belief.acts[i].coords for i in 1:(length(final_belief.acts) - 2)];
    #vertical_line_x_coords = unique([c[2] for c in coords])

    # plot map
    if which_map == :smooth
        map_to_plot = final_state.smooth_map
        max_val = m.grid_dim[1] * m.upscale_factor
    else
        map_to_plot = final_state.ore_map
        max_val = m.grid_dim[1]
        #vertical_line_x_coords = [ceil(Int, x / m.upscale_factor) for x in vertical_line_x_coords]
    end
    title = "$(final_belief.acts[end].type) at t=$(n_fly+1) with belief $mn & truth $r_massive"
    p = plot_map(map_to_plot, title, axis=false)

    # if baseline
    #     for x in vertical_line_x_coords
    #         add_agent_trajectory_to_plot!(p, [x, x], [0, max_val], add_start=false)
    #         if grid
    #             # plot horizontal_lines
    #             add_agent_trajectory_to_plot!(p, [0, max_val], [x, x], add_start=false)
    #         end
    #     end
    # else
    if !baseline
        x, y = normalize_agent_coordinates(final_state.agent_pos_x, final_state.agent_pos_y, m.base_grid_element_length)
        add_agent_trajectory_to_plot!(p, x, y, add_start=false)
     end

    # save map
    type_for_file = which_map == :smooth ? "smooth" : "base"
    savefig(p, "$(dir)trial_$(seed)_$(type_for_file)_trajectory.pdf")
    plot!(p, title="")
    savefig(p, "$(dir)trial_$(seed)_$(type_for_file)_trajectory_no_title.pdf")
    empty!(p)

    # plot and save belief
    p_mean, p_std = plot(final_belief, return_individual=true)
    # remove title
    plot!(p_mean, title="")
    plot!(p_std, title="")

    savefig(p_mean, "$(dir)trial_$(seed)_beliefmn.pdf")
    savefig(p_std, "$(dir)trial_$(seed)_beliefstd.pdf")
    empty!(p_mean)
    empty!(p_std)

    # write info
    open(string(dir, "trial_$(seed)_info.txt"), "w") do io
        println(io, "Seed: $seed")
        println(io, "Mean: $mn")
        println(io, "Std: $std")
        println(io, "Massive: $r_massive")
        println(io, "Extraction cost: $(m.extraction_cost)")
        println(io, "Decision: $(final_belief.acts[end].type)")
        println(io, "Fly: $n_fly")
        println(io, "Reward: $reward")
        println(io, )
    end

    # write mean and std history
    if means != nothing
        open(string(dir, "trial_$(seed)_mn_history.txt"), "w") do io
            for i in 1:length(means)
                println(io, "$(means[i])")
            end
        end
    end
    if stds != nothing
        open(string(dir, "trial_$(seed)_std_history.txt"), "w") do io
            for i in 1:length(stds)
                println(io, "$(stds[i])")
            end
        end
    end
end


function extract_value(line::String, keyword::String)
    pattern = Regex("$keyword:\\s*(.*)")
    match_result = Base.match(pattern, line)
    return match_result !== nothing ? match_result.captures[1] : nothing
end

function get_results_dir(;baseline::Bool)
    if baseline
        dir = "/Users/benbarlow/dev/MineralExploration/data/experiments/baselinestopearly/"
    else
        dir = "/Users/benbarlow/dev/MineralExploration/data/experiments/intelligent/"
    end
    return dir
end

function get_all_results(;baseline::Bool, dir=nothing)
    # Initialize dictionaries to store extracted values
    DIS_RETURN = Dict{Int, Float64}()
    ORES = Dict{Int, Int}()
    N_FLY = Dict{Int, Int}()
    FINAL_ACTION = Dict{Int, String}()
    MEAN = Dict{Int, Float64}()
    STD_DEV = Dict{Int, Float64}()
    EX_COST = Dict{Int, Float64}()

    if dir == nothing
        dir = get_results_dir(baseline=baseline)
    end
    txt_files = Glob.glob("*.txt", dir)


    for file in txt_files
        current_seed = nothing
        open(file, "r") do f
            for line in eachline(f)
                if occursin("Seed", line)
                    current_seed = parse(Int, extract_value(line, "Seed"))
                elseif occursin("Extraction cost", line)
                    EX_COST[current_seed] = parse(Float64, extract_value(line, "Extraction cost"))
                elseif occursin("Mean", line)
                    MEAN[current_seed] = parse(Float64, extract_value(line, "Mean"))
                elseif occursin("Std", line)
                    STD_DEV[current_seed] = parse(Float64, extract_value(line, "Std"))
                elseif occursin("Massive", line)
                    ORES[current_seed] = parse(Int, extract_value(line, "Massive"))
                elseif occursin("Fly", line)
                    N_FLY[current_seed] = parse(Int, extract_value(line, "Fly"))
                elseif occursin("Decision", line)
                    FINAL_ACTION[current_seed] = extract_value(line, "Decision")
                elseif occursin("Reward", line)
                    DIS_RETURN[current_seed] = parse(Float64, extract_value(line, "Reward"))
                end
            end
        end
    end

    return DIS_RETURN, ORES, N_FLY, FINAL_ACTION, MEAN, STD_DEV, EX_COST
end

function output_results(;baseline::Bool)
    DIS_RETURN, ORES, N_FLY, FINAL_ACTION, MEAN, STD_DEV, EX_COST = get_all_results(baseline=baseline)

    DIS_RETURN = collect(values(DIS_RETURN))
    ORES = collect(values(ORES))
    N_FLY = collect(values(N_FLY))
    FINAL_ACTION = collect(values(FINAL_ACTION))
    MEAN = collect(values(MEAN))
    STD_DEV = collect(values(STD_DEV))
    EX_COST = collect(values(EX_COST))

    @assert length(unique(EX_COST)) == 1
    @assert length(unique(FINAL_ACTION)) == 2
    if baseline
        @assert length(unique(N_FLY)) == 1
    end
    
    extraction_cost = EX_COST[1]
    
    abandoned = [a == "abandon" for a in FINAL_ACTION]
    mined = [a == "mine" for a in FINAL_ACTION]
    
    # Determine categories with ±20 cutoff
    major_unprofitable = ORES .< (extraction_cost - 20)
    minor_unprofitable = (ORES .>= (extraction_cost - 20)) .& (ORES .< extraction_cost)
    minor_profitable = (ORES .>= extraction_cost) .& (ORES .<= (extraction_cost + 20))
    major_profitable = ORES .> (extraction_cost + 20)

    n_major_unprofitable = sum(major_unprofitable)
    n_minor_unprofitable = sum(minor_unprofitable)
    n_minor_profitable = sum(minor_profitable)
    n_major_profitable = sum(major_profitable)

    major_unprofitable_mined = sum(mined .* major_unprofitable)
    major_unprofitable_abandoned = sum(abandoned .* major_unprofitable)

    minor_unprofitable_mined = sum(mined .* minor_unprofitable)
    minor_unprofitable_abandoned = sum(abandoned .* minor_unprofitable)

    minor_profitable_mined = sum(mined .* minor_profitable)
    minor_profitable_abandoned = sum(abandoned .* minor_profitable)

    major_profitable_mined = sum(mined .* major_profitable)
    major_profitable_abandoned = sum(abandoned .* major_profitable)

    major_profitable_mined_profit = sum(mined .* major_profitable .* (ORES .- extraction_cost))
    minor_profitable_mined_profit = sum(mined .* minor_profitable .* (ORES .- extraction_cost))

    major_profitable_available_profit = sum(major_profitable .* (ORES .- extraction_cost))
    minor_profitable_available_profit = sum(minor_profitable .* (ORES .- extraction_cost))

    mined_flys = sum(N_FLY .* mined) / sum(mined)
    abandoned_flys = sum(N_FLY .* abandoned) / sum(abandoned)

    println("Count based")
    println("N Major Unprofitable: $n_major_unprofitable, N Minor Unprofitable: $n_minor_unprofitable, N Minor Profitable: $n_minor_profitable, N Major Profitable: $n_major_profitable")
    println("Major Unprofitable: $n_major_unprofitable, Mined: $major_unprofitable_mined, Abandoned: $major_unprofitable_abandoned")
    println("Minor Unprofitable: $n_minor_unprofitable, Mined: $minor_unprofitable_mined, Abandoned: $minor_unprofitable_abandoned")
    println("Minor Profitable: $n_minor_profitable, Mined: $minor_profitable_mined, Abandoned: $minor_profitable_abandoned")
    println("Major Profitable: $n_major_profitable, Mined: $major_profitable_mined, Abandoned: $major_profitable_abandoned")
    println("")
    println("Profit based")
    println("Major Profitable - available Profit: $major_profitable_available_profit, Mined Profit: $major_profitable_mined_profit")
    println("Minor Profitable - available Profit: $minor_profitable_available_profit, Mined Profit: $minor_profitable_mined_profit")    
end

function spot_check()
    DIS_RETURN, ORES, N_FLY, FINAL_ACTION, MEAN, STD_DEV, EX_COST = get_all_results(baseline=true)

    seed = first(keys(DIS_RETURN))
    println(seed)
    println(MEAN[seed])
    println(STD_DEV[seed])
    println(ORES[seed])
    println(EX_COST[seed])
    println(FINAL_ACTION[seed])
    println(N_FLY[seed])
    println(DIS_RETURN[seed])
end


function get_all_seeds()
    file_path = "/Users/benbarlow/dev/MineralExploration/experiments/seeds.txt"  # Replace with the actual path to your file
    seeds = []

    open(file_path, "r") do file
        for line in eachline(file)
            push!(seeds, parse(Int, strip(line)))
        end
    end
    return seeds
end

function extract_numbers(s::String)
    return parse.(Int, eachmatch(r"\d+", s) .|> x -> x.match)
end

function get_uncompleted_seeds(;baseline::Bool)
    seeds_done = get_completed_seeds(baseline=baseline)
    all_seeds = get_all_seeds()
    return collect(setdiff(all_seeds, seeds_done))
end

# Get all filenames in the directory
function get_completed_seeds(;baseline::Bool)
    dir = get_results_dir(baseline=baseline)
    filenames = readdir(dir)
    all_numbers = Int[]
    
    for filename in filenames
        numbers = extract_numbers(filename)
        append!(all_numbers, numbers)
    end
    
    return collect(Set(all_numbers))
end
    

function get_masses(cost::Float64=157.0; verbose::Bool=false)
    seeds = get_seeds()

    m = MineralExplorationPOMDP()
    ds0 = POMDPs.initialstate(m)
    
    masses = []
    for seed in seeds
        Random.seed!(seed)
        s = rand(ds0)
        push!(masses, calc_massive(m, s))
    end
    if verbose

        println(mean(masses))
        println(std(masses))
        println(sum(masses .> cost) / length(masses))
    end
    return masses
end

function plot_mass_distribution()
    
    masses = get_masses()
    mean_mass = round(mean(masses), digits=2)
    std_mass = round(std(masses), digits=2)
    h = histogram(masses, title="μ=$mean_mass, σ=$std_mass", normalize=:probability)
    p = plot(h)
    return p
end

function summarize_experiment_count()
    baseline_seeds = get_uncompleted_seeds(baseline=true)
    intelligent_seeds = get_uncompleted_seeds(baseline=false)
    println("Baseline: $(length(baseline_seeds))")
    println("Intelligent: $(length(intelligent_seeds))")
end

function compare_results()
    dir1 = "/Users/benbarlow/dev/MineralExploration/data/experiments/baselinegrid/"
    dir2 = "/Users/benbarlow/dev/MineralExploration/data/experiments/baselinenogrid/"

    DIS_RETUR_1, ORE_1, N_FLY_1, FINAL_ACTIO_1, MEA_1, STD_DE_1, EX_COS_1 = get_all_results(baseline=true, dir=dir1)
    DIS_RETUR_2, ORE_2, N_FLY_2, FINAL_ACTIO_2, MEA_2, STD_DE_2, EX_COS_2 = get_all_results(baseline=true, dir=dir2)

    common_keys = intersect(Set(keys(DIS_RETUR_1)), Set(keys(DIS_RETUR_2)))
    # print last component of directory
    println("Directory 1: $(split(dir1, "/")[end-1]) Directory 2: $(split(dir2, "/")[end-1])")
    for key in common_keys
        # output ore1, ore 2, mean 1, mean 2, std 1, std 2
        println("Seed: $key Ore1: $(ORE_1[key]) Ore2: $(ORE_2[key]) Mean1: $(MEA_1[key]) Mean2: $(MEA_2[key]) Std1: $(STD_DE_1[key]) Std2: $(STD_DE_2[key])")
    end
end


function plot_std(;baseline::Bool, dir=nothing)
    if dir == nothing
        dir = get_results_dir(baseline=baseline)
    end
    files = glob("*_std_history.txt", dir)

    # Initialize a list to store all the standard deviation values by line
    all_std_values = []

    # Loop through each file to read the data
    for file in files
        # Read the lines and parse them as Float64
        std_values = readlines(file) |> x -> parse.(Float64, x)
        
        # Append the values to the respective line index in the all_std_values array
        for i in 1:length(std_values)
            if length(all_std_values) < i
                push!(all_std_values, [])
            end
            push!(all_std_values[i], std_values[i])
        end
    end

    # Compute the average standard deviation for each line
    average_std_by_line = [mean(vals) for vals in all_std_values]

    # Plot the results
    plot(1:length(average_std_by_line), average_std_by_line, xlabel="Line Number", ylabel="Average Std Dev", title="Average Std Dev by Line Number", legend=false)
end