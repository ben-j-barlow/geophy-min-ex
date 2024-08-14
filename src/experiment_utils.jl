using Glob

write_baseline_result_to_file(m, final_belief, final_state; n_fly, reward, seed, r_massive, grid, which_map=:base) = write_result_to_file(m, final_belief, final_state; n_fly=n_fly, reward=reward, seed=seed, r_massive=r_massive, baseline=true, grid, which_map=which_map)
write_intelligent_result_to_file(m, final_belief, final_state; n_fly, reward, seed, r_massive, which_map=:smooth) = write_result_to_file(m, final_belief, final_state; n_fly=n_fly, reward=reward, seed=seed, r_massive=r_massive, baseline=false, nothing, which_map=which_map)
function write_result_to_file(m, final_belief, final_state; n_fly, reward, seed, r_massive, baseline, grid, which_map=:base)
    dir = get_results_dir(baseline=baseline)
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)

    # vertical lines setup
    coords = [final_belief.acts[i].coords for i in 1:(length(final_belief.acts) - 2)];
    #vertical_line_x_coords = unique([c[2] for c in coords]);

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
    #     x, y = normalize_agent_coordinates(final_state.agent_pos_x, final_state.agent_pos_y, m.base_grid_element_length)
    #     add_agent_trajectory_to_plot!(p, x, y, add_start=false)
    # end

    # save map
    type_for_file = which_map == :smooth ? "smooth" : "base"
    savefig(p, "$(dir)trial_$(seed)_$(type_for_file)_trajectory.png")
    empty!(p)

    # plot and save belief
    p_mean, p_std = plot(final_belief, return_individual=true)
    savefig(p_mean, "$(dir)trial_$(seed)_beliefmn.png")
    savefig(p_std, "$(dir)trial_$(seed)_beliefstd.png")
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

end


function extract_value(line::String, keyword::String)
    pattern = Regex("$keyword:\\s*(.*)")
    match_result = Base.match(pattern, line)
    return match_result !== nothing ? match_result.captures[1] : nothing
end

function get_results_dir(;baseline::Bool)
    if baseline
        dir = "/Users/benbarlow/dev/MineralExploration/data/experiments/baselinegrid/"
    else
        dir = "/Users/benbarlow/dev/MineralExploration/data/experiments/intelligent/"
    end
    return dir
end

function get_all_results(;baseline::Bool)
    # Initialize dictionaries to store extracted values
    DIS_RETURN = Dict{Int, Float64}()
    ORES = Dict{Int, Int}()
    N_FLY = Dict{Int, Int}()
    FINAL_ACTION = Dict{Int, String}()
    MEAN = Dict{Int, Float64}()
    STD_DEV = Dict{Int, Float64}()
    EX_COST = Dict{Int, Float64}()

    dir = get_results_dir(baseline=baseline)
    txt_files = glob("*.txt", dir)


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
    # Determine categories
    unprofitable = ORES .< (extraction_cost - 20)
    borderline_profitable = (ORES .>= (extraction_cost - 20)) .& (ORES .<= (extraction_cost + 20))
    profitable = ORES .> (extraction_cost + 20)

    n_unprofitable = sum(unprofitable)
    n_borderline_profitable = sum(borderline_profitable)
    n_profitable = sum(profitable)

    unprofitable_mined = sum(mined .* unprofitable)
    unprofitable_abandoned = sum(abandoned .* unprofitable)

    borderline_profitable_mined = sum(mined .* borderline_profitable)
    borderline_profitable_abandoned = sum(abandoned .* borderline_profitable)

    profitable_mined = sum(mined .* profitable)
    profitable_abandoned = sum(abandoned .* profitable)

    mined_profit = sum(mined .* (ORES .- extraction_cost))
    available_profit = sum(profitable .* (ORES .- extraction_cost))

    mean_flys = mean(N_FLY)
    mined_flys = sum(N_FLY .* mined) / sum(mined)
    abandoned_flys = sum(N_FLY .* abandoned) / sum(abandoned)

    println("N unprofitable: $n_unprofitable, N borderline profitable: $n_borderline_profitable, N profitable: $n_profitable")
    println("Available Profit: $available_profit, Mined Profit: $mined_profit")
    
    println("Unprofitable: $n_unprofitable, Mined: $unprofitable_mined, Abandoned: $unprofitable_abandoned")
    println("Borderline Profitable: $n_borderline_profitable, Mined: $borderline_profitable_mined, Abandoned: $borderline_profitable_abandoned")
    println("Profitable: $n_profitable, Mined: $profitable_mined, Abandoned: $profitable_abandoned")
    
    println("Mean Bores: $mean_flys, Mined Flys: $mined_flys, Abandon Flys: $abandoned_flys")
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
