mutable struct BetaZeroTrainingData
    b # current belief
    π # current policy estimate (using N(s,a))
    z # final outcome (discounted return of the episode)
end

function plot_history(hs::Vector, n_max::Int64=10,
    title::Union{Nothing,String}=nothing,
    y_label::Union{Nothing,String}=nothing,
    box_plot::Bool=false)
    @info "plot_history(hs::Vector, n_max::Int64=10, title::Union{Nothing, String}=nothing, y_label::Union{Nothing, String}=nothing, box_plot::Bool=false)"
    μ = Float64[]
    σ = Float64[]
    vals_vector = Vector{Float64}[]
    for i = 1:n_max
        vals = Float64[]
        for h in hs
            if length(h) >= i
                push!(vals, h[i])
            end
        end
        push!(μ, mean(vals))
        push!(σ, std(vals) / sqrt(length(vals)))
        push!(vals_vector, vals)
    end
    σ .*= 1.0 .- isnan.(σ)
    if isa(title, String)
        if box_plot
            fig = plot(μ, legend=:none, title=title, ylabel=y_label)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, repeat([i], outer=length(vals)), vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none, title=title, ylabel=y_label)
        end
    else
        if box_plot
            fig = plot(μ, legend=:none)
            for (i, vals) in enumerate(vals_vector)
                boxplot!(fig, i, vals, color=:white, outliers=false)
            end
        else
            fig = plot(μ, yerror=σ, legend=:none)
        end
    end
    return (fig, μ, σ)
end

function gen_cases(ds0::MEInitStateDist, n::Int64, save_dir::Union{String,Nothing}=nothing)
    @info "gen_cases(ds0::MEInitStateDist, n::Int64, save_dir::Union{String, Nothing}=nothing)"
    states = MEState[]
    for i = 1:n
        push!(states, rand(ds0.rng, ds0))
    end
    if isa(save_dir, String)
        save(save_dir, "states", states)
    end
    return states
end

function run_trial(m::MineralExplorationPOMDP, up::POMDPs.Updater,
    policy::POMDPs.Policy, s0::MEState, b0::MEBelief;
    display_figs::Bool=true, save_dir::Union{Nothing,String}=nothing,
    return_final_belief=false, return_all_trees=false, collect_training_data=false,
    cmap=:viridis, verbose::Bool=true)
    @info "start of run trial"

    if verbose
        println("Initializing belief...")
    end
    if verbose
        println("Belief Initialized!")
    end

    ore_fig = plot_ore_map(s0.ore_map, cmap)
    mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true)
    b0_fig = plot(b0; cmap=cmap)
    b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose)

    if isa(save_dir, String)
        save_dir = joinpath(abspath(save_dir), "")

        path = string(save_dir, "ore_map.png")
        savefig(ore_fig, path)

        path = string(save_dir, "mass_map.png")
        savefig(mass_fig, path)

        path = string(save_dir, "b0.png")
        savefig(b0_fig, path)

        path = string(save_dir, "b0_hist.png")
        savefig(b0_hist, path)
    end

    if display_figs
        display(ore_fig)
        display(mass_fig)
        display(b0_fig)
        display(b0_hist)
    end

    b_mean, b_std = MineralExploration.summarize(b0)
    if isa(save_dir, String)
        path = string(save_dir, "belief_mean.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_mean, :, 1))
        end
        path = string(save_dir, "belief_std.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_std, :, 1))
        end
    end

    last_action = :drill
    n_drills = 0
    discounted_return = 0.0
    ae = mean(abs.(vols .- r_massive))
    re = mean(vols .- r_massive)
    abs_errs = Float64[ae]
    rel_errs = Float64[re]
    vol_stds = Float64[std_vols]
    final_belief = nothing
    trees = []
    if verbose
        println("Entering Simulation...")
    end
    if collect_training_data
        training_data = [BetaZeroTrainingData(get_input_representation(b0), nothing, nothing)]
    end
    @info "\n\n First timestep"
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=m.max_bores + 2, rng=m.rng)
        @info "\n\n timestep $t"
        discounted_return += POMDPs.discount(m)^(t - 1) * r
        last_action = a.type

        b_fig = plot(bp, t; cmap=cmap)
        b_hist, vols, mean_vols, std_vols = plot_volume(m, bp, r_massive; t=t, verbose=false)

        if verbose
            @show t
            @show a.type
            println("Vols: $mean_vols ± $std_vols")
        end

        if a.type == :drill
            n_drills += 1
            ae = mean(abs.(vols .- r_massive))
            re = mean(vols .- r_massive)
            push!(abs_errs, ae)
            push!(rel_errs, re)
            push!(vol_stds, std_vols)

            if isa(save_dir, String)
                path = string(save_dir, "b$t.png")
                savefig(b_fig, path)

                path = string(save_dir, "b$(t)_hist.png")
                savefig(b_hist, path)
            end
            if display_figs
                display(b_fig)
                display(b_hist)
            end
            b_mean, b_std = MineralExploration.summarize(bp)
            if isa(save_dir, String)
                path = string(save_dir, "belief_mean.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_mean, :, 1))
                end
                path = string(save_dir, "belief_std.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_std, :, 1))
                end
                path = string(save_dir, "pos.txt")
                open(path, "w") do file
                    # Iterate over the vectors and write each pair to the file
                    for i in 1:length(sp.agent_pos_x)
                        x = sp.agent_pos_x[i]
                        y = sp.agent_pos_y[i]
                        println(file, "$x $y")
                    end
                end
            end
        end
        final_belief = bp
        if return_all_trees
            push!(trees, deepcopy(policy.tree))
        end
        if collect_training_data && a.type == :drill # NOTE that :stop and beyond have the same belief representation and obs.
            data = BetaZeroTrainingData(get_input_representation(bp), nothing, nothing)
            push!(training_data, data)
        end
    end
    if verbose
        println("Discounted Return: $discounted_return")
    end
    if collect_training_data
        for d in training_data
            d.z = discounted_return
        end
    end
    ts = [1:length(abs_errs);] .- 1

    abs_err_fig = plot(ts, abs_errs, title="absolute volume error",
        xlabel="time step", ylabel="absolute error", legend=:none, lw=2, c=:crimson)

    rel_err_fig = plot(ts, rel_errs, title="relative volume error",
        xlabel="time step", ylabel="relative error", legend=:none, lw=2, c=:crimson)
    rel_err_fig = plot!([xlims()...], [0, 0], lw=1, c=:black, xlims=xlims())

    vols_fig = plot(ts, vol_stds ./ vol_stds[1], title="volume standard deviation",
        xlabel="time step", ylabel="standard deviation", legend=:none, lw=2, c=:crimson)
    if isa(save_dir, String)
        path = string(save_dir, "abs_err.png")
        savefig(abs_err_fig, path)

        path = string(save_dir, "rel_err.png")
        savefig(rel_err_fig, path)

        path = string(save_dir, "vol_std.png")
        savefig(vols_fig, path)
    end
    if display_figs
        display(abs_err_fig)
        display(rel_err_fig)
        display(vols_fig)
    end

    @info "decision $last_action"
    @info "n drills $n_drills"
    return_values = (discounted_return, abs_errs, rel_errs, vol_stds, n_drills, r_massive, last_action)
    if return_final_belief
        return_values = (return_values..., final_belief)
    end
    if return_all_trees
        return_values = (return_values..., trees)
    end
    if collect_training_data
        return_values = (return_values..., training_data)
    end
    return return_values
end

function run_geophysical_trial(m::MineralExplorationPOMDP, up::POMDPs.Updater,
    policy::POMDPs.Policy, s0::MEState, b0::MEBelief;
    display_figs::Bool=true, save_dir::Union{Nothing,String}=nothing,
    cmap=:viridis, verbose::Bool=true)
    @info "start of run trial"

    ore_fig = plot_ore_map(s0.ore_map, cmap, "base map")
    smooth_fig = plot_ore_map(s0.smooth_map, cmap, "smooth map")

    # r_massive, which is the amount of ore, is calculated using ore map (not smooth map)
    # mass_fig is the plot of the massive ore
    mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true)

    # plot initial belief and histogram
    b0_fig = plot(b0; cmap=cmap) #TODO: add flying lines
    b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose)

    if isa(save_dir, String)
        save_dir = joinpath(abspath(save_dir), "")

        path = string(save_dir, "base_map.png")
        savefig(ore_fig, path)

        path = string(save_dir, "smooth_map.png")
        savefig(smooth_fig, path)

        path = string(save_dir, "mass_map.png")
        savefig(mass_fig, path)

        path = string(save_dir, "b0.png")
        savefig(b0_fig, path)

        path = string(save_dir, "b0_hist.png")
        savefig(b0_hist, path)
    end

    if display_figs
        display(ore_fig)
        display(smooth_fig)
        display(mass_fig)
        display(b0_fig)
        display(b0_hist)
    end

    # summarize built from ore map (not smooth map)
    b_mean, b_std = MineralExploration.summarize(b0)
    if isa(save_dir, String)
        path = string(save_dir, "belief_mean.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_mean, :, 1))
        end
        path = string(save_dir, "belief_std.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_std, :, 1))
        end
    end

    last_action = :fly
    n_flys = 0
    discounted_return = 0.0
    ae = mean(abs.(vols .- r_massive))
    re = mean(vols .- r_massive)
    abs_errs = Float64[ae]
    rel_errs = Float64[re]
    vol_stds = Float64[std_vols]
    final_belief = nothing
    final_state = nothing
    trees = []
    
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=3, rng=m.rng)
        
        discounted_return += POMDPs.discount(m)^(t - 1) * r
        last_action = a.type

        b_fig = plot(bp, t; cmap=cmap)
        b_hist, vols, mean_vols, std_vols = plot_volume(m, bp, r_massive; t=t, verbose=false)

        if verbose
            @info "\n\n timestep $t"
            @info "a type $(a.type)"
        end

        if a.type == :fly
            n_flys += 1
            #ae = mean(abs.(vols .- r_massive))
            #re = mean(vols .- r_massive)
            #push!(abs_errs, ae)
            #push!(rel_errs, re)
            #push!(vol_stds, std_vols)

            if isa(save_dir, String)
                path = string(save_dir, "b$t.png")
                savefig(b_fig, path)

                path = string(save_dir, "b$(t)_hist.png")
                savefig(b_hist, path)
            end
            if display_figs
                display(b_fig)
                display(b_hist)
            end
            #b_mean, b_std = MineralExploration.summarize(bp)
            if isa(save_dir, String)
            #    path = string(save_dir, "belief_mean.txt")
            #    open(path, "a") do io
            #        writedlm(io, reshape(b_mean, :, 1))
            #    end
            #    path = string(save_dir, "belief_std.txt")
            #    open(path, "a") do io
            #        writedlm(io, reshape(b_std, :, 1))
            #    end
            end
        end
        final_belief = bp
        final_state = sp
        
        push!(trees, deepcopy(policy.tree))
        if isa(save_dir, String)
            path = string(save_dir, "plane_trajectory.png")
            savefig(plot_plane_trajectory(sp.agent_pos_x, sp.agent_pos_y), path)
        end
    end
    if verbose
        println("Discounted Return: $discounted_return")
    end
    
    ts = [1:length(abs_errs);] .- 1

    #abs_err_fig = plot(ts, abs_errs, title="absolute volume error",
    #    xlabel="time step", ylabel="absolute error", legend=:none, lw=2, c=:crimson)

    #rel_err_fig = plot(ts, rel_errs, title="relative volume error",
    #    xlabel="time step", ylabel="relative error", legend=:none, lw=2, c=:crimson)
    #rel_err_fig = plot!([xlims()...], [0, 0], lw=1, c=:black, xlims=xlims())

    #vols_fig = plot(ts, vol_stds ./ vol_stds[1], title="volume standard deviation",
    #    xlabel="time step", ylabel="standard deviation", legend=:none, lw=2, c=:crimson)
    if isa(save_dir, String)
        path = string(save_dir, "pos_meters.txt")
        open(path, "w") do file
            # Iterate over the vectors and write each pair to the file
            for i in 1:length(final_state.agent_pos_x)
                x, y = final_state.agent_pos_x[i], final_state.agent_pos_y[i]
                println(file, "$x $y")
            end
        end

        path = string(save_dir, "pos_base_map.txt")
        open(path, "w") do file
            # Iterate over the vectors and write each pair to the file
            for i in 1:length(final_state.agent_pos_x)
                x, y = get_base_map_coordinates(final_state.agent_pos_x[i], final_state.agent_pos_y[i], m)
                println(file, "$x $y")
            end
        end

        path = string(save_dir, "pos_smooth_map.txt")
        open(path, "w") do file
            # Iterate over the vectors and write each pair to the file
            for i in 1:length(final_state.agent_pos_x)
                x, y = get_smooth_map_coordinates(final_state.agent_pos_x[i], final_state.agent_pos_y[i], m)
                println(file, "$x $y")
            end
        end

        #path = string(save_dir, "abs_err.png")
        #savefig(abs_err_fig, path)

        #path = string(save_dir, "rel_err.png")
        #savefig(rel_err_fig, path)

        #path = string(save_dir, "vol_std.png")
        #savefig(vols_fig, path)
    end
    #if display_figs
        #display(abs_err_fig)
        #display(rel_err_fig)
        #display(vols_fig)
    #end

    #return_values = (discounted_return, abs_errs, rel_errs, vol_stds, n_drills, r_massive, last_action)
    display(plot_ore_map(final_state.ore_map, cmap, "ore map"))
    display(plot_ore_map(final_state.smooth_map, cmap, "smooth map"))

    return discounted_return, n_flys, final_belief, trees
end

function plot_plane_trajectory(x_list::Vector{Float64}, y_list::Vector{Float64})
    @info "plot_plane_trajectory(x_list::Vector{Float64}, y_list::Vector{Float64})"
    return plot(x_list, y_list, label="Trajectory", xlabel="X Position (meters)", ylabel="Y Position (meters)",
        title="Aircraft Trajectory", legend=:topright, grid=true, axis=:equal)
end


function plot_ore_map(ore_map, cmap=:viridis, title="true ore map")
    @info "plot_ore_map(ore_map, cmap=:viridis, title=\"true ore map\")"
    xl = (0.5, size(ore_map, 1) + 0.5)
    yl = (0.5, size(ore_map, 2) + 0.5)
    return heatmap(ore_map[:, :, 1], title=title, fill=true, clims=(0.0, 1.0), aspect_ratio=1, xlims=xl, ylims=yl, c=cmap)
end


function plot_mass_map(ore_map, massive_threshold, cmap=:viridis; dim_scale=1, truth=false)
    @info "plot_mass_map(ore_map, massive_threshold, cmap=:viridis; dim_scale=1, truth=false)"
    xl = (0.5, size(ore_map, 1) + 0.5)
    yl = (0.5, size(ore_map, 2) + 0.5)
    s_massive = ore_map .>= massive_threshold
    r_massive = dim_scale * sum(s_massive)
    mass_fig = heatmap(s_massive[:, :, 1], title="massive ore deposits: $(round(r_massive, digits=2))", fill=true, clims=(0.0, 1.0), aspect_ratio=1, xlims=xl, ylims=yl, c=cmap)
    return (mass_fig, r_massive)
end

function plot_volume(m::MineralExplorationPOMDP, b0::MEBelief, r_massive::Real; t=0, verbose::Bool=true)
    @info "plot_volume(m::MineralExplorationPOMDP, b0::MEBelief, r_massive::Real; t=0, verbose::Bool=true)"
    vols = [calc_massive(m, p) for p in b0.particles]
    mean_vols = round(mean(vols), digits=2)
    std_vols = round(std(vols), digits=2)
    if verbose
        println("Vols: $mean_vols ± $std_vols")
    end
    h = fit(Histogram, vols, [0:10:400;])
    h = normalize(h, mode=:probability)

    b0_hist = plot(h, title="belief volumes t=$t, μ=$mean_vols, σ=$std_vols", legend=:none, c=:cadetblue)
    h_height = maximum(h.weights)

    # plot true volume in solid red
    plot!(b0_hist, [r_massive, r_massive], [0.0, h_height], linecolor=:crimson, linewidth=3)
    
    # plot mean volume in dashed red
    plot!([mean_vols, mean_vols], [0.0, h_height], linecolor=:crimson, linestyle=:dash, linewidth=2, label=false)

    # plot extraction cost in gold
    plot!([m.extraction_cost, m.extraction_cost], [0.0, h_height / 3], linecolor=:gold, linewidth=4, label=false)
    ylims!(0, h_height * 1.05)

    return (b0_hist, vols, mean_vols, std_vols)
end
