function run_trial(m::MineralExplorationPOMDP, up::POMDPs.Updater,
    policy::POMDPs.Policy, s0::MEState, b0::MEBelief;
    display_figs::Bool=true, save_dir::Union{Nothing,String}=nothing,
    return_final_belief=false, return_all_trees=false, collect_training_data=false,
    cmap=:viridis, verbose::Bool=true)
    if verbose
        println("Initializing belief...") # Print initialization message if verbose is true
    end
    if verbose
        println("Belief Initialized!") # Print initialized message if verbose is true
    end

    ore_fig = plot_ore_map(s0.ore_map, cmap) # Plot the ore map
    mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true) # Plot the mass map and get the massive ore deposit
    b0_fig = plot(b0; cmap=cmap) # Plot the initial belief
    b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose) # Plot volume histogram for initial belief

    if isa(save_dir, String)
        save_dir = joinpath(abspath(save_dir), "") # Get absolute path for save directory

        path = string(save_dir, "ore_map.png")
        savefig(ore_fig, path) # Save ore map figure

        path = string(save_dir, "mass_map.png")
        savefig(mass_fig, path) # Save mass map figure

        path = string(save_dir, "b0.png")
        savefig(b0_fig, path) # Save initial belief figure

        path = string(save_dir, "b0_hist.png")
        savefig(b0_hist, path) # Save initial belief histogram
    end

    if display_figs
        display(ore_fig) # Display ore map figure
        display(mass_fig) # Display mass map figure
        display(b0_fig) # Display initial belief figure
        display(b0_hist) # Display initial belief histogram
    end

    b_mean, b_std = MineralExploration.summarize(b0) # Summarize initial belief
    if isa(save_dir, String)
        path = string(save_dir, "belief_mean.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_mean, :, 1)) # Save belief mean to file
        end
        path = string(save_dir, "belief_std.txt")
        open(path, "w") do io
            writedlm(io, reshape(b_std, :, 1)) # Save belief standard deviation to file
        end
    end

    last_action = :drill # Initialize last action as drill
    n_drills = 0 # Initialize number of drills
    discounted_return = 0.0 # Initialize discounted return
    ae = mean(abs.(vols .- r_massive)) # Calculate initial absolute error
    re = mean(vols .- r_massive) # Calculate initial relative error
    abs_errs = Float64[ae] # Initialize absolute errors array
    rel_errs = Float64[re] # Initialize relative errors array
    vol_stds = Float64[std_vols] # Initialize volume standard deviations array
    dists = Float64[] # Initialize distances array
    final_belief = nothing # Initialize final belief
    trees = [] # Initialize trees array
    if verbose
        println("Entering Simulation...") # Print entering simulation message if verbose is true
    end
    if collect_training_data
        training_data = [BetaZeroTrainingData(get_input_representation(b0), nothing, nothing)] # Initialize training data if collecting training data
    end
    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", max_steps=m.max_bores + 2, rng=m.rng)  # QFR: why +2?
        discounted_return += POMDPs.discount(m)^(t - 1) * r # Update discounted return
        dist = sqrt(sum(([a.coords[1], a.coords[2]] .- 25.0) .^ 2)) # Calculate distance for fixed coordinates
        last_action = a.type # Update last action

        b_fig = plot(bp, t; cmap=cmap) # Plot belief at time t
        b_hist, vols, mean_vols, std_vols = plot_volume(m, bp, r_massive; t=t, verbose=false) # Plot volume histogram for belief at time t

        if verbose
            @show t # Show current time step
            @show a.type # Show current action type
            @show a.coords # Show current action coordinates
            println("Vols: $mean_vols ± $std_vols") # Print volumes with mean and standard deviation
        end

        if a.type == :drill
            n_drills += 1 # Increment number of drills
            ae = mean(abs.(vols .- r_massive)) # Calculate absolute error for current step
            re = mean(vols .- r_massive) # Calculate relative error for current step
            push!(dists, dist) # Add distance to distances array
            push!(abs_errs, ae) # Add absolute error to errors array
            push!(rel_errs, re) # Add relative error to errors array
            push!(vol_stds, std_vols) # Add volume standard deviation to array

            if isa(save_dir, String)
                path = string(save_dir, "b$t.png")
                savefig(b_fig, path) # Save belief figure at time t

                path = string(save_dir, "b$(t)_hist.png")
                savefig(b_hist, path) # Save belief histogram at time t
            end
            if display_figs
                display(b_fig) # Display belief figure at time t
                display(b_hist) # Display belief histogram at time t
            end
            b_mean, b_std = MineralExploration.summarize(bp) # Summarize belief at time t
            if isa(save_dir, String)
                path = string(save_dir, "belief_mean.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_mean, :, 1)) # Append belief mean to file
                end
                path = string(save_dir, "belief_std.txt")
                open(path, "a") do io
                    writedlm(io, reshape(b_std, :, 1)) # Append belief standard deviation to file
                end
            end
        end
        final_belief = bp # Update final belief
        if return_all_trees
            push!(trees, deepcopy(policy.tree)) # Add policy tree to trees array
        end
        if collect_training_data && a.type == :drill
            data = BetaZeroTrainingData(get_input_representation(bp), nothing, nothing) # Create new training data entry
            push!(training_data, data) # Add training data entry to array
        end
    end
    if verbose
        println("Discounted Return: $discounted_return") # Print discounted return if verbose is true
    end
    if collect_training_data
        for d in training_data
            d.z = discounted_return # Update final outcome in training data
        end
    end
    ts = [1:length(abs_errs);] .- 1 # Create time steps array
    dist_fig = plot(ts[2:end], dists, title="bore distance to center",
        xlabel="time step", ylabel="distance", legend=:none, lw=2, c=:crimson) # Plot bore distances

    abs_err_fig = plot(ts, abs_errs, title="absolute volume error",
        xlabel="time step", ylabel="absolute error", legend=:none, lw=2, c=:crimson) # Plot absolute errors

    rel_err_fig = plot(ts, rel_errs, title="relative volume error",
        xlabel="time step", ylabel="relative error", legend=:none, lw=2, c=:crimson) # Plot relative errors
    rel_err_fig = plot!([xlims()...], [0, 0], lw=1, c=:black, xlims=xlims()) # Add horizontal line at y=0

    vols_fig = plot(ts, vol_stds ./ vol_stds[1], title="volume standard deviation",
        xlabel="time step", ylabel="standard deviation", legend=:none, lw=2, c=:crimson) # Plot volume standard deviations
    if isa(save_dir, String)
        path = string(save_dir, "dists.png")
        savefig(dist_fig, path) # Save distances plot

        path = string(save_dir, "abs_err.png")
        savefig(abs_err_fig, path) # Save absolute errors plot

        path = string(save_dir, "rel_err.png")
        savefig(rel_err_fig, path) # Save relative errors plot

        path = string(save_dir, "vol_std.png")
        savefig(vols_fig, path) # Save volume standard deviations plot
    end
    if display_figs
        display(dist_fig) # Display distances plot
        display(abs_err_fig) # Display absolute errors plot
        display(rel_err_fig) # Display relative errors plot
        display(vols_fig) # Display volume standard deviations plot
    end
    return_values = (discounted_return, dists, abs_errs, rel_errs, vol_stds, n_drills, r_massive, last_action) # Prepare return values
    if return_final_belief
        return_values = (return_values..., final_belief) # Add final belief to return values if required
    end
    if return_all_trees
        return_values = (return_values..., trees) # Add trees to return values if required
    end
    if collect_training_data
        return_values = (return_values..., training_data) # Add training data to return values if required
    end
    return return_values # Return the final values
end
