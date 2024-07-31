function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
    a::MEAction, o::MEObservation)
    if a.type != :drill
        bp_particles = MEState[] # Create an empty array for updated particles
        for p in b.particles
            s = MEState(p.ore_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided) # Update the state with new observations
            push!(bp_particles, s) # Add the updated state to the particles array
        end
        bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
            coordinates=deepcopy(b.rock_obs.coordinates)) # Deep copy the rock observations
        # Create an updated GeoStatsDistribution or GSLIBDistribution based on the geodist_type
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock,
                b.geostats.domain, b.geostats.mean,
                b.geostats.variogram, b.geostats.lu_params)
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                b.geostats.variogram, b.geostats.target_histogram_file,
                b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                b.geostats.transform_data, b.geostats.mn,
                b.geostats.sz)
        end
    else
        bp_rock = deepcopy(b.rock_obs) # Deep copy the rock observations
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]]) # Concatenate new coordinates
        push!(bp_rock.ore_quals, o.ore_quality) # Add the new ore quality observation
        # Update geostats with new rock observations
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock),
                b.geostats.domain, b.geostats.mean,
                b.geostats.variogram, b.geostats.lu_params)
            update!(bp_geostats, bp_rock) # Update GeoStats distribution
        elseif up.m.geodist_type == GSLIBDistribution
            bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                b.geostats.variogram, b.geostats.target_histogram_file,
                b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                b.geostats.transform_data, b.geostats.mn,
                b.geostats.sz)
        end
        # Reweight and resample particles based on the updated geostats and rock observations
        f_update_particles = up.abc ? update_particles_abc : update_particles
        bp_particles = f_update_particles(up, b.particles, bp_geostats, bp_rock, a, o)
    end

    bp_acts = MEAction[] # Initialize updated actions array
    for act in b.acts
        push!(bp_acts, act) # Add previous actions to the updated actions array
    end
    push!(bp_acts, a) # Add the new action to the actions array

    bp_obs = MEObservation[] # Initialize updated observations array
    for obs in b.obs
        push!(bp_obs, obs) # Add previous observations to the updated observations array
    end
    push!(bp_obs, o) # Add the new observation to the observations array

    bp_stopped = o.stopped # Update stopped status
    bp_decided = o.decided # Update decided status

    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
        bp_decided, bp_geostats, up) # Return the updated belief
end


function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    ws = Float64[] # Initialize weights array
    bore_coords = rock_obs.coordinates # Get borehole coordinates
    n = size(bore_coords)[2] # Number of boreholes
    ore_obs = [o for o in rock_obs.ore_quals] # List of ore quality observations
    K = calc_K(geostats, rock_obs) # Calculate the covariance matrix using geostats and rock observations
    mu = zeros(Float64, n) .+ up.m.gp_mean # Mean vector filled with the Gaussian process mean
    gp_dist = MvNormal(mu, K) # Multivariate normal distribution based on mean vector and covariance matrix
    for s in particles
        mb_map = s.mainbody_map # Get the main body map from the state
        o_n = zeros(Float64, n) # Initialize the array for normalized ore observations
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]] # Main body ore quality at the borehole location
            o_n[i] = ore_obs[i] - o_mainbody # Normalize the ore observation
        end
        w = pdf(gp_dist, o_n) # Calculate the weight as the probability density of the normalized observations
        push!(ws, w) # Add the weight to the weights array
    end
    ws .+= 1e-6 # Add a small value to weights to avoid zero values
    ws ./= sum(ws) # Normalize the weights
    return ws # Return the weights
end


function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
                geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation;
                apply_perturbation=true, resample_background_noise::Bool=true, n=up.n)
    sampled_particles = sample(up.rng, particles, StatsBase.Weights(wp), n, replace=true) # Resample particles based on weights
    mainbody_params = [] # Initialize array for main body parameters
    particles = MEState[] # Initialize array for resampled particles
    ore_quals = deepcopy(rock_obs.ore_quals) # Deep copy of ore quality observations
    for s in sampled_particles
        mainbody_param = s.mainbody_params # Get the main body parameters from the sampled state
        mainbody_map = s.mainbody_map # Get the main body map from the sampled state
        ore_map = s.ore_map # Get the ore map from the sampled state
        gp_ore_map = ore_map - mainbody_map # Calculate the Gaussian process ore map
        if apply_perturbation
            if mainbody_param ∈ mainbody_params
                mainbody_map, mainbody_param = perturb_sample(up.m.mainbody_gen, mainbody_param, up.noise) # Perturb the main body map and parameters
                max_lode = maximum(mainbody_map) # Get the maximum value of the main body map
                mainbody_map ./= max_lode # Normalize the main body map
                mainbody_map .*= up.m.mainbody_weight # Scale the main body map by the main body weight
                mainbody_map = reshape(mainbody_map, up.m.grid_dim) # Reshape the main body map to grid dimensions
                # clamp!(ore_map, 0.0, 1.0) # Optionally clamp the ore map values between 0 and 1
            end
        end
        n_ore_quals = Float64[] # Initialize array for new ore qualities
        for (i, ore_qual) in enumerate(ore_quals)
            prior_ore = mainbody_map[rock_obs.coordinates[1, i], rock_obs.coordinates[2, i], 1] # Get the prior ore quality from the main body map
            n_ore_qual = (ore_qual - prior_ore) # Normalize the ore quality
            push!(n_ore_quals, n_ore_qual) # Add the normalized ore quality to the array
        end
        geostats.data.ore_quals = n_ore_quals # Update geostats with the new ore qualities
        if resample_background_noise
            gp_ore_map = Base.rand(up.rng, geostats) # Resample the Gaussian process ore map
        end
        ore_map = gp_ore_map .+ mainbody_map # Combine the Gaussian process ore map and the main body map
        rock_obs_p = RockObservations(rock_obs.ore_quals, rock_obs.coordinates) # Create a new RockObservations object with the updated coordinates and ore qualities
        sp = MEState(ore_map, mainbody_param, mainbody_map, rock_obs_p, # Create a new state with the updated ore map, parameters, and observations
                    o.stopped, o.decided)
        push!(mainbody_params, mainbody_param) # Add the main body parameters to the array
        push!(particles, sp) # Add the new state to the particles array
    end
    return particles # Return the resampled particles
end
