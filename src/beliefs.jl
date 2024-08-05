struct MEBeliefUpdater{G} <: POMDPs.Updater
    m::MineralExplorationPOMDP
    geostats::G
    n::Int64
    noise::Float64  # only used in perturb_sample (og min ex code)
    abc::Bool
    abc_ϵ::Float64
    abc_dist::Function
    rng::AbstractRNG
    sigma::Float64
    upscale_factor::Int
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, noise::Float64=1.0; abc::Bool=false, abc_ϵ::Float64=1e-1, abc_dist::Function=(x, x′) -> abs(x - x′))
    geostats = m.geodist_type(m)
    return MEBeliefUpdater(m, geostats, n, noise, abc, abc_ϵ, abc_dist, m.rng, m.sigma, m.upscale_factor)
end


struct MEBelief{G}
    particles::Vector{MEState} # Vector of vars & lode maps
    rock_obs::RockObservations
    acts::Vector{MEAction}
    obs::Vector{MEObservation}
    stopped::Bool
    decided::Bool
    geostats::G #GSLIB or GeoStats
    up::MEBeliefUpdater ## include the belief updater
    geophysical_obs::GeophysicalObservations
    agent_bank_angle::Vector{Int64}
end

# Ensure MEBeliefs can be compared when adding them to dictionaries (using `hash`, `isequal` and `==`)
Base.hash(r::RockObservations, h::UInt) = hash(Tuple(getproperty(r, p) for p in propertynames(r)), h)
Base.isequal(r1::RockObservations, r2::RockObservations) = all(isequal(getproperty(r1, p), getproperty(r2, p)) for p in propertynames(r1))
Base.:(==)(r1::RockObservations, r2::RockObservations) = isequal(r1, r2)

Base.hash(g::GeoStatsDistribution, h::UInt) = hash(Tuple(getproperty(g, p) for p in propertynames(g)), h)
Base.isequal(g1::GeoStatsDistribution, g2::GeoStatsDistribution) = all(isequal(getproperty(g1, p), getproperty(g2, p)) for p in propertynames(g1))
Base.:(==)(g1::GeoStatsDistribution, g2::GeoStatsDistribution) = isequal(g1, g2)

Base.hash(b::MEBelief, h::UInt) = hash(Tuple(getproperty(b, p) for p in propertynames(b)), h)
Base.isequal(b1::MEBelief, b2::MEBelief) = all(isequal(getproperty(b1, p), getproperty(b2, p)) for p in propertynames(b1))
Base.:(==)(b1::MEBelief, b2::MEBelief) = isequal(b1, b2)


function POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)
    #@info "POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)"
    particles = rand(up.rng, d, up.n)
    if up.m.mineral_exploration_mode == "borehole"
        rock_obs = RockObservations(up.m.initial_data.ore_quals, up.m.initial_data.coordinates)
        geophysical_obs = GeophysicalObservations()
    else
        rock_obs = RockObservations()
        geophysical_obs = GeophysicalObservations(up.m.initial_geophysical_data.reading, up.m.initial_geophysical_data.smooth_map_coordinates, up.m.initial_geophysical_data.base_map_coordinates)
    end
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.geostats, up, geophysical_obs, [up.m.init_bank_angle])
end

# TODO: ParticleFilters.particles
particles(b::MEBelief) = b.particles
# TODO: ParticleFilters.support
POMDPs.support(b::MEBelief) = POMDPs.support(particles(b))


function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    #@info "calc_K(geostats::GeoDist, rock_obs::RockObservations)"
    # Check if the geostats object is of type GeoStatsDistribution
    if isa(geostats, GeoStatsDistribution)
        # If true, use the domain and variogram from the geostats object
        pdomain = geostats.domain
        γ = geostats.variogram
    else
        # Otherwise, create a CartesianGrid using the grid dimensions from geostats
        pdomain = CartesianGrid(geostats.grid_dims[1], geostats.grid_dims[2])
        # Create a SphericalVariogram using the sill, range, and nugget from geostats
        γ = SphericalVariogram(sill=geostats.sill, range=geostats.variogram[6], nugget=geostats.nugget)
    end

    # table = DataFrame(ore=rock_obs.ore_quals .- geostats.mean)

    # Define the domain as a PointSet using the coordinates from rock_obs
    domain = PointSet(rock_obs.coordinates)

    # pdata = georef(table, domain)
    # vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    # dlocs = Int64[m[1] for m in vmapping]
    # 𝒟d = [centroid(pdomain, i) for i in dlocs]

    # Transform the points in the domain by adding 0.5 to their coordinates
    𝒟d = [GeoStats.Point(p.coords[1] + 0.5, p.coords[2] + 0.5) for p in domain]

    # Calculate the covariance matrix K
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)

    #@info "covar matrix $(typeof(K))"
    #@info "covar matrix $(size(K))"
    # Return the covariance matrix K
    return K
end

function calc_K(geostats::GeoDist, geophysical_obs::GeophysicalObservations)
    #@info "calc_K(geostats::GeoDist, geophysical_obs::GeophysicalObservations)"
    # exactly the same implementation as calc_K(geostats::GeoDist, rock_obs::RockObservations)
    # difference is some indexes of K will correspond to the same coordinates rather than all being unique
    γ = geostats.variogram
    𝒟d = [GeoStats.Point(p.coords[1] + 0.5, p.coords[2] + 0.5) for p in PointSet(geophysical_obs.base_map_coordinates)]
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)
    return K
end


function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    #@info "reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)"
    ws = Float64[]
    bore_coords = rock_obs.coordinates
    n = size(bore_coords)[2]
    ore_obs = [o for o in rock_obs.ore_quals]
    K = calc_K(geostats, rock_obs)
    mu = zeros(Float64, n) .+ up.m.gp_mean
    gp_dist = MvNormal(mu, K)
    for s in particles
        mb_map = s.mainbody_map
        o_n = zeros(Float64, n)
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]]
            o_n[i] = ore_obs[i] - o_mainbody
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-6
    ws ./= sum(ws)
    return ws
end


function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, geophysical_obs::GeophysicalObservations)
    #@info "reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, geophysical_obs::GeophysicalObservations)"
    ws = Float64[]
    
    # dedupe observations from same square
    geo_obs_dedupe = aggregate_base_map_duplicates(deepcopy(geophysical_obs))
    coords = geo_obs_dedupe.base_map_coordinates

    n = size(coords)[2]
    if n == 0
        error("no geophysical observations")
    end

    ore_obs = [o for o in geo_obs_dedupe.reading]
    K = calc_K(geostats, geo_obs_dedupe)
    #@info "coordinates $(coords)"
    #@info "K: $(size(K))"
    #@info "K $K"
    mu = zeros(Float64, n) .+ up.m.gp_mean
    gp_dist = MvNormal(mu, K)
    for p in particles
        mb_map = p.mainbody_map
        o_n = zeros(Float64, n)
        for i = 1:n
            o_mainbody = mb_map[coords[1, i], coords[2, i]]
            o_n[i] = ore_obs[i] - o_mainbody
        end
        w = pdf(gp_dist, o_n)
        push!(ws, w)
    end
    ws .+= 1e-6
    ws ./= sum(ws)
    return ws
end


function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
    geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation;
    apply_perturbation=true, resample_background_noise::Bool=true, n=up.n)
    #@info "resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64}, geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)"
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

        smooth_map = smooth_map_with_filter(ore_map, up.sigma, up.upscale_factor)
        sp = MEState(ore_map, smooth_map, mainbody_param, mainbody_map, rock_obs_p, # Create a new state with the updated ore map, parameters, and observations
            o.stopped, o.decided, s.agent_heading, s.agent_pos_x, s.agent_pos_y, s.agent_bank_angle, s.geophysical_obs)
        push!(mainbody_params, mainbody_param) # Add the main body parameters to the array
        push!(particles, sp) # Add the new state to the particles array
    end
    return particles # Return the resampled particles
end



function resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64},
    geostats::GeoDist, geo_obs::GeophysicalObservations, a::MEAction, o::MEObservation;
    apply_perturbation=true, resample_background_noise::Bool=false, n=up.n)
    #@info "resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64}, geostats::GeoDist, geo_obs::GeophysicalObservations, a::MEAction, o::MEObservation)"
    sampled_particles = sample(up.rng, particles, StatsBase.Weights(wp), n, replace=true) # Resample particles based on weights
    mainbody_params = []
    particles = MEState[]

    dummy_geo_obs = GeophysicalObservations() # used to force multiple dispatch to the correct method

    for s in sampled_particles
        gp_map = s.ore_map - s.mainbody_map  # gp represents noise linking the mainbody map to smooth map

        # perform perturbation
        mainbody_param = s.mainbody_params
        mainbody_map = s.mainbody_map
        if apply_perturbation
            if mainbody_param ∈ mainbody_params
                mainbody_map, mainbody_param = perturb_sample(up.m.mainbody_gen, mainbody_param, up.noise)
                max_lode = maximum(mainbody_map)
                mainbody_map ./= max_lode
                mainbody_map .*= up.m.mainbody_weight
                mainbody_map = reshape(mainbody_map, up.m.grid_dim)
                # clamp!(ore_map, 0.0, 1.0) 
            end
        end

        # take average in case of multiple readings at the same location
        geo_obs_dedupe = aggregate_base_map_duplicates(geo_obs)

        # initialize ahead of normalization
        n_normalized_reading = Float64[]
        readings = deepcopy(geo_obs_dedupe.reading)

        # normalize readings using mainbody value
        for (i, value) in enumerate(readings)
            prior_ore = mainbody_map[geo_obs_dedupe.base_map_coordinates[1, i], geo_obs_dedupe.base_map_coordinates[2, i], 1]
            to_append = (value - prior_ore)
            push!(n_normalized_reading, to_append)
        end
        geostats.geophysical_data.reading = n_normalized_reading  # update geostats to contain the normalized readings
        geostats.geophysical_data.base_map_coordinates = geo_obs_dedupe.base_map_coordinates  # update geostats to contain the deduped coordinates

        gp_ore_map = s.ore_map - s.mainbody_map
        if resample_background_noise
            gp_ore_map = Base.rand(up.rng, geostats, dummy_geo_obs)
        end
        ore_map = gp_ore_map .+ mainbody_map

        geo_obs_p = GeophysicalObservations(geo_obs.reading, geo_obs.smooth_map_coordinates, geo_obs.base_map_coordinates)

        smooth_map = smooth_map_with_filter(ore_map, up.sigma, up.upscale_factor)

        sp = MEState(ore_map, smooth_map, mainbody_param, mainbody_map, s.rock_obs,
            o.stopped, o.decided, s.agent_heading, s.agent_pos_x, s.agent_pos_y, s.agent_bank_angle, geo_obs_p)
        push!(mainbody_params, mainbody_param)
        push!(particles, sp)
    end
    return particles # Return the resampled particles
end



function update_particles(up::MEBeliefUpdater, particles::Vector{MEState},
    geostats::GeoDist, obs::Union{GeophysicalObservations,RockObservations}, a::MEAction, o::MEObservation)
    #@info "update_particles"
    wp = reweight(up, geostats, particles, obs)
    pp = resample(up, particles, wp, geostats, obs, a, o)
    return pp
end


function update_particles_perturbed_inject(up::MEBeliefUpdater, particles::Vector{MEState},
    geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    #@info "update_particles_perturbed_inject"
    m = 50 # TODO: parameterize `m`
    wp = reweight(up, geostats, particles, rock_obs)
    injected_particles = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=true, n=m)
    particles = vcat(particles, injected_particles)
    wp2 = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp2, geostats, rock_obs, a, o; apply_perturbation=false)
    return pp
end


function reweight_abc(up::MEBeliefUpdater, particles::Vector, rock_obs::RockObservations)
    #@info "reweight_abc(up::MEBeliefUpdater, particles::Vector, rock_obs::RockObservations)"
    ws = Float64[]
    ϵ = up.abc_ϵ
    rho = up.abc_dist
    ore_quals = deepcopy(rock_obs.ore_quals)
    actions = deepcopy(rock_obs.coordinates)
    for particle in particles
        w = 1
        for a in eachcol(actions)
            for o in ore_quals
                b_o = particle.ore_map[a[1], a[2]]
                w = rho(b_o, o)
                w = w ≤ ϵ ? w : 0 # acceptance tolerance
            end
        end
        push!(ws, w)
    end
    ws .+= 1e-6
    normalize!(ws, 1)
    # effective_particles = 1 / sum(ws .^ 2); @show effective_particles
    return ws
end

function update_particles_abc(up::MEBeliefUpdater, particles::Vector{MEState},
    geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    #@info "update_particles_abc"
    wp = reweight_abc(up, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=false, resample_background_noise=false)
    return pp
end

function inject_particles(up::MEBeliefUpdater, n::Int64)
    #@info "inject_particles(up::MEBeliefUpdater, n::Int64)"
    d = POMDPs.initialstate_distribution(up.m) # TODO. Keep as part of `MEBeliefUpdater`
    return rand(up.rng, d, n)
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
    a::MEAction, o::MEObservation)
    #@info "POMDPs.update(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)"

    if up.m.mineral_exploration_mode == "borehole"
        bp_geophysical_obs = deepcopy(b.geophysical_obs)
        if a.type != :drill
            bp_particles = MEState[] # MEState[p for p in b.particles]
            for p in b.particles
                s = MEState(p.ore_map, p.smooth_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided, p.agent_heading, p.agent_pos_x, p.agent_pos_y, p.agent_bank_angle, p.geophysical_obs) # Update the state with new observations
                push!(bp_particles, s)
            end
            bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
                coordinates=deepcopy(b.rock_obs.coordinates))
            # TODO Make this a little more general in future
            if up.m.geodist_type == GeoStatsDistribution
                bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock, bp_geophysical_obs,
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
        elseif a.type == :drill
            bp_rock = deepcopy(b.rock_obs)
            bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
            push!(bp_rock.ore_quals, o.ore_quality)
            if up.m.geodist_type == GeoStatsDistribution
                bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock), b.geophysical_obs,
                    b.geostats.domain, b.geostats.mean,
                    b.geostats.variogram, b.geostats.lu_params)
                update!(bp_geostats, bp_rock)
            elseif up.m.geodist_type == GSLIBDistribution
                bp_geostats = GSLIBDistribution(b.geostats.grid_dims, b.geostats.grid_dims,
                    bp_rock, b.geostats.mean, b.geostats.sill, b.geostats.nugget,
                    b.geostats.variogram, b.geostats.target_histogram_file,
                    b.geostats.columns_for_vr_and_wt, b.geostats.zmin_zmax,
                    b.geostats.lower_tail_option, b.geostats.upper_tail_option,
                    b.geostats.transform_data, b.geostats.mn,
                    b.geostats.sz)
            end
            f_update_particles = up.abc ? update_particles_abc : update_particles
            bp_particles = f_update_particles(up, b.particles, bp_geostats, bp_rock, a, o)
        end
    elseif up.m.mineral_exploration_mode == "geophysical"
        bp_rock = deepcopy(b.rock_obs) # create dummy variable ahead of instantiation of MEBelief

        if a.type == :fly && !is_empty(o.geophysical_obs)
            bp_geophysical_obs = append_geophysical_obs_sequence(b.geophysical_obs, o.geophysical_obs)            
            bp_dedupe_geophysical_obs = aggregate_base_map_duplicates(bp_geophysical_obs)

            if up.m.geodist_type == GeoStatsDistribution
                bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock), bp_dedupe_geophysical_obs,
                    b.geostats.domain, b.geostats.mean,
                    b.geostats.variogram, b.geostats.lu_params)
                update!(bp_geostats, bp_dedupe_geophysical_obs)
            else
                error("GSLIBDistribution not implemented for fly action")
            end
            bp_particles = update_particles(up, b.particles, bp_geostats, bp_geophysical_obs, a, o)
            
        else # mine, abandon, stop, or (fly & is_empty(o.geophysical_obs))
            bp_particles = MEState[] # MEState[p for p in b.particles]
            for p in b.particles
                # behaviour built into o::MEObservation: plane dynamics are same as in the previous timestep for a in {mine, abandon, or stop}
                # behaviour built into o::MEObservation: plane dynamics update if (a = fly) & (is_empty(o.geophysical_obs))
                agent_pos_x_p = push!(deepcopy(p.agent_pos_x), deepcopy(o.agent_pos_x))
                agent_pos_y_p = push!(deepcopy(p.agent_pos_y), deepcopy(o.agent_pos_y))
                bank_angle_p = push!(deepcopy(p.agent_bank_angle), deepcopy(o.agent_bank_angle))
                s = MEState(p.ore_map, p.smooth_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided, o.agent_heading, agent_pos_x_p, agent_pos_y_p, bank_angle_p, p.geophysical_obs) # Update the state with new observations
                push!(bp_particles, s)
            end    
            bp_geophysical_obs = deepcopy(b.geophysical_obs)
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, bp_rock, bp_geophysical_obs, b.geostats.domain, b.geostats.mean, b.geostats.variogram, b.geostats.lu_params)
        end
    end

    bp_acts = MEAction[]
    for act in b.acts
        push!(bp_acts, act)
    end
    push!(bp_acts, a)

    bp_obs = MEObservation[]
    for obs in b.obs
        push!(bp_obs, obs)
    end
    push!(bp_obs, o)

    bp_stopped = o.stopped
    bp_decided = o.decided

    agent_bank_angle_p = push!(deepcopy(b.agent_bank_angle), o.agent_bank_angle)

    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
        bp_decided, bp_geostats, up, bp_geophysical_obs, agent_bank_angle_p)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    #@info "Base.rand(rng::AbstractRNG, b::MEBelief)"
    return rand(rng, b.particles)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    #@info "summarize(b::MEBelief)"
    (x, y, z) = size(b.particles[1].ore_map)
    μ = zeros(Float64, x, y, z)
    w = 1.0 / length(b.particles)
    for p in b.particles
        ore_map = p.ore_map
        μ .+= ore_map .* w
    end
    σ² = zeros(Float64, x, y, z)
    for p in b.particles
        ore_map = p.ore_map
        σ² .+= w * (ore_map - μ) .^ 2
    end
    return (μ, σ²)
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::MEBelief)
    #@info "POMDPs.actions(m::MineralExplorationPOMDP, b::MEBelief)"
    if m.mineral_exploration_mode == "borehole"
        if b.stopped
            return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
        else
            action_set = OrderedSet(POMDPs.actions(m))
            n_initial = length(m.initial_data)
            if !isempty(b.rock_obs.ore_quals)
                n_obs = length(b.rock_obs.ore_quals) - n_initial
                if m.max_movement != 0 && n_obs > 0
                    d = m.max_movement
                    drill_s = b.rock_obs.coordinates[:, end]
                    x = drill_s[1]
                    y = drill_s[2]
                    reachable_coords = CartesianIndices((x-d:x+d, y-d:y+d))
                    reachable_acts = MEAction[]
                    for coord in reachable_coords
                        dx = abs(x - coord[1])
                        dy = abs(y - coord[2])
                        d2 = sqrt(dx^2 + dy^2)
                        if d2 <= d
                            push!(reachable_acts, MEAction(coords=coord))
                        end
                    end
                    push!(reachable_acts, MEAction(type=:stop))
                    reachable_acts = OrderedSet(reachable_acts)
                    # reachable_acts = OrderedSet([MEAction(coords=coord) for coord in collect(reachable_coords)])
                    action_set = intersect(reachable_acts, action_set)
                end
                for i = 1:n_obs
                    coord = b.rock_obs.coordinates[:, i+n_initial]
                    x = Int64(coord[1])
                    y = Int64(coord[2])
                    keepout = collect(CartesianIndices((x-m.delta:x+m.delta, y-m.delta:y+m.delta)))
                    keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
                    setdiff!(action_set, keepout_acts)
                end
                if n_obs < m.min_bores
                    delete!(action_set, MEAction(type=:stop))
                end
            elseif m.min_bores > 0
                delete!(action_set, MEAction(type=:stop))
            end
            delete!(action_set, MEAction(type=:mine))
            delete!(action_set, MEAction(type=:abandon))
            return collect(action_set)
        end
    elseif m.mineral_exploration_mode == "geophysical"
        # if stopped, return mine & abandon
        if b.stopped
            return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
        end

        # if not stopped but stop bound satisfied, return stop
        tmp = false
        if tmp # calculate_stop_bound(m, b)
            return MEAction[MEAction(type=:stop)]
        end
        
        # if not stopped and stop bound not satisfied, return 3 flying actions subject to bank angle (-45 deg, 45 deg) constraints
        return collect(get_flying_actions(m, last(b.agent_bank_angle)))
    end
    error("Invalid mineral exploration mode")
end

function calculate_stop_bound(m::MineralExplorationPOMDP, b::MEBelief)
    #@info "calculate_stop_bound()"
    volumes = Float64[]
    for p in b.particles
        v = calc_massive(m, p)
        push!(volumes, v)
    end
    mean_volume = Statistics.mean(volumes)
    volume_std = Statistics.std(volumes)
    #@info "mean_volume $(mean_volume)"
    #@info "volume_std $(volume_std)"

    lcb = mean_volume - volume_std*m.extraction_lcb
    ucb = mean_volume + volume_std*m.extraction_ucb

    #@info "lcb is $(lcb) = $(mean_volume - volume_std*m.extraction_lcb) >= $(m.extraction_cost) which is extraction cost"
    #@info "ucb is $(ucb) = $(mean_volume + volume_std*m.extraction_lcb) <= $(m.extraction_cost) which is extraction cost"

    cond1 = lcb >= m.extraction_cost
    cond2 = ucb <= m.extraction_cost
    
    #@info "cond1 $(cond1)"
    #@info "cond2 $(cond2)"
    
    to_return =  cond1 || cond2
    #@info "calculate stop bound returning $(to_return)"
    return to_return
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)
    #@info "POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)"
    if m.mineral_exploration_mode == "borehole"
        o = b.sr_belief.o
        
        #@info "b $(typeof(b))"  # POMCPOW.StateBelief{POWNodeBelief{MEState, MEAction, MEObservation, MineralExplorationPOMDP}}
        #@info "b.sr_belief $(typeof(b.sr_belief))"
        #@info "b.sr_belief.dist $(typeof(b.sr_belief.dist))" # CategoricalVector{Tuple{MEState, Float64}}
        #@info "b.sr_belief.particles $(typeof(b.sr_belief.particles))" # ErrorException("type POWNodeBelief has no field particles")
        #@info "b.sr_belief properties $(propertynames(b.sr_belief))" # properties (:model, :a, :o, :dist)
        #@info "b.sr_belief fieldnames $(fieldnames(typeof(b.sr_belief)))"
        s = rand(m.rng, b.sr_belief.dist)[1]
        #@info "s type $(typeof(s))" # s type MEState{Vector{Any}}

        if o.stopped
            return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
        else
            action_set = OrderedSet(POMDPs.actions(m))
            n_initial = length(m.initial_data)
            if !isempty(s.rock_obs.ore_quals)
                n_obs = length(s.rock_obs.ore_quals) - n_initial
                if m.max_movement != 0 && n_obs > 0
                    d = m.max_movement
                    drill_s = s.rock_obs.coordinates[:, end]
                    x = drill_s[1]
                    y = drill_s[2]
                    reachable_coords = CartesianIndices((x-d:x+d, y-d:y+d))
                    reachable_acts = MEAction[]
                    for coord in reachable_coords
                        dx = abs(x - coord[1])
                        dy = abs(y - coord[2])
                        d2 = sqrt(dx^2 + dy^2)
                        if d2 <= d
                            push!(reachable_acts, MEAction(coords=coord))
                        end
                    end
                    push!(reachable_acts, MEAction(type=:stop))
                    reachable_acts = OrderedSet(reachable_acts)
                    # reachable_acts = OrderedSet([MEAction(coords=coord) for coord in collect(reachable_coords)])
                    action_set = intersect(reachable_acts, action_set)
                end
                for i = 1:n_obs
                    coord = s.rock_obs.coordinates[:, i+n_initial]
                    x = Int64(coord[1])
                    y = Int64(coord[2])
                    keepout = collect(CartesianIndices((x-m.delta:x+m.delta, y-m.delta:y+m.delta)))
                    keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
                    setdiff!(action_set, keepout_acts)
                end
                if n_obs < m.min_bores
                    delete!(action_set, MEAction(type=:stop))
                end
            elseif m.min_bores > 0
                delete!(action_set, MEAction(type=:stop))
            end
            # delete!(action_set, MEAction(type=:mine))
            # delete!(action_set, MEAction(type=:abandon))
            return collect(action_set)
        end
        return MEAction[]
    elseif m.mineral_exploration_mode == "geophysical"
        o = b.sr_belief.o
        s = rand(m.rng, b.sr_belief.dist)[1]
        if o.stopped
            return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
        else
            # add stop and flying actions to the belief tree
            to_return = get_flying_actions(m, last(s.agent_bank_angle))
            push!(to_return, MEAction(type=:stop))
            return collect(to_return)
        end
    end
end

function get_flying_actions(m::MineralExplorationPOMDP, current_bank_angle::Int)
    #@info "get_flying_actions(m::MineralExplorationPOMDP, current_bank_angle::Int)"
    acts = MEAction[]
    if !(current_bank_angle + m.bank_angle_intervals > m.max_bank_angle)
        #@info "bank angle is $(current_bank_angle) so adding action with bank angle $(current_bank_angle + m.bank_angle_intervals)"
        push!(acts, MEAction(type=:fly, change_in_bank_angle=m.bank_angle_intervals))
    end
    if !(current_bank_angle - m.bank_angle_intervals < -m.max_bank_angle)
        #@info "bank angle is $(current_bank_angle) so adding action with bank angle $(current_bank_angle - m.bank_angle_intervals)"
        push!(acts, MEAction(type=:fly, change_in_bank_angle=-m.bank_angle_intervals))
    end
    #@info "adding action with bank angle $(current_bank_angle)"
    push!(acts, MEAction(type=:fly, change_in_bank_angle=0))
    return acts
end

function POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)
    #@info "POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)"

    if o.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        action_set = OrderedSet(POMDPs.actions(m))
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end

function mean_var(b::MEBelief)
    #@info "mean_var(b::MEBelief)"

    vars = [s[1] for s in b.particles]
    mean(vars)
end

function std_var(b::MEBelief)
    #@info "std_var(b::MEBelief)"

    vars = [s[1] for s in b.particles]
    std(vars)
end



function Plots.plot(b::MEBelief, t=nothing; cmap=:viridis)
    #@info "Plots.plot(b::MEBelief, t=nothing; cmap=:viridis)"
    mean, var = summarize(b)
    if t == nothing
        mean_title = "belief mean"
        std_title = "belief std"
    else
        mean_title = "belief mean t=$t"
        std_title = "belief std t=$t"
    end
    xl = (1, size(mean, 1))
    yl = (1, size(mean, 2))
    fig1 = heatmap(mean[:, :, 1], title=mean_title, fill=true, clims=(0.0, 1.0), legend=:none, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig2 = heatmap(sqrt.(var[:, :, 1]), title=std_title, fill=true, legend=:none, clims=(0.0, 0.2), ratio=1, c=cmap, xlims=xl, ylims=yl)
    if !isempty(b.rock_obs.ore_quals)
        x = b.rock_obs.coordinates[2, :]
        y = b.rock_obs.coordinates[1, :]
        plot!(fig1, x, y, seriestype=:scatter)
        plot!(fig2, x, y, seriestype=:scatter)
        n = length(b.rock_obs)
        if n > 1
            for i = 1:n-1
                x = b.rock_obs.coordinates[2, i:i+1]
                y = b.rock_obs.coordinates[1, i:i+1]
                plot!(fig1, x, y, arrow=:closed, color=:black)
            end
        end
    end
    fig = plot(fig1, fig2, layout=(1, 2), size=(600, 250))
    return fig
end


function get_belief_plot_title(t, type)
    allowable_type = ["base", "geophysical"]
    if !(type in allowable_type)
        error("type must be one of $allowable_type")
    end

    if t == nothing
        mean_title = "belief mean ($type)"
        std_title = "belief std ($type)"
    else
        mean_title = "belief mean ($type; t=$t)"
        std_title = "belief std  ($type; t=$t)"
    end
    return mean_title, std_title
end

function Plots.plot(b::MEBelief, m::MineralExplorationPOMDP, t=nothing)
    #@info "Plots.plot(b::MEBelief, m::MineralExplorationPOMDP, t=nothing)"
    if m.mineral_exploration_mode == "borehole"
        return Plots.plot(b, t)
    elseif m.mineral_exploration_mode == "geophysical"
        # prepare data for all plots
        base_map, var = summarize(b)
        smooth_map_mean = smooth_map_with_filter(base_map, m.sigma, m.upscale_factor)
        smooth_map_std = smooth_map_with_filter(var, m.sigma, m.upscale_factor)

        # get title of all plots
        base_mean_title, base_std_title = get_belief_plot_title(t, "base")
        smooth_mean_title, smooth_std_title = get_belief_plot_title(t, "geophysical")

        # create mean plots
        p_base_mean = plot_map(base_map, base_mean_title)
        p_smooth_mean = plot_map(smooth_map_mean, smooth_mean_title)

        # get agent path on respective coordinate systems
        #x, y = get_agent_trajectory(b.agent_bank_angle, m)
        #x_base, y_base = normalize_agent_coordinates(x, y, m.base_grid_element_length)
        #x_smooth, y_smooth = normalize_agent_coordinates(x, y, m.smooth_grid_element_length)

        # make std plots
        xl = (0.5, size(var, 1) + 0.5)
        yl = (0.5, size(var, 2) + 0.5)
        p_base_std = heatmap(sqrt.(var[:, :, 1]), title=base_std_title, fill=true, legend=:none, clims=(0.0, 0.2), ratio=1, c=:viridis, xlims=xl, ylims=yl)
        #p_smooth_std = heatmap(sqrt.(smooth_map_std[:, :, 1]), title=smooth_std_title, fill=true, legend=:none, clims=(0.0, 0.2), ratio=1, c=:viridis, xlims=xl, ylims=yl)
        
        # add agent path
        #add_agent_trajectory_to_plot!(p_base_mean, x_base, y_base)
        #add_agent_trajectory_to_plot!(p_base_std, x_base, y_base)
        #add_agent_trajectory_to_plot!(p_smooth_mean, x_smooth, y_smooth)
        #add_agent_trajectory_to_plot!(p_smooth_std, x_smooth, y_smooth)

        # make plots
        sz = (600, 250)
        fig_base = plot(p_base_mean, p_base_std, layout=(1, 2), size=sz)
        #fig_smooth = plot(p_smooth_mean, p_smooth_std, layout=(1, 2), size=sz)
        return fig_base #, fig_smooth
    end
end


data_skewness(D) = [skewness(D[x, y, 1:end-1]) for x in 1:size(D, 1), y in 1:size(D, 2)]
data_kurtosis(D) = [kurtosis(D[x, y, 1:end-1]) for x in 1:size(D, 1), y in 1:size(D, 2)]


function convert2data(b::MEBelief)
    #@info "convert2data(b::MEBelief)"
    states = cat([p.ore_map[:, :, 1] for p in particles(b)]..., dims=3)
    observations = zeros(size(states)[1:2])
    for (i, a) in enumerate(b.acts)
        if a.type == :drill
            x, y = a.coords.I
            observations[x, y] = b.obs[i].ore_quality
        end
    end
    return cat(states, observations; dims=3)
end


function get_input_representation(b::MEBelief)
    #@info "get_input_representation(b::MEBelief)"

    D = convert2data(b)
    μ = mean(D[:, :, 1:end-1], dims=3)[:, :, 1]
    σ² = std(D[:, :, 1:end-1], dims=3)[:, :, 1]
    sk = data_skewness(D)
    kurt = data_kurtosis(D)
    obs = D[:, :, end]
    return cat(μ, σ², sk, kurt, obs; dims=3)
end


plot_input_representation(b::MEBelief) = plot_input_representation(get_input_representation(b))
function plot_input_representation(B::Array{<:Real,3})
    #@info "plot_input_representation(B::Array{<:Real,3)"
    μ = B[:, :, 1]
    σ² = B[:, :, 2]
    sk = B[:, :, 3]
    kurt = B[:, :, 4]
    obs = B[:, :, 5]
    xl = (1, size(μ, 1))
    yl = (1, size(μ, 2))
    cmap = :viridis
    fig1 = heatmap(μ, title="mean", fill=true, clims=(0, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig2 = heatmap(σ², title="stdev", fill=true, clims=(0.0, 0.2), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig3 = heatmap(sk, title="skewness", fill=true, clims=(-1, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig4 = heatmap(kurt, title="kurtosis", fill=true, clims=(-3, 3), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    fig5 = heatmap(obs, title="obs", fill=true, clims=(0, 1), legend=false, ratio=1, c=cmap, xlims=xl, ylims=yl)
    return plot(fig1, fig2, fig3, fig4, fig5, layout=(1, 5), size=(300 * 5, 250), margin=3Plots.mm)
end
