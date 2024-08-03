const DEG_TO_RAD = π / 180

function GeoStatsDistribution(p::MineralExplorationPOMDP; truth=false)
    #@info "GeoStatsDistribution(p::MineralExplorationPOMDP; truth=false)"
    grid_dims = truth ? p.high_fidelity_dim : p.grid_dim
    variogram = SphericalVariogram(sill=p.variogram[1], range=p.variogram[2],
                                    nugget=p.variogram[3])
    #domain = CartesianGrid(convert(Float64, grid_dims[1]), convert(Float64, grid_dims[2]))
    domain = CartesianGrid(grid_dims[1], grid_dims[2])
    #return GeoStatsDistribution(rng=p.rng,
    if p.mineral_exploration_mode == "borehole"
        return GeoStatsDistribution(grid_dims=grid_dims,
                                    data=deepcopy(p.initial_data),
                                    geophysical_data=GeophysicalObservations(),
                                    domain=domain,
                                    mean=p.gp_mean,
                                    variogram=variogram)
    elseif p.mineral_exploration_mode == "geophysical"
        return GeoStatsDistribution(grid_dims=grid_dims,
                                    data=RockObservations(),
                                    geophysical_data=deepcopy(p.initial_geophysical_data),
                                    domain=domain,
                                    mean=p.gp_mean,
                                    variogram=variogram)
    else
        error("Invalid Mineral Exploration Mode: $(p.mineral_exploration_mode)")
    end
end

function GSLIBDistribution(p::MineralExplorationPOMDP)
    #@info "GSLIBDistribution(p::MineralExplorationPOMDP)"
    variogram = (1, 1, 0.0, 0.0, 0.0, p.variogram[2], p.variogram[2], 1.0)
    # variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    return GSLIBDistribution(grid_dims=p.grid_dim, n=p.grid_dim,
                            data=deepcopy(p.initial_data), mean=p.gp_mean,
                            sill=p.variogram[1], variogram=variogram,
                            nugget=p.variogram[3])
end

"""
    sample_coords(dims::Tuple{Int, Int}, n::Int)
Sample coordinates from a Cartesian grid of dimensions given by dims and return
them in an array
"""
function sample_coords(dims::Tuple{Int, Int, Int}, n::Int)
    #@info "sample_coords(dims::Tuple{Int, Int, Int}, n::Int)"
    idxs = CartesianIndices(dims)
    samples = sample(idxs, n)
    sample_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(samples)
        sample_array[1, i] = sample[1]
        sample_array[2, i] = sample[2]
    end
    return (samples, sample_array)
end

function sample_initial(p::MineralExplorationPOMDP, n::Integer)
    #@info "sample_initial(p::MineralExplorationPOMDP, n::Integer)"
    if p.mineral_exploration_mode == "borehole"
        coords, coords_array = sample_coords(p.grid_dim, n)
        dist = GeoStatsDistribution(p)
        state = rand(p.rng, dist)
        ore_quality = state[coords]
        return RockObservations(ore_quality, coords_array)
    elseif p.mineral_exploration_mode == "geophysical"
        error("Geophysical mode not implemented yet")
    else
        error("Invalid Mineral Exploration Mode: $(p.mineral_exploration_mode)")
    end
end

function sample_initial(p::MineralExplorationPOMDP, coords::Array)
    #@info "sample_initial(p::MineralExplorationPOMDP, coords::Array)"
    error("not considered geophysical case, see function defined above")
    n = length(coords)
    coords_array = Array{Int64}(undef, 2, n)
    for (i, sample) in enumerate(coords)
        coords_array[1, i] = sample[1]
        coords_array[2, i] = sample[2]
    end
    dist = GeoStatsDistribution(p)
    state = rand(p.rng, dist)
    ore_quality = state[coords]
    return RockObservations(ore_quality, coords_array)
end

function initialize_data!(p::MineralExplorationPOMDP, n::Integer)
    #@info "initialize_data!(p::MineralExplorationPOMDP, n::Integer)"
    if p.mineral_exploration_mode == "geophysical" 
        error("Geophysical mode not implemented yet")
    end
    new_rock_obs = sample_initial(p, n)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

function initialize_data!(p::MineralExplorationPOMDP, coords::Array)
    #@info "initialize_data!(p::MineralExplorationPOMDP, coords::Array)"

    new_rock_obs = sample_initial(p, coords)
    append!(p.initial_data.ore_quals, new_rock_obs.ore_quals)
    p.initial_data.coordinates = hcat(p.initial_data.coordinates, new_rock_obs.coordinates)
    return p
end

POMDPs.discount(::MineralExplorationPOMDP) = 0.99
POMDPs.isterminal(m::MineralExplorationPOMDP, s::MEState) = s.decided

function POMDPs.initialstate(m::MineralExplorationPOMDP)
    #@info "POMDPs.initialstate(m::MineralExplorationPOMDP)"
    true_gp_dist = m.geodist_type(m; truth=true)
    gp_dist = m.geodist_type(m)
    MEInitStateDist(true_gp_dist, gp_dist, m.mainbody_weight,
                    m.true_mainbody_gen, m.mainbody_gen,
                    m.massive_threshold, m.dim_scale, m.target_dim_scale,
                    m.target_mass_params[1], m.target_mass_params[2], m.rng, m.sigma, m.upscale_factor, m) #m.rng passes global
end

function smooth_map_with_filter(map::Array{Float64}, sigma::Float64, upscale_factor::Int)
    map_2d = reshape(map, 50, 50) # Remove the third dimension since it is 1
    interpolated_map = interpolate(map_2d, BSpline(Linear())) # Increase the resolution using interpolation

    # Generate the high resolution grid points and image
    new_dims = (size(map_2d, 1) * upscale_factor, size(map_2d, 2) * upscale_factor)
    high_res_x = range(1, stop=size(map_2d, 1), length=new_dims[1])  
    high_res_y = range(1, stop=size(map_2d, 2), length=new_dims[2])
    high_res_map_array = [interpolated_map(x, y) for x in high_res_x, y in high_res_y]

    # Apply Gaussian filter to smooth the image
    smooth_map = imfilter(high_res_map_array, Kernel.gaussian(sigma))
    
    #reshape to original dimensionality
    return reshape(smooth_map, new_dims[1], new_dims[2], 1)
end

function Base.rand(rng::Random.AbstractRNG, d::MEInitStateDist, n::Int=1; truth::Bool=false, apply_scale::Bool=false)
    #@info "Base.rand(rng::Random.AbstractRNG, d::MEInitStateDist, n::Int=1; truth::Bool=false, apply_scale::Bool=false)"
    gp_dist = truth ? d.true_gp_distribution : d.gp_distribution
    gp_ore_maps = Base.rand(rng, gp_dist, n)
    if n == 1
        gp_ore_maps = Array{Float64, 3}[gp_ore_maps]
    end

    states = MEState[]
    x_dim = gp_dist.grid_dims[1]
    y_dim = gp_dist.grid_dims[2]
    mainbody_gen = truth ? d.true_mainbody_gen : d.mainbody_gen
    for i = 1:n
        lode_map, lode_params = rand(rng, mainbody_gen)  # sample mainbody
        lode_map = normalize_and_weight(lode_map, d.mainbody_weight)

        gp_ore_map = gp_ore_maps[i]
        ore_map = lode_map + gp_ore_map  # mineralization and geo noise
        if apply_scale
            ore_map, lode_params = scale_sample(d, mainbody_gen, lode_map, gp_ore_map, lode_params; target_μ=d.target_μ, target_σ=d.target_σ)
        end
        smooth_map = smooth_map_with_filter(ore_map, d.sigma, d.upscale_factor)

        state = MEState(ore_map, smooth_map, lode_params, lode_map, RockObservations(), false, false, convert(Float64, d.m.init_heading), [convert(Float64, d.m.init_pos_x)], [convert(Float64, d.m.init_pos_y)], [d.m.init_bank_angle], GeophysicalObservations())
        push!(states, state)
    end
    if n == 1
        return states[1]
    else
        return states
    end
end

Base.rand(d::MEInitStateDist, n::Int=1; kwargs...) = rand(d.rng, d, n; kwargs...)

function normalize_and_weight(lode_map::AbstractArray, mainbody_weight::Real)
    max_lode = maximum(lode_map)
    lode_map ./= max_lode
    lode_map .*= mainbody_weight
    lode_map = repeat(lode_map, outer=(1, 1, 1))
    return lode_map
end

calc_massive(pomdp::MineralExplorationPOMDP, s::MEState) = calc_massive(s.ore_map, pomdp.massive_threshold, pomdp.dim_scale)
function calc_massive(ore_map::AbstractArray, massive_threshold::Real, dim_scale::Real)
    return dim_scale*sum(ore_map .>= massive_threshold)
end

function extraction_reward(m::MineralExplorationPOMDP, s::MEState)
    #@info "extraction_reward(m::MineralExplorationPOMDP, s::MEState)"
    truth = size(s.mainbody_map) == m.high_fidelity_dim
    dim_scale = truth ? m.target_dim_scale : m.dim_scale
    r_massive = calc_massive(s.ore_map, m.massive_threshold, dim_scale)
    r = m.strike_reward*r_massive
    r -= m.extraction_cost
    return r
end

function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, b::MEBelief, rng::Random.AbstractRNG)
    error("POMDPs.gen with a belief passed is has not been implemented (yet) — Robert Moss")
end



function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)
    #@info "POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)"
    if a.type == :fly
        direction = a.change_in_bank_angle < 0 ? "right" : "left"
        #@info "chosen action is fly $(direction) by changing bank angle $(a.change_in_bank_angle) degrees"
    else
        #@info "chosen action is $(a.type)"
    end

    if a ∉ POMDPs.actions(m, s)  # has no check on confidence when stop is chosen in geophysical case
        error("Invalid Action from state")
    end
    stopped = s.stopped
    decided = s.decided
    a_type = a.type
    
    # drill then stop then mine or abandon
    if a_type == :stop && !stopped && !decided
        obs = MEObservation(nothing, true, true, nothing, s.agent_heading, last(s.agent_pos_x), last(s.agent_pos_y), last(s.agent_bank_angle))
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = false
        
        pos_x_p, pos_y_p, heading_p, bank_angle_p, geo_obs_p = s.agent_pos_x, s.agent_pos_y, s.agent_heading, s.agent_bank_angle, deepcopy(s.geophysical_obs)
    elseif a_type == :abandon && stopped && !decided
        heading_p = convert(Float64, 0)
        obs = MEObservation(nothing, true, true, nothing, s.agent_heading, last(s.agent_pos_x), last(s.agent_pos_y), last(s.agent_bank_angle))
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
        
        pos_x_p, pos_y_p, heading_p, bank_angle_p, geo_obs_p = s.agent_pos_x, s.agent_pos_y, s.agent_heading, s.agent_bank_angle, deepcopy(s.geophysical_obs)
    elseif a_type == :mine && stopped && !decided
        obs = MEObservation(nothing, true, true, nothing, s.agent_heading, last(s.agent_pos_x), last(s.agent_pos_y), last(s.agent_bank_angle))
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true

        pos_x_p, pos_y_p, heading_p, bank_angle_p, geo_obs_p = s.agent_pos_x, s.agent_pos_y, s.agent_heading, s.agent_bank_angle, deepcopy(s.geophysical_obs)
    elseif a_type ==:drill && !stopped && !decided
        ore_obs = high_fidelity_obs(m, s.ore_map, a)
        a_coords = reshape(Int64[a.coords[1] a.coords[2]], 2, 1)
        rock_obs_p = deepcopy(s.rock_obs)
        rock_obs_p.coordinates = hcat(rock_obs_p.coordinates, a_coords)
        push!(rock_obs_p.ore_quals, ore_obs)
        n_bores = length(rock_obs_p)
        stopped_p = n_bores >= m.max_bores
        decided_p = false
        # include meaningless bank angle since non-nothing bank angle needed for MEBelief construction
        obs = MEObservation(ore_obs, stopped_p, decided_p, nothing, nothing, nothing, nothing, last(s.agent_bank_angle))

        # create dummy variables
        pos_x_p, pos_y_p, heading_p, bank_angle_p, geo_obs_p = s.agent_pos_x, s.agent_pos_y, s.agent_heading, s.agent_bank_angle, deepcopy(s.geophysical_obs)
    elseif a_type == :fly
        # get new geophysical observation(s)

        new_bank_angle = convert(Int64, last(s.agent_bank_angle) + a.change_in_bank_angle)
        #@info "new bank angle is $(new_bank_angle)"
        current_geophysical_obs, pos_x, pos_y, heading_p = generate_geophysical_obs_sequence(m, s, a, new_bank_angle)
    
        # build MEObservation
        stopped_p = false 
        decided_p = false
        obs = MEObservation(nothing, stopped_p, decided_p, current_geophysical_obs, heading_p, pos_x, pos_y, new_bank_angle)

        # prepare for MEState
        geo_obs_p = append_geophysical_obs_sequence(deepcopy(s.geophysical_obs), current_geophysical_obs)
        pos_x_p = push!(deepcopy(s.agent_pos_x), deepcopy(pos_x))
        pos_y_p = push!(deepcopy(s.agent_pos_y), deepcopy(pos_y))
        bank_angle_p = push!(deepcopy(s.agent_bank_angle), deepcopy(new_bank_angle))

        # create dummy variable
        rock_obs_p = deepcopy(s.rock_obs)
    else
        error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
    end

    r = reward(m, s, a)
    sp = MEState(s.ore_map, s.smooth_map, s.mainbody_params, s.mainbody_map, rock_obs_p, stopped_p, decided_p, heading_p, pos_x_p, pos_y_p, bank_angle_p, geo_obs_p)
    return (sp=sp, o=obs, r=r)
end


function POMDPs.reward(m::MineralExplorationPOMDP, s::MEState, a::MEAction)
    #@info "POMDPs.reward(m::MineralExplorationPOMDP, s::MEState, a::MEAction)"
    stopped = s.stopped
    decided = s.decided
    a_type = a.type

    if m.mineral_exploration_mode == "borehole"
        if a_type in [:stop, :abandon]
            r = 0.0
        elseif a_type == :mine
            r = extraction_reward(m, s)
        elseif a_type ==:drill
            r = -m.drill_cost
        else
            error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
        end
    elseif m.mineral_exploration_mode == "geophysical"
        if a_type == :fly
            if check_plane_within_region(m, last(s.agent_pos_x), last(s.agent_pos_y), m.out_of_bounds_tolerance)
                r = -m.fly_cost
                #@info "negative flying cost $(r)"
            else
                r = - (m.fly_cost + m.out_of_bounds_cost)
                #@info "negative flying cost with out of bounds cost $(r)"
            end
        elseif a_type == :mine
            r = extraction_reward(m, s)
            #@info "mining so positive extraction reward $(r)"
        elseif a_type in [:stop, :abandon]
            r = 0.0
            #@info "stop or abandon so zero reward $(r)"
        end
    else
        error("Invalid Mineral Exploration Mode: $(m.mineral_exploration_mode)")
    end
    return r
end


function POMDPs.actions(m::MineralExplorationPOMDP)
    #@info "POMDPs.actions(m::MineralExplorationPOMDP)"
    if m.mineral_exploration_mode == "geophysical"
        error("Geophysical mode not implemented yet")
    end
    idxs = CartesianIndices(m.grid_dim[1:2])
    bore_actions = reshape(collect(idxs), prod(m.grid_dim[1:2]))
    actions = MEAction[MEAction(type=:stop), MEAction(type=:mine),
                        MEAction(type=:abandon)]
    grid_step = m.grid_spacing + 1
    for coord in bore_actions[1:grid_step:end]
        push!(actions, MEAction(coords=coord))
    end
    return actions
end

function POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)
    #@info "POMDPs.actions(m::MineralExplorationPOMDP, s::MEState)"
    if s.decided
        return MEAction[]
    elseif s.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        if m.mineral_exploration_mode == "borehole"
            action_set = OrderedSet(POMDPs.actions(m))
            n_initial = length(m.initial_data)
            n_obs = length(s.rock_obs.ore_quals) - n_initial
            if m.max_movement != 0 && n_obs > 0
                d = m.max_movement
                drill_s = s.rock_obs.coordinates[:,end]
                x = drill_s[1]
                y = drill_s[2]
                reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
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
            for i=1:n_obs
                coord = s.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = OrderedSet([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
            # delete!(action_set, MEAction(type=:mine))
            # delete!(action_set, MEAction(type=:abandon))
            return collect(action_set)
        elseif m.mineral_exploration_mode == "geophysical"
            #@info "no use of confidence when checking if stop is a permitted action"
            acts = get_flying_actions(m, last(s.agent_bank_angle))
            push!(acts, MEAction(type=:stop))
            return collect(acts)
        else
            error("Invalid Mineral Exploration Mode: $(m.mineral_exploration_mode)")
        end
    end
    return MEAction[]
end

function POMDPModelTools.obs_weight(m::MineralExplorationPOMDP, s::MEState,
                    a::MEAction, sp::MEState, o::MEObservation)
    #@info "POMDPModelTools.obs_weight(m::MineralExplorationPOMDP, s::MEState, a::MEAction, sp::MEState, o::MEObservation)"
    # this function tries to capture the likelihood of observing a particular magnitude of noise
    # the value of noise is the difference between the observation and the mainbody value at the location

    # in the borehole version, the noise only stems from the background variation in the subsurface
    # in the geophysical version, the noise stems from the smoothing of the map, the sensor noise, and the background variation in the subsurface
    
    if m.mineral_exploration_mode == "borehole"
        w = 0.0
        if a.type != :drill
            w = o.ore_quality == nothing ? 1.0 : 0.0
        else
            o_mainbody = high_fidelity_obs(m, s.mainbody_map, a)  # mainbody value at drill location

            # what is the likelihood of a particular value given the Gaussian process on noise that we expect?

            o_gp = (o.ore_quality - o_mainbody)  # difference between observation and mainbody value
            mu = m.gp_mean  # pre-defined mean of the noise GP
            sigma = sqrt(m.variogram[1])  # pre-defined sill of the noise GP
            point_dist = Normal(mu, sigma)  # dist
            w = pdf(point_dist, o_gp)  # weight
        end
    elseif m.mineral_exploration_mode == "geophysical"
        if a.type == :fly && (!is_empty(o.geophysical_obs)) # flying outside of the map region (& mine, abdndon, stop) all lead to o.geophysical_reading == nothing evaluating to true
            # for #observations
            # get mainbody value at drill location
            n_obs = length(o.geophysical_obs.reading)
            #@info "number of obserations $(n_obs)"
            o_n = zeros(Float64, n_obs)

            for i in 1:length(n_obs)
                #@info "observation number $(i)"
                dummy_drill_action = MEAction(type=:drill, coords=CartesianIndex(o.geophysical_obs.base_map_coordinates[1, i], o.geophysical_obs.base_map_coordinates[2, i]))
                o_mainbody = high_fidelity_obs(m, s.mainbody_map, dummy_drill_action)
                #@info "at coordinate $(o.geophysical_obs.base_map_coordinates[1, i]), $(o.geophysical_obs.base_map_coordinates[2, i])"
                #@info "mainbody value  is $(o_mainbody)"
                o_n[i] = (o.geophysical_obs.reading[i] - o_mainbody)  # difference between observation and mainbody value
                #@info "GP noise is $(o_n[i])"
            end
            
            if n_obs == 1
                point_dist = Normal(m.gp_mean, sqrt(m.variogram[1]))  # dist
                w = pdf(point_dist, o_n)[1]
                #@info "weight is $(w) & type is $(typeof(w))"
            else
                # be careful when I implement
                # note w = pdf(point_dist, o_n) above returns a vector
                error("Obs weight not implemented for multiple observations")
            end            
        else  
            if o.geophysical_obs == nothing # mine, abandon, stop lead to == nothing
                w = 1.0
            elseif is_empty(o.geophysical_obs) # flying outside of region leads to empty GeophysicalObservations
                w = 1.0
            else
                error("o.geophysical_reading unexpectedly not nothing")
            end
        end
    else
        error("Invalid Mineral Exploration Mode: $(m.mineral_exploration_mode)")
    end
    return w
end

function high_fidelity_obs(m::MineralExplorationPOMDP, subsurface_map::Array, a::MEAction)
    #@info "high_fidelity_obs(m::MineralExplorationPOMDP, subsurface_map::Array, a::MEAction)"
    if size(subsurface_map) == m.grid_dim
        return subsurface_map[a.coords[1], a.coords[2], 1]
    else
        # truncate drill coordinates to map to high-fidelity grid
        hf_coords = trunc.(Int, a.coords.I ./ m.ratio[1:2])
        return subsurface_map[hf_coords[1], hf_coords[2], 1]
    end
end

function geophysical_obs(x::Int64, y::Int64, smooth_map::Array{Float64}, std_dev::Float64)
    # geophysical equivalent of high_fidelity_obs
    # add noise to the observation

    # ideas
    # use continuous coordinates to generate weighted average of 4 map points
    # let bank angle influence noise
    if x > size(smooth_map)[1]
        error("x coordinate out of bounds")
    elseif y > size(smooth_map)[2]
        error("y coordinate out of bounds")
    else
        #plot_ore_map(smooth_map, title="smooth map in geophysical obs")
        noiseless_geo_obs = smooth_map[y, x, 1]
        if std_dev == 0
            #@info "returning noiseless obs $(noiseless_geo_obs)"
            return noiseless_geo_obs
        end
        noise = rand(Normal(0, std_dev), 1)[1]
        to_return = noiseless_geo_obs + noise
        #@info "noiseless geo obs $(noiseless_geo_obs) with noise $(noise) gives $(to_return)"
        return to_return
    end
end


function update_agent_state(x::Float64, y::Float64, psi::Float64, phi::Float64, v::Int, dt::Float64 = 1.0, g::Float64 = 9.81)
    # psi - current heading
    # phi - current bank angle, in radians
    # v - current velocity (remains constant)
    # normalizing_factor - factor to normalize x and y position by, corresponds to arbitrary length represented by 1 grid square
    #@info "bank angle received in update_agent_state $(phi)"
    if dt != 1.0
        error("stick to dt = 1 for now")
    end
    # get updated heading
    psi_dot = g * tan(phi) / v
    to_return_psi = psi + (psi_dot * dt)

    # get change in x and y
    x_dot = v * cos(to_return_psi)
    y_dot = v * sin(to_return_psi)

    # update x and y
    to_return_x = x + (x_dot * dt)
    to_return_y = y + (y_dot * dt)
    #@info "heading_dot $(psi_dot), x_dot $(x_dot), y_dot $(y_dot)"
    #@info "new heading $(to_return_psi), new x $(to_return_x), new y $(to_return_y)"
    return to_return_x, to_return_y, to_return_psi
end


function generate_geophysical_obs_sequence(m::MineralExplorationPOMDP, s::MEState, a::MEAction, bank_angle::Int)
    # get sequence of positions in meters, convert to smooth grid coordinates and observe geophysical data
    # return in GeophysicalObservations format

    pos_x, pos_y, heading = last(s.agent_pos_x), last(s.agent_pos_y), s.agent_heading  # position in meters

    dt = m.timestep_in_seconds / m.observations_per_timestep

    tmp_go = GeophysicalObservations()

    for i in 1:m.observations_per_timestep
        #@info "\ngenerating observation $(i)"

        #@info "bank angle being parsed to update_agent_state"
        pos_x, pos_y, heading = update_agent_state(pos_x, pos_y, heading, bank_angle * DEG_TO_RAD, m.velocity, dt)  # position in meters
        "new position $(pos_x), $(pos_y), $(heading)"
        # convert position in meters to coordinates (including check for map boundaries)
        
        if check_plane_within_region(m, pos_x, pos_y)
            x_smooth_map, y_smooth_map = get_smooth_map_coordinates(pos_x, pos_y, m)
            x_base_map, y_base_map = get_base_map_coordinates(pos_x, pos_y, m)
        
            # generate observation
            #@info "smooth map coordinates $(x_smooth_map), $(y_smooth_map)"
            obs = geophysical_obs(x_smooth_map, y_smooth_map, s.smooth_map, m.geophysical_noise_std_dev)
            tmp_go.reading = push!(tmp_go.reading, obs)
            tmp_go.smooth_map_coordinates = hcat(tmp_go.smooth_map_coordinates, reshape(Int64[y_smooth_map x_smooth_map], 2, 1))
            tmp_go.base_map_coordinates = hcat(tmp_go.base_map_coordinates, reshape(Int64[y_base_map x_base_map], 2, 1))
        else
            #@info "plane out of region"
        end
    end

    return tmp_go, pos_x, pos_y, heading
end


function append_geophysical_obs_sequence(history::GeophysicalObservations, new_obs::GeophysicalObservations)
    history.reading = vcat(history.reading, new_obs.reading)
    history.base_map_coordinates = hcat(history.base_map_coordinates, new_obs.base_map_coordinates)
    history.smooth_map_coordinates = hcat(history.smooth_map_coordinates, new_obs.smooth_map_coordinates)
    return history
end

function check_plane_within_region(m::MineralExplorationPOMDP, pos_x::Float64, pos_y::Float64, tolerance::Int=0)
    # tolerance - number of grid squares that the plane can be outside the region
    max = (m.grid_dim[1] + tolerance) * m.base_grid_element_length
    min = (0 - tolerance) * m.base_grid_element_length
    pos_x_on_map = min < pos_x <= max
    pos_y_on_map = min < pos_y <= max

    to_return = pos_x_on_map && pos_y_on_map
    return to_return
end

function is_empty(obs::GeophysicalObservations)
    return length(obs.reading) == 0
end