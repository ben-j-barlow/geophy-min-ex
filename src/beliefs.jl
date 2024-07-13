struct MEBeliefUpdater{G} <: POMDPs.Updater
    m::MineralExplorationPOMDP
    geostats::G
    n::Int64
    noise::Float64  # only used in perturb_sample (og min ex code)
    abc::Bool
    abc_ϵ::Float64
    abc_dist::Function
    rng::AbstractRNG
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, noise::Float64=1.0; abc::Bool=false, abc_ϵ::Float64=1e-1, abc_dist::Function=(x, x′) -> abs(x - x′))
    geostats = m.geodist_type(m)
    return MEBeliefUpdater(m, geostats, n, noise, abc, abc_ϵ, abc_dist, m.rng)
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
    @info "POMDPs.initialize_belief(up::MEBeliefUpdater, d::MEInitStateDist)"
    particles = rand(up.rng, d, up.n)
    init_rocks = up.m.initial_data
    rock_obs = RockObservations(init_rocks.ore_quals, init_rocks.coordinates)
    acts = MEAction[]
    obs = MEObservation[]
    return MEBelief(particles, rock_obs, acts, obs, false, false, up.geostats, up)
end

# TODO: ParticleFilters.particles
particles(b::MEBelief) = b.particles
# TODO: ParticleFilters.support
POMDPs.support(b::MEBelief) = POMDPs.support(particles(b))


function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    @info "calc_K(geostats::GeoDist, rock_obs::RockObservations)"
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

    # Return the covariance matrix K
    return K
end


function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    @info "reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)"
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
    @info "resample(up::MEBeliefUpdater, particles::Vector, wp::Vector{Float64}, geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)"
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
                    o.stopped, o.decided, s.agent_heading, s.agent_pos_x, s.agent_pos_y, s.agent_velocity, s.agent_bank_angle, s.geophysical_obs)
        push!(mainbody_params, mainbody_param) # Add the main body parameters to the array
        push!(particles, sp) # Add the new state to the particles array
    end
    return particles # Return the resampled particles
end


function update_particles(up::MEBeliefUpdater, particles::Vector{MEState},
    geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    @info "update_particles"
    wp = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o)
    return pp
end


function update_particles_perturbed_inject(up::MEBeliefUpdater, particles::Vector{MEState},
    geostats::GeoDist, rock_obs::RockObservations, a::MEAction, o::MEObservation)
    @info "update_particles_perturbed_inject"
    m = 50 # TODO: parameterize `m`
    wp = reweight(up, geostats, particles, rock_obs)
    injected_particles = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=true, n=m)
    particles = vcat(particles, injected_particles)
    wp2 = reweight(up, geostats, particles, rock_obs)
    pp = resample(up, particles, wp2, geostats, rock_obs, a, o; apply_perturbation=false)
    return pp
end


function reweight_abc(up::MEBeliefUpdater, particles::Vector, rock_obs::RockObservations)
    @info "reweight_abc(up::MEBeliefUpdater, particles::Vector, rock_obs::RockObservations)"
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
    @info "update_particles_abc"
    wp = reweight_abc(up, particles, rock_obs)
    pp = resample(up, particles, wp, geostats, rock_obs, a, o; apply_perturbation=false, resample_background_noise=false)
    return pp
end

function inject_particles(up::MEBeliefUpdater, n::Int64)
    @info "inject_particles(up::MEBeliefUpdater, n::Int64)"
    d = POMDPs.initialstate_distribution(up.m) # TODO. Keep as part of `MEBeliefUpdater`
    return rand(up.rng, d, n)
end

function POMDPs.update(up::MEBeliefUpdater, b::MEBelief,
    a::MEAction, o::MEObservation)
    @info "POMDPs.update(up::MEBeliefUpdater, b::MEBelief, a::MEAction, o::MEObservation)"

    if a.type != :drill
        bp_particles = MEState[] # MEState[p for p in b.particles]
        for p in b.particles
            s = MEState(p.ore_map, p.mainbody_params, p.mainbody_map, p.rock_obs, o.stopped, o.decided, p.agent_heading, p.agent_pos_x, p.agent_pos_y, p.agent_velocity, p.agent_bank_angle, p.geophysical_obs) # Update the state with new observations
            push!(bp_particles, s)
        end
        bp_rock = RockObservations(ore_quals=deepcopy(b.rock_obs.ore_quals),
            coordinates=deepcopy(b.rock_obs.coordinates))
        # TODO Make this a little more general in future
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
        bp_rock = deepcopy(b.rock_obs)
        bp_rock.coordinates = hcat(bp_rock.coordinates, [a.coords[1], a.coords[2]])
        push!(bp_rock.ore_quals, o.ore_quality)
        if up.m.geodist_type == GeoStatsDistribution
            bp_geostats = GeoStatsDistribution(b.geostats.grid_dims, deepcopy(bp_rock),
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

    return MEBelief(bp_particles, bp_rock, bp_acts, bp_obs, bp_stopped,
        bp_decided, bp_geostats, up)
end

function Base.rand(rng::AbstractRNG, b::MEBelief)
    return rand(rng, b.particles)
end

Base.rand(b::MEBelief) = rand(Random.GLOBAL_RNG, b)

function summarize(b::MEBelief)
    @info "summarize(b::MEBelief)"
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
    @info "POMDPs.actions(m::MineralExplorationPOMDP, b::MEBelief)"

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
    return MEAction[]
end

function POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)
    @info "POMDPs.actions(m::MineralExplorationPOMDP, b::POMCPOW.StateBelief)"

    o = b.sr_belief.o
    s = rand(m.rng, b.sr_belief.dist)[1]
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
end

function POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)
    @info "POMDPs.actions(m::MineralExplorationPOMDP, o::MEObservation)"

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
    @info "mean_var(b::MEBelief)"

    vars = [s[1] for s in b.particles]
    mean(vars)
end

function std_var(b::MEBelief)
    @info "std_var(b::MEBelief)"

    vars = [s[1] for s in b.particles]
    std(vars)
end

function Plots.plot(b::MEBelief, t=nothing; cmap=:viridis)
    @info "Plots.plot(b::MEBelief, t=nothing; cmap=:viridis)"
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


data_skewness(D) = [skewness(D[x, y, 1:end-1]) for x in 1:size(D, 1), y in 1:size(D, 2)]
data_kurtosis(D) = [kurtosis(D[x, y, 1:end-1]) for x in 1:size(D, 1), y in 1:size(D, 2)]


function convert2data(b::MEBelief)
    @info "convert2data(b::MEBelief)"
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
    @info "get_input_representation(b::MEBelief)"

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
    @info "plot_input_representation(B::Array{<:Real,3)"
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
