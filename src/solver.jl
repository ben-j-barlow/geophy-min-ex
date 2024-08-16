@with_kw struct NextActionSampler
    ucb::Float64 = 1.0
end

function sample_ucb_drill(mean, var, idxs)
    #@info "sample_ucb_drill(mean, var, idxs)"
    scores = belief_scores(mean, var)
    weights = Float64[]
    for idx in idxs
        push!(weights, scores[idx])
    end
    coords = sample(idxs, StatsBase.Weights(weights))
    return MEAction(coords=coords)
end

function belief_scores(m, v)
    #@info "belief_scores(m, v)"
    norm_mean = m[:,:,1]./(maximum(m[:,:,1]) - minimum(m[:,:,1]))
    norm_mean .-= minimum(norm_mean)
    s = v[:,:,1]
    norm_std = s./(maximum(s) - minimum(s)) # actualy using variance
    norm_std .-= minimum(norm_std)
    scores = (norm_mean .* norm_std).^2
    # scores = norm_mean .+ norm_std
    # scores .+= 1.0/length(scores)
    scores ./= sum(scores)
    return scores
end

using Infiltrator

function POMCPOW.next_action(o::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::MEBelief, h)
    #@info "POMCPOW.next_action(o::NextActionSampler, pomdp::MineralExplorationPOMDP, b::MEBelief, h)"
    if h isa Vector
        tried_idxs = h
    elseif h.tree isa POMCPOWTree
        tried_idxs = h.tree.tried[h.node]
    elseif h.tree isa MCTS.DPWTree
        tried_idxs = h.tree.children[h.index]
    end
    action_set = POMDPs.actions(pomdp, b)
    if b.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:abandon)
        else
            return MEAction(type=:mine)
        end
    else
        volumes = Float64[]
        for s in b.particles
            v = calc_massive(pomdp, s)
            push!(volumes, v)
        end
        # volumes = Float64[sum(p[2]) for p in b.particles]
        mean_volume = Statistics.mean(volumes)
        volume_std = Statistics.std(volumes)
        lcb = mean_volume - volume_std*pomdp.extraction_lcb
        ucb = mean_volume + volume_std*pomdp.extraction_ucb
        stop_bound = lcb >= pomdp.extraction_cost || ucb <= pomdp.extraction_cost
        if MEAction(type=:stop) ∈ action_set && (length(tried_idxs) <= 0 || length(action_set) == 1) && stop_bound
            return MEAction(type=:stop)
        else
            mean, var = summarize(b)
            coords = [a.coords for a in action_set if a.type == :drill]
            return sample_ucb_drill(mean, var, coords)
        end
    end
end

function POMCPOW.next_action(obj::NextActionSampler, pomdp::MineralExplorationPOMDP,
                            b::POMCPOW.StateBelief, h)
    ##@info "POMCPOW.next_action(obj::NextActionSampler, pomdp::MineralExplorationPOMDP, b::POMCPOW.StateBelief, h)"
    o = b.sr_belief.o
    # s = rand(pomdp.rng, b.sr_belief.dist)[1]
    tried_idxs = h.tree.tried[h.node]
    action_set = POMDPs.actions(pomdp, b)
    if o.stopped
        if length(tried_idxs) == 0
            return MEAction(type=:abandon)
        else
            return MEAction(type=:mine)
        end
    else
        if MEAction(type=:stop) ∈ action_set && (length(tried_idxs) <= 0 || length(action_set) == 1)
            return MEAction(type=:stop)
        else
            ore_maps = Array{Float64, 3}[]
            weights = Float64[]
            for (idx, item) in enumerate(b.sr_belief.dist.items)
                weight = b.sr_belief.dist.cdf[idx]
                state = item[1]
                push!(ore_maps, state.mainbody_map)
                push!(weights, weight)
            end
            weights ./= sum(weights)
            mean = sum(weights.*ore_maps)
            var = sum([weights[i]*(ore_map - mean).^2 for (i, ore_map) in enumerate(ore_maps)])
            coords = [a.coords for a in action_set if a.type == :drill]
            return sample_ucb_drill(mean, var, coords)
        end
    end
end

struct ExpertPolicy <: Policy
    m::MineralExplorationPOMDP
end

POMCPOW.updater(p::ExpertPolicy) = MEBeliefUpdater(p.m, 1)

function POMCPOW.BasicPOMCP.extract_belief(p::MEBeliefUpdater, node::POMCPOW.BeliefNode)
    error("This function hasn't been implemented for geophysical version, returns 0 for bank angle")
    #@info "POMCPOW.BasicPOMCP.extract_belief(p::MEBeliefUpdater, node::POMCPOW.BeliefNode)"
    srb = node.tree.sr_beliefs[node.node]
    cv = srb.dist
    particles = MEState[]
    weights = Float64[]
    state = nothing
    coords = nothing
    stopped = false
    for (idx, item) in enumerate(cv.items)
        weight = cv.cdf[idx]
        state = item[1]
        coords = state.bore_coords
        stopped = state.stopped
        push!(particles, state)
        push!(weights, weight)
    end
    acts = MEAction[]
    obs = MEObservation[]
    for i = 1:size(state.bore_coords)[2]
        a = MEAction(coords=CartesianIndex((state.bore_coords[1, i], state.bore_coords[2, i])))
        ore_qual = state.ore_map[state.bore_coords[1, i], state.bore_coords[2, i], 1]
        o = MEObservation(ore_qual, state.stopped, state.decided, nothing)
        push!(acts, a)
        push!(obs, o)
    end
    return MEBelief(coords, stopped, particles, acts, obs, p, 0)
end

function POMDPs.action(p::ExpertPolicy, b::MEBelief)
    #@info "POMDPs.action(p::ExpertPolicy, b::MEBelief)"
    volumes = Float64[]
    for s in b.particles
        v = calc_massive(p.m, s)
        push!(volumes, v)
    end
    # volumes = Float64[sum(p[2]) for p in b.particles]
    mean_volume = Statistics.mean(volumes)
    volume_var = Statistics.var(volumes)
    volume_std = sqrt(volume_var)
    lcb = mean_volume - volume_std*p.m.extraction_lcb
    ucb = mean_volume + volume_std*p.m.extraction_ucb
    stop_bound = lcb >= p.m.extraction_cost || ucb <= p.m.extraction_cost
    if b.stopped
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    elseif stop_bound
        return MEAction(type=:stop)
    else
        ore_maps = Array{Float64, 3}[s.ore_map for s  in b.particles]
        w = 1.0/length(ore_maps)
        mean = sum(ore_maps)./length(ore_maps)
        var = sum([w*(ore_map - mean).^2 for (i, ore_map) in enumerate(ore_maps)])
        action_set = POMDPs.actions(p.m, b)
        coords = [a.coords for a in action_set if a.type == :drill]
        return sample_ucb_drill(mean, var, coords)
    end
end

mutable struct RandomSolver <: POMDPs.Solver
    rng::AbstractRNG
end

RandomSolver(;rng=Random.GLOBAL_RNG) = RandomSolver(rng)
POMDPs.solve(solver::RandomSolver, problem::Union{POMDP,MDP}) = POMCPOW.RandomPolicy(solver.rng, problem, BeliefUpdaters.PreviousObservationUpdater())

function leaf_estimation(pomdp::MineralExplorationPOMDP, s::MEState, h::POMCPOW.BeliefNode, ::Any)
    #@info "leaf_estimation(pomdp::MineralExplorationPOMDP, s::MEState, h::POMCPOW.BeliefNode, ::Any)"
    if s.stopped
        γ = POMDPs.discount(pomdp)
    else
        if isempty(s.rock_obs.ore_quals)
            bores = 0
        else
            bores = length(s.rock_obs.ore_quals)
        end
        t = pomdp.max_bores - bores + 1
        γ = POMDPs.discount(pomdp)^t
    end
    if s.decided
        return 0.0
    else
        r_extract = extraction_reward(pomdp, s)
        if r_extract >= 0.0
            return γ*r_extract*0.9
        else
            return γ*r_extract*0.1
        end
        # return γ*r_extract
    end
end

function geophysical_leaf_estimation(pomdp::MineralExplorationPOMDP, s::MEState, h::POMCPOW.BeliefNode, ::Any)
    #@info "leaf_estimation(pomdp::MineralExplorationPOMDP, s::MEState, h::POMCPOW.BeliefNode, ::Any)"
    # prepare discount
    if s.stopped
        γ = POMDPs.discount(pomdp)
    else
        if is_empty(s.geophysical_obs)
            n_readings = 0
        else
            if pomdp.observations_per_timestep != 1
                error("assumed one observation per timestep")
            end
            n_readings = length(s.geophysical_obs.reading)
        end
        t = pomdp.max_timesteps - n_readings + 1
        γ = POMDPs.discount(pomdp)^t
    end

    # prepare return value
    if s.decided
        return 0.0
    else
        r_extract = extraction_reward(pomdp, s)
        if r_extract >= 0.0
            if check_plane_within_region(pomdp, last(s.agent_pos_x), last(s.agent_pos_y), 0)
                return γ*r_extract*0.9
            else
                return 0 # penalty for going out of region
            end
        else
            return γ*r_extract*0.1
        end
        # return γ*r_extract
    end
end

struct GridPolicy <: Policy
    m::MineralExplorationPOMDP
    n::Int64 # Number of grid points per dimension (n x n)
    grid_size::Int64 # Size of grid area, centered on map center
    grid_coords::Vector{CartesianIndex{2}}
end

function GridPolicy(m::MineralExplorationPOMDP, n::Int, grid_size::Int)
    #@info "GridPolicy(m::MineralExplorationPOMDP, n::Int, grid_size::Int)"
    grid_start_i = (m.grid_dim[1] - grid_size)/2
    grid_start_j = (m.grid_dim[2] - grid_size)/2
    grid_end_i = grid_start_i + grid_size
    grid_end_j = grid_start_j + grid_size
    grid_i = LinRange(grid_start_i, grid_end_i, n)
    grid_j = LinRange(grid_start_j, grid_end_j, n)

    coords = CartesianIndex{2}[]
    for i=1:n
        for j=1:n
            coord = CartesianIndex(Int(floor(grid_i[i])), Int(floor(grid_j[j])))
            push!(coords, coord)
        end
    end
    return GridPolicy(m, n, grid_size, coords)
end

function POMDPs.action(p::GridPolicy, b::MEBelief)
    #@info "POMDPs.action(p::GridPolicy, b::MEBelief)"
    n_bores = length(b.rock_obs)
    if b.stopped
        volumes = Float64[]
        for s in b.particles
            v = calc_massive(p.m, s)
            push!(volumes, v)
        end
        mean_volume = Statistics.mean(volumes)
        volume_var = Statistics.var(volumes)
        volume_std = sqrt(volume_var)
        lcb = mean_volume - volume_std*p.m.extraction_lcb
        if lcb >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    elseif n_bores >= p.n^2
        return MEAction(type=:stop)
    else
        coords = p.grid_coords[n_bores + 1]
        return MEAction(coords=coords)
    end
end


mutable struct BaselineGeophysicalPolicy <: Policy
    m::MineralExplorationPOMDP
    max_coord::Int64
    smooth_move_size::Int64
    smooth_sidestep_size::Int64
    init_coords::CartesianIndex{2}
    early_stop::Bool
    moves::Vector{MEAction}  # Field to store all the moves
    current_move::Int64  # Field to track the current move
    grid::Bool

    # Inner constructor
    function BaselineGeophysicalPolicy(m::MineralExplorationPOMDP, max_coord::Int64, smooth_move_size::Int64, smooth_sidestep_size::Int64, init_coords::CartesianIndex{2}, early_stop::Bool, grid::Bool)
        instance = new(m, max_coord, smooth_move_size, smooth_sidestep_size, init_coords, early_stop, Vector{MEAction}(), 1, grid)
        instance.moves = get_all_moves(instance)
        return instance
    end
end

function POMDPs.action(p::BaselineGeophysicalPolicy, b::MEBelief)
    #@info "POMDPs.action(p::GridPolicy, b::MEBelief)"
    if b.stopped
        vols = [calc_massive(p.m, s) for s in b.particles]
        mean_vols = round(mean(vols), digits=2)
        if mean_vols >= p.m.extraction_cost
            return MEAction(type=:mine)
        else
            return MEAction(type=:abandon)
        end
    end

    if p.early_stop && calculate_stop_bound(p.m, b)
        return MEAction(type=:stop)
    end

    # Return the next action from the moves list
    action = deepcopy(p.moves[p.current_move])
    p.current_move += 1  # Increment the counter
    return action
end

function get_next_baseline_coord(p::BaselineGeophysicalPolicy, b::MEBelief)
    if length(b.acts) == 0  
        # first move: return init coordinates
        return p.init_coords
    end
    init_x = p.init_coords[2]
    return calculate_move(init_x, p.max_coord, p.smooth_move_size, p.smooth_sidestep_size, length(b.acts))
end

function get_all_moves(p::BaselineGeophysicalPolicy)
    init_x = p.init_coords[2]
    
    vertical_moves = [p.init_coords]
    num_actions = 1
    new_x = 1
    
    while new_x < p.max_coord
        # Determine the number of steps in each column
        moves_in_column = floor(Int, p.max_coord / p.smooth_move_size)

        # Determine how many complete columns we've finished
        completed_columns = div(num_actions, moves_in_column)

        # Determine the position within the current column
        steps_in_current_column = mod(num_actions, moves_in_column)

        # Determine the direction of travel: north for odd columns, south for even columns
        head_north = mod(completed_columns, 2) == 0
        
        # Calculate the new x position after sidesteps
        new_x = init_x + completed_columns * p.smooth_sidestep_size
        
        # Calculate new _y
        new_y = head_north ? (1 + steps_in_current_column * p.smooth_move_size) : (p.max_coord - steps_in_current_column * p.smooth_move_size)
        push!(vertical_moves, CartesianIndex(new_y, new_x))

        num_actions += 1
    end
    
    horizontal_moves = [CartesianIndex(m[2], m[1]) for m in vertical_moves]

    vertical_lines = MEAction[MEAction(type=:fake_fly, coords=coord) for coord in vertical_moves]

    if p.grid
        horizontal_lines = MEAction[MEAction(type=:fake_fly, coords=coord) for coord in horizontal_moves]
        all_lines = vcat(vertical_lines, horizontal_lines)
    else
        all_lines = vertical_lines
    end
    # concat the two lists and add a stop action
    return vcat(all_lines, MEAction(type=:stop))
end