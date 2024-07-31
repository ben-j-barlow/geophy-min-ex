function POMDPs.gen(m::MineralExplorationPOMDP, s::MEState, a::MEAction, rng::Random.AbstractRNG)
    if a ∉ POMDPs.actions(m, s)
        error("Invalid Action $a from state $s")
    end
    stopped = s.stopped
    decided = s.decided
    a_type = a.type

    # drill then stop then mine or abandon
    if a_type == :stop && !stopped && !decided
        obs = MEObservation(nothing, true, false)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = false
    elseif a_type == :abandon && stopped && !decided
        obs = MEObservation(nothing, true, true)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type == :mine && stopped && !decided
        obs = MEObservation(nothing, true, true)
        rock_obs_p = s.rock_obs
        stopped_p = true
        decided_p = true
    elseif a_type == :drill && !stopped && !decided
        ore_obs = high_fidelity_obs(m, s.ore_map, a) # Obtain high fidelity observation for the action
        a_coords = reshape(Int64[a.coords[1] a.coords[2]], 2, 1) # Reshape the action coordinates
        rock_obs_p = deepcopy(s.rock_obs) # Create a deep copy of current rock observations
        rock_obs_p.coordinates = hcat(rock_obs_p.coordinates, a_coords) # Concatenate new coordinates to rock observations
        push!(rock_obs_p.ore_quals, ore_obs) # Add the new ore quality observation
        n_bores = length(rock_obs_p) # Count the number of boreholes
        stopped_p = n_bores >= m.max_bores # Check if the maximum number of boreholes is reached
        decided_p = false # Set decided to false since drilling decision is not final
        obs = MEObservation(ore_obs, stopped_p, false) # Create a new observation with current ore quality and status
    else
        error("Invalid Action! Action: $(a.type), Stopped: $stopped, Decided: $decided")
    end
    r = reward(m, s, a)
    sp = MEState(s.ore_map, s.mainbody_params, s.mainbody_map, rock_obs_p, stopped_p, decided_p)
    return (sp=sp, o=obs, r=r)
end
