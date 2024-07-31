using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using POMDPModelTools
using ParticleFilters
using Statistics

using MineralExploration

C_EXP = 2

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

m = MineralExplorationPOMDP(
    c_exp=C_EXP,
    sigma=20,
    init_heading=45,
    out_of_bounds_cost=0,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1
)

up = MEBeliefUpdater(m, g, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked

ds0 = POMDPs.initialstate(m)

b0 = POMDPs.initialize_belief(up, ds0) #Checked

display(plot(b0))

# define max number of timesteps here
hr = HistoryRecorder(max_steps=2)

# prepare POMCPOW
solver = POMCPOWSolver(
    tree_queries=3,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    k_observation=2.0,
    alpha_observation=0.1,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    max_depth=2,
    estimate_value=leaf_estimation,
    tree_in_info=false,
)
planner = POMDPs.solve(solver, m)

# for i in 1:n
s0 = rand(ds0)

a, ai = action_info(planner, b0)

out = gen(m, s0, a, m.rng)
outnt = NamedTuple{(:sp,:o,:r,:info)}(out)

x = out[:o].geophysical_obs.smooth_map_coordinates[1]
y = out[:o].geophysical_obs.smooth_map_coordinates[2]

if m.init_heading == 45 && m.init_pos_x == 0 && m.init_pos_y == 0
    if a.change_in_bank_angle == -5
        @assert x > y
    elseif a.change_in_bank_angle == 0
        @assert x == y
    elseif a.change_in_bank_angle == 5
        @assert x < y
    else
        error("change_in_bank_angle must be -5, 0, or 5 for testing")
    end
else
    error("plane dynamics not permitted for testing")
end

h = simulate(hr, m, planner, up, b0, s0)
n_flys = 0

tmp_stp = nothing
returns = 0.0

for stp in h
    returns += POMDPs.discount(m)^(stp[:t] - 1)*stp[:r]
    if stp[:a].type == :fly
        n_flys += 1
    end
    tmp_stp = stp
    act = last(stp[:o].geophysical_obs.reading)

end

x = tmp_stp[:o].geophysical_obs.smooth_map_coordinates[1]
y = tmp_stp[:o].geophysical_obs.smooth_map_coordinates[2]
exp = s0.smooth_map[x, y, 1]
act = tmp_stp[:o].geophysical_obs.reading[1]

found = any(x -> x == act, s0.smooth_map)

mooth map coordinates 6, 9
[ Info: returning noiseless obs 0.24855268705255312

[ Info: region check ... x & y on map: true; pos x 26.266099440886485 and pos y 42.54517622670592
[ Info: smooth map coordinates 6, 9
[ Info: returning noiseless obs 0.3223548511944943

[ Info: region check ... x & y on map: true; pos x 26.266099440886485 and pos y 42.54517622670592
[ Info: smooth map coordinates 6, 9
[ Info: returning noiseless obs 0.24855268705255312

[ Info: region check ... x & y on map: true; pos x 26.266099440886485 and pos y 42.54517622670592
[ Info: smooth map coordinates 6, 9
[ Info: returning noiseless obs 0.18626342280485725

tmp_stp[:sp].geophysical_obs.smooth_map_coordinates
keys(tmp_stp)

tmp_stp[:o].geophysical_obs
tmp_stp[:action_info][:tree_queries]
tmp_stp[:action_info][:search_time]
