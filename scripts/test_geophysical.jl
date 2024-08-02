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
    init_heading=convert(Float64, 0),
    init_pos_x=0,
    init_pos_y=0,
    velocity=40,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=5
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked

ds0 = POMDPs.initialstate(m)

b0 = POMDPs.initialize_belief(up, ds0) #Checked


# define max number of timesteps here
hr = HistoryRecorder(max_steps=2)

# prepare POMCPOW
solver = POMCPOWSolver(
    tree_queries=2000,
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
    max_depth=5,
    estimate_value=leaf_estimation,
    tree_in_info=false,
)
planner = POMDPs.solve(solver, m)

# for i in 1:n
s0 = rand(ds0);

plot_map(s0.ore_map, "ore map")

a, ai = action_info(planner, b0);

#using D3Trees
#inbrowser(D3Tree(planner.tree, init_expand=1), "safari")

@assert a.type == :fly
a.change_in_bank_angle

s = deepcopy(s0);
@assert s.agent_pos_x[1] == 0
@assert s.agent_pos_y[1] == 0
@assert s.agent_heading == 0
@assert s.agent_bank_angle[1] == 0
@assert length(s.agent_pos_x) == 1
@assert length(s.agent_pos_y) == 1
@assert length(s.agent_bank_angle) == 1

out = gen(m, s0, a, m.rng);

o = out[:o]
o.agent_pos_x
o.agent_pos_y
bp, ui = update_info(up, b0, a, outnt.o)

#if isterminal(it.pomdp, is[2]) || is[1] > it.max_steps 
#    return nothing 
#end 
#t = is[1]
#s = is[2]
#b = is[3]
#a, ai = action_info(it.policy, b)
#out = @gen(:sp,:o,:r,:info)(it.pomdp, s, a, it.rng)
#outnt = NamedTuple{(:sp,:o,:r,:info)}(out)
#bp, ui = update_info(it.updater, b, a, outnt.o)
#nt = merge(outnt, (t=t, b=b, s=s, a=a, action_info=ai, bp=bp, update_info=ui))
#return (out_tuple(it, nt), (t+1, nt.sp, nt.bp))


x = out[:o].geophysical_obs.smooth_map_coordinates[1]
y = out[:o].geophysical_obs.smooth_map_coordinates[2]

h = simulate(hr, m, planner, up, b0, s0) #ds0 instead of b0?
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
