using Revise
import BSON: @save, @load
using POMDPs
using POMCPOW
using Plots;
default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
using Statistics
using Images
using Random
using Infiltrator
using MineralExploration
using POMDPPolicies

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0
C_EXP = 2

grid_dims = (50, 50, 1)
m = MineralExplorationPOMDP(
    grid_dim=grid_dims,
    c_exp=C_EXP,
    mainbody_gen=BlobNode(grid_dims=grid_dims),
    true_mainbody_gen=BlobNode(grid_dims=grid_dims),
    timestep_in_seconds=2,
    observations_per_timestep=1,
    velocity=50,
    geophysical_noise_std_dev=0
)

ds0 = POMDPs.initialstate(m)

Random.seed!(52992)
s0 = rand(ds0; truth=true) #Checked
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION); #Checked
b0 = POMDPs.initialize_belief(up, ds0); #Checked


solver = POMCPOWSolver(
    tree_queries=4000,
    max_depth=6,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    k_observation=2.0,
    alpha_observation=0.1,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=leaf_estimation,
    tree_in_info=false,
)
planner = POMDPs.solve(solver, m)
discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=30, save_dir="./data/sandbox/tmp");

#using D3Trees
#tree_1 = trees[1]
#inbrowser(D3Tree(tree_1, init_expand=1), "safari")


# plot normal map with custom title
plot_map(s0.smooth_map, "geophysical map")
plot_map(s0.ore_map, "ore map")
plot_map(final_state.smooth_map, "geophysical map")
plot_map(final_state.ore_map, "ore map")

# plot map and plane trajectory
plot_smooth_map_and_plane_trajectory(final_state, m)

# plot map of observation locations
plot_base_map_at_observation_locations(final_state)
plot_smooth_map_at_observation_locations(final_state)

fig1, fig2 = plot(final_belief, m)
display(fig1)
display(fig2)

# test x and y
tmp_map = deepcopy(s0.ore_map)
tmp_map[20, 1, 1] = NaN
plot_map(tmp_map, "x = 20, y = 1 is empty")