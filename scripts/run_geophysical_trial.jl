using D3Trees
using Revise
import BSON: @save, @load
using POMDPs
using POMCPOW
using Plots;default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
using Statistics
using Images
using Random
using Infiltrator
using MineralExploration
using POMDPPolicies

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 0.1
# noise
# 10 - belief didnt really converge
C_EXP = 125
GET_TREES = true

SAVE_DIR = "./data/geophys_trial" #instead of +string(variable) OR $(var1+var2)
!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

#io = MineralExploration.prepare_logger()

grid_dims = (50, 50, 1)

m = MineralExplorationPOMDP(
    grid_dim=grid_dims,
    c_exp=C_EXP,
    sigma=5,
    init_heading=HEAD_NORTH,
    mainbody_gen=BlobNode(grid_dims=grid_dims),
    true_mainbody_gen=BlobNode(grid_dims=grid_dims),
    geophysical_noise_std_dev=0.0,
    observations_per_timestep=1,
    timestep_in_seconds=1,
    init_pos_x=30 * 25,
    init_pos_y=0,
    bank_angle_intervals=15,
    max_bank_angle=55,
    velocity=25,
    base_grid_element_length=25.0
)

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

Random.seed!(42)
s0 = rand(ds0; truth=true) #Checked

# get min amd max value of ore map
min_ore = minimum(s0.ore_map)
max_ore = maximum(s0.ore_map)
minimum(s0.smooth_map)
maximum(s0.smooth_map)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
b0 = POMDPs.initialize_belief(up, ds0) #Checked

solver = POMCPOWSolver(
    tree_queries=50000,
    k_observation=2.0,
    alpha_observation=0.1,
    max_depth=7,
    check_repeat_obs=false,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=leaf_estimation,
    tree_in_info=GET_TREES,
)
planner = POMDPs.solve(solver, m)

discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=3, output_t=1, return_all_trees=GET_TREES);

plot_base_map_and_plane_trajectory(final_state, m)

for i in 1:3
    inbrowser(D3Tree(trees[i], init_expand=1), "safari")
    @info ""
end
