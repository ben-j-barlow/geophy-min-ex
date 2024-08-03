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
NOISE_FOR_PERTURBATION = 2.0
C_EXP = 2

SAVE_DIR = "./data/sandbox/tmp" #instead of +string(variable) OR $(var1+var2)
!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

#io = MineralExploration.prepare_logger()

grid_dims = (50, 50, 1)

m = MineralExplorationPOMDP(
    grid_dim=grid_dims,
    c_exp=C_EXP,
    sigma=20,
    init_heading=HEAD_NORTH,
    out_of_bounds_cost=1000,
    mainbody_gen=BlobNode(grid_dims=grid_dims),
    true_mainbody_gen=BlobNode(grid_dims=grid_dims),
    geophysical_noise_std_dev=0.0,
    observations_per_timestep=1,
    timestep_in_seconds=1,
    init_pos_x=300,
    init_pos_y=400,
    bank_angle_intervals=10,
    max_bank_angle=55,
    velocity=25,
    base_grid_element_length=25.0
)

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

Random.seed!(42)
s0 = rand(ds0; truth=true) #Checked

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
b0 = POMDPs.initialize_belief(up, ds0) #Checked

tree_queries = [3, 5, 10, 100, 1_000, 2000, 5000, 10_000]
i_tree_queries = 6

solver = POMCPOWSolver(
    tree_queries=4000,
    k_observation=2.0,
    alpha_observation=0.1,
    max_depth=5,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=leaf_estimation,
    tree_in_info=true,
)
planner = POMDPs.solve(solver, m)

discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=100);

#plot_smooth_map_and_plane_trajectory(final_state, m)


#plot(final_belief)
#MineralExploration.close_logger(io)

#minimum(s0.smooth_map[:, :, 1])

# smooth map coordinates 7, 10 noiseless geo obs 0.26859071353819397 

#using D3Trees
#tree_1 = trees[1]
#inbrowser(D3Tree(tree_1, init_expand=1), "safari")
