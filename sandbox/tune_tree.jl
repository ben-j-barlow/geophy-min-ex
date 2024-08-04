using Revise

using POMDPs
using POMCPOW
using Plots
using POMDPModelTools
using Random

using MineralExploration

# key parameters are
# C_EXP: exploration constant
# alpha
# k


C_EXP = 100

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

grid_dims = (50,50,1)

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
    init_pos_x=15*25,
    init_pos_y=15*25,
    bank_angle_intervals=10,
    max_bank_angle=55,
    velocity=25,
    base_grid_element_length=25.0
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0) #Checked
Random.seed!(42)
s0 = rand(ds0);

# prepare POMCPOW
solver = POMCPOWSolver(
    tree_queries=10000,
    k_observation=2,
    alpha_observation=0.05,
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
    tree_in_info=false,
)
planner = POMDPs.solve(solver, m)

#@info "ground truth s.smooth_map[80,75,1] is $(s0.smooth_map[80,75,1])"
#a, ai = action_info(planner, b0);


#using D3Trees
#inbrowser(D3Tree(planner.tree, init_expand=1), "safari")

discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=80);
