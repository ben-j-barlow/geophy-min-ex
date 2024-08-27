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
GET_TREES = false

SEED = 42

SAVE_DIR = "./data/multiple_trial_until_termination/" 


if isdir(SAVE_DIR)
    rm(SAVE_DIR; recursive=true)
    #error("directory already exists")
end 

mkdir(SAVE_DIR)

#io = MineralExploration.prepare_logger()

grid_dims = (50, 50, 1)

m = MineralExplorationPOMDP(
    grid_dim=grid_dims,
    upscale_factor=3,
    c_exp=C_EXP,
    sigma=5,
    mainbody_gen=BlobNode(grid_dims=grid_dims),
    true_mainbody_gen=BlobNode(grid_dims=grid_dims),
    geophysical_noise_std_dev=0.0,
    observations_per_timestep=1,
    timestep_in_seconds=1,
    init_pos_x=2.0,
    init_pos_y=-5.0,
    init_heading=HEAD_NORTH,
    bank_angle_intervals=15,
    max_bank_angle=55,
    velocity=25,
    base_grid_element_length=20.0,
    extraction_cost=150.0,
    extraction_lcb = 0.7,
    extraction_ucb = 0.7,
    out_of_bounds_cost=0.0,
    max_timesteps=100
)

# do not call
#initialize_data!(m, N_INITIAL)

solver = POMCPOWSolver(
    tree_queries=10000,
    k_observation=2.0,
    alpha_observation=0.3,
    max_depth=3,
    check_repeat_obs=false,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=geophysical_leaf_estimation,
    tree_in_info=GET_TREES,
)
planner = POMDPs.solve(solver, m)

b = nothing
trees = nothing
final_belief = nothing

SEEDS = [convert(Int64, floor(rand() * 10000)) for _ in 1:5]

for seed in SEEDS
    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0)
    r_massive = sum(s0.ore_map .>= m.massive_threshold)
    up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
    b0 = POMDPs.initialize_belief(up, ds0)

    discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=110, output_t=10, return_all_trees=GET_TREES, display_figs=false);
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)

    p = plot_base_map_and_plane_trajectory(final_state, m, t=n_flys)
    
    title!(p, "$(final_belief.acts[end].type) at t=$(n_flys+1) with belief $mn & truth $r_massive")
    savefig(p, "$(SAVE_DIR)trial_$(seed).png")
end

#final_belief.acts
#for i in 1:3
#    inbrowser(D3Tree(trees[i], init_expand=1), "safari")
#    @info ""
#end