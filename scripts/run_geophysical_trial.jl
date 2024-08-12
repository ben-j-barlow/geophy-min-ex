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


SAVE_DIR = "./data/trials/" 


#if isdir(SAVE_DIR)
#    rm(SAVE_DIR; recursive=true)
    #error("directory already exists")
#end 

#mkdir(SAVE_DIR)

#io = MineralExploration.prepare_logger()

m = MineralExplorationPOMDP(
    grid_dim = (50, 50, 1),
    c_exp = 100.0,
    base_grid_element_length = 40.0,
    upscale_factor = 4,
    sigma = 5,
    geophysical_noise_std_dev = 0.01,
    fly_cost = 0.01,
    out_of_bounds_cost = 0.0,
    out_of_bounds_tolerance = 0,
    init_pos_x = 100.0,
    init_pos_y = 0.0,
    init_heading = HEAD_NORTH,
    max_bank_angle = 55,
    bank_angle_intervals = 15,
    velocity = 50,
    extraction_cost = 160.0,
    extraction_lcb = 0.7,
    extraction_ucb = 0.7
)

# do not call
#initialize_data!(m, N_INITIAL)

solver = get_geophysical_solver(m.c_exp)
planner = POMDPs.solve(solver, m)

b = nothing
trees = nothing
final_belief = nothing

ds0 = POMDPs.initialstate(m)

SEED = 42 # convert(Int64, floor(rand() * 10000))
@info "$(SEED)"
Random.seed!(SEED)

for i in 1:N
s0 = rand(ds0)
r_massive = sum(s0.ore_map .>= m.massive_threshold)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=110, output_t=5, return_all_trees=GET_TREES, display_figs=false);

p = plot_base_map_and_plane_trajectory(final_state, m, t=n_flys)

#b_hist, vols, mn, std = plot_volume(m, bp, r_massive; t=t, verbose=false)
#title!(p, "$(final_belief.acts[end].type) at t=$(n_flys+1) with belief $mn ± $std & truth $r_massive")
#savefig(p, "$(SAVE_DIR)trial_$(seed).png")

#p = plot_base_map_and_plane_trajectory(final_state, m, t=n_flys)

#final_belief.acts
#for i in 1:3
#    inbrowser(D3Tree(trees[i], init_expand=1), "safari")
#    @info ""
#end