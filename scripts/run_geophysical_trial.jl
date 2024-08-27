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

NOISE_FOR_PERTURBATION = 0.85
N_PARTICLES = 1000
C_EXP = 125.0

GET_TREES = true


SAVE_DIR = "./data/trials/" 

#if isdir(SAVE_DIR)
#    rm(SAVE_DIR; recursive=true)
    #error("directory already exists")
#end 

#mkdir(SAVE_DIR)

#io = MineralExploration.prepare_logger()

m = MineralExplorationPOMDP(
    # DO NOT CHANGE - same as baseline
    upscale_factor=4,
    sigma=3,
    geophysical_noise_std_dev=0.02,
    ### 
    grid_dim=(48,48,1),
    c_exp=C_EXP,
    init_heading=HEAD_NORTH,
    max_bank_angle=55,
    max_timesteps=150,
    massive_threshold=0.7,
    out_of_bounds_cost=10.0,
    out_of_bounds_tolerance=4.0,
    fly_cost=0.01,
    velocity=60,
    min_readings=30,
    bank_angle_intervals=18,
    timestep_in_seconds=1,
    observations_per_timestep=1,
    extraction_cost=150.0,
    extraction_lcb=0.8,
    extraction_ucb=0.8,
    init_pos_y=10.0,
    init_pos_x=20.5*25.0,
)


# do not call
#initialize_data!(m, N_INITIAL)

solver = get_geophysical_solver(m.c_exp)
planner = POMDPs.solve(solver, m)

b = nothing
trees = nothing
final_belief = nothing


#SEED = 42 # convert(Int64, floor(rand() * 10000))
#@info "$(SEED)"
#Random.seed!(SEED)

seeds = get_all_seeds()

for seed in seeds[10]
    seed = 4136
    println("seed $seed")
    Random.seed!(seed)
    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0)
    r_massive = sum(s0.ore_map .>= m.massive_threshold)
    up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
    b0 = POMDPs.initialize_belief(up, ds0)

    discounted_return, n_flys, final_belief, final_state, trees = run_geophysical_trial(m, up, planner, s0, b0, max_t=100, output_t=5, save_dir=SAVE_DIR, return_all_trees=GET_TREES, display_figs=true);
end

inbrowser(D3Tree(trees[1], init_expand=1), "safari")
inbrowser(D3Tree(trees[6], init_expand=1), "safari")