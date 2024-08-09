using Revise

import BSON: @save, @load
using POMDPs
using POMCPOW
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using Statistics
using Images
using Random
using Infiltrator
using MineralExploration
using POMDPPolicies

#rng = MersenneTwister(3)
Random.seed!(1004) # determinism

N_INITIAL = 0
MAX_BORES = 10
MIN_BORES = 2
GRID_SPACING = 0
MAX_MOVEMENT = 20

C_EXP = 2
SAVE_DIR = "./data/moss_mfms" #instead of +string(variable) OR $(var1+var2)

!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

grid_dims = (50, 50, 1)
true_mainbody = BlobNode(grid_dims=grid_dims, factor=4)
mainbody = BlobNode(grid_dims=grid_dims)


m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            true_mainbody_gen=true_mainbody, mainbody_gen=mainbody, original_max_movement=MAX_MOVEMENT,
                            min_bores=MIN_BORES, grid_dim=grid_dims, c_exp=C_EXP)
initialize_data!(m, N_INITIAL)
@show m.max_movement

ds0 = POMDPs.initialstate(m)

s0 = rand(ds0; truth=true) #Checked

up = MEBeliefUpdater(m, 1000, 2.0) #Checked
b0 = POMDPs.initialize_belief(up, ds0) #Checked

next_action = NextActionSampler()
tree_queries = [5, 1_000, 10_000]
i_tree_queries = 1

usepomcpow = true
if usepomcpow == true
    solver = POMCPOWSolver(tree_queries=100,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        next_action=next_action,
                        k_action=2.0,
                        alpha_action=0.25,
                        k_observation=2.0,
                        alpha_observation=0.1,
                        criterion=POMCPOW.MaxUCB(100.0),
                        final_criterion=POMCPOW.MaxQ(),
                        # final_criterion=POMCPOW.MaxTries(),
                        estimate_value=0.0,
                        max_depth=3,
                        # estimate_value=leaf_estimation,
                        )
    planner = POMDPs.solve(solver, m)
else
    #planner = FunctionPolicy(b->MEAction(coords=CartesianIndex(10,10)))
    planner = RandomPolicy(m, updater = up)
end

timing = @timed results = run_trial(m, up, planner, s0, b0, save_dir=SAVE_DIR, display_figs=false, return_all_trees=true)
@show timing.time


beliefs = results[9];
final_belief = beliefs[end];

final_belief.acts

beliefs[1].stopped
beliefs[1].decided

beliefs[2].stopped
beliefs[2].decided

beliefs[3].stopped
beliefs[3].decided

beliefs[4].stopped
beliefs[4].decided

using D3Trees
trees = results[10];
inbrowser(D3Tree(trees[4], init_expand=1), "safari")