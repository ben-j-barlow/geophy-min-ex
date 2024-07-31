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
MAX_BORES = 2
MIN_BORES = 2
GRID_SPACING = 0
MAX_MOVEMENT = 20

C_EXP = 2
SAVE_DIR = "./data/sandbox/w3_$C_EXP" #instead of +string(variable) OR $(var1+var2)

!isdir(SAVE_DIR) && mkdir(SAVE_DIR)

io = MineralExploration.prepare_logger()

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
tree_queries = [3, 5, 10, 100, 1_000, 10_000]
i_tree_queries = 5

usepomcpow = true
if usepomcpow == true
    solver = POMCPOWSolver(tree_queries=tree_queries[i_tree_queries],
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        enable_action_pw=m.mineral_exploration_mode == "borehole" ? true : false,
                        next_action=m.mineral_exploration_mode == "borehole" ? next_action : nothing,
                        k_action=m.mineral_exploration_mode == "borehole" ? 2.0 : nothing,
                        alpha_action=m.mineral_exploration_mode == "borehole" ? 0.25 : nothing,
                        k_observation=2.0,
                        alpha_observation=0.1,
                        criterion=POMCPOW.MaxUCB(100.0),
                        final_criterion=POMCPOW.MaxQ(),
                        # final_criterion=POMCPOW.MaxTries(),
                        max_depth=5,
                        estimate_value=leaf_estimation, # or 0.0
                        tree_in_info=true,
                        )
    planner = POMDPs.solve(solver, m)
else
    #planner = FunctionPolicy(b->MEAction(coords=CartesianIndex(10,10)))
    planner = RandomPolicy(m, updater = up)
end

timing = @timed run_trial(m, up, planner, s0, b0, save_dir=SAVE_DIR, display_figs=false, return_all_trees=true, verbose=false)

MineralExploration.close_logger(io)

@show timing.time