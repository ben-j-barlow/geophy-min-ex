N_PROCS = 2

using Distributed
addprocs(N_PROCS)

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using POMDPModelTools
using ParticleFilters
using Random
using JLD
using MineralExploration

@everywhere using MineralExploration
@everywhere using POMDPs
@everywhere using POMDPSimulators
@everywhere using POMCPOW
@everywhere using ParticleFilters

# define constants
const NSIM = 2
const MAX_STEPS = 5
const C_EXP = 2
N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

# define starting configs
START_X = [30.0, 30.0, 25.1*40.1, 25.1*25.1]
START_Y = [30.0, 25.1*50.1-30, 25.1*40.1, 25,1*25.1]
INIT_HEADING = [HEAD_NORTH, HEAD_SOUTH, HEAD_EAST, HEAD_WEST]

# prepare POMCPOW
GeophysicalPOMDPSolver = POMCPOWSolver(
    tree_queries=4000,
    k_observation=2.0,
    alpha_observation=0.1,
    max_depth=5,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    criterion=POMCPOW.MaxUCB(C_EXP),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=leaf_estimation,
    tree_in_info=false,
)

# prepare for simulations
m = [MineralExplorationPOMDP(init_heading=INIT_HEADING[i],init_pos_x=START_X[i],init_pos_y=START_Y[i]) for i in 1:NSIM]
up = [MEBeliefUpdater(m[i], N_PARTICLES, NOISE_FOR_PERTURBATION) for i in 1:NSIM]
ds0 = [POMDPs.initialstate(m[i]) for i in 1:NSIM]
b0 = [POMDPs.initialize_belief(up[i], ds0[i]) for i in 1:NSIM]
Random.seed!(42)
s0 = [rand(ds0[i]) for i in 1:NSIM]
planner = [POMDPs.solve(GeophysicalPOMDPSolver, m[i]) for i in 1:NSIM]

# run simulations
queue = POMDPSimulators.Sim[]
for i = 1:NSIM
    r_massive = sum(s0[i].ore_map[:,:,1] .>= m[i].massive_threshold)
    push!(queue, POMDPSimulators.Sim(
        m[i], 
        planner[i], 
        up[i], 
        b0[i], 
        s0[i], 
        metadata=Dict(:massive_ore=>r_massive),
        max_steps=MAX_STEPS))
end

println("Starting Simulations...")
data = POMDPSimulators.run_parallel(queue, show_progress=true)
println("Simulations Complete!")
JLD.save("./data/POMCPOW_test_4.jld", "results", data)
# extracting speicifc data 
# https://juliapomdp.github.io/POMDPs.jl/stable/POMDPTools/simulators/#Specifying-information-to-be-recordedp
data[!, :reward]
data[!, :massive_ore]