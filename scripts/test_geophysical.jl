# Import necessary packages
using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration
using Random

# Constants for the problem setup
NOISE_FOR_PERTURBATION = 2.5
N_PARTICLES = 500
C_EXP = 100.0
SEEDS = get_uncompleted_seeds(baseline=false)

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    c_exp=C_EXP,
    init_pos_x=7.5*25.0 # align with baseline starting in x=8
)

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)

# Define the belief updater for the POMDP with 1000 particles and a spread factor of 2.0
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

# set up the solver
solver = get_geophysical_solver(C_EXP)
planner = POMDPs.solve(solver, m)

println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    

for (i, seed) in enumerate(SEEDS[1:200])
    Random.seed!(seed)
    s0 = rand(ds0)  # Sample a starting state
    s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
    @assert m.dim_scale == 1
    r_massive = sum(s_massive)  # Calculate total massive ore
    println("Massive Ore: $r_massive")
    
    # Simulate a sequence of actions and states
    #h = simulate(hr, m, policy, up, b0, s0)
    v = 0.0  # Initialize the return
    n_fly = 0  # Initialize the fly count
    vols = []

    for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)
        if t % 60 == 0
            println("t: $t")
        end

        if a.type == :fly
            n_fly += 1

        end
        v += POMDPs.discount(m)^(t - 1) * r

        b_vol = mean([calc_massive(m, p) for p in bp.particles])
        push!(vols, b_vol)

        final_belief = bp
        final_state = sp
    end
    
    #println("Steps: $(length(h))")
    println("Decision: $(final_belief.acts[end].type)")
    println("======================")
    
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)
    write_intelligent_result_to_file(m, final_belief, final_state; n_fly=n_fly, reward=v, seed=seed, r_massive=r_massive, which_map=:base)
    GC.gc()
end