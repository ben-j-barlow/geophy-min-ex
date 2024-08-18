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
using D3Trees

# Constants for the problem setup
NOISE_FOR_PERTURBATION = 0.7
N_PARTICLES = 1000
C_EXP = 125.0
SEEDS = get_uncompleted_seeds(baseline=false)

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    # DO NOT CHANGE - same as baseline
    upscale_factor=4,
    sigma=3,
    geophysical_noise_std_dev=0.02,
    ### 
    grid_dim=(48,48,1),
    c_exp=C_EXP,
    init_pos_x=7.5*25.0, # align with baseline starting in x=8
    init_pos_y=0.0,
    init_heading=HEAD_NORTH,
    max_bank_angle=55,
    max_timesteps=150,
    massive_threshold=0.7,
    out_of_bounds_cost=0.1,
    out_of_bounds_tolerance=4.0,
    fly_cost=0.01,
    velocity=60,
    min_readings=30,
    bank_angle_intervals=18,
    timestep_in_seconds=1,
    observations_per_timestep=1,
    extraction_cost=150.0,
    extraction_lcb=0.8,
    extraction_ucb=0.8
)


# set up the solver
solver = get_geophysical_solver(C_EXP, true)
planner = POMDPs.solve(solver, m)

println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    
trees = []

for (i, seed) in enumerate(SEEDS[1:10])
    Random.seed!(seed)
    println(seed)

    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0)  # Sample a starting state
    up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
    b0 = POMDPs.initialize_belief(up, ds0)

    s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
    @assert m.dim_scale == 1
    r_massive = sum(s_massive)  # Calculate total massive ore
    println("Massive Ore: $r_massive")
    
    # Simulate a sequence of actions and states
    #h = simulate(hr, m, policy, up, b0, s0)
    v = 0.0  # Initialize the return
    n_fly = 0  # Initialize the fly count
    mns = []
    stds = []
    #a, ai = POMCPOW.action_info(planner, b0, tree_in_info=true)
    #inbrowser(D3Tree(ai[:tree], init_expand=1), "safari")

    for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)
        push!(trees, deepcopy(planner.tree))
        if !check_plane_within_region(m, last(sp.agent_pos_x), last(sp.agent_pos_y), 0)
            inbrowser(D3Tree(trees[t-2], init_expand=1), "safari")
            inbrowser(D3Tree(trees[t-1], init_expand=1), "safari")
            break;
        end
        
        if a.type == :fly
            n_fly += 1

        end
        v += POMDPs.discount(m)^(t - 1) * r

        if t % 10 == 2
            println("Mean Ore: t $t $(last(mns)) ± $(last(stds))")
            p = plot_smooth_map_and_plane_trajectory(sp, m; t=t)
            display(p)
        end

        b_vol = [calc_massive(m, p) for p in bp.particles]
        push!(mns, mean(b_vol))
        push!(stds, Statistics.std(b_vol))

        final_belief = bp
        final_state = sp
    end
    
    #println("Steps: $(length(h))")
    println("Mean Ore: $(last(mns)) ± $(last(stds))")
    println("Decision: $(final_belief.acts[end].type)")
    println("======================")
    
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)
    write_intelligent_result_to_file(m, final_belief, final_state, mns, stds; n_fly=n_fly, reward=v, seed=seed, r_massive=r_massive, which_map=:base)
    GC.gc()
end

