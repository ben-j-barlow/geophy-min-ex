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
NOISE_FOR_PERTURBATION = 0.8
N_PARTICLES = 1000
C_EXP = 125.0
GET_TREES = false
SEEDS = get_uncompleted_seeds(baseline=false)

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    # DO NOT CHANGE - same as baseline
    upscale_factor=4,
    sigma=2,
    geophysical_noise_std_dev=0.005,
    ### 
    grid_dim=(48,48,1),
    c_exp=C_EXP,
    init_pos_x=7.5*25.0, # align with baseline starting in x=8
    init_pos_y=0.0,
    init_heading=(HEAD_NORTH + HEAD_EAST) / 2,
    max_bank_angle=55,
    max_timesteps=250,
    massive_threshold=0.7,
    out_of_bounds_cost=0, # greater than -10 so 0.1*extraction_reward() is a smaller negative
    out_of_bounds_tolerance=0.0,
    fly_cost=0.01,
    velocity=40,
    min_readings=100,
    bank_angle_intervals=18,
    timestep_in_seconds=1,
    observations_per_timestep=1,
    extraction_cost=150.0,
    extraction_lcb=0.7,
    extraction_ucb=0.7
)


# set up the solver
#solver = get_geophysical_solver(C_EXP, false)
solver = POMCPOWSolver(
    tree_queries=15000,
    k_observation=2.0,
    alpha_observation=0.3,
    max_depth=5,
    check_repeat_obs=false,
    check_repeat_act=true,
    enable_action_pw=false,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=geophysical_leaf_estimation,
    #estimate_value=0.0,
    tree_in_info=GET_TREES,
)
planner = POMDPs.solve(solver, m)

println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    
trees = []

for (i, seed) in enumerate(SEEDS[1:61])
    Random.seed!(seed)
    println("$seed & $i")

    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0)  # Sample a starting state
    up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
    b0 = POMDPs.initialize_belief(up, ds0)

    s_massive = s0.mainbody_map .>= m.massive_threshold  # Identify massive ore locations
    @assert m.dim_scale == 1
    r_massive = sum(s_massive)  # Calculate total massive ore
    println("Massive Ore: $r_massive")
    
    # Simulate a sequence of actions and states
    #h = simulate(hr, m, policy, up, b0, s0)
    v = 0.0  # Initialize the return
    n_fly = 0  # Initialize the fly count
    mns = []
    stds = []
    in_region = true
    
    #a, ai = POMCPOW.action_info(planner, b0, tree_in_info=true)
    #inbrowser(D3Tree(ai[:tree], init_expand=1), "safari")

    for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)        
        #if in_region && !check_plane_within_region(m, last(sp.agent_pos_x), last(sp.agent_pos_y), 0)
        #    in_region = false
        #    a, ai = POMCPOW.action_info(planner, beliefs[end], tree_in_info=true)
        #    inbrowser(D3Tree(ai[:tree], init_expand=1), "safari")
        #end
        if a.type == :fly
            n_fly += 1

        end
        v += POMDPs.discount(m)^(t - 1) * r

        #if t % 10 == 0
        #    println("Mean Ore: t $t $(last(mns)) ± $(last(stds))")
        #end

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
    write_intelligent_result_to_file(m, final_belief, final_state, mns, stds; n_fly=n_fly, reward=v, seed=seed, r_massive=r_massive, which_map=:smooth)
    GC.gc()
end
