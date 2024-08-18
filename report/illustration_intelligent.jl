using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration
using Random

save_dir = "./data/report/illustration/intelligent/"

# Constants for the problem setup
NOISE_FOR_PERTURBATION = 0.75
N_PARTICLES = 1000
C_EXP = 100.0
SEEDS = get_uncompleted_seeds(baseline=false)

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    grid_dim=(48, 48, 1),
    geophysical_noise_std_dev=0.0,
    upscale_factor=4,
    sigma=5.0,
    c_exp=C_EXP,
    base_grid_element_length=25.0,
    init_pos_x=7.5 * 25.0, # align with baseline starting in x=8
    init_pos_y=0.0,
    init_heading=HEAD_NORTH,
    max_bank_angle=55,
    max_timesteps=120,
    massive_threshold=0.7,
    out_of_bounds_cost=0.0,
    out_of_bounds_tolerance=0,
    fly_cost=0.01,
    velocity=50,
    min_readings=30,
    bank_angle_intervals=18,
    timestep_in_seconds=1,
    observations_per_timestep=1,
    extraction_cost=150.0,
    extraction_lcb=0.8,
    extraction_ucb=0.8,
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

for (i, seed) in enumerate(SEEDS[2:200])
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
    stds = []

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
        push!(stds, Statistics.std(b_vol))

        final_belief = bp
        final_state = sp
    end

    #println("Steps: $(length(h))")
    println("Decision: $(final_belief.acts[end].type)")
    println("======================")

    write_intelligent_result_to_file(m, final_belief, final_state, vols, stds; n_fly=n_fly, reward=v, seed=seed, r_massive=r_massive, which_map=:base)
    GC.gc()
end


function write_intelligent_during_execution(m, belief, state, r_massive, t, save_dir)
    p, _, mn, std = plot_volume(m, belief, r_massive, t=t, verbose=false)
    @info "Vols at time $t: $mn ± $std"
    savefig(p, string(save_dir, "$(t)volume.png"))
    display(p)
    empty!(p)

    p = plot(belief, m, state, t=t)
    savefig(p, string(save_dir, "$(t)b.png"))
    display(p)
    empty!(p)

    p = plot_base_map_and_plane_trajectory(state, m, t=t)
    savefig(p, string(save_dir, "$(t)trajectory.png"))
    display(p)
    empty!(p)
end

function write_initial_setup_to_file()
    # initial belief
    p = plot(b0, m, s0, t=0)
    savefig(p, string(save_dir, "0b.png"))
    empty!(p)

    # ore map
    p = plot_map(s0.ore_map, "ore map", axis=false)
    savefig(p, string(save_dir, "0ore_map.png"))
    empty!(p)

    p = plot_map(s0.smooth_map, "geophysical map", axis=false)
    savefig(p, string(save_dir, "0smooth.png"))
    empty!(p)

    # mass map
    p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
    savefig(p, string(save_dir, "0mass_map.png"))
    empty!(p)

    # initial volume
    p, _, mn, std = plot_volume(m, b0, r_massive, t=0, verbose=false)
    @info "Vols at time 0: $mn ± $std"
    savefig(p, string(save_dir, "0volume.png"))
    empty!(p)
end