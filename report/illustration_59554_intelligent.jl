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


SEED = 59554
save_dir = "/Users/benbarlow/dev/diss-plots/help_plots/illustrative/intelligent/$(SEED)_"
DPI = 300
NOISE_FOR_PERTURBATION = 0.8
N_PARTICLES = 1000
C_EXP = 125.0
GET_TREES = false

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
)
planner = POMDPs.solve(solver, m)

println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    
trees = []


Random.seed!(SEED)

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

t = 0

path = string(save_dir, "belief_mean.txt")
open(path, "w") do io
    println(io, "Init")
end

p = plot_map(s0.ore_map, "", axis=false, colorbar=false)
plot!(p, dpi=DPI)
path = string(save_dir, "0ore_map.png")
savefig(p, path)

p = plot_map(s0.mainbody_map, "", axis=false, colorbar=false)
plot!(p, dpi=DPI)
path = string(save_dir, "0mass_map.png")
savefig(p, path)

p = plot_map(s0.smooth_map, "", axis=false, colorbar=false)
plot!(p, dpi=DPI)
path = string(save_dir, "0smooth_map.png")
savefig(p, path)

b_hist, vols, mn, std = plot_volume(m, b0, r_massive; t=t, verbose=false)
plot!(b_hist, dpi=DPI)
path = string(save_dir, "$(t)volume.png")
savefig(b_hist, path)

b_mn, b_std = plot(b0, return_individual=true)
plot!(b_mn, dpi=DPI)
plot!(b_std, dpi=DPI)
path = string(save_dir, "$(t)b.png")
savefig(b_mn, path)
path = string(save_dir, "$(t)bstd.png")
savefig(b_std, path)

for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)        
    if a.type == :fly
        n_fly += 1

    end
    v += POMDPs.discount(m)^(t - 1) * r

    b_vol = [calc_massive(m, p) for p in bp.particles]
    push!(mns, mean(b_vol))
    push!(stds, Statistics.std(b_vol))

    final_belief = bp
    final_state = sp

    b_hist, vols, mn, std = plot_volume(m, bp, r_massive; t=t, verbose=false)
    push!(mns, mn)
    push!(stds, std)

    if t % 10 == 0 || final_belief.decided
        b_mn, b_std = plot(bp, return_individual=true)
        map_and_plane = plot_smooth_map_and_plane_trajectory(sp, m, t=t)

        if isa(save_dir, String)
            path = string(save_dir, "$(t)b.png")
            plot!(b_mn, dpi=DPI)
            savefig(b_mn, path)

            path = string(save_dir, "$(t)bstd.png")
            plot!(b_std, dpi=DPI)
            savefig(b_std, path)

            path = string(save_dir, "$(t)volume.png")
            plot!(b_hist, dpi=DPI)
            savefig(b_hist, path)

            path = string(save_dir, "$(t)trajectory.png")
            plot!(map_and_plane, dpi=DPI)
            savefig(map_and_plane, path)

            path = string(save_dir, "belief_mean.txt")
            open(path, "a") do io
                println(io, "Vols at time $t: $(mn) ± $(std)")
            end
        end
    end
end

open(string(save_dir, "trial_$(SEED)_info.txt"), "w") do io
    println(io, "Seed: $SEED")
    println(io, "Mean: $(last(mns))")
    println(io, "Std: $(last(stds))")
    println(io, "Massive: $r_massive")
    println(io, "Extraction cost: $(m.extraction_cost)")
    println(io, "Decision: $(final_belief.acts[end].type)")
    println(io, "Fly: $n_fly")
    println(io, "Reward: NA")
end