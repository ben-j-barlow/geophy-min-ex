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
#GRID_DIMS = (50, 50, 1)
NOISE_FOR_PERTURBATION = 2.0
N_PARTICLES = 1000
MOVE_MULT = 6  # assuming 50ms plane, 25m grid cell lengths
SIDESTEP_MULT = 8 # line spacing of 200m and 50m grid cells
INIT_X_BASE = 8 # go 8, 16, ..., 40
EARLY_STOP = false
SEEDS = get_uncompleted_seeds(baseline=true)
GRID_LINES = true


# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    upscale_factor=4,
    sigma=3,
    geophysical_noise_std_dev=0.02
    )

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)

# Define the belief updater for the POMDP with 1000 particles and a spread factor of 2.0
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

# set up the solver
max_coord = m.grid_dim[1] * m.upscale_factor
move_size = MOVE_MULT * m.upscale_factor
sidestep_size = SIDESTEP_MULT * m.upscale_factor
init_coords = CartesianIndex(1, INIT_X_BASE * m.upscale_factor)


println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    


for (i, seed) in enumerate(SEEDS[1:150])
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
    mns = []
    stds = []
    policy = BaselineGeophysicalPolicy(m, max_coord, move_size, sidestep_size, init_coords, EARLY_STOP, GRID_LINES)

    for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)
        # if t % 60 == 0
        #     println("t: $t")
        # end
        # if t > 120 && t % 5 == 0
        #     _, _, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)
        #     println("t: $t, mn: $mn, std: $std")
        # end

        if a.type == :fake_fly
            n_fly += 1

        end
        v += POMDPs.discount(m)^(t - 1) * r
        
        final_belief = bp
        final_state = sp

        b_vol = [calc_massive(m, p) for p in bp.particles]
        push!(mns, mean(b_vol))
        push!(stds, Statistics.std(b_vol))
    end
    
    #println("Steps: $(length(h))")
    println("Mean Ore: $(last(mns)) ± $(last(stds))")
    println("Decision: $(final_belief.acts[end].type)")
    println("======================")
    
    b_hist, vols, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)
    write_baseline_result_to_file(m, final_belief, final_state, mns, stds; n_fly=n_fly, reward=v, seed=seed, r_massive=r_massive, grid=GRID_LINES, which_map=:base)
    
    GC.gc()
end