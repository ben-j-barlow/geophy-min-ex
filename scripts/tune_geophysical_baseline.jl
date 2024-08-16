using JSON
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
NOISE_FOR_PERTURBATION = [1.0, 2.5, 3, 5]
N_PARTICLES = [250, 500, 750, 100]
MOVE_MULT = 2  # assuming 50ms plane, 25m grid cell lengths
SIDESTEP_MULT = 6 # line spacing of 200m and 50m grid cells
INIT_X_BASE = 6 # go 8, 16, ..., 40
EARLY_STOP = false
GRID_LINES = true
dir = "/Users/benbarlow/dev/MineralExploration/data/tune/"
SEEDS = get_all_seeds()

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP()

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)

# set up the solver
max_coord = m.grid_dim[1] * m.upscale_factor
move_size = MOVE_MULT * m.upscale_factor
sidestep_size = SIDESTEP_MULT * m.upscale_factor
init_coords = CartesianIndex(1, INIT_X_BASE * m.upscale_factor)


println("Starting simulations")

# Run the simulation for N trials
final_state = nothing
final_belief = nothing    

# create a Np: noise: seed: error dictionary
d = Dict{Int, Dict{Float64, Dict{Int, Float64}}}()
for Np in N_PARTICLES
    if !haskey(d, Np)
        d[Np] = Dict{Float64, Dict{Int, Float64}}()
    end
    for noise in NOISE_FOR_PERTURBATION
        if !haskey(d[Np], noise)
            d[Np][noise] = Dict{Int, Float64}()
        end
    end
end


for Np in N_PARTICLES
    if !isfile(string(dir, "np_$(Np).txt"))
        open(string(dir, "np_$(Np).txt"), "w") do io
            println(io, "N_PARTICLES: $Np")
        end
    end
    
    for noise in NOISE_FOR_PERTURBATION
        open(string(dir, "np_$(Np).txt"), "a") do io
            println(io, "======================")
            println(io, "")
            println(io, "noise: $noise")
        end
        println("N_PARTICLES: $Np, NOISE_FOR_PERTURBATION: $noise")
        
        
        for (i, seed) in enumerate(SEEDS[20:23])
            Random.seed!(seed)
            s0 = rand(ds0)  # Sample a starting state
            up = MEBeliefUpdater(m, Np, noise)
            b0 = POMDPs.initialize_belief(up, ds0)
            
            s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
            @assert m.dim_scale == 1
            r_massive = sum(s_massive)  # Calculate total massive ore
            println("Massive Ore: $r_massive")
            
            
            # Simulate a sequence of actions and states
            #h = simulate(hr, m, policy, up, b0, s0)
            v = 0.0  # Initialize the return
            n_fly = 0  # Initialize the fly count
            policy = BaselineGeophysicalPolicy(m, max_coord, move_size, sidestep_size, init_coords, EARLY_STOP, GRID_LINES)

            for (sp, a, r, bp, t) in stepthrough(m, policy, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)
                if t % 60 == 0
                    println("t: $t")
                end
                
                if a.type == :fake_fly
                    n_fly += 1

                end
                v += POMDPs.discount(m)^(t - 1) * r
                
                final_belief = bp
                final_state = sp
            end

            b_vol = [calc_massive(m, p) for p in final_belief.particles]
            err = mean(b_vol .- r_massive)
            d[Np][noise][seed] = err

            _, _, mn, std = plot_volume(m, final_belief, r_massive; verbose=false)
            open(string(dir, "np_$(Np).txt"), "a") do io
                println(io, "seed: $seed, t: $n_fly, mn: $mn, truth: $r_massive, std: $std")
            end
            
            #println("Steps: $(length(h))")
            println("Decision: $(final_belief.acts[end].type)")
            println("seed: $seed, t: $n_fly, mn: $mn, truth: $r_massive, std: $std")
            println("======================")
            
            
            GC.gc()
        end
    end
end

open("baseline_errors.json", "w") do io
    JSON.print(io, d)
end

# summarise results grouped by N_PARTICLES, NOISE_FOR_PERTURBATION
for (Np, noise_dict) in d
    for (noise, seed_dict) in noise_dict
        println("N_PARTICLES: $Np, NOISE_FOR_PERTURBATION: $noise")
        println("Mean Error: $(mean([abs(val) for val in values(seed_dict)]))")
        println("Std Error: $(std(values(seed_dict)))")
        println("======================")
    end
end