using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration
using Random


save_dir = "./data/real_comparison/"

mutable struct ManualPolicy <: Policy
    m::MineralExplorationPOMDP
    actions::Vector{MEAction}
end

POMDPs.action(p::ManualPolicy, b::MEBelief) = pop!(p.actions)

Random.seed!(30)
# Constants for the problem setup
#GRID_DIMS = (50, 50, 1)
NOISE_FOR_PERTURBATION = 1.0
N_PARTICLES = 1000

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    base_grid_element_length=10,
    upscale_factor=1, 
    grid_dim=(50,50,1), 
    max_timesteps=5,
    init_pos_x=-25,
    init_pos_y=295,
    velocity=30,
    init_heading=HEAD_EAST,
    )

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

s0 = rand(ds0; truth=true) 

# Run the simulation for N trials
final_state = nothing
final_belief = nothing

b = b0
s = s0

p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)

p = plot_ore_map(s0.ore_map)
savefig(p, string(save_dir, "real_ore_map.png"))

println(r_massive)

path = string(save_dir, "real_belief_mean.txt")
open(path, "w") do io
    println(io, "")
end

a = MEAction(type=:fly, change_in_bank_angle=0)

for i in 1:5
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp = update(up, b, a, o);
    
    if i % 1 == 0
        p, _, mn, std_dev = plot_volume(m, bp, r_massive, t=i, verbose=false)
        @info "Vols at time $i: $mn ± $std_dev"
        savefig(p, string(save_dir, "real_$(i)volume.png"))
        #display(p)
        empty!(p)
        
        p_mean, p_std = plot(bp)
        savefig(p_mean, string(save_dir, "real_$(i)b.png"))
        #display(p)
        empty!(p_mean)

        path = string(save_dir, "real_belief_mean.txt")
        open(path, "a") do io
            println(io, "Vols at time $i: $(mn) ± $(std_dev)")
        end
    end

    b = bp
    s = sp
end