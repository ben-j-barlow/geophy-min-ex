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
m = MineralExplorationPOMDP(upscale_factor=1, grid_dim=(50,50,1), max_timesteps=5)

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

s0 = rand(ds0; truth=true) 

# set up the solver
coords = [CartesianIndex(30,16), CartesianIndex(30,13), CartesianIndex(30,10), CartesianIndex(30,7), CartesianIndex(30,4), CartesianIndex(30,1)]
acts = MEAction[MEAction(type=:stop, coords=CartesianIndex(0,0))]
for i in 1:6
    push!(acts, MEAction(type=:fake_fly, coords=coords[i]))
end
policy = ManualPolicy(m, acts)


# Run the simulation for N trials
final_state = nothing
final_belief = nothing

b = b0
s = s0

p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)

p = plot_ore_map(s0.ore_map)
savefig(p, string(save_dir, "fake_ore_map.png"))

println(r_massive)

path = string(save_dir, "fake_belief_mean.txt")
open(path, "w") do io
    println(io, "")
end

for i in 1:5
    a = action(policy, b0)
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp = update(up, b, a, o);
    
    if i % 1 == 0
        p, _, mn, std_dev = plot_volume(m, bp, r_massive, t=i, verbose=false)
        @info "Vols at time $i: $mn ± $std_dev"
        savefig(p, string(save_dir, "fake_$(i)volume.png"))
        #display(p)
        empty!(p)
        
        p_mean, p_std = plot(bp)
        savefig(p_mean, string(save_dir, "fake_$(i)b.png"))
        #display(p)
        empty!(p_mean)

        path = string(save_dir, "fake_belief_mean.txt")
        open(path, "a") do io
            println(io, "Vols at time $i: $(mn) ± $(std_dev)")
        end
    end

    b = bp
    s = sp
end