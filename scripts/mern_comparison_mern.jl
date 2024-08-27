
## MERN COMPARISON

using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration
using Random


save_dir = "./data/mern_comparison/"

mutable struct ManualPolicy <: Policy
    m::MineralExplorationPOMDP
    actions::Vector{MEAction}
end

POMDPs.action(p::ManualPolicy, b::MEBelief) = pop!(p.actions)

Random.seed!(10)
# Constants for the problem setup
#GRID_DIMS = (50, 50, 1)
NOISE_FOR_PERTURBATION = 1.0
N_PARTICLES = 1000


m = MineralExplorationPOMDP(mineral_exploration_mode="borehole", grid_dim=(50,50,1))

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

acts = MEAction[MEAction(type=:stop, coords=CartesianIndex(0,0))]
coords = [CartesianIndex(5,1), CartesianIndex(15,10), CartesianIndex(35,8), CartesianIndex(45,10), CartesianIndex(40,5)]
for i in 1:5
    push!(acts, MEAction(type=:drill, coords=coords[i]))
end
policy = ManualPolicy(m, acts)

s0 = rand(ds0; truth=true) 
p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
println(r_massive)

b = b0
s = s0

p = plot_ore_map(s0.ore_map)
savefig(p, string(save_dir, "mern_ore_map.png"))

path = string(save_dir, "mern_belief_mean.txt")
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
        savefig(p, string(save_dir, "mern$(i)volume.png"))
        #display(p)
        empty!(p)
        
        p_mean, p_std = plot(bp)
        savefig(p_mean, string(save_dir, "mern$(i)b.png"))
        #display(p)
        empty!(p_mean)

        path = string(save_dir, "mern_belief_mean.txt")
        open(path, "a") do io
            println(io, "Vols at time $i: $(mn) ± $(std_dev)")
        end
    end

    b = bp
    s = sp
end


b.obs
b.geostats.data.ore_quals
