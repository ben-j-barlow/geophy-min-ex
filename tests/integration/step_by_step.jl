using POMCPOW
using POMDPs
using Plots
using MineralExploration
using POMDPModelTools
using Random
using DelimitedFiles
using LinearAlgebra
using D3Trees

include("../helpers.jl")


save_dir = "./data/step_by_step/"


if isdir(save_dir)
    rm(save_dir; recursive=true)
    #error("directory already exists")
end
mkdir(save_dir)

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 1.0
# noise
# 10 - belief didnt really converge

C_EXP = 125
GET_TREES = true


#io = MineralExploration.prepare_logger()

m = MineralExplorationPOMDP()

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

#Random.seed!(SEED)
seed = convert(Int64, floor(rand() * 10000))
#Random.seed!(seed)  # Reseed with a new random seed
s0 = rand(ds0; truth=true) #Checked

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
b0 = POMDPs.initialize_belief(up, ds0) #Checked

solver = get_geophysical_solver(m.c_exp)
planner = POMDPs.solve(solver, m)

b = b0;
s = s0;

# initial belief
p = plot(b0, m, s0, t=0)
savefig(p, string(save_dir, "0b.png"))
empty!(p)

# ore map
p = plot_map(s0.ore_map, "ore map")
savefig(p, string(save_dir, "0ore_map.png"))
empty!(p)

# mass map
p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
savefig(p, string(save_dir, "0mass_map.png"))
display(p)
empty!(p)

# initial volume
p, _, mn, std_dev = plot_volume(m, b0, r_massive, t=0, verbose=false)
@info "Vols at time 0: $mn ± $std_dev"
savefig(p, string(save_dir, "0volume.png"))
empty!(p)

path = string(save_dir, "belief_mean.txt")
open(path, "w") do io
    println(io, "Vols at time 0: $(mn) ± $(std_dev)")
end

trees = []

beliefs = []
states = []

bank_angle = 0
T = 15

for i in 1:10
    a, ai = action_info(planner, b, tree_in_info=GET_TREES)
    if GET_TREES
        tree = deepcopy(ai[:tree]);
        #if i % 1 == 0 # && i >= 9
            #push!(trees, tree);
            
        #end    
    end
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);

#if i % 1 == 2
#    p, _, mn, std_dev = plot_volume(m, bp, r_massive, t=i, verbose=false)
#    @info "Vols at time $i: $mn ± $std_dev"
#    savefig(p, string(save_dir, "$(i)volume.png"))
#    #display(p)
#    empty!(p)
#    
#    p = plot(bp, m, sp, t=i)
#    savefig(p, string(save_dir, "$(i)b.png"))
#    #display(p)
#    empty!(p)
#
#    p = plot_base_map_and_plane_trajectory(sp, m, t=i)
#    savefig(p, string(save_dir, "$(i)trajectory.png"))
#    display(p)
#    empty!(p)
#
#    path = string(save_dir, "belief_mean.txt")
#    open(path, "a") do io
#        println(io, "Vols at time $i: $(mn) ± $(std_dev)")
#    end
#end

b = bp
s = sp

push!(beliefs, b)
push!(states, s)

#if b.decided 
#    break
#end
end

display(plot(b))

inbrowser(D3Tree(tree, init_expand=1), "safari")