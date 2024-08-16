
using MineralExploration
using Plots
using Random
using POMDPs
using POMDPModelTools

SEEDS = [42, 100, 6]
NAMES = ["prof", "unprof", "border"]
GRID_DIMS = (30,30,1)
UPSCALE = 3

save_dir = "./data/report/map_examples/"
if isdir(save_dir)
    rm(save_dir; recursive=true)
end
mkdir(save_dir)

m = MineralExplorationPOMDP(
    grid_dim=GRID_DIMS,
    upscale_factor=3
)
ds0 = POMDPs.initialstate(m)


function save_maps_for_report(ds::MEInitStateDist, m::MineralExplorationPOMDP, save_dir::String, seed::Int64)
    Random.seed!(seed)
    s0 = rand(ds, save_dir=save_dir);

    p = plot_map(s0.ore_map, "ore map", axis=false)
    savefig(p, string(save_dir, "0ore_map.pdf"))
    empty!(p)

    p = plot_map(s0.smooth_map, "geophysical map", axis=false)
    savefig(p, string(save_dir, "0smooth.pdf"))
    empty!(p)

    # mass map
    p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
    @info "$seed & $r_massive"
    savefig(p, string(save_dir, "0mass_map.pdf"))
    empty!(p)
end

for (seed, name) in zip(SEEDS, NAMES)
    seed_save_dir = string("./data/report/map_examples/$(name)/", )
    if isdir(seed_save_dir)
        rm(seed_save_dir; recursive=true)
    end
    mkdir(seed_save_dir)
    save_maps_for_report(ds0, m, seed_save_dir, seed)
end


Random.seed!(SEEDS[1])
s0 = rand(ds0, save_dir=save_dir);

mainbody_param = s0.mainbody_params
mainbody_gen = m.mainbody_gen
NOISE = 2
p = plot_map(s0.mainbody_map, "mainbody map", axis=false)
savefig(p, string(save_dir, "$(0)perturb.pdf"))
empty!(p)
for i in 1:3
    mainbody_map, mainbody_param = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
    p = plot_map(mainbody_map, axis=false)
    savefig(p, string(save_dir, "$(i)perturb.pdf"))
    empty!(p) 
end