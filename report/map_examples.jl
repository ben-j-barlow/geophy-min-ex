
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
    savefig(p, string(save_dir, "0ore_map.png"))
    empty!(p)

    p = plot_map(s0.smooth_map, "geophysical map", axis=false)
    savefig(p, string(save_dir, "0smooth.png"))
    empty!(p)

    # mass map
    p, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
    @info "$seed & $r_massive"
    savefig(p, string(save_dir, "0mass_map.png"))
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