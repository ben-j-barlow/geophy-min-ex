using Random
using MineralExploration
using POMDPs
using Plots
using Plots.PlotMeasures

DIR = "/Users/benbarlow/dev/diss-plots/help_plots/maps_to_combine/pb3/"
DPI = 300

m = MineralExplorationPOMDP(grid_dim=(48,48,1))
ds0 = POMDPs.initialstate(m)
Random.seed!(10)
s0 = rand(ds0);

p1 = plot_map(s0.mainbody_map, "", axis=false, colorbar=false)


mainbody_param = s0.mainbody_params
mainbody_gen = m.mainbody_gen

NOISE = 1

mainbody_map2, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p2 = plot_map(mainbody_map2, "", axis=false, colorbar=false)

mainbody_map3, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p3 = plot_map(mainbody_map3, "", axis=false, colorbar=false)

mainbody_map4, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p4 = plot_map(mainbody_map4, "", axis=false, colorbar=false)

plot!(p1, dpi=DPI)
savefig(p1, string(DIR, "og.png"))

plot!(p2, dpi=DPI)
savefig(p2, string(DIR, "perturb1.png"))

plot!(p3, dpi=DPI)
savefig(p3, string(DIR, "perturb2.png"))

plot!(p4, dpi=DPI)
savefig(p4, string(DIR, "perturb3.png"))


