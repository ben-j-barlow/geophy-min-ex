
dir = "/Users/benbarlow/dev/MineralExploration/data/method_plots/"

Random.seed!(10)
s0 = rand(ds0);

p1 = plot_map(s0.mainbody_map, "(a)", axis=false, colorbar=false)


mainbody_param = s0.mainbody_params
mainbody_gen = m.mainbody_gen

NOISE = 1

mainbody_map2, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p2 = plot_map(mainbody_map2, "(b)", axis=false, colorbar=false)

mainbody_map3, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p3 = plot_map(mainbody_map3, "(c)", axis=false, colorbar=false)

mainbody_map4, _ = MineralExploration.perturb_sample(mainbody_gen, mainbody_param, NOISE) # Perturb the main body map and parameters
p4 = plot_map(mainbody_map4, "(d)", axis=false, colorbar=false)

sz = (800, 250)
all = plot(p1, p2, p3, p4, layout=(1, 4), size=sz, margin = 0mm)

#vline!(all[1], [1.5], color=:black, linewidth=2, linestyle=:solid)

savefig(all, string(dir, "pb3.pdf"))

mean(mainbody_map2)
mean(mainbody_map3)