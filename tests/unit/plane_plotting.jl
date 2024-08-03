using POMDPs
using Plots
using MineralExploration
using Random

m = MineralExplorationPOMDP();
ds0 = POMDPs.initialstate(m);
Random.seed!(100)
s0 = rand(ds0);

x_list = [25, 30, 35]
y_list = [20, 20, 20]

p = plot_map(s0.ore_map, "ore")
plot!(p, x_list, y_list, seriestype = :scatter, label="Plot by hand", color="red")
add_agent_trajectory_to_plot!(p, x_list, y_list)