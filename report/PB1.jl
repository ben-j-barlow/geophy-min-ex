# Import necessary packages
using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration
using Random
using Plots.PlotMeasures

function adjust_plot(map, title="")
    xl = (0, size(map, 1))
    yl = (0, size(map, 2))
    return heatmap(map[:, :, 1], fill=true, title=title, clims=(0.0, 1.0), aspect_ratio=1, xlims=xl, ylims=yl, c=:viridis, left_margin=0mm, bottom_margin = 0mm, framestyle = :none, axis=false, colorbar=false)
end

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    grid_dim=(48,48,1),
    init_pos_x=7.5*25.0 # align with baseline starting in x=8
)

ds0 = POMDPs.initialstate(m)

Random.seed!(10)
s1 = rand(ds0)  # Sample a starting state

Random.seed!(20)
s2 = rand(ds0)  # Sample a starting state

Random.seed!(30)
s3 = rand(ds0)  # Sample a starting state

mn = zeros(size(s1.ore_map))
for s in [s1, s2, s3]
    mn .+= s.ore_map
end
mn ./= 3

dir = "/Users/benbarlow/dev/MineralExploration/data/method_plots/"

p1 = adjust_plot(s1.ore_map, "(a)")
p2 = adjust_plot(s2.ore_map, "(b)")
p3 = adjust_plot(s3.ore_map, "(c)")
p4 = adjust_plot(mn, "(d)")

sz = (800, 250)
all = plot(p1, p2, p3, p4, layout=(1, 4), size=sz, margin = 0mm)
savefig(all, string(dir, "pb1.pdf"))
#fig_smooth = plot(p_smooth_mean, p_smooth_std, layout=(1, 2), size=sz)
