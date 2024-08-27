# TEST: compare flying action by action in a straight line with vertical line of observations at once

using MineralExploration
using Plots
using Random
using POMDPs
using POMDPModelTools

include("../helpers.jl")

save_dir = "./data/belief_fly_line_comparison_noise_dot5/"
!isdir(save_dir) && mkdir(save_dir)

# constants
N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 0.1

# CIRCLE TEST
#init_pos_x=30 * 25.0, # cell 31
#init_pos_y=20 * 25.0, # cell 0 (outside grid so first flight will be to cell 1)
#init_heading=HEAD_EAST,
#init_bank_angle=15,

# setup
m = MineralExplorationPOMDP(
    init_pos_x=30 * 25.0, # cell 31
    init_pos_y=20, # cell 0 (outside grid so first flight will be to cell 1)
    init_heading=HEAD_NORTH,
    init_bank_angle=0,
    base_grid_element_length=25,
    velocity=25,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=1
)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0);
Random.seed!(42)
s0 = rand(ds0);
a = MEAction(type=:fly, change_in_bank_angle=0);

plot(b0)

# fly straight
a = MEAction(type=:fly, change_in_bank_angle=0)

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
empty!(p)

# initial volume
p, _, mn, std = plot_volume(m, b0, r_massive, t=0, verbose=false)
@info "Vols at time 0: $mn ± $std"
savefig(p, string(save_dir, "0volume.png"))
empty!(p)

for i in 1:50
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);

    if i % 5 == 0
        p, _, mn, std = plot_volume(m, bp, r_massive, t=i, verbose=false)
        @info "Vols at time $i: $mn ± $std"
        savefig(p, string(save_dir, "$(i)volume.png"))
        display(p)
        empty!(p)
        
        p = plot(bp, m, sp, t=i)
        savefig(p, string(save_dir, "$(i)b.png"))
        display(p)
        empty!(p)

        p = plot_base_map_and_plane_trajectory(sp, m, t=i)
        savefig(p, string(save_dir, "$(i)trajectory.png"))
        display(p)
        empty!(p)
    end
    
    s = sp
    b = bp
end



# get vertical line comparison
obs_vertical = observe_line(s0.ore_map, ceil(m.init_pos_x / m.base_grid_element_length), :vertical);
o_vertical = MEObservation(nothing,false,false,obs_vertical,m.init_heading,m.init_pos_x,m.init_pos_y,0)
b_vertical, ui = update_info(up, b0, a, o_vertical);
plot(b0)
plot(b_vertical)
p, _, _, _ = plot_volume(m, b_vertical, r_massive)
display(p)
empty!(p)
