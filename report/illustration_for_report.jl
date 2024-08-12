# TEST: compare flying action by action in a straight line with vertical line of observations at once

using MineralExploration
using Plots
using Random
using POMDPSimulators
using POMDPs
using POMDPModelTools


save_dir = "./data/illustration_fly_over/"
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
    grid_dim = (50, 50, 1),
    c_exp = 100.0,
    base_grid_element_length = 40.0,
    upscale_factor = 4,
    sigma = 5,
    geophysical_noise_std_dev = 0.01,
    fly_cost = 0.01,
    out_of_bounds_cost = 0.0,
    out_of_bounds_tolerance = 0,
    init_pos_x = 62.5,
    init_pos_y = 0.0,
    init_heading = HEAD_NORTH,
    max_bank_angle = 55,
    bank_angle_intervals = 15,
    velocity = 50,
    extraction_cost = 150.0,
    extraction_lcb = 0.3,
    extraction_ucb = 0.3
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0);
Random.seed!(42)
s0 = rand(ds0, save_dir=save_dir);
a = MEAction(type=:fly, change_in_bank_angle=0);

solver = get_geophysical_solver(m.c_exp)
planner = POMDPs.solve(solver, m)

# fly straight
#a = MEAction(type=:fly, change_in_bank_angle=0)

b = b0;
s = s0;

# initial belief
p = plot(b0, m, s0, t=0)
savefig(p, string(save_dir, "0b.png"))
empty!(p)

# ore map
p = plot_map(s0.ore_map, "ore map", axis=false)
savefig(p, string(save_dir, "0ore_map.png"))
empty!(p)

p = plot_map(s0.smooth_map, "geophysical map", axis=false)
savefig(p, string(save_dir, "0smooth.png"))
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
    a = action_info(planner, b0)
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);

    if i % 1 == 0
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



hr = HistoryRecorder(max_steps=m.max_timesteps+3)  # Record history of the simulation
h = simulate(hr, m, planner, up, b0, s0)

v = 0.0  # Initialize the return
n_fly = 0  # Initialize the fly count

states = []
beliefs = []

# Calculate the discounted return and count drills
for stp in h
    v += POMDPs.discount(m)^(stp[:t] - 1) * stp[:r]
    if stp[:a].type == :fly
        n_fly += 1
    end
    b = step[:b]
    s = step[:sp]
    # push!(states, )
    # push(beliefs)
end

final_action = h[end][:a].type