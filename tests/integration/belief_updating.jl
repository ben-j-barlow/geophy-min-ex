# TEST: update using extreme case where line of observations stretches the whole grid

using MineralExploration
using Plots
using Random
using POMDPs
using POMDPModelTools

include("../helpers.jl")

# constants
N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

# setup
m = MineralExplorationPOMDP()
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0);
Random.seed!(42)
s0 = rand(ds0);
a = MEAction(type=:fly, change_in_bank_angle=0);


# get horizontal line comparison
obs_horizontal = observe_line(s0.ore_map, 25, :horizontal);
o_horizontal = MEObservation(nothing,false,false,obs_horizontal,m.init_heading,m.init_pos_x,m.init_pos_y,0)
b_horizontal, ui = update_info(up, b0, a, o_horizontal);
plot(b0)
plot(b_horizontal)

# get vertical line comparison
obs_vertical = observe_line(s0.ore_map, 25, :vertical);
o_vertical = MEObservation(nothing,false,false,obs_vertical,m.init_heading,m.init_pos_x,m.init_pos_y,0)
b_vertical, ui = update_info(up, b0, a, o_vertical);
plot(b0)
plot(b_vertical)

# combine both lines
b_both_1, ui = update_info(up, b_horizontal, a, o_vertical);
b_both_2, ui = update_info(up, b_vertical, a, o_horizontal);
plot(b0)
plot(b_both_1)
plot(b_both_2)
