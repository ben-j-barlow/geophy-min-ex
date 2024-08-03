using POMCPOW
using POMDPs
using Plots
using MineralExploration
using POMDPModelTools

C_EXP = 2

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0
m = MineralExplorationPOMDP(
    c_exp=C_EXP,
    sigma=20,
    init_pos_x=500,
    init_pos_y=20,
    init_heading=HEAD_NORTH,
    velocity=20,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=1
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0) #Checked
s0 = rand(ds0);

a = MEAction(type=:fly, change_in_bank_angle=5)

s = deepcopy(s0)
b = deepcopy(b0)

s_ = [deepcopy(s)]

for i in 1:8
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    r = out[:r];
    bp, ui = update_info(up, b, a, o);

    s = deepcopy(sp);
    b = deepcopy(bp);

    push!(s_, deepcopy(s))
end

s_[3].agent_pos_x
s_[3].agent_pos_y

#x, y = get_agent_trajectory(s.agent_bank_angle, m)

x, y = normalize_agent_coordinates(s_[8].agent_pos_x, s_[8].agent_pos_y, m.smooth_grid_element_length)
p = plot_map(s.smooth_map, "geophysical map with plane trajectory")
plot!(p, x, y, label="plane position at t=3", lw=2, color=:red)



add_agent_trajectory_to_plot!(p, x, y)

s_[8].agent_pos_y

x
s_[8].agent_pos_x
#straight_x = deepcopy(s.agent_pos_x)
#straight_y = deepcopy(s.agent_pos_y)