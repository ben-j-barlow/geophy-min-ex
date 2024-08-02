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
    init_heading=45.0,
    velocity=20,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=2
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0) #Checked
s0 = rand(ds0);

a = MEAction(type=:fly, change_in_bank_angle=5)

s = deepcopy(s0)
b = deepcopy(b0)

for i in 1:20
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    r = out[:r];
    bp, ui = update_info(up, b, a, o);

    s = deepcopy(sp);
    b = deepcopy(bp);
end

plot_smooth_map_and_plane_trajectory(s, m)

#straight_x = deepcopy(s.agent_pos_x)
#straight_y = deepcopy(s.agent_pos_y)