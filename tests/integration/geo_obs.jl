using POMCPOW
using POMDPs
using Plots
using MineralExploration
using POMDPModelTools
using Random

C_EXP = 2

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

m = MineralExplorationPOMDP(
    base_grid_element_length=4.0,
    upscale_factor=4,
    c_exp=C_EXP,
    sigma=20,
    init_pos_x=80, # coord 20
    init_pos_y=40, # coord 10
    init_heading=HEAD_NORTH,
    velocity=20,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=1,
    bank_angle_intervals=5
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
ds0 = POMDPs.initialstate(m)
b0 = POMDPs.initialize_belief(up, ds0) #Checked

Random.seed!(100)
s0 = rand(ds0);


a = MEAction(type=:fly, change_in_bank_angle=5)

s = deepcopy(s0);
b = deepcopy(b0);

s_list = [deepcopy(s)];
b_list = [deepcopy(b)];

for i in 1:5
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    r = out[:r];
    bp, ui = update_info(up, b, a, o);

    s = deepcopy(sp);
    b = deepcopy(bp);
    push!(s_list, deepcopy(s))
    push!(b_list, deepcopy(b))
end

#s_list[8].agent_pos_x
#s_list[3].agent_pos_x

#s_list[1].agent_pos_x
#s_list[3].agent_pos_x
#s_list[3].agent_pos_y

#s_list[3].geophysical_obs.smooth_map_coordinates
#s_list[5].geophysical_obs.smooth_map_coordinates

#s_list[3].geophysical_obs.reading
#s_list[5].geophysical_obs.reading

# coords (103,8) have reading 0.4494573606441881
# coords (105,11) have reading 0.48222639129246203
#s_list[1].smooth_map[103,8]

#to_plot = deepcopy(s_list[1].smooth_map)
#to_plot[103,8] = NaN
# OBS LOCATION IS NOT CORRESPONDING TO THE PLANE'S PATH
#plot_map(to_plot, "moment of truth")
p = plot_smooth_map_and_plane_trajectory(s_list[end], m)
#plot!(p, s_list[end].agent_pos_y, s_list[end].agent_pos_x, label="Plane", marker=:circle, color=:red)

timestep1_x = s_list[end].agent_pos_x[2]
timestep1_y = s_list[end].agent_pos_y[2]

smooth_length = 1
base_length = 4
smooth_x = convert(Int64, ceil(timestep1_x / smooth_length))
smooth_y = convert(Int64, ceil(timestep1_y / smooth_length))
base_x = convert(Int64, ceil(timestep1_x/base_length))
base_y = convert(Int64, ceil(timestep1_y/base_length))

reading = s_list[end].smooth_map[smooth_y, smooth_x,1]
s_list[2].geophysical_obs

s_list[end].geophysical_obs.reading

#s_list[5].agent_pos_x
#s_list[5].agent_pos_y

