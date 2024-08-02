using POMDPs
using MineralExploration

const DEG_TO_RAD = π / 180

m = MineralExplorationPOMDP(
    init_heading=convert(Float64, 0),
    init_pos_x=0,
    init_pos_y=0,
    velocity=40,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=5
)

ds0 = POMDPs.initialstate(m);
s0 = rand(ds0);

dummy_action = MEAction(type=:fly, change_in_bank_angle=-5);

# go straight
bank_angle = 0
go1, a, b, c = generate_geophysical_obs_sequence(m, s0, dummy_action, bank_angle)
@assert is_empty(go1)
@assert a == 40 * 5
@asset b = 0

# turn right
bank_angle = 10
go2, d, e, f = generate_geophysical_obs_sequence(m, s0, dummy_action, bank_angle)
@assert !is_empty(go2) # turned right off map

map_coords_x, map_coord_y = get_smooth_map_coordinates(d, e, m)
@assert s0.smooth_map[map_coords_x, map_coord_y, 1] == go2.reading[1]

# turn left
bank_angle = -10
go3, g, h, i = generate_geophysical_obs_sequence(m, s0, dummy_action, bank_angle)
@assert is_empty(go3) # turned left off map
@assert g == d # turning same amount right and left should give same coordinate value
@assert h == -e  # turning same amount right and left should result in coordinate value reflected in zero line




# multiple observations
m_multiple_obs = MineralExplorationPOMDP(
    init_heading=convert(Float64, 0),
    init_pos_x=0,
    init_pos_y=0,
    velocity=40,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=5,
    timestep_in_seconds=5
)

ds0 = POMDPs.initialstate(m);
s0 = rand(ds0);

dummy_action = MEAction(type=:fly, change_in_bank_angle=-5);

# turn right
bank_angle = 10
go4, g, h, i = generate_geophysical_obs_sequence(m_multiple_obs, s0, dummy_action, bank_angle)
@assert !is_empty(go2) # turned right off map

go4.reading
go4.base_map_coordinates
go4.smooth_map_coordinates
#map_coords_x, map_coord_y = get_smooth_map_coordinates(d, e, m)
#@assert s0.smooth_map[map_coords_x, map_coord_y, 1] == go2.reading[1]
