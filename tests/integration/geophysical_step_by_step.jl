using POMCPOW
using POMDPs
using Plots
using MineralExploration
using POMDPModelTools
using Random

include("helpers.jl")

C_EXP = 100
N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

m = MineralExplorationPOMDP(
    c_exp=C_EXP,
    sigma=20,
    init_heading=HEAD_EAST,
    init_pos_x=201,
    init_pos_y=700,
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
s0_copy = deepcopy(s0);
b0_copy = deepcopy(b0);

solver = POMCPOWSolver(
    tree_queries=2000,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    k_observation=2.0,
    alpha_observation=0.1,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    max_depth=5,
    estimate_value=leaf_estimation,
    tree_in_info=false,
)
planner = POMDPs.solve(solver, m);


# action info : inspect b0 vs copied version

a = MEAction(type=:fly, change_in_bank_angle=0);

b = b0;
s = s0;

r_massive = sum(s0.ore_map[:,:,1] .>= m.massive_threshold)


mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, :viridis; truth=true)
b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=true)
p = plot_map(s0.ore_map, "ore")

display(mass_fig)
display(p)
display(b0_hist)

empty!(mass_fig)
empty!(p)
empty!(b0_hist)



for i in 1:30
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);
    
    @info ""
    @info "timestep $i"
    @info "b geostats base map coords  $(b.geostats.geophysical_data.base_map_coordinates)"
    @info "s norm (base) agent pos     x $(last(sp.agent_pos_x) / m.base_grid_element_length) y $(last(sp.agent_pos_y) / m.base_grid_element_length)"
    @info "obs base map coords         $(o.geophysical_obs.base_map_coordinates)"
    @info "s norm (smooth) agent pos   x $(last(sp.agent_pos_x) / m.smooth_grid_element_length) y $(last(sp.agent_pos_y) / m.smooth_grid_element_length)"
    @info "obs smooth map coords       $(o.geophysical_obs.smooth_map_coordinates)"
    @info "bp geostats base map coords $(bp.geostats.geophysical_data.base_map_coordinates)"
    
    if i % 4 == 0
        p, _, _, _ = plot_volume(m, bp, r_massive, t=i)
        display(p)
        empty!(p)
        
        p = plot(bp, m, sp)
        display(p)
        empty!(p)

        p = plot_smooth_map_and_plane_trajectory(sp, m)
        display(p)
        empty!(p)
    end
    
    b = bp
    s = sp
end



a = MEAction(type=:fly, change_in_bank_angle=10);
for i in 1:4
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);

    s = sp
    b = bp
end

a = MEAction(type=:fly, change_in_bank_angle=-10);
for i in 1:5
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);

    s = sp
    b = bp
end


t = 39
a = MEAction(type=:fly, change_in_bank_angle=0);
for i in t:(t+10)
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);
    
    @info ""
    @info "timestep $i"
    @info "b geostats base map coords  $(b.geostats.geophysical_data.base_map_coordinates)"
    @info "s norm (base) agent pos     x $(last(sp.agent_pos_x) / m.base_grid_element_length) y $(last(sp.agent_pos_y) / m.base_grid_element_length)"
    @info "obs base map coords         $(o.geophysical_obs.base_map_coordinates)"
    @info "s norm (smooth) agent pos   x $(last(sp.agent_pos_x) / m.smooth_grid_element_length) y $(last(sp.agent_pos_y) / m.smooth_grid_element_length)"
    @info "obs smooth map coords       $(o.geophysical_obs.smooth_map_coordinates)"
    @info "bp geostats base map coords $(bp.geostats.geophysical_data.base_map_coordinates)"
    
    if i % 4 == 0
        p, _, _, _ = plot_volume(m, bp, r_massive, t=i)
        display(p)
        empty!(p)
        
        p = plot(bp, m, sp)
        display(p)
        empty!(p)

        p = plot_smooth_map_and_plane_trajectory(sp, m)
        display(p)
        empty!(p)
    end
    
    b = bp
    s = sp
end