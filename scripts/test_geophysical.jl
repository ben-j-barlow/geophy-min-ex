using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using POMDPModelTools
using ParticleFilters
using Statistics
using Random
using Dates

using MineralExploration

C_EXP = 2

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 2.0

# define max number of timesteps here
hr = HistoryRecorder(max_steps=10)

# prepare POMCPOW
solver = POMCPOWSolver(
    tree_queries=4000,
    k_observation=2.0,
    alpha_observation=0.1,
    max_depth=5,
    check_repeat_obs=true,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    criterion=POMCPOW.MaxUCB(C_EXP),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=leaf_estimation,
    tree_in_info=true,
)

START_X = [30.0, 30.0, 25.1*40.1, 25.1*25.1]
START_Y = [30.0, 25.1*50.1-30, 25.1*40.1, 25,1*25.1]
INIT_HEADING = [HEAD_NORTH, HEAD_SOUTH, HEAD_EAST, HEAD_WEST]

path_map = []
volumes = []

start_time = now();
h = nothing
for i in 1:4
    @info "iteration $i"
    m = MineralExplorationPOMDP(
        init_heading=INIT_HEADING[i],
        init_pos_x=START_X[i],
        init_pos_y=START_Y[i],
        out_of_bounds_cost=50,
        geophysical_noise_std_dev=convert(Float64, 0.0),
        observations_per_timestep=1,
        timestep_in_seconds=1,
        bank_angle_intervals=10,
        max_bank_angle=55,
        velocity=25,
        base_grid_element_length=25.0
    )
    up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
    ds0 = POMDPs.initialstate(m)
    b0 = POMDPs.initialize_belief(up, ds0); #Checked

    Random.seed!(42)
    s0 = rand(ds0);

    planner = POMDPs.solve(solver, m)
    h = simulate(hr, m, planner, up, b0, s0) #ds0 instead of b0?

    push!(path_map, plot_smooth_map_and_plane_trajectory(h[end][:sp], m))

    r_massive = sum(s0.ore_map .>= m.massive_threshold)
    hist_start, _, _, _ = plot_volume(m, b0, r_massive; t="start")
    hist_end, _, _, _ = plot_volume(m, h[end][:bp], r_massive; t="end")
    push!(volumes, hist_start)
    push!(volumes, hist_end)
end
end_time = now();
elapsed_time = end_time - start_time;

#plot(volumes..., layout=(2, 2), size=(800, 800))


#plot(path_map[1], path_map[2], path_map[3], path_map[4], layout=(2, 2), size=(800, 800))