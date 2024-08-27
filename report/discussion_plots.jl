# Import necessary packages
#using Revise
using POMDPs
#using POMDPSimulators
#using POMCPOW
using Plots
#using ParticleFilters
#using Statistics
using MineralExploration
using Random

NAME = "deeper_planning_"

SEED = 4804
save_dir = "/Users/benbarlow/dev/diss-plots/help_plots/discussion/$NAME"
DPI = 300

#NOISE_FOR_PERTURBATION = 0.8
#N_PARTICLES = 1000
#C_EXP = 125.0
#GET_TREES = false

# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    # DO NOT CHANGE - same as baseline
    upscale_factor=4,
    sigma=2,
    geophysical_noise_std_dev=0.005,
    ### 
    grid_dim=(48,48,1),
    init_pos_x=20*25.0,
    init_pos_y=25*25.0,
    init_heading=HEAD_EAST,
    max_bank_angle=55,
    bank_angle_intervals=18,
    max_timesteps=250,
    velocity=40
)

Random.seed!(SEED)
ds0 = POMDPs.initialstate(m)
s0 = rand(ds0)
#up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
#b0 = POMDPs.initialize_belief(up, ds0)

#s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
#r_massive = sum(s_massive)  # Calculate total massive ore

a_left = MEAction(type=:fly, change_in_bank_angle=m.bank_angle_intervals)
a_right = MEAction(type=:fly, change_in_bank_angle=-m.bank_angle_intervals)
a_straight = MEAction(type=:fly, change_in_bank_angle=0)

sp_l = s0
sp_r = s0
sp_s = s0
rnd = Random.GLOBAL_RNG

for i in 1:3
    sp_l = POMDPs.gen(m,sp_l,a_left, rnd)[:sp];
    sp_r = POMDPs.gen(m,sp_r,a_right, rnd)[:sp];
    sp_s = POMDPs.gen(m,sp_s,a_straight, rnd)[:sp];
end

p = plot_map(s0.smooth_map, "");

x, y = normalize_agent_coordinates(sp_l.agent_pos_x, sp_l.agent_pos_y, m.smooth_grid_element_length)
add_agent_trajectory_to_plot!(p, x, y, add_start=false)

x, y = normalize_agent_coordinates(sp_r.agent_pos_x, sp_r.agent_pos_y, m.smooth_grid_element_length)
add_agent_trajectory_to_plot!(p, x, y, add_start=false)

x, y = normalize_agent_coordinates(sp_s.agent_pos_x, sp_s.agent_pos_y, m.smooth_grid_element_length)
add_agent_trajectory_to_plot!(p, x, y, add_start=false)

p


for (sp, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "sp,a,r,bp,t", rng=m.rng)        
    if a.type == :fly
        n_fly += 1

    end
    v += POMDPs.discount(m)^(t - 1) * r

    b_vol = [calc_massive(m, p) for p in bp.particles]
    push!(mns, mean(b_vol))
    push!(stds, Statistics.std(b_vol))

    final_belief = bp
    final_state = sp

    b_hist, vols, mn, std = plot_volume(m, bp, r_massive; t=t, verbose=false)
    push!(mns, mn)
    push!(stds, std)

    if t % 10 == 0 || final_belief.decided
        b_mn, b_std = plot(bp, return_individual=true)
        map_and_plane = plot_smooth_map_and_plane_trajectory(sp, m, t=t)

        if isa(save_dir, String)
            path = string(save_dir, "$(t)b.png")
            plot!(b_mn, dpi=DPI)
            savefig(b_mn, path)

            path = string(save_dir, "$(t)bstd.png")
            plot!(b_std, dpi=DPI)
            savefig(b_std, path)

            path = string(save_dir, "$(t)volume.png")
            plot!(b_hist, dpi=DPI)
            savefig(b_hist, path)

            path = string(save_dir, "$(t)trajectory.png")
            plot!(map_and_plane, dpi=DPI)
            savefig(map_and_plane, path)

            path = string(save_dir, "belief_mean.txt")
            open(path, "a") do io
                println(io, "Vols at time $t: $(mn) ± $(std)")
            end
        end
    end
end

open(string(save_dir, "trial_$(SEED)_info.txt"), "w") do io
    println(io, "Seed: $SEED")
    println(io, "Mean: $(last(mns))")
    println(io, "Std: $(last(stds))")
    println(io, "Massive: $r_massive")
    println(io, "Extraction cost: $(m.extraction_cost)")
    println(io, "Decision: $(final_belief.acts[end].type)")
    println(io, "Fly: $n_fly")
    println(io, "Reward: NA")
end