using POMCPOW
using POMDPs
using Plots
using MineralExploration
using POMDPModelTools
using Random
using DelimitedFiles
using LinearAlgebra
using D3Trees

include("../helpers.jl")


save_dir = "./data/step_by_step/"


if isdir(save_dir)
    rm(save_dir; recursive=true)
    #error("directory already exists")
end
mkdir(save_dir)

N_PARTICLES = 1000
NOISE_FOR_PERTURBATION = 0.1
# noise
# 10 - belief didnt really converge

C_EXP = 125
GET_TREES = true

SEED = 42


#io = MineralExploration.prepare_logger()

grid_dims = (30, 30, 1)

m = MineralExplorationPOMDP(
    grid_dim=grid_dims,
    upscale_factor=3,
    c_exp=C_EXP,
    sigma=5,
    mainbody_gen=BlobNode(grid_dims=grid_dims),
    true_mainbody_gen=BlobNode(grid_dims=grid_dims),
    geophysical_noise_std_dev=0.0,
    observations_per_timestep=1,
    timestep_in_seconds=1,
    init_pos_x=2.0,
    init_pos_y=-5.0,
    init_heading=HEAD_NORTH,
    bank_angle_intervals=15,
    max_bank_angle=55,
    velocity=25,
    base_grid_element_length=25.0,
    extraction_cost=70.0,
    extraction_lcb = 0.7,
    extraction_ucb = 0.7,
    out_of_bounds_cost=0.0,
)

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

#Random.seed!(SEED)
#seed = 4293 #convert(Int64, floor(rand() * 10000))
#Random.seed!(seed)  # Reseed with a new random seed
s0 = rand(ds0; truth=true) #Checked

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
b0 = POMDPs.initialize_belief(up, ds0) #Checked

solver = POMCPOWSolver(
    tree_queries=10000,
    k_observation=2.0,
    alpha_observation=0.3,
    max_depth=3,
    check_repeat_obs=false,
    check_repeat_act=true,
    enable_action_pw=false,
    #next_action=nothing,
    #alpha_action=nothing,
    #k_action=nothing,
    criterion=POMCPOW.MaxUCB(m.c_exp),
    final_criterion=POMCPOW.MaxQ(),
    estimate_value=geophysical_leaf_estimation,
    tree_in_info=GET_TREES,
)
planner = POMDPs.solve(solver, m)

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
display(p)
empty!(p)

# initial volume
p, _, mn, std = plot_volume(m, b0, r_massive, t=0, verbose=false)
@info "Vols at time 0: $mn ± $std"
savefig(p, string(save_dir, "0volume.png"))
empty!(p)

path = string(save_dir, "belief_mean.txt")
open(path, "w") do io
    println(io, "Vols at time 0: $(mn) ± $(std)")
end

trees = []

beliefs = []
states = []

bank_angle = 0
T = 100
for i in 1:T
    a = nothing
    ai = nothing
    try
        a, ai = action_info(planner, b, tree_in_info=GET_TREES);
        if a.type != :fly
            @info "$(a.type) at $i"
        end
    catch e
        @info "error was caught"
        inbrowser(D3Tree(tree, init_expand=1), "safari")
        break
    end
    if a.type == :fly
        bank_angle += a.change_in_bank_angle
        @info "Bank angle at $i: $bank_angle"
    end
    if GET_TREES
        tree = deepcopy(ai[:tree]);
        #if i % 1 == 0 # && i >= 9
            #push!(trees, tree);
            #inbrowser(D3Tree(tree, init_expand=1), "safari")
        #end    
    end
    out = gen(m, s, a, m.rng);
    o = out[:o];
    sp = out[:sp];
    bp, ui = update_info(up, b, a, o);
    
    # @info ""
    # @info "timestep $i"
    # @info "b geostats base map coords  $(b.geostats.geophysical_data.base_map_coordinates)"
    # @info "s norm (base) agent pos     x $(last(sp.agent_pos_x) / m.base_grid_element_length) y $(last(sp.agent_pos_y) / m.base_grid_element_length)"
    # @info "obs base map coords         $(o.geophysical_obs.base_map_coordinates)"
    # @info "s norm (smooth) agent pos   x $(last(sp.agent_pos_x) / m.smooth_grid_element_length) y $(last(sp.agent_pos_y) / m.smooth_grid_element_length)"
    # @info "obs smooth map coords       $(o.geophysical_obs.smooth_map_coordinates)"
    # @info "bp geostats base map coords $(bp.geostats.geophysical_data.base_map_coordinates)"
    
    if i % 5 == 0
        p, _, mn, std = plot_volume(m, bp, r_massive, t=i, verbose=false)
        @info "Vols at time $i: $mn ± $std"
        savefig(p, string(save_dir, "$(i)volume.png"))
        #display(p)
        empty!(p)
        
        p = plot(bp, m, sp, t=i)
        savefig(p, string(save_dir, "$(i)b.png"))
        #display(p)
        empty!(p)

        p = plot_base_map_and_plane_trajectory(sp, m, t=i)
        savefig(p, string(save_dir, "$(i)trajectory.png"))
        display(p)
        empty!(p)

        path = string(save_dir, "belief_mean.txt")
        open(path, "a") do io
            println(io, "Vols at time $i: $(mn) ± $(std)")
        end
    end
    
    b = bp
    s = sp

    push!(beliefs, b)
    push!(states, s)

    if b.decided 
        break
    end
end


#for i in 2:(T+1)
#    @info "at time $(i-1) plane is $(check_plane_within_region(m, s.agent_pos_x[i], s.agent_pos_y[i])) in region with (x,y) coord ($(s.agent_pos_x[i]),$(s.agent_pos_y[i]))"
#end]
#last(b.agent_bank_angle)


s.agent_bank_angle
b.agent_bank_angle
isequal(b.particles[1].agent_bank_angle, s.agent_bank_angle)