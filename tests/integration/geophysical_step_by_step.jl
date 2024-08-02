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
    init_heading=convert(Float64, 0),
    init_pos_x=0,
    init_pos_y=0,
    velocity=40,
    out_of_bounds_cost=0.5,
    geophysical_noise_std_dev=convert(Float64, 0.0),
    observations_per_timestep=1,
    timestep_in_seconds=5
)

up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION) #Checked
ds0 = POMDPs.initialstate(m)

b0 = POMDPs.initialize_belief(up, ds0) #Checked
s0 = rand(ds0);
s0_copy = deepcopy(s0)
b0_copy = deepcopy(b0)


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
planner = POMDPs.solve(solver, m)


# action info : inspect b0 vs copied version

a = MEAction(type=:fly, change_in_bank_angle=0)
a, ai = action_info(planner, b0);
a_copy = deepcopy(a)
typeof(b0)
mylog(b0, b0_copy)

a
println("Mismatched properties: " * join([p for p in propertynames(b0) if !isequal(getproperty(b0, p), getproperty(b0_copy, p))], ", "))

p500_current = b0.particles[500];
p500_original = b0_copy.particles[500];
println("Mismatched properties: " * join([p for p in propertynames(p500_current) if !isequal(getproperty(p500_current, p), getproperty(p500_original, p))], ", "))
p500_current.geophysical_obs
p500_original.geophysical_obs
a
plot_map(b0.particles[500].ore_map, "particle 500 current")
plot_map(b0_copy.particles[500].ore_map, "particle 500 original")
plot_map(b0.particles[500].smooth_map, "particle 500 current")
plot_map(b0_copy.particles[500].smooth_map, "particle 500 original")


# run gen : inspect s0 and a vs copied version
out = gen(m, s0, a, m.rng);
o = out[:o];
s1 = out[:sp];
r = out[:r];

o_copy = deepcopy(o);
s1_copy = deepcopy(s1);

println("Mismatched properties: " * join([p for p in propertynames(s0) if !isequal(getproperty(s0, p), getproperty(s0_copy, p))], ", "))
s0.geophysical_obs
#s0_copy.geophysical_obs
# I HAVE IMPLEMENTED THE ISEQUAL FUNCTION FOR GeophysicalObservations BUT I HAVE NOT TESTED IT

x, y = o.agent_pos_x, o.agent_pos_y;
@assert last(s1.agent_pos_x) == x
@assert last(s1.agent_pos_y) == y
x
s1.agent_pos_x
s1.agent_pos_y
o.geophysical_obs.smooth_map_coordinates

# run update : inspect b0, a, o vs copied version
bp, ui = update_info(up, b0, a, o)

plot(bp, m)
x, y = get_agent_trajectory(bp.agent_bank_angle, m);


function pad_line(description::String, original::String, current::String, pad_length::Int)
    pad = " " ^ (pad_length - length(description))
    println(description * pad * " | " * original * " | " * current)
end

function mylog(current::MEBelief{GeoStatsDistribution}, original::MEBelief{GeoStatsDistribution})
    #println("Beliefs equal: $(isequal(current, original))")
    #println("Mismatched properties: " * join([p for p in propertynames(current) if !isequal(getproperty(current, p), getproperty(original, p))], ", "))
    pad_line("obs base map coords", string(original.geophysical_obs.base_map_coordinates), string(current.geophysical_obs.base_map_coordinates), 20)
    pad_line("bank angle", string(original.agent_bank_angle), string(current.agent_bank_angle), 20)
    println("geostats distributions are equal: $(isequal(current.geostats, original.geostats))")
end

function log(current::MEState, original::MEState)
    pad_line("heading", string(original.agent_heading), string(current.agent_heading), 20)
    pad_line("pos x", string(original.agent_pos_x), string(current.agent_pos_x), 20)
    pad_line("pos y", string(original.agent_pos_y), string(current.agent_pos_y), 20)
end