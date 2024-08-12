using Revise
using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics
using MineralExploration

# Constants for the problem setup
GRID_DIMS = (30, 30, 1)
N_TRIALS = 1
NOISE_FOR_PERTURBATION = 0.3
N_PARTICLES = 1000
C_EXP = 100
MOVE_SIZE = 5
SIDESTEP_SIZE = 15
INIT_COORDS = CartesianIndex(1, 3) # (y, x)
EARLY_STOP = true

# pomdp
m = MineralExplorationPOMDP(
    c_exp=C_EXP
)

# setup
ds0 = POMDPs.initialstate(m)
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

# set up the solver
max_coord = m.grid_dim[1] * m.upscale_factor
policy = BaselineGeophysicalPolicy(m, max_coord, MOVE_SIZE, SIDESTEP_SIZE, INIT_COORDS, EARLY_STOP)

# more setup
hr = HistoryRecorder(max_steps=m.max_timesteps+3)  # Record history of the simulation
s0 = rand(ds0)  # Sample a starting state
s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
r_massive = sum(s_massive)  # Calculate total massive ore
println("Massive Ore: $r_massive")

# Simulate a sequence of actions and states
h = simulate(hr, m, policy, up, b0, s0);

v = 0.0  # Initialize the return
n_fly = 0  # Initialize the fly count

# Calculate the discounted return and count drills

for stp in h
    v += POMDPs.discount(m)^(stp[:t] - 1) * stp[:r]
    if stp[:a].type == :fake_fly
        n_fly += 1
    end
end

final = h[end][:a].type

errors = Float64[]  # Array to store errors
stds = Float64[]  # Array to store standard deviations

# Calculate errors and standard deviations for each step
for step in h
    b = step[:b]
    b_vol = [calc_massive(m, p) for p in b.particles]
    push!(errors, mean(b_vol .- r_massive))
    push!(stds, std(b_vol))
end

final_belief = h[end][:bp];
final_belief.acts[end].type

# plotting

# get trajectory
coords = [final_belief.acts[i].coords for i in 1:(length(final_belief.acts) - 2)];
x_path = [c[2] for c in coords];
y_path = [c[1] for c in coords];
x_to_plot = [ele - 0.5 for ele in x_path];
y_to_plot = [ele - 0.5 for ele in y_path];

# plot map and trajectory
p = plot_map(s0.smooth_map, "smooth map");
plot!(p, x_to_plot, y_to_plot, lw=2)


plot(final_belief)

using Plots
using PlotUtils
using Images

png_path = "/Users/benbarlow/dev/MineralExploration/img/plane.png"



