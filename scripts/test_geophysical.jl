# Import necessary packages
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
N_TRIALS = 10
NOISE_FOR_PERTURBATION = 0.3
N_PARTICLES = 1000
C_EXP = 100


# Create a POMDP model for mineral exploration with specified parameters
m = MineralExplorationPOMDP(
    c_exp=C_EXP
)

# Initialize the state distribution for the POMDP model
ds0 = POMDPs.initialstate(m)

# Define the belief updater for the POMDP with 1000 particles and a spread factor of 2.0
up = MEBeliefUpdater(m, N_PARTICLES, NOISE_FOR_PERTURBATION)
b0 = POMDPs.initialize_belief(up, ds0)

# Set up the POMCPOW solver with specific parameters
solver = get_geophysical_solver(C_EXP)

# Solve the POMDP problem using the defined solver
planner = POMDPs.solve(solver, m)

# Initialize data storage and simulator for the results
# rs = RolloutSimulator(max_steps=MAX_BORES+5)
hr = HistoryRecorder(max_steps=m.max_timesteps+3)  # Record history of the simulation
DIS_RETURN = Float64[]  # Array to store discounted returns
ORES = Float64[]  # Array to store ore values
N_FLY = Int64[]  # Array to store the number of drills used
FINAL_ACTION = Symbol[]  # Array to store the final action taken
ME = Vector{Float64}[]  # Array to store mean errors
STD = Vector{Float64}[]  # Array to store standard deviations
println("Starting simulations")

# Run the simulation for N trials
for i in 1:N_TRIALS
    if (i % 1) == 0
        println("Trial $i")
    end
    s0 = rand(ds0, apply_scale=true)  # Sample a starting state
    s_massive = s0.ore_map .>= m.massive_threshold  # Identify massive ore locations
    @assert m.dim_scale == 1
    r_massive = sum(s_massive)  # Calculate total massive ore
    println("Massive Ore: $r_massive")
    
    # Simulate a sequence of actions and states
    h = simulate(hr, m, planner, up, b0, s0)
    
    v = 0.0  # Initialize the return
    n_fly = 0  # Initialize the fly count
    
    # Calculate the discounted return and count drills
    for stp in h
        v += POMDPs.discount(m)^(stp[:t] - 1) * stp[:r]
        if stp[:a].type == :fly
            n_fly += 1
        end
    end
    
    # Store the results
    push!(N_FLY, n_fly)
    push!(FINAL_ACTION, h[end][:a].type)
    
    errors = Float64[]  # Array to store errors
    stds = Float64[]  # Array to store standard deviations
    
    # Calculate errors and standard deviations for each step
    for step in h
        bp = step[:bp]
        b_vol = [calc_massive(m, p) for p in bp.particles]
        push!(errors, mean(b_vol .- r_massive))
        push!(stds, std(b_vol))
    end
    
    # Store errors and standard deviations
    push!(ME, errors)
    push!(STD, stds)
    
    println("Steps: $(length(h))")
    println("Decision: $(h[end][:a].type)")
    println("======================")
    
    # Store ore value and discounted return
    push!(ORES, r_massive)
    push!(DIS_RETURN, v)

    GC.gc()
end

# Post-processing: Calculate profits and statistics based on simulation results

# mean_v = mean(V)
# se_v = std(V) / sqrt(N)
# println("Discounted Return: $mean_v ± $se_v")

abandoned = [a == :abandon for a in FINAL_ACTION]
mined = [a == :mine for a in FINAL_ACTION]

profitable = ORES .> m.extraction_cost
lossy = ORES .<= m.extraction_cost

n_profitable = sum(profitable)
n_lossy = sum(lossy)

profitable_mined = sum(mined .* profitable)
profitable_abandoned = sum(abandoned .* profitable)

lossy_mined = sum(mined .* lossy)
lossy_abandoned = sum(abandoned .* lossy)

mined_profit = sum(mined .* (ORES .- m.extraction_cost))
available_profit = sum(profitable .* (ORES .- m.extraction_cost))

mean_flys = mean(N_FLY)
mined_flys = sum(N_FLY .* mined) / sum(mined)
abandoned_flys = sum(N_FLY .* abandoned) / sum(abandoned)

# Print the final statistics and results
println("Available Profit: $available_profit, Mined Profit: $mined_profit")
println("Profitable: $(sum(profitable)), Mined: $profitable_mined, Abandoned: $profitable_abandoned")
println("Lossy: $(sum(lossy)), Mined: $lossy_mined, Abandoned: $lossy_abandoned")
println("Mean Bores: $mean_flys, Mined Flys: $mined_flys, Abandon Flys: $abandoned_flys")
