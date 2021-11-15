using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Plots
using ParticleFilters
using Statistics

using MineralExploration

N_INITIAL = 0
MAX_BORES = 10

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=2)
initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate_distribution(m)

g = GeoStatsDistribution(m)

up = MEBeliefUpdater(m, g, 1000, 2.0, 1)
println("Initializing belief...")
b0 = POMDPs.initialize_belief(up, ds0)
println("Belief Initialized!")

next_action = NextActionSampler()

solver = POMCPOWSolver(tree_queries=1000,
                       check_repeat_obs=true,
                       check_repeat_act=true,
                       next_action=next_action,
                       k_action=2.0,
                       alpha_action=0.25,
                       k_observation=2.0,
                       alpha_observation=0.25,
                       criterion=POMCPOW.MaxUCB(20.0),
                       final_criterion=POMCPOW.MaxQ(),
                       # final_criterion=POMCPOW.MaxTries(),
                       # estimate_value=0.0
                       estimate_value=leaf_estimation
                       )
planner = POMDPs.solve(solver, m)

N = 100
# rs = RolloutSimulator(max_steps=MAX_BORES+5)
hr = HistoryRecorder(max_steps=MAX_BORES+5)
V = Float64[]
ores = Float64[]
D = Int64[]
A = Symbol[]
println("Starting simulations")
for i in 1:N
    if (i%1) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    s_massive = s0.ore_map .>= m.massive_threshold
    r_massive = sum(s_massive)
    println("Massive Ore: $r_massive")
    h = simulate(hr, m, planner, up, b0, s0)
    v = 0.0
    d = 0
    for stp in h
        v += POMDPs.discount(m)^(stp[:t] - 1)*stp[:r]
        if stp[:a].type == :drill
            d += 1
        end
    end
    push!(D, d)
    push!(A, h[end][:a].type)

    println("Steps: $(length(h))")
    println("Decision: $(h[end][:a].type)")
    println("======================")
    push!(ores, r_massive)
    push!(V, v)
end
# mean_v = mean(V)
# se_v = std(V)/sqrt(N)
# println("Discounted Return: $mean_v ± $se_v")

abandoned = [a == :abandon for a in A]
mined = [a == :mine for a in A]

profitable = ores .> m.extraction_cost
lossy = ores .<= m.extraction_cost

n_profitable = sum(profitable)
n_lossy = sum(lossy)

profitable_mined = sum(mined.*profitable)
profitable_abandoned = sum(abandoned.*profitable)

lossy_mined = sum(mined.*lossy)
lossy_abandoned = sum(abandoned.*lossy)

mined_profit = sum(mined.*(ores .- m.extraction_cost))
available_profit = sum(profitable.*(ores .- m.extraction_cost))

mean_drills = mean(D)
mined_drills = sum(D.*mined)/sum(mined)
abandoned_drills = sum(D.*abandoned)/sum(abandoned)

println("Available Profit: $available_profit, Mined Profit: $mined_profit")
println("Profitable: $(sum(profitable)), Mined: $profitable_mined, Abandoned: $profitable_abandoned")
println("Lossy: $(sum(lossy)), Mined: $lossy_mined, Abandoned: $lossy_abandoned")
println("Mean Bores: $mean_drills, Mined Bores: $mined_drills, Abandon Bores: $abandoned_drills")
