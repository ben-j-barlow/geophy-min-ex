using MineralExploration
using POMDPs
using Statistics

GRID_DIMS = [(30, 30, 1),(50, 50, 1)]
N = 1000

OUTPUT = []

push!(OUTPUT, "=============================")
push!(OUTPUT, "Running standardisation tests")
push!(OUTPUT, "N = $N; extraction_cost = 63.0")

for dim in GRID_DIMS
    m = MineralExplorationPOMDP(
        grid_dim=dim,
        extraction_cost=63.0,
    )
    ds0 = POMDPs.initialstate(m)
   
    # get masses
    for scale in [true, false]
        ORES = Float64[]
        for i in 1:N
            s0 = rand(ds0, apply_scale=scale)
            s_massive = s0.ore_map .>= m.massive_threshold
            r_massive = sum(s_massive)
            push!(ORES, r_massive)
        end
        mn = round(mean(ORES), digits=2)
        std_dev = round(std(ORES), digits=2)
        push!(OUTPUT, "$dim with scale $scale: $mn ± $std_dev (extraction cost $(m.extraction_cost))")
    end
end

# push!(OUTPUT, "=============================")
# push!(OUTPUT, "Running standardisation tests")
# push!(OUTPUT, "N = $N; extraction_cost = 70.0")

# for dim in GRID_DIMS
#     m = MineralExplorationPOMDP(
#         grid_dim=dim,
#         extraction_cost=70.0,
#     )
#     ds0 = POMDPs.initialstate(m)
   
#     # get masses
#     for scale in [true, false]
#         ORES = Float64[]
#         for i in 1:N
#             s0 = rand(ds0, apply_scale=scale)
#             s_massive = s0.ore_map .>= m.massive_threshold
#             r_massive = sum(s_massive)
#             push!(ORES, r_massive)
#         end
#         mn = round(mean(ORES), digits=2)
#         std_dev = round(std(ORES), digits=2)
#         push!(OUTPUT, "$dim with scale $scale: $mn ± $std_dev (extraction cost $(m.extraction_cost))")
#     end
# end

for line in OUTPUT
    println(line)
end