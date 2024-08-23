using Random
using MineralExploration
using POMDPs
using Statistics

seeds = get_all_seeds()
GRID_DIM = (48,48,1)

ORE_MAP_VOL = Float64[]
ORE_TH = 0.7
for seed in seeds[1:1]

    Random.seed!(seed)
    m = MineralExplorationPOMDP(
        grid_dim=GRID_DIM,
        mainbody_gen=CircleNode(grid_dims=GRID_DIM)
    )
    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0, apply_scale=false)

    s_massive = s0.ore_map .>= ORE_TH
    r_massive = sum(s_massive)
    push!(ORE_MAP_VOL, r_massive)

    display(plot_map(s0.mainbody_map,""))
    display(plot_map(s0.ore_map,""))
    
    display(plot_map(s0.smooth_map,""))
end


MAIN_MAP_VOL = Float64[]
MAIN_TH = 0.46
for seed in seeds[1:150]
    Random.seed!(seed)
    m = MineralExplorationPOMDP(
        grid_dim=GRID_DIM,
    )
    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0, apply_scale=false)

    s_massive = s0.mainbody_map .>= MAIN_TH
    r_massive = sum(s_massive)
    push!(MAIN_MAP_VOL, r_massive) 
end

# plot a histogram of the each list, narrow bin width
using Plots
BINS = 0:25:350
histogram(ORE_MAP_VOL, label="Ore Map", bins = BINS)
histogram(MAIN_MAP_VOL, label="Main Body Map", bins = BINS)

# count # > 150
length(ORE_MAP_VOL)
ORE_COUNT = sum(ORE_MAP_VOL .> 150)
length(MAIN_MAP_VOL)
MAIN_COUNT = sum(MAIN_MAP_VOL .> 150)


    # get masses
    #for scale in [true, false]
    #    ORES = Float64[]
    #    for i in 1:N
    #        s_massive = s0.ore_map .>= m.massive_threshold
            
#    push!(ORES, r_massive)
#end
#mn = round(mean(ORES), digits=2)
#std_dev = round(std(ORES), digits=2)
#push!(OUTPUT, "$dim with scale $scale: $mn ± $std_dev (extraction cost $(m.extraction_cost))")
#end

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

#for line in OUTPUT
#    println(line)
#end