using Revise

using POMDPs
using POMCPOW
using Plots
using Statistics

using MineralExploration

N_INITIAL = 0
MAX_BORES = 25
MIN_BORES = 10
GRID_SPACING = 1
MAX_MOVEMENT = 10
SAVE_DIR = "./data/demos/two_constrained_demo/"

function generate_new_ore()
    mainbody = MultiVarNode()
    #mainbody = SingleFixedNode()
    # mainbody = SingleVarNode()

    m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                                mainbody_gen=mainbody, max_movement=MAX_MOVEMENT, min_bores=MIN_BORES)
    initialize_data!(m, N_INITIAL)

    ds0 = POMDPs.initialstate(m)
    s0 = rand(ds0)

    up = MEBeliefUpdater(m, 1000, 2.0)
    b0 = POMDPs.initialize_belief(up, ds0)


    cmap=:viridis
    verbose=true
    ore_fig = plot_ore_map(s0.ore_map, cmap)

end

generate_new_ore()

mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true)
b0_fig = plot(b0; cmap=cmap)
b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose)

display(ore_fig)
display(mass_fig)
display(b0_fig)
display(b0_hist)
