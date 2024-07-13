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

    @show size(s0.ore_map)

    cmap=:viridis
    verbose=true
    ore_fig = plot_ore_map(s0.ore_map, cmap)
    ore_fig
    return s0.ore_map
end

s0_ore_map = generate_new_ore()


mass_fig, r_massive = plot_mass_map(s0.ore_map, m.massive_threshold, cmap; truth=true)
b0_fig = plot(b0; cmap=cmap)
b0_hist, vols, mean_vols, std_vols = plot_volume(m, b0, r_massive; t=0, verbose=verbose)

display(ore_fig)
display(mass_fig)
display(b0_fig)
display(b0_hist)



using Interpolations
using ImageFiltering

# Remove the third dimension since it is 1
s0_ore_map_2d = reshape(s0_ore_map, 50, 50)

# Increase the resolution using interpolation
interpolated_map = interpolate(s0_ore_map_2d, BSpline(Linear()))

# factor for upscale
factor = 10

# Define the new dimensions for the higher resolution grid
new_dims = (size(s0_ore_map_2d, 1) * factor, size(s0_ore_map_2d, 2) * factor)

# Generate the high resolution grid points
high_res_x = range(1, stop=size(s0_ore_map_2d, 1), length=new_dims[1])
high_res_y = range(1, stop=size(s0_ore_map_2d, 2), length=new_dims[2])

# Create a high resolution image by evaluating the interpolation at the new grid points
high_res_map_array = [interpolated_map(x, y) for x in high_res_x, y in high_res_y]

# Apply Gaussian filter to smooth the image
sigma = 1.0  # You can adjust the sigma value for more or less smoothing
smoothed_map = imfilter(high_res_map_array, Kernel.gaussian(sigma))

cmap=:viridis
plot_ore_map(s0_ore_map, cmap, "my_title")


MineralExploration.plot_ore_map