using ImageFiltering
using Plots
using OffsetArrays

# Create the Gaussian kernel with sigma = 3
sigma = 3.0
gaussian_kernel = Kernel.gaussian(sigma)


# Extract the x and y ranges from the offset indices
x_range = axes(gaussian_kernel, 1)
y_range = axes(gaussian_kernel, 2)

# Plot the 2D Gaussian kernel using heatmap
heatmap(x_range, y_range, gaussian_kernel, xlabel="X index", ylabel="Y index", title="13x13 2D Offset Array (Gaussian Filter)", color=:viridis)

heatmap(gaussian_kernel, xlabel="X index", ylabel="Y index", title="13x13 2D Offset Array (Gaussian Filter)", color=:viridis)

# Get the x and y coordinates from the kernel's indices
x = range(first(axes(gaussian_kernel, 1)), stop=last(axes(gaussian_kernel, 1)), length=length(gaussian_kernel))
y = range(first(axes(gaussian_kernel, 2)), stop=last(axes(gaussian_kernel, 2)), length=length(gaussian_kernel))

# Plot the Gaussian kernel
plot(x, y, st=:stem, marker=:circle, label="Gaussian Kernel (σ = $sigma)", xlabel="Index", ylabel="Value")
title!("Gaussian Filter from ImageFiltering.jl (σ = 3)")