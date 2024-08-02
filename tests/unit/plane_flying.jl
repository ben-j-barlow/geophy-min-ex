using Random
using LinearAlgebra
using Plots
using MineralExploration

const DEG_TO_RAD = π / 180

# PROOF UPDATE AGENT STATE WORKS
# X and Y SWAPPED IN plot(y_list, x_list, ...)


# Simulation loop
function run_and_plot_simulation(total_time, dt, v=20, g=9.81)
    # Initial conditions
    x, y, psi = 0.0, 0.0, 45.0  # initial positions and heading angle
    phi = 0  # initial bank angle

    # Lists to store trajectory
    x_list, y_list = [x], [y]

    for _ in 1:Int(total_time / dt)
        phi_change = rand([-5, 5]) * DEG_TO_RAD
        phi += phi_change
        phi = -5 * DEG_TO_RAD

        a, b, heading = update_agent_state(x, y, psi, phi, v, dt, g)
        push!(x_list, deepcopy(a))
        push!(y_list, deepcopy(b))

        x = deepcopy(a)
        y = deepcopy(b)
        psi = deepcopy(heading)
    end

    # Plotting the trajectory
    plot(y_list, x_list, label="Trajectory", xlabel="X Position (meters)", ylabel="Y Position (meters)", 
        title="Aircraft Trajectory", legend=:topright, grid=true, axis=:equal, aspect_ratio=1)
end

# Example usage
run_and_plot_simulation(140, 1.0)
