using Random
using LinearAlgebra
using Plots

const DEG_TO_RAD = π / 180

# Function to update positions and heading
function update_position(x, y, psi, phi, v, dt, g)
    psi_dot = g * tan(phi) / v
    x_dot = v * cos(psi)
    y_dot = v * sin(psi)
    psi += psi_dot * dt
    x += x_dot * dt
    y += y_dot * dt
    return x, y, psi
end

# Simulation loop
function run_and_plot_simulation(total_time, dt, v=20, g=9.81)
    # Initial conditions
    x, y, psi = 0.0, 0.0, 45  # initial positions and heading angle
    phi = 0.0  # initial bank angle

    # Lists to store trajectory
    x_list, y_list = [x], [y]

    for _ in 1:Int(total_time / dt)
        phi_change = rand([-5, 5]) * DEG_TO_RAD
        phi += phi_change
    
        x, y, psi = update_position(x, y, psi, phi, v, dt, g)
    
        push!(x_list, x)
        push!(y_list, y)
    end

    # Plotting the trajectory
    plot!(x_list, y_list, label="Trajectory", xlabel="X Position (meters)", ylabel="Y Position (meters)", 
        title="Aircraft Trajectory", legend=:topright, grid=true, axis=:equal)
end

# Example usage
run_and_plot_simulation(40, 1)
