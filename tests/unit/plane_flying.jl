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
    x, y, psi = 30.0, 0.0, 0.0  # initial positions and heading angle
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
    p = plot(y_list, x_list, label="Trajectory", xlabel="X Position (meters)", ylabel="Y Position (meters)", 
        title="Aircraft Trajectory", legend=:topright, grid=true, axis=:equal, aspect_ratio=1)
    return x_list, y_list, p    
end

# Example usage
x_out, y_out, p1 = run_and_plot_simulation(80, 1.0);
p1

a = (230, 10)
plot!(p1, [a[2]], [a[1]], seriestype = :scatter, label="Target", color="red")



using POMDPs
using Plots
using MineralExploration
using Random

m = MineralExplorationPOMDP();
ds0 = POMDPs.initialstate(m);
Random.seed!(100)
s0 = rand(ds0);

init_x = 25.0
init_y = 5.0
init_heading = 90.0 * DEG_TO_RAD
velocity = 20

x,y,heading = update_agent_state(init_x, init_y, init_heading, 0.0, velocity)

x_list = [init_x, x]
y_list = [init_y, y]

p = plot_map(s0.ore_map, "ore")
plot!(p, x_list, y_list, seriestype = :scatter, label="Plot by hand", color="red")
add_agent_trajectory_to_plot!(p, x_list, y_list)

