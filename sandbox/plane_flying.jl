using Random
using LinearAlgebra
using Plots

const DEG_TO_RAD = π / 180

# Function to update positions and heading
function update_position(x, y, psi, phi, v, dt, g)
    psi_dot = g * tan(phi) / v
    psi += psi_dot * dt
    
    x_dot = v * cos(psi)
    y_dot = v * sin(psi)

    x += x_dot * dt
    y += y_dot * dt
    return x, y, psi
end

# Simulation loop
function run_and_plot_simulation(total_time, dt, v=20, g=9.81)
    # Initial conditions
    x, y, psi = 0.0, 0.0, 45  # initial positions and heading angle
    phi = 0  # initial bank angle

    # Lists to store trajectory
    x_list, y_list = [x], [y]

    for _ in 1:Int(total_time / dt)
        phi_change = rand([-5, 5]) * DEG_TO_RAD
        phi += phi_change
        phi = -5 * DEG_TO_RAD

        a, b, heading = update_position(x, y, psi, phi, v, dt, g)
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
run_and_plot_simulation(140, 1)

a = [1,2,3]
a[end] = 4

last(a)




# PLANE FLYING SETUP

const DEG_TO_RAD = π / 180

init_x::Float64 = 0.0
init_y::Float64 = 0.0
init_heading::Float64 = 45
velocity::Int = 60
timestep_in_seconds::Int = 1
dt::Float64 = 0.1
#observations_per_timestep::Int = 1
bank_angle_history::Vector{Int} = [5, 5, 5, 5, 5,0,0,-5, -5,-5,-5,-5, -10,-10]


updates_per_timestep = timestep_in_seconds / dt
pos_x, pos_y, heading = init_x, init_y, init_heading

pos_x_outer_loop_history, pos_y_outer_loop_history = [], []
pos_x_history, pos_y_history = [init_x], [init_y]

function get_base_map_coordinates(x::Float64, y::Float64, m::MineralExplorationPOMDP)
    # use ceil() because plane at position (10.2, 12.7) in continuous scale should map to (11, 13) in discrete scale
    return convert(Int64, ceil(x / m.base_grid_element_length)), convert(Int64, ceil(y / m.base_grid_element_length))
end

function get_smooth_map_coordinates(x::Float64, y::Float64, m::MineralExplorationPOMDP)
    # use ceil() because plane at position (10.2, 12.7) in continuous scale should map to (11, 13) in discrete scale
    return convert(Int64, ceil(x / m.smooth_grid_element_length)), convert(Int64, ceil(y / m.smooth_grid_element_length))
end

for bank_angle in bank_angle_history
    for i in 1:updates_per_timestep
        pos_x, pos_y, heading = update_agent_state(pos_x, pos_y, heading, bank_angle * DEG_TO_RAD, velocity, dt)
        push!(pos_x_history, get_base_map_coordinates(pos_x, pos_y, m)[1])
        push!(pos_y_history, get_base_map_coordinates(pos_x, pos_y, m)[2])
    end
    push!(pos_x_outer_loop_history, deepcopy(pos_x_history))
    push!(pos_y_outer_loop_history, deepcopy(pos_y_history))
    pos_x_history, pos_y_history = [get_base_map_coordinates(pos_x, pos_y, m)[1]], [get_base_map_coordinates(pos_x, pos_y, m)[2]]
end


# PLOTTING 

using Plots

max_x_value = maximum(maximum.(pos_x_outer_loop_history))
max_y_value = maximum(maximum.(pos_y_outer_loop_history))
min_x_value = minimum(minimum.(pos_x_outer_loop_history))
min_y_value = minimum(minimum.(pos_y_outer_loop_history))
# get overall min
lim_min = minimum([min_x_value, min_y_value])
lim_max = maximum([max_x_value, max_y_value])


# aesthetics
lw = 3
color1, color2 = palette(:tab10)[1], palette(:tab10)[2]

for (i, _) in enumerate(bank_angle_history)
    color_i = mod(i, 2) == 0 ? color1 : color2
    if i == 1
        display(plot(pos_x_outer_loop_history[1], pos_y_outer_loop_history[1], xlabel="X Position (meters)", ylabel="Y Position (meters)", color=color_i, 
                title="Aircraft Trajectory", legend=:none, grid=true, xlim=(lim_min, lim_max), ylim=(lim_min, lim_max), axis=:equal, aspect_ratio=1, linewidth=lw))
    else
        display(plot!(pos_x_outer_loop_history[i], pos_y_outer_loop_history[i], linecolor=color_i, linewidth=lw))
    end
end
# get final pos per timestep

# REVERSE ENGINEERING

function cost_function(current_x, current_y, target_x, target_y)
    return sqrt((current_x - target_x)^2 + (current_y - target_y)^2)
end

using Optim

function optimize_path(init_x, init_y, init_heading, velocity, dt, target_coordinates)
    bank_angle_sequence = []
    pos_x, pos_y, heading = init_x, init_y, init_heading

    for (target_x, target_y) in target_coordinates
        result = optimize(bank_angle -> begin
            x, y, ψ = pos_x, pos_y, heading
            for i in 1:updates_per_timestep
                x, y, ψ = update_agent_state(x, y, ψ, bank_angle * DEG_TO_RAD, velocity, dt)
            end
            return cost_function(x, y, target_x, target_y)
        end, 0.0, -π/4, π/4)  # adjust the bounds for bank angles as needed

        best_bank_angle = result.minimizer
        push!(bank_angle_sequence, best_bank_angle)

        # Update the plane's position and heading with the best bank angle
        for i in 1:updates_per_timestep
            pos_x, pos_y, heading = update_agent_state(pos_x, pos_y, heading, best_bank_angle * DEG_TO_RAD, velocity, dt)
        end
    end

    return bank_angle_sequence
end

bank_angles = optimize_path(init_x, init_y, init_heading, velocity, dt, target_coordinates)
println("Optimal bank angles: ", bank_angles)
