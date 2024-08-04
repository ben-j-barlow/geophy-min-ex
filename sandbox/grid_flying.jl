using Printf
using Plots

const DEG_TO_RAD = π / 180

const HEAD_EAST = 0.0
const HEAD_NORTH = 90 * DEG_TO_RAD
const HEAD_WEST = 180 * DEG_TO_RAD
const HEAD_SOUTH = 270 * DEG_TO_RAD

#tan(HEAD_EAST)
#tan(HEAD_SOUTH)
#tan(HEAD_NORTH)

function plot_path(pos_x, pos_y)
    # Plotting the path
    normalized_x = pos_x ./ cell_length
    normalized_y = pos_y ./ cell_length

    plot(normalized_x, normalized_y, seriestype = :path, lw = 2, label = "Plane Path")
    xlims!(-3, 53)
    ylims!(-3, 53)
    hline!([0, 50], lw = 1, color = :red, label = "Grid Border")
    vline!([0, 50], lw = 1, color = :red, label = "")

    xlabel!("Grid X Coordinate")
    ylabel!("Grid Y Coordinate")
    title!("Plane Path on Grid")
    #legend(:topright)
end

# Function to update the plane's state
function update_agent_state(x::Float64, y::Float64, psi::Float64, phi::Float64, v::Int, dt::Float64 = 1.0, g::Float64 = 9.81)
    if dt != 1.0
        error("stick to dt = 1 for now")
    end
    # normalize to 0 to 2pi
    psi = normalize_angle(psi)

    psi_dot = g * tan(phi) / v
    to_return_psi = psi + (psi_dot * dt)
    x_dot = v * cos(to_return_psi)
    y_dot = v * sin(to_return_psi)
    to_return_x = x + (x_dot * dt)
    to_return_y = y + (y_dot * dt)
    return to_return_x, to_return_y, to_return_psi
end

# Constants
grid_size = 50
cell_length = 25.0  # meters
v = 25  # m/s
time_step = 1.0  # second
g = 9.81  # m/s^2
turn_angle = 40 * DEG_TO_RAD  # radians, equivalent to 10 degrees

# Initialize state at the center of the first cell
x, y = cell_length / 2, 0.0
psi = HEAD_NORTH
phi = 0.0

# History of positions for plotting
positions_x = [x]
positions_y = [y]

# Function to fly straight for a number of timesteps
function fly_straight(x, y, psi, phi, v, timesteps)
    for t in 1:timesteps
        x, y, psi = update_agent_state(x, y, psi, phi, v)
        push!(positions_x, x)
        push!(positions_y, y)
    end
    return x, y, psi
end

function compute_bank_angle_for_final_turn_timestep(current_heading::Float64, target_heading::Float64, v::Float64, dt::Float64 = 1.0, g::Float64 = 9.81)
    # Calculate the required change in heading per time step
    psi_dot = (target_heading - current_heading) / dt
    
    # Calculate the required bank angle
    phi = atan(psi_dot * v / g)
    
    return phi
end

function normalize_angle(angle::Float64)
    return mod(angle, 2 * π)
end

function check_turn_too_far(psi, psi_dot, target_heading)
    candidate_psi = psi + psi_dot
    if candidate_psi < target_heading && target_heading == HEAD_EAST && 
        return true
    elseif normalize_angle(candidate_psi) < target_heading && target_heading == HEAD_SOUTH
        println("candidate_psi: ", candidate_psi / DEG_TO_RAD, " target_heading: ", target_heading / DEG_TO_RAD)
        return true
    else
        return false
    end
end

function check_sign_difference(psi::Float64, psi_dot::Float64, target_heading::Float64)
    # Calculate the expressions
    expr1 = psi - target_heading
    expr2 = psi + psi_dot - target_heading
    
    # Check if their signs are different
    if sign(expr1) != sign(expr2)
        return true
    else
        return false
    end
end

# Function to turn the plane
function turn_plane(x, y, psi, phi, v, angle, target_heading, max_iterations=100)
    iteration = 0
    phi = angle
    psi_dot = g * tan(phi) / v
    println("psi_dot: ", psi_dot / DEG_TO_RAD, "psi: ", psi / DEG_TO_RAD, "target_heading: ", target_heading / DEG_TO_RAD)
    while !check_sign_difference(psi, psi_dot, target_heading)
        iteration += 1
        if iteration > max_iterations
            println("Max iterations reached, exiting turn loop")
            break
        end
        x, y, psi = update_agent_state(x, y, psi, phi, v)
        push!(positions_x, x)
        push!(positions_y, y)
        @printf("Turning: x=%.2f, y=%.2f, psi=%.2f, phi=%.2f, target_heading=%.2f\n", x, y, psi / DEG_TO_RAD, phi / DEG_TO_RAD, target_heading / DEG_TO_RAD)
    end

    # Calculate the bank angle for the final turn
    phi = compute_bank_angle_for_final_turn_timestep(psi, target_heading, convert(Float64, v))
    x, y, psi = update_agent_state(x, y, psi, phi, v)
    push!(positions_x, x)
    push!(positions_y, y)
    
    phi = 0.0  # Level the wings after turn
    return x, y, psi
end

# Fly the path
spacing = 5  # grid cells
timesteps_per_cell = cell_length / v
timesteps_vertical = grid_size * timesteps_per_cell

#for i in 1:div(grid_size, spacing)
    # Fly up the column
x, y, psi = fly_straight(x, y, psi, phi, v, timesteps_vertical)
plot_path(positions_x, positions_y)

# Turn right
target_heading = HEAD_EAST
x, y, psi = turn_plane(x, y, psi, phi, v, -turn_angle, target_heading)
plot_path(positions_x, positions_y)

# Fly horizontally for the spacing
x, y, psi = fly_straight(x, y, psi, phi, v, spacing * timesteps_per_cell)
plot_path(positions_x, positions_y)

# Turn right again
target_heading = HEAD_SOUTH
x, y, psi = turn_plane(x, y, psi, phi, v, -turn_angle, target_heading)
plot_path(positions_x, positions_y)
# Fly down the column
x, y, psi = fly_straight(x, y, psi, phi, v, timesteps_vertical)

# Turn left
target_heading = 0.0
x, y, psi = turn_plane(x, y, psi, phi, v, turn_angle, target_heading)
plot_path(positions_x, positions_y)
# Fly horizontally for the spacing
x, y, psi = fly_straight(x, y, psi, phi, v, spacing * timesteps_per_cell)

plot_path(positions_x, positions_y)
# Turn left again
target_heading = psi + π/2
x, y, psi = turn_plane(x, y, psi, phi, v, turn_angle, target_heading)
#end

println("Completed flying the grid!")





Turning: x=279.54, y=1303.50, psi=-18.87, phi=-40.00, target_heading=270.00
Turning: x=299.31, y=1288.20, psi=-37.73, phi=-40.00, target_heading=270.00
Turning: x=313.07, y=1267.33, psi=-56.60, phi=-40.00, target_heading=270.00
Turning: x=319.35, y=1243.13, psi=-75.46, phi=-40.00, target_heading=270.00
Turning: x=317.46, y=1218.20, psi=-94.33, phi=-40.00, target_heading=270.00
Turning: x=307.62, y=1195.22, psi=-113.19, phi=-40.00, target_heading=270.00