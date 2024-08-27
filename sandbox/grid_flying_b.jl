const DEG_TO_RAD = π / 180

function normalize_angle(angle::Float64)
    return mod(angle, 2 * π)
end

function update_agent_state(x::Float64, y::Float64, psi::Float64, phi::Float64, v::Int, dt::Float64 = 1.0, g::Float64 = 9.81)
    if dt != 1.0
        error("stick to dt = 1 for now")
    end
    # normalize to 0 to 2pi
    psi = normalize_angle(psi)

    psi_dot = g * tan(phi) / v
    to_return_psi = normalize_angle(psi + (psi_dot * dt))
    x_dot = v * cos(to_return_psi)
    y_dot = v * sin(to_return_psi)
    to_return_x = x + (x_dot * dt)
    to_return_y = y + (y_dot * dt)
    return to_return_x, to_return_y, to_return_psi
end

function compute_turn_sequence(v::Int, max_phi::Float64, timesteps::Int)
    # Initialize variables
    x, y, psi, phi = 0.0, 0.0, 0.0, 0.0
    dt = 1.0
    g = 9.81
    
    # Target total heading change is π/2 radians (90 degrees)
    total_heading_change = π / 2
    cumulative_heading_change = 0.0
    
    # Create a list to store bank angles
    bank_angles = []
    
    # Generate a sequence of bank angles for the right turn
    for t in 1:timesteps
        # If we haven't yet reached the required heading change, increase phi
        if cumulative_heading_change < total_heading_change / 2
            phi = max_phi
        else
            # Start reducing phi after reaching halfway in heading change
            phi = max_phi * (2 * (total_heading_change - cumulative_heading_change) / total_heading_change)
        end
        
        # Calculate the change in heading for this timestep
        psi_dot = g * tan(phi) / v
        cumulative_heading_change += psi_dot * dt
        
        # Store the current bank angle
        push!(bank_angles, phi)
        
        # Update plane's state
        x, y, psi = update_agent_state(x, y, psi, phi, v, dt, g)
    end
    
    return bank_angles, x, y, psi
end

# Example usage
v = 20  # velocity in m/s
max_phi = π / 6  # maximum bank angle in radians (~30 degrees)
timesteps = 10

bank_angles, final_x, final_y, final_psi = compute_turn_sequence(v, max_phi, timesteps)

println("Bank Angles: ", bank_angles)
println("Final Position: (x, y) = ($final_x, $final_y)")
println("Final Heading: $(final_psi/DEG_TO_RAD) degrees")

