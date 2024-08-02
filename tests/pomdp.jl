using MineralExploration

# Test Case 1: No bank angle (phi = 0)
x, y, heading = update_agent_state(0.0, 0.0, 0.0, 0.0, 40)
@assert round(x, digits=2) ≈ 40.0
@assert round(y, digits=2) ≈ 0.0
@assert round(heading, digits=2) ≈ 0.0

# Test Case 2: Small positive bank angle (phi = π/6)
x, y, heading = update_agent_state(0.0, 0.0, 0.0, π/6, 40)
@assert round(x, digits=2) ≈ 40.0
@assert round(y, digits=2) ≈ 0.0
@assert round(heading, digits=2) ≈ 0.14

# Test Case 3: Small negative bank angle (phi = -π/6)
x, y, heading = update_agent_state(0.0, 0.0, 0.0, -π/6, 40)
@assert round(x, digits=2) ≈ 40.0
@assert round(y, digits=2) ≈ 0.0
@assert round(heading, digits=2) ≈ -0.14

# Test Case 4: Heading (psi = π/4), bank angle (phi = π/6)
x, y, heading = update_agent_state(0.0, 0.0, π/4, π/6, 40)
@assert round(x, digits=2) ≈ 28.28
@assert round(y, digits=2) ≈ 28.28
@assert round(heading, digits=2) ≈ 0.93
