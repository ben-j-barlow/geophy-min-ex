using Revise
import BSON: @save, @load
using POMDPs
using POMCPOW
using Plots;
default(fontfamily="Computer Modern", framestyle=:box); # LaTex-style
using Statistics
using Images
using Random
using Infiltrator
using MineralExploration
using POMDPPolicies

C_EXP = 2
const DEG_TO_RAD = π / 180

m = MineralExplorationPOMDP(c_exp=C_EXP, sigma=20)

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

s0 = rand(ds0; truth=true) #Checked


bank_angle_history::Vector{Int} = [5, 5, 5, 5, 5,0,0,-5, -5,-5,-5,-5, -10,-10]

function plot_map_and_plane_trajectory(map, bank_angle_history::Vector{Int64}, m::MineralExplorationPOMDP, grid_element_length::Union{Int64,Float64})
    @info "plot_map_and_plane_trajectory(map, bank_angle_history::Vector{Int64}, m::MineralExplorationPOMDP)"
    if m.grid_dim[1] != m.grid_dim[2]
        error("function assumes square grid")
    end
    to_plot_x, to_plot_y = get_plane_trajectory(bank_angle_history, m)
    to_plot_x = [x / grid_element_length for x in to_plot_x]
    to_plot_y = [y / grid_element_length for y in to_plot_y]
    p = plot_ore_map(map)
    plot!(p, to_plot_x, to_plot_y, color="red", lw=2, )
end

function get_plane_trajectory(bank_angle_history::Vector{Int64}, m::MineralExplorationPOMDP, dt::Float64=0.2)
    updates_per_timestep = m.timestep_in_seconds / dt
    pos_x, pos_y, heading = convert(Float64, m.init_pos_x), convert(Float64, m.init_pos_y), convert(Float64, m.init_heading)

    pos_x_outer_loop_history, pos_y_outer_loop_history = [], []
    pos_x_history, pos_y_history = [deepcopy(pos_x)], [deepcopy(pos_y)]

    # create list of lists
    for bank_angle in bank_angle_history
        for i in 1:updates_per_timestep
            pos_x, pos_y, heading = update_agent_state(pos_x, pos_y, heading, bank_angle * DEG_TO_RAD, m.velocity, dt)
            push!(pos_x_history, deepcopy(pos_x))
            push!(pos_y_history, deepcopy(pos_y))
        end
        push!(pos_x_outer_loop_history, deepcopy(pos_x_history))
        push!(pos_y_outer_loop_history, deepcopy(pos_y_history))
        pos_x_history, pos_y_history = [deepcopy(pos_x)], [deepcopy(pos_y)]
    end
    
    # reduce lists of lists to single list
    to_plot_x = reduce(vcat, pos_x_outer_loop_history)
    to_plot_y = reduce(vcat, pos_y_outer_loop_history)
    return to_plot_x, to_plot_y
end

function plot_plane_trajectory(bank_angle_history::Vector{Int64}, m::MineralExplorationPOMDP)
    if m.grid_dim[1] != m.grid_dim[2]
        error("function assumes square grid")
    end
    
    to_plot_x, to_plot_y = get_plane_trajectory(bank_angle_history, m)
    return visualize_plane_trajectory(to_plot_x, to_plot_y, m.grid_dim[1], m.base_grid_element_length)
end

function add_plane_trajectory!(p, x, y)
    plot!(p, x, y, color="red", lw=2)
end

function normalize_plane_coordinates(x::Vector{Float64}, y::Vector{Float64}, grid_element_length::Float64)
    # normalize for plotting on map
    x = [x / grid_element_length for x in x]
    y = [y / grid_element_length for y in y]
    return x, y
end

plot_map_and_plane_trajectory(s0.smooth_map, bank_angle_history, m, m.smooth_grid_element_length)

x,y = get_plane_trajectory(bank_angle_history, m)
