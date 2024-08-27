### MAP PLOTTING

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
using LinearAlgebra
using MineralExploration
using POMDPPolicies

C_EXP = 2
const DEG_TO_RAD = π / 180

m = MineralExplorationPOMDP(c_exp=C_EXP, sigma=20)

# do not call
#initialize_data!(m, N_INITIAL)

ds0 = POMDPs.initialstate(m)

s0 = rand(ds0; truth=true) #Checked


# Function to generate correlated readings
function generate_correlated_readings(n, start_value, delta_range)
    readings = Float64[start_value]
    for i in 2:n
        new_value = clamp(readings[end] + rand(delta_range), 0.6, 0.7)
        push!(readings, new_value)
    end
    return readings
end

# Initialize coordinates from (1,5) to (50,5)
coordinates = hcat(1:50, fill(5, 50))

# Generate correlated readings
start_value = 0.65  # Starting value within the range 0.6 to 0.7
delta_range = (-0.02, 0.02)  # Small changes to ensure correlation
readings = generate_correlated_readings(50, start_value, delta_range)

# Create the GeophysicalObservations object
go = GeophysicalObservations(
    reading=readings,
    smooth_map_coordinates=nothing,
    base_map_coordinates=coordinates
)

function check_coordinates(coordinates::Matrix{Int64})
    if size(coordinates, 1) != 2
        error("problem with coordinates 1")
    end
    check_duplicate_coordinates(coordinates)
end

# Function to check for duplicate coordinates
function check_duplicate_coordinates(coordinates::Matrix{Int64})
    coord_set = Set{Tuple{Int64, Int64}}()

    for i in 1:size(coordinates, 2)
        coord = (coordinates[1, i], coordinates[2, i])
        if coord in coord_set
            error("Duplicate coordinates")
        end
        push!(coord_set, coord)
    end
end

function check_coordinates_and_readings(coordinates::Matrix{Int64}, readings::Vector{Float64})
    check_coordinates(coordinates)    
    if size(coordinates, 2) != length(readings)
        error("number of readings must match number of coordinates")
    end
end


function get_transpose(matrix::Matrix{Int64})
    return convert(Matrix{Int64}, matrix')
end


# check GeophysicalObs coordinates are accepted
check_coordinates(go.base_map_coordinates)

# check 2x1 is accepted
random_matrix = rand(Int64, 2, 1)
check_coordinates(random_matrix)

# check 2x2 is accepted
random_matrix = rand(Int64, 2, 2)
check_coordinates(random_matrix)

# check 3 x 2 is not accepted
random_matrix = rand(Int64, 3, 2)
check_coordinates(random_matrix)

# Matrix with unique coordinates
coordinates_unique = [1 2 3 4 5; 1 2 3 4 5]
check_duplicate_coordinates(coordinates_unique)

# Matrix with duplicate coordinates
coordinates_duplicate = [1 2 3 4 1; 1 2 3 4 1]  # (1, 1) is duplicated
check_duplicate_coordinates(coordinates_duplicate)



function nan_unvisited_cells(matrix::Array{Float64, 3}, coordinates::Union{Matrix{Int64}, Array{Int64, 2}})
    check_coordinates(coordinates)

    # Create a copy of the matrix to avoid modifying the original
    result_matrix = fill(NaN, size(matrix))
    
    for i in 1:size(coordinates, 2)
        x, y = coordinates[:, i]
        result_matrix[x, y, 1] = matrix[x, y, 1]
    end
    
    return result_matrix
end

# plot line of map
tmp = nan_unvisited_cells(s0.ore_map, get_transpose(go.base_map_coordinates))
plot_ore_map(tmp)

# plot line of readings

# Function to update the matrix with readings at the specified coordinates
function set_readings_in_map(matrix::Array{Float64, 3}, coordinates::Matrix{Int64}, readings::Vector{Float64})
    check_coordinates_and_readings(coordinates, readings)
    # Create a copy of the matrix to avoid modifying the original
    result_matrix = fill(NaN, size(matrix))
    
    for i in 1:size(coordinates, 2)
        x, y = coordinates[:, i]
        result_matrix[x, y, 1] = readings[i]
    end
    
    return result_matrix
end

tmp = set_readings_in_map(s0.ore_map, get_transpose(go.base_map_coordinates), go.reading)
plot_ore_map(tmp)

