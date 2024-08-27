go = GeophysicalObservations()

obs, x, y = 10, 1, 1
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))

obs, x, y = 20, 1, 1
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))
a
obs, x, y = 30, 2, 2
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))

unique_elements = unique(go.base_map_coordinates)

coordinates = Dict{Int, Vector{Tuple{Int, Int}}}()

for element in unique_elements
    coords = findall(x -> x == element, go.base_map_coordinates)
    coordinates[element] = coords
end

coordinates

go.base_map_coordinates



using DataFrames
using Statistics
using Parameters: @with_kw

@with_kw mutable struct GeophysicalObservations
    reading::Vector{Float64} = Vector{Float64}()
    smooth_map_coordinates::Union{Matrix{Int64}, Nothing} = zeros(Int64, 2, 0)
    base_map_coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

function reduce_duplicates(obs::GeophysicalObservations)
    # Convert base_map_coordinates to a DataFrame for easier manipulation
    df = DataFrame(x=obs.base_map_coordinates[1, :], y=obs.base_map_coordinates[2, :], reading=obs.reading)

    # Group by coordinates and calculate the mean reading for each group
    grouped_df = combine(groupby(df, [:x, :y]), :reading => mean => :average_reading)

    # Create new GeophysicalObservations object with reduced duplicates
    new_reading = grouped_df.average_reading
    new_base_map_coordinates = hcat(grouped_df.x, grouped_df.y)

    return GeophysicalObservations(
        reading=new_reading,
        smooth_map_coordinates=nothing,
        base_map_coordinates=new_base_map_coordinates
    )
end


# Example usage
original_obs = GeophysicalObservations(
    reading=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    base_map_coordinates=[1 1 2 2 2 3; 1 1 2 2 2 3]
)

new_obs = reduce_duplicates(original_obs)


println("Original readings: ", original_obs.reading)
println("Original base map coordinates: ", original_obs.base_map_coordinates)
println("New readings: ", new_obs.reading)
println("New base map coordinates: ", new_obs.base_map_coordinates)