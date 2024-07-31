using Parameterss

@with_kw mutable struct GeophysicalObservations
    reading::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

function append_geophysical_obs_sequence(history::GeophysicalObservations, new_obs::GeophysicalObservations)
    history.reading = vcat(history.reading, new_obs.reading)
    history.coordinates = hcat(history.coordinates, new_obs.coordinates)
    return history
end


go = GeophysicalObservations()

obs = 10
x = 1
y = 1

go.reading = push!(go.reading, obs)
go.coordinates = hcat(go.coordinates, reshape(Int64[x y], 2, 1))



my_go = GeophysicalObservations()

obs = 20
x = 2
y = 2

my_go.reading = push!(my_go.reading, obs)
my_go.coordinates = hcat(my_go.coordinates, reshape(Int64[x y], 2, 1))


out = append_geophysical_obs_sequence(go, GeophysicalObservations(reading=[20], coordinates=reshape(Int64[2 2], 2, 1)))

out.coordinates



# test dedupe

@with_kw mutable struct GeophysicalObservations
    reading::Vector{Float64} = Vector{Float64}()
    smooth_map_coordinates::Union{Matrix{Int64}, Nothing} = zeros(Int64, 2, 0)
    base_map_coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end


function aggregate_base_map_duplicates(obs::GeophysicalObservations)
    # Create a dictionary to store the sums and counts of readings for each coordinate
    coord_sum = Dict{Tuple{Int64, Int64}, Float64}()
    coord_count = Dict{Tuple{Int64, Int64}, Int64}()
    
    # Iterate over the coordinates and readings to populate the dictionaries
    for i in 1:length(obs.reading)
        coord = (obs.base_map_coordinates[1, i], obs.base_map_coordinates[2, i])
        if haskey(coord_sum, coord)
            coord_sum[coord] += obs.reading[i]
            coord_count[coord] += 1
        else
            coord_sum[coord] = obs.reading[i]
            coord_count[coord] = 1
        end
    end
    
    # Create new vectors for the deduplicated coordinates and their average readings
    new_coords = zeros(Int64, 2, length(coord_sum))
    new_readings = Vector{Float64}(undef, length(coord_sum))
    
    idx = 1
    for (coord, sum_reading) in coord_sum
        new_coords[1, idx] = coord[1]
        new_coords[2, idx] = coord[2]
        new_readings[idx] = sum_reading / coord_count[coord]
        idx += 1
    end
    
    return GeophysicalObservations(
        reading=new_readings,
        smooth_map_coordinates=nothing,
        base_map_coordinates=new_coords
    )
end


go = GeophysicalObservations()

obs,x,y = 20,1,1
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))

obs,x,y = 10,1,1
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))

obs,x,y = 30,2,2
go.reading = push!(go.reading, obs)
go.base_map_coordinates = hcat(go.base_map_coordinates, reshape(Int64[x y], 2, 1))

res = aggregate_base_map_duplicates(go)

res.base_map_coordinates