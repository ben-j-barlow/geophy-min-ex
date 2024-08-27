using MineralExploration

function pad_line(description::String, original::String, current::String, pad_length::Int)
    pad = " " ^ (pad_length - length(description))
    println(description * pad * " | " * original * " | " * current)
end

function mylog(current::MEBelief{GeoStatsDistribution}, original::MEBelief{GeoStatsDistribution})
    #println("Beliefs equal: $(isequal(current, original))")
    #println("Mismatched properties: " * join([p for p in propertynames(current) if !isequal(getproperty(current, p), getproperty(original, p))], ", "))
    pad_line("obs base map coords", string(original.geophysical_obs.base_map_coordinates), string(current.geophysical_obs.base_map_coordinates), 20)
    pad_line("bank angle", string(original.agent_bank_angle), string(current.agent_bank_angle), 20)
    println("geostats distributions are equal: $(isequal(current.geostats, original.geostats))")
end

function log(current::MEState, original::MEState)
    pad_line("heading", string(original.agent_heading), string(current.agent_heading), 20)
    pad_line("pos x", string(original.agent_pos_x), string(current.agent_pos_x), 20)
    pad_line("pos y", string(original.agent_pos_y), string(current.agent_pos_y), 20)
end

# fake observation 
function observe_line(map, fixed, orientation)
    to_return = GeophysicalObservations()

    for i in 1:size(map)[1]
        if orientation == :horizontal
            push!(to_return.reading, map[fixed, i, 1])
            to_return.base_map_coordinates = hcat(to_return.base_map_coordinates, reshape(Int64[fixed i], 2, 1))
        else
            push!(to_return.reading, map[i, fixed, 1])
            to_return.base_map_coordinates = hcat(to_return.base_map_coordinates, reshape(Int64[i fixed], 2, 1))
        end
    end
    
    return to_return
end