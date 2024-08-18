const DEG_TO_RAD = π / 180

const HEAD_EAST = 0
const HEAD_NORTH = 90 * DEG_TO_RAD
const HEAD_WEST = 180 * DEG_TO_RAD
const HEAD_SOUTH = 270 * DEG_TO_RAD

@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

@with_kw mutable struct GeophysicalObservations
    reading::Vector{Float64} = Vector{Float64}()
    smooth_map_coordinates::Union{Matrix{Int64}, Nothing} = zeros(Int64, 2, 0)
    base_map_coordinates::Union{Matrix{Int64}, Nothing} = zeros(Int64, 2, 0)
end


function aggregate_coordinates(reading::Vector{Float64}, coordinates::Matrix{Int64})
    coord_sum = Dict{Tuple{Int64, Int64}, Float64}()
    coord_count = Dict{Tuple{Int64, Int64}, Int64}()
    
    # Iterate over the coordinates and readings to populate the dictionaries
    for i in 1:length(reading)
        coord = (coordinates[1, i], coordinates[2, i])
        if haskey(coord_sum, coord)
            coord_sum[coord] += reading[i]
            coord_count[coord] += 1
        else
            coord_sum[coord] = reading[i]
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
    
    return new_readings, new_coords
end

function aggregate_base_map_duplicates(obs::GeophysicalObservations)
    # Create a dictionary to store the sums and counts of readings for each coordinate
    readings, coords = aggregate_coordinates(obs.reading, obs.base_map_coordinates)
    
    return GeophysicalObservations(
        reading=readings,
        smooth_map_coordinates=nothing,
        base_map_coordinates=coords
    )
end

function aggregate_smooth_map_duplicates(obs::GeophysicalObservations)
    # Create a dictionary to store the sums and counts of readings for each coordinate
    readings, coords = aggregate_coordinates(obs.reading, obs.smooth_map_coordinates)
    
    return GeophysicalObservations(
        reading=readings,
        smooth_map_coordinates=coords,
        base_map_coordinates=nothing
    )
end

function Base.isequal(go1::GeophysicalObservations, go2::GeophysicalObservations)
    error("this functions has not been tested")
    if isempty(go1)
        if isempty(go2)
            return true
        else
            return false
        end
    else
        if isempty(go2)
            return false
        else
            base_coords = go1.base_map_coordinates == go2.base_map_coordinates 
            smooth_coords = go1.smooth_map_coordinates == go2.smooth_map_coordinates
            reading = go1.reading == go2.reading
            return base_coords & smooth_coords & reading
        end
    end
    
end

struct MEState{MB}
    ore_map::Array{Float64}  # 3D array of ore_quality values for each grid-cell
    smooth_map::Array{Float64}  # 3D array of smoothed values for each grid-cell
    mainbody_params::MB #  Diagonal variance of main ore-body generator
    mainbody_map::Array{Float64}
    rock_obs::RockObservations
    stopped::Bool # Whether or not STOP action has been taken
    decided::Bool # Whether or not the extraction decision has been made
    agent_heading::Float64
    # pos: will contain coordinates of agent at every timestep (which is only a subset of coordinates at which observations are made)
    agent_pos_x::Vector{Float64}  
    agent_pos_y::Vector{Float64}
    agent_bank_angle::Vector{Int64}  # bank angle of agent
    geophysical_obs::GeophysicalObservations
end

function Base.length(obs::RockObservations)
    return length(obs.ore_quals)
end

struct MEObservation
    ore_quality::Union{Float64, Nothing}
    stopped::Bool
    decided::Bool
    geophysical_obs::Union{GeophysicalObservations, Nothing}
    agent_heading::Union{Float64, Nothing}
    agent_pos_x::Union{Float64, Nothing}
    agent_pos_y::Union{Float64, Nothing}
    agent_bank_angle::Union{Int64, Nothing}
end

@with_kw struct MEAction
    type::Symbol = :drill
    coords::CartesianIndex = CartesianIndex(0, 0)
    change_in_bank_angle::Int = 0
end

abstract type GeoDist end

abstract type MainbodyGen end

@with_kw struct MineralExplorationPOMDP <: POMDP{MEState, MEAction, MEObservation}
    reservoir_dims::Tuple{Float64, Float64, Float64} = (2000.0, 2000.0, 30.0) #  lat x lon x thick in meters
    grid_dim::Tuple{Int64, Int64, Int64} = (48, 48, 1) #  dim x dim grid size
    high_fidelity_dim::Tuple{Int64, Int64, Int64} = grid_dim # grid dimensions for high-fidelity case (the "truth" uses this)
    target_dim::Tuple{Int64, Int64, Int64} = grid_dim # grid dimension as the "intended" high-fidelity (i.e., the standard grid dimension that was used to select `extraction_cost` etc.)
    ratio::Tuple{Float64, Float64, Float64} = grid_dim ./ target_dim # scaling "belief" ratio relative to default grid dimensions of 50x50
    target_ratio::Tuple{Float64, Float64, Float64} = high_fidelity_dim ./ target_dim # scaling "truth" ratio relative to default grid dimensions of 50x50
    dim_scale::Float64 = 1/prod(ratio) # scale ore value per cell (for "belief")
    target_dim_scale::Float64 = 1/prod(target_ratio) # scale ore value per cell (for "truth")
    max_bores::Int64 = 10 # Maximum number of bores
    min_bores::Int64 = 1 # Minimum number of bores
    original_max_movement::Int64 = 0 # Original maximum distanace between bores in the default 50x50 grid. 0 denotes no restrictions
    max_movement::Int64 = round(Int, original_max_movement*ratio[1]) # Maximum distanace between bores (scaled based on the ratio). 0 denotes no restrictions
    initial_data::RockObservations = RockObservations() # Initial rock observations
    initial_geophysical_data::GeophysicalObservations = GeophysicalObservations() # Initial geophysical observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 0 # Number of cells in between each cell in which wells can be placed
    drill_cost::Float64 = 0.1
    variogram::Tuple = (0.005, 30.0, 0.0001) #sill, range, nugget
    # nugget::Tuple = (1, 0)
    geodist_type::Type = GeoStatsDistribution # GeoDist type for geo noise
    gp_mean::Float64 = 0.25
    mainbody_weight::Float64 = 0.6  
    true_mainbody_gen::MainbodyGen = BlobNode(grid_dims=high_fidelity_dim) # high-fidelity true mainbody generator
    mainbody_gen::MainbodyGen = BlobNode(grid_dims=grid_dim)
    rng::AbstractRNG = Random.GLOBAL_RNG
    c_exp::Float64 = 100.0

    base_grid_element_length::Float64 = 50.0 # length of each grid element in meters, 50x50 grid with grid_element_length = 100 models a 5km x 5km region 
    upscale_factor::Int = 4  # factor to upscale the grid by for smooth, higher resolution map
    smooth_grid_element_length::Float64 = base_grid_element_length / upscale_factor
    sigma::Float64 = 3  # for smoothing map with gaussian filter
    geophysical_noise_std_dev::Float64 = 0.00
    max_timesteps::Int = 120
    mineral_exploration_mode = "geophysical" # borehole or geophysical
    fly_cost::Float64 = 0.01
    out_of_bounds_cost::Float64 = 5.0  # reward gets penalized if the plane position is out of bounds at a timestep, does not penalize if the plane is out of bounds between timesteps
    out_of_bounds_tolerance::Int = -1 # number of grid base map grid squares the agent can be out of bounds before incurring cost
    massive_threshold::Float64 = 0.7
    strike_reward::Float64 = 1.0
    init_bank_angle::Int = 0
    init_pos_x::Float64 = 600.0
    init_pos_y::Float64 = 0.0
    init_heading::Float64 = HEAD_NORTH
    max_bank_angle::Int = 55
    min_readings::Int = 30
    bank_angle_intervals::Int = 18
    timestep_in_seconds::Int = 1
    observations_per_timestep::Int = 1
    velocity::Int = 50
    extraction_cost::Float64 = 150.0
    extraction_lcb::Float64 = 0.8
    extraction_ucb::Float64 = 0.8
    target_mass_params::Tuple{Real, Real} = (extraction_cost, extraction_cost/3) # target mean and std when standardizing ore mass distributions
end

struct MEInitStateDist  # prior over state space
    true_gp_distribution::GeoDist  #
    gp_distribution::GeoDist  # background noise on tope of mainbody, no conditioning on samples
    # mainbody - some shape of what the main ore body looks like
    mainbody_weight::Float64
    true_mainbody_gen::MainbodyGen  # the way to sample a shape, e.g., with circle sample a radius or centre
    # only need one of true_mainbody_gen and mainbody_gen as don't need to generate different shapes
    mainbody_gen::MainbodyGen
    massive_thresh::Float64
    dim_scale::Float64
    target_dim_scale::Float64
    target_μ::Float64
    target_σ::Float64
    rng::AbstractRNG
    sigma::Float64  # for smoothing map with gaussian filter
    upscale_factor::Int
    m::MineralExplorationPOMDP
end

# prepare POMCPOW
function get_geophysical_solver(c_exp::Float64, get_tree::Bool=false)
    return POMCPOWSolver(
        tree_queries=10000,
        k_observation=1.5,
        alpha_observation=0.15,
        max_depth=6,
        check_repeat_obs=false,
        check_repeat_act=true,
        enable_action_pw=false,
        criterion=POMCPOW.MaxUCB(c_exp),
        final_criterion=POMCPOW.MaxQ(),
        estimate_value=geophysical_leaf_estimation,
        #estimate_value=0.0,
        tree_in_info=get_tree,
    )
end