@with_kw mutable struct RockObservations
    ore_quals::Vector{Float64} = Vector{Float64}()
    coordinates::Matrix{Int64} = zeros(Int64, 2, 0)
end

@with_kw mutable struct GeophysicalObservations
    reading::Matrix{Vector{Float64}} = Matrix{Vector{Float64}}(undef, 0, 0)
    function GeophysicalObservations(x_dim::Int, y_dim::Int)
        obj = new(Matrix{Vector{Float64}}(undef, x_dim, y_dim))
        for i in 1:x_dim, j in 1:y_dim
            obj.reading[i, j] = Vector{Float64}()
        end
        return obj
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
    agent_pos_x::Vector{Float64}
    agent_pos_y::Vector{Float64}
    agent_velocity::Int
    agent_bank_angle::Int  # bank angle of agent
    geophysical_obs::GeophysicalObservations
end

function Base.length(obs::RockObservations)
    return length(obs.ore_quals)
end

struct MEObservation
    ore_quality::Union{Float64, Nothing}
    stopped::Bool
    decided::Bool
    geophysical_reading::Union{Float64, Nothing}
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
    grid_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) #  dim x dim grid size
    high_fidelity_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) # grid dimensions for high-fidelity case (the "truth" uses this)
    target_dim::Tuple{Int64, Int64, Int64} = (50, 50, 1) # grid dimension as the "intended" high-fidelity (i.e., the standard grid dimension that was used to select `extraction_cost` etc.)
    ratio::Tuple{Float64, Float64, Float64} = grid_dim ./ target_dim # scaling "belief" ratio relative to default grid dimensions of 50x50
    target_ratio::Tuple{Float64, Float64, Float64} = high_fidelity_dim ./ target_dim # scaling "truth" ratio relative to default grid dimensions of 50x50
    dim_scale::Float64 = 1/prod(ratio) # scale ore value per cell (for "belief")
    target_dim_scale::Float64 = 1/prod(target_ratio) # scale ore value per cell (for "truth")
    max_bores::Int64 = 10 # Maximum number of bores
    min_bores::Int64 = 1 # Minimum number of bores
    original_max_movement::Int64 = 0 # Original maximum distanace between bores in the default 50x50 grid. 0 denotes no restrictions
    max_movement::Int64 = round(Int, original_max_movement*ratio[1]) # Maximum distanace between bores (scaled based on the ratio). 0 denotes no restrictions
    initial_data::RockObservations = RockObservations() # Initial rock observations
    delta::Int64 = 1 # Minimum distance between wells (grid coordinates)
    grid_spacing::Int64 = 0 # Number of cells in between each cell in which wells can be placed
    drill_cost::Float64 = 0.1
    strike_reward::Float64 = 1.0
    extraction_cost::Float64 = 150.0
    extraction_lcb::Float64 = 0.1
    extraction_ucb::Float64 = 0.1
    variogram::Tuple = (0.005, 30.0, 0.0001) #sill, range, nugget
    # nugget::Tuple = (1, 0)
    geodist_type::Type = GeoStatsDistribution # GeoDist type for geo noise
    gp_mean::Float64 = 0.25
    mainbody_weight::Float64 = 0.6  
    true_mainbody_gen::MainbodyGen = BlobNode(grid_dims=high_fidelity_dim) # high-fidelity true mainbody generator
    mainbody_gen::MainbodyGen = BlobNode(grid_dims=grid_dim)
    massive_threshold::Float64 = 0.7
    target_mass_params::Tuple{Real, Real} = (extraction_cost, extraction_cost/3) # target mean and std when standardizing ore mass distributions
    rng::AbstractRNG = Random.GLOBAL_RNG
    c_exp::Float64 = 1.0

    grid_element_length::Int = 60  # length of each grid element in meters, 50x50 grid with grid_element_length = 100 models a 5km x 5km region 
    upscale_factor::Int = 10  # factor to upscale the grid by for smooth, higher resolution map
    smooth_grid_element_length::Float64 = grid_element_length / upscale_factor
    sigma::Float64 = 10  # for smoothing map with gaussian filter
    geophysical_noise_std_dev::Float64 = 0.25
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
end
