function calc_K(geostats::GeoDist, rock_obs::RockObservations)
    @info "calc_K(geostats::GeoDist, rock_obs::RockObservations)"
    # Check if the geostats object is of type GeoStatsDistribution
    if isa(geostats, GeoStatsDistribution)
        # If true, use the domain and variogram from the geostats object
        pdomain = geostats.domain
        γ = geostats.variogram
    else
        # Otherwise, create a CartesianGrid using the grid dimensions from geostats
        pdomain = CartesianGrid(geostats.grid_dims[1], geostats.grid_dims[2])
        # Create a SphericalVariogram using the sill, range, and nugget from geostats
        γ = SphericalVariogram(sill=geostats.sill, range=geostats.variogram[6], nugget=geostats.nugget)
    end

    # table = DataFrame(ore=rock_obs.ore_quals .- geostats.mean)

    # Define the domain as a PointSet using the coordinates from rock_obs
    domain = PointSet(rock_obs.coordinates)

    # pdata = georef(table, domain)
    # vmapping = map(pdata, pdomain, (:ore,), GeoStats.NearestMapping())[:ore]
    # dlocs = Int[]
    # for (loc, dloc) in vmapping
    #     push!(dlocs, loc)
    # end
    # dlocs = Int64[m[1] for m in vmapping]
    # 𝒟d = [centroid(pdomain, i) for i in dlocs]

    # Transform the points in the domain by adding 0.5 to their coordinates
    𝒟d = [GeoStats.Point(p.coords[1] + 0.5, p.coords[2] + 0.5) for p in domain]

    # Calculate the covariance matrix K
    K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)

    @info "covar matrix $(typeof(K))"
    @info "covar matrix $(size(K))"
    # Return the covariance matrix K
    return K
end

    
function reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)
    @info "reweight(up::MEBeliefUpdater, geostats::GeoDist, particles::Vector, rock_obs::RockObservations)"
    ws = Float64[] # Initialize weights array
    bore_coords = rock_obs.coordinates # Get borehole coordinates
    n = size(bore_coords)[2] # Number of boreholes
    ore_obs = [o for o in rock_obs.ore_quals] # List of ore quality observations
    K = calc_K(geostats, rock_obs) # Calculate the covariance matrix using geostats and rock observations
    mu = zeros(Float64, n) .+ up.m.gp_mean # Mean vector filled with the Gaussian process mean
    gp_dist = MvNormal(mu, K) # Multivariate normal distribution based on mean vector and covariance matrix
    for s in particles
        mb_map = s.mainbody_map # Get the main body map from the state
        o_n = zeros(Float64, n) # Initialize the array for normalized ore observations
        for i = 1:n
            o_mainbody = mb_map[bore_coords[1, i], bore_coords[2, i]] # Main body ore quality at the borehole location
            o_n[i] = ore_obs[i] - o_mainbody # Normalize the ore observation
        end
        w = pdf(gp_dist, o_n) # Calculate the weight as the probability density of the normalized observations
        push!(ws, w) # Add the weight to the weights array
    end
    ws .+= 1e-6 # Add a small value to weights to avoid zero values
    ws ./= sum(ws) # Normalize the weights
    return ws # Return the weights
end



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
    target_mass_params::Tuple{Real, Real} = (extraction_cost, extraction_cost/3) # target mean and std when standardizing ore mass distributions
    rng::AbstractRNG = Random.GLOBAL_RNG
    c_exp::Float64 = 1.0

    grid_element_length::Int = 60  # length of each grid element in meters, 50x50 grid with grid_element_length = 100 models a 5km x 5km region 
    upscale_factor::Int = 10  # factor to upscale the grid by for smooth, higher resolution map
    smooth_grid_element_length::Float64 = grid_element_length / upscale_factor
    sigma::Float64 = 10  # for smoothing map with gaussian filter
    geophysical_noise_std_dev::Float64 = 0.25
    max_timesteps::Int = 100
    mineral_exploration_mode = "borehole" # borehole or geophysical
    fly_cost::Float64 = 0.01
    massive_threshold::Float64 = 0.7
    strike_reward::Float64 = 1.0
end

@with_kw struct GeoStatsDistribution <: GeoDist
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    data::RockObservations = RockObservations()
    domain::CartesianGrid = CartesianGrid(grid_dims[1], grid_dims[2])
    mean::Float64 = 0.3
    variogram::Variogram = SphericalVariogram(sill=0.005, range=30.0,
                                            nugget=0.0001)
    #rng::AbstractRNG = Random.GLOBAL_RNG
    #lu_params::LUParams = LUParams(rng, variogram, domain)
    lu_params::LUParams = LUParams(variogram, domain)
end

function MEBeliefUpdater(m::MineralExplorationPOMDP, n::Int64, noise::Float64=1.0; abc::Bool=false, abc_ϵ::Float64=1e-1, abc_dist::Function=(x,x′)->abs(x-x′))
    geostats = m.geodist_type(m)
    return MEBeliefUpdater(m, geostats, n, noise, abc, abc_ϵ, abc_dist, m.rng)
end

function GeoStatsDistribution(p::MineralExplorationPOMDP; truth=false)
    @info "GeoStatsDistribution(p::MineralExplorationPOMDP; truth=false)"
    grid_dims = truth ? p.high_fidelity_dim : p.grid_dim
    variogram = SphericalVariogram(sill=p.variogram[1], range=p.variogram[2],
                                    nugget=p.variogram[3])
    #domain = CartesianGrid(convert(Float64, grid_dims[1]), convert(Float64, grid_dims[2]))
    domain = CartesianGrid(grid_dims[1], grid_dims[2])
    #return GeoStatsDistribution(rng=p.rng,
    return GeoStatsDistribution(grid_dims=grid_dims,
                                data=deepcopy(p.initial_data),
                                domain=domain,
                                mean=p.gp_mean,
                                variogram=variogram)
end


using MineralExploration
using GeoStats

m = MineralExplorationPOMDP()

geostats = m.geodist_type(m)

geostats.grid_dims

ro = RockObservations()

coord = reshape(Int64[1, 1], 2, 1)
ro.coordinates = hcat(ro.coordinates, coord)
push!(ro.ore_quals, 5)      

coord = reshape(Int64[2, 1], 2, 1)
ro.coordinates = hcat(ro.coordinates, coord)
push!(ro.ore_quals, 10)      

coord = reshape(Int64[4, 5], 2, 1)
ro.coordinates = hcat(ro.coordinates, coord)
push!(ro.ore_quals, 15)

coord = reshape(Int64[1, 1], 2, 1)
ro.coordinates = hcat(ro.coordinates, coord)
push!(ro.ore_quals, 20)


n = size(ro.coordinates)[2] # Number of boreholes
K = calc_K(geostats, ro) # Calculate the covariance matrix using geostats and rock observations
@show K


γ = geostats.variogram

domain = PointSet(ro.coordinates)
𝒟d = [GeoStats.Point(p.coords[1] + 0.5, p.coords[2] + 0.5) for p in domain]

@show 𝒟d
@show GeoStats.pairwise(γ, 𝒟d)

K = GeoStats.sill(γ) .- GeoStats.pairwise(γ, 𝒟d)

mu = zeros(Float64, n) .+ m.gp_mean # Mean vector filled with the Gaussian process mean
gp_dist = MvNormal(mu, K) # Multivariate normal distribution based on mean vector and covariance matrix
    

@show geostats.variogram
pdomain = CartesianGrid(3, 3)