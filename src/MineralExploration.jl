module MineralExploration

using BeliefUpdaters
using CSV
using DataFrames
using Dates
using DelimitedFiles
using Distributions
using Distances # for KL and JS
using GeoStats
using ImageFiltering
using Infiltrator # for debugging
using Interpolations
using JLD
using JSON
using KernelDensity
using Luxor
using LinearAlgebra
using Logging
using MCTS
using Parameters
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using POMCPOW
using POMDPModelTools
using POMDPSimulators
using POMDPs
using Random
using StatsBase
using StatsPlots
using Statistics
using OrderedCollections


export
        MEState,
        MEObservation,
        MEAction,
        RockObservations,
        GeoDist,
        MineralExplorationPOMDP,
        MEInitStateDist,
        MEBelief,
        HEAD_NORTH,
        HEAD_SOUTH,
        HEAD_EAST,
        HEAD_WEST,
        MainbodyGen,
        GeophysicalObservations,
        aggregate_base_map_duplicates,
        aggregate_smooth_map_duplicates
include("common.jl")

export
        GeoStatsDistribution,
        kriging
include("geostats.jl")

export
        GSLIBDistribution,
        kriging
include("gslib.jl")

export
        SingleFixedNode,
        SingleVarNode,
        MultiVarNode
include("mainbody.jl")

export
        MEBeliefUpdater,
        particles,
        support,
        get_input_representation,
        plot_input_representation
include("beliefs.jl")

export
        initialize_data!,
        high_fidelity_obs,
        calc_massive,
        update_agent_state,
        generate_geophysical_obs_sequence,
        is_empty,
        check_plane_within_region
include("pomdp.jl")

export
        ShapeNode,
        CircleNode,
        EllipseNode,
        BlobNode,
        MultiShapeNode
include("shapes.jl")

export
        NextActionSampler,
        ExpertPolicy,
        RandomSolver,
        GridPolicy,
        leaf_estimation
include("solver.jl")

export
        GPNextAction
include("action_selection.jl")

export
    standardize,
    standardize_scale,
    calculate_standardization,
    save_standardization,
    generate_ore_mass_samples
include("standardization.jl")

export
        plot_history,
        run_trial,
        run_geophysical_trial,
        gen_cases,
        plot_ore_map,
        plot_map,
        plot_mass_map,
        plot_volume,
        nan_unvisited_cells,
        set_readings_in_map,
        normalize_agent_coordinates,
        add_agent_trajectory_to_plot!,
        get_agent_trajectory,
        plot_smooth_map_and_plane_trajectory,
        plot_base_map_and_plane_trajectory,
        get_base_map_coordinates,
        get_smooth_map_coordinates,
        plot_base_map_at_observation_locations,
        plot_smooth_map_at_observation_locations,
        normalize_agent_coordinates
include("utils.jl")

export 
        generate_log_file_name,
        prepare_logger,
        close_logger
include("logging.jl")

end
