using POMDPs
using Random # for AbstractRNG
using POMDPModelTools # for Deterministic
using POMCPOW
using POMDPSimulators
using POMDPPolicies

struct BabyPOMDP <: POMDP{Bool, Bool, Bool}
    r_feed::Float64
    r_hungry::Float64
    p_become_hungry::Float64
    p_cry_when_hungry::Float64
    p_cry_when_not_hungry::Float64
    discount::Float64   
end

BabyPOMDP() = BabyPOMDP(-5., -10., 0.1, 0.8, 0.1, 0.9);

function POMDPs.gen(m::BabyPOMDP, s, a, rng)
    # transition model
    if a # feed
        sp = false
    elseif s # hungry
        sp = true
    else # not hungry
        sp = rand(rng) < m.p_become_hungry
    end
    
    # observation model
    if sp # hungry
        o = rand(rng) < m.p_cry_when_hungry
    else # not hungry
        o = rand(rng) < m.p_cry_when_not_hungry
    end
    
    # reward model
    r = s*m.r_hungry + a*m.r_feed
    
    # create and return a NamedTuple
    return (sp=sp, o=o, r=r)
end

solver = POMCPOWSolver(tree_queries=5,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        enable_action_pw=false,
                        
                        k_observation=2.0,
                        alpha_observation=0.1,
                        criterion=POMCPOW.MaxUCB(100.0),
                        final_criterion=POMCPOW.MaxQ(),
                        
                        estimate_value=0.0,
                        max_depth=10,
                        # estimate_value=leaf_estimation,
                        tree_in_info=true,
                        )

POMDPs.initialstate(m::BabyPOMDP) = Deterministic(false)

m = BabyPOMDP()

planner = POMDPs.solve(solver, m)

# policy that maps every input to a feed (true) action
policy = FunctionPolicy(o->true)

for (s, a, r) in stepthrough(m, policy, "s,a,r", max_steps=10)
    @show s
    @show a
    @show r
    println()
end