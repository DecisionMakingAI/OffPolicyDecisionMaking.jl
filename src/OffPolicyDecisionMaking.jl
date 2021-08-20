module OffPolicyDecisionMaking

using DecisionMakingPolicies

using Random, Statistics, Distributions
using Zygote
using DecisionMakingEnvironments




export BanditExperience, Trajectory
export length, push!

export ope
export estimate_returns, estimate_returns!, estimate_returns_withentropy, estimate_returns_withentropy!

export AbstractImportanceSampling, UnweightedIS, WeightedIS, IS, PDIS, WIS, WPDIS

export andersons_ci, tdist_ci

export AbstractSeldonianProblem, SeldonianProblem
export solve

export AbstractSplitMethod, SplitLastK, SplitLastKKeepTest
export collect_and_split!
export HICOPI


include("importance_sampling/importancesampling.jl")
include("confidenceintervals.jl")
include("ope.jl")
include("seldonian.jl")
include("highconfidence.jl")

end
