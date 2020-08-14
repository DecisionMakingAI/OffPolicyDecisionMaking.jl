module OffPolicy

using Statistics, Distributions
using Policies
using Zygote
import Base:length, push!


export BanditExperience, Trajectory
export length, push!

export ope
export estimate_returns, estimate_returns!, estimate_returns_withentropy, estimate_returns_withentropy!

export AbstractImportanceSampling, UnweightedIS, WeightedIS, IS, PDIS, WIS, WPDIS

export andersons_ci, tdist_ci


include("history.jl")
include("importance_sampling/importancesampling.jl")
include("confidenceintervals.jl")
include("ope.jl")


end
