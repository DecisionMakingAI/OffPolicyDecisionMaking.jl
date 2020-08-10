module OffPolicy

import Base:length,push!

# export AbstractHistory, History, BanditHistory, Trajectory
export BanditExperience, Trajectory
export length, push!

export evaluate_policy
export estimate_returns, estimate_returns!, estimate_returns_withentropy, estimate_returns_withentropy!

export AbstractImportanceSampling, UnweightedIS, WeightedIS, IS, PDIS, WIS, WPDIS

include("history.jl")
include("importancesampling.jl")

end
