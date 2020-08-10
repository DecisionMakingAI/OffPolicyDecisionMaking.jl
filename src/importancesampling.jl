abstract type AbstractImportanceSampling end
abstract type UnweightedIS <: AbstractImportanceSampling end
abstract type WeightedIS <: AbstractImportanceSampling end

struct IS <: UnweightedIS end
struct PDIS <: UnweightedIS end
struct WIS <: WeightedIS end
struct WPDIS <: WeightedIS end


include("bandit_is.jl")
include("trajectory_is.jl")
