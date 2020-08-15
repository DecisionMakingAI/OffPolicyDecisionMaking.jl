abstract type AbstractImportanceSampling end
abstract type UnweightedIS <: AbstractImportanceSampling end
abstract type WeightedIS <: AbstractImportanceSampling end

struct IS <: UnweightedIS end
struct PDIS <: UnweightedIS end
struct WIS <: WeightedIS end
struct WPDIS <: WeightedIS end
struct CWPDIS <: WeightedIS end


include("bandit_is.jl")
include("trajectory_is.jl")

function estimate_returns(ism::AbstractImportanceSampling, π, θ, H)
    return @. estimate_return((ism,), (π,), (θ,), H)
end

function estimate_returns!(G, ism::AbstractImportanceSampling, π, θ, H)
    @. G = estimate_return((ism,), (π,), (θ,), H)
end

function estimate_returns_withentropy(ism::AbstractImportanceSampling, π, θ, H, α)
    return @. estimate_return_withentropy((ism,), (π,), (θ,), H, α)
end

function estimate_returns_withentropy!(G, ism::AbstractImportanceSampling, π, θ, H, α)
    @. G = estimate_return_withentropy((ism,), (π,), (θ,), H, α)
end

function estimate_returns(ism::WIS, π, θ, H)
    N = length(H)
    ρtot = 0.0
    rets = estimate_return.((ism,), (π,), (θ,), H)
    G = [r[1] for r in rets]

    for i in 1:N
        ρi = rets[i][2]
        ρtot += ρi
    end
    G ./ ρtot
end

function estimate_returns_withentropy(ism::WIS, π, θ, H, α)
    N = length(H)
    ρtot = 0.0
    rets = estimate_return_withentropy.((ism,), (π,), (θ,), H, (α,))
    G = [r[1] for r in rets]

    for i in 1:N
        ρi = rets[i][2]
        ρtot += ρi
    end
    G / ρtot
end

function estimate_returns!(G, ism::WIS, logpdf_fn, H)
    N = length(H)
    ρtot = 0.0
    for i in 1:N
        Gi, ρi = estimate_return(ism, logpdf_fn, H[i])
        G[i] = Gi
        ρtot += ρi
    end
    @. G ./= ρtot
end

function estimate_returns_withentropy!(G, ism::WIS, π, θ, H, α)
    N = length(H)
    ρtot = 0.0
    for i in 1:N
        Gi, ρi = estimate_return_withentropy(ism, π, θ, H[i], α)
        G[i] = Gi
        ρtot += ρi
    end
    @. G ./= ρtot
end
