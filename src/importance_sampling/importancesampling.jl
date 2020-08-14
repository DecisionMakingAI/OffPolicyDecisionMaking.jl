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

function estimate_returns(ism::AbstractImportanceSampling, logpdf_fn, H)
    return @. estimate_return((ism,), (logpdf_fn,), H)
end

function estimate_returns!(G, ism::AbstractImportanceSampling, logpdf_fn, H)
    @. G = estimate_return((ism,), (logpdf_fn,), H)
end

function estimate_returns_withentropy(ism::AbstractImportanceSampling, logpdf_fn, H, α)
    return @. estimate_return_withentropy((ism,), (logpdf_fn,), H, α)
end

function estimate_returns_withentropy!(G, ism::AbstractImportanceSampling, logpdf_fn, H, α)
    @. G = estimate_return_withentropy((ism,), (logpdf_fn,), H, α)
end

function estimate_returns(ism::WIS, logpdf_fn, H)
    N = length(H)
    # G = zeros(N)
    ρtot = 0.0
    rets = estimate_return.((ism,), (logpdf_fn,), H)
    G = [r[1] for r in rets]

    for i in 1:N
        ρi = rets[i][2]
        # Gi, ρi = estimate_return_withentropy(ism, logpdf_fn, H[i], α)
        # G[i] = Gi
        # push!(G, Gi)
        ρtot += ρi
    end
    G ./ ρtot
end

function estimate_returns_withentropy(ism::WIS, logpdf_fn, H, α)
    N = length(H)
    # G = zeros(N)
    ρtot = 0.0
    # G = Vector{Float64}()
    rets = estimate_return_withentropy.((ism,), (logpdf_fn,), H, (α,))
    G = [r[1] for r in rets]

    for i in 1:N
        ρi = rets[i][2]
        # Gi, ρi = estimate_return_withentropy(ism, logpdf_fn, H[i], α)
        # G[i] = Gi
        # push!(G, Gi)
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

function estimate_returns_withentropy!(G, ism::WIS, logpdf_fn, H, α)
    N = length(H)
    ρtot = 0.0
    for i in 1:N
        Gi, ρi = estimate_return_withentropy(ism, logpdf_fn, H[i], α)
        G[i] = Gi
        ρtot += ρi
    end
    @. G ./= ρtot
end
