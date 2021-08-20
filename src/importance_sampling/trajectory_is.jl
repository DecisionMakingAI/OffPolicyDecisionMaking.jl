
function estimate_return(::IS, π, θ, τ::Trajectory)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    logp = sum(@. logpdf((π,), τ.states, τ.actions))
    ρ = exp(logp - blogp)
    return exp(logp - blogp) * G
end

function estimate_return(::WIS, (π,), τ::Trajectory)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    logp = sum(@. logpdf((π,), τ.states, τ.actions))
    ρ = exp(logp - blogp)
    return ρ * G, ρ
end

function iterativePDIS(π, τ::Trajectory{T}, t, lnρ) where {T}
    lnρ += logpdf(π, τ.states[t], τ.actions[t]) - τ.blogps[t]
    ρ = exp(lnρ)
    return ρ * τ.rewards[t], ρ, lnρ
end

function estimate_return(::PDIS, π, τ::Trajectory{T}) where {T}
    N = length(τ)
    lnρ = T(0.0)
    G = T(0.0)
    for t in 1:N
        ρR, _, lnρ = iterativePDIS(π, τ, t, lnρ)
        G += ρR
    end
    return G
end




function estimate_returns!(G, ism::CWPDIS, π, H)
    N = length(H)
    lnρs = zeros(N)
    ρtot = 0.0
    L = maximum(length.(H))
    for i in 1:N
        for t in 1:L
            ptot = 0.0
            if length(H[i]) ≤ t
                ρR, ρ, lnρ = iterativePDIS(π, H[i], t, lnρs[i])
                G[i] += ρR
                lnρs[i] += lnρ
                ptot += ρ
            else
                ptot += exp(lnρs[i])
            end
            @. G /= ptot
        end
    end
end

function estimate_returns(ism::CWPDIS, π, H)
    N = length(H)
    G = zeros(N)
    estimate_returns!(G, ism, π, H)
    return G
end
