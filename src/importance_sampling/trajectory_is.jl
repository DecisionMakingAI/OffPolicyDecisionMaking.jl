
function estimate_return(::IS, π, τ::Trajectory)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    states = reduce(hcat,τ.states)
    actions = reduce(hcat,τ.actions)
    logps = logpdf(π, states, actions)
    # logp = sum(@. logpdf((π,), τ.states, τ.actions))
    logp = sum(logps)
    ρ = exp(logp - blogp)
    return exp(logp - blogp) * G
end

function estimate_return_withentropy(::IS, π, τ::Trajectory, α)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    states = reduce(hcat,τ.states)
    actions = reduce(hcat,τ.actions)
    logps = logpdf(π, states, actions)
    # logp = sum(@. logpdf((π,), τ.states, τ.actions))
    logp = sum(logps)
    ρ = exp(logp - blogp)
    return exp(logp - blogp) * (G - α * logp)
end

function estimate_return(::WIS, π, τ::Trajectory)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    logp = sum(@. logpdf((π,), τ.states, τ.actions))
    ρ = exp(logp - blogp)
    return ρ * G, ρ
end

function estimate_return_withentropy(::WIS, π, τ::Trajectory, α)
    G = sum(τ.rewards)
    blogp = sum(τ.blogps)
    logp = sum(@. logpdf((π,), τ.states, τ.actions))
    ρ = exp(logp - blogp)
    return ρ * (G - α * logp), ρ
end

function iterativePDIS(π, τ::Trajectory{T}, t, lnρ) where {T}
    lnρ += logpdf(π, τ.states[t], τ.actions[t]) - τ.blogps[t]
    ρ = exp(lnρ)
    return ρ * τ.rewards[t], ρ, lnρ
end

function iterativePDIS_withentropy(π, τ::Trajectory{T}, t, lnρ, α) where {T}
    logp = logpdf(π, τ.states[t], τ.actions[t])
    lnρ += logp - τ.blogps[t]
    ρ = exp(lnρ)
    return ρ * (τ.rewards[t] - α * logp), ρ, lnρ
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

function estimate_return_withentropy(::PDIS, π, τ::Trajectory{T}, α) where {T}
    N = length(τ)
    lnρ = T(0.0)
    G = T(0.0)
    for t in 1:N
        ρR, _, lnρ = iterativePDIS_withentropy(π, τ, t, lnρ, α)
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

function estimate_returns_withentropy!(G, ism::CWPDIS, π, H, α)
    N = length(H)
    lnρs = zeros(N)
    ρtot = 0.0
    L = maximum(length.(H))
    for i in 1:N
        for t in 1:L
            ptot = 0.0
            if length(H[i]) ≤ t
                ρR, ρ, lnρ = iterativePDIS_withentropy(π, H[i], t, lnρs[i], α)
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

function estimate_returns_withentropy(ism::CWPDIS, π, H, α)
    N = length(H)
    G = zeros(N)
    estimate_returns_withentropy!(G, ism, π, H, α)
    return G
end
