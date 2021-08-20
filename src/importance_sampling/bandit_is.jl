function estimate_return(::UnweightedIS, π, τ::BanditExperience)
    return exp(logpdf(π, τ.action) - τ.blogp) * τ.reward
end

function estimate_return(::WeightedIS, π, τ::BanditExperience)
    ρ = exp(logpdf(π, τ.action) - τ.blogp)
    return ρ * τ.reward, ρ
end

function estimate_return_withentropy(::UnweightedIS, π, τ::BanditExperience, α)
    logp = logpdf(π, τ.action)
    return exp(logp - τ.blogp) * (τ.reward - α * logp)
end

function estimate_return_withentropy(::WeightedIS, π, τ::BanditExperience, α)
    logp = logpdf(π, τ.action)
    ρ = exp(logp - τ.blogp)
    return ρ * (τ.reward - α * logp), ρ
end
