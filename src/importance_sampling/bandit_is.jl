function estimate_return(::UnweightedIS, π, θ, τ::BanditExperience)
    return exp(logpdf(π, θ, τ.action) - τ.blogp) * τ.reward
end

function estimate_return(::WeightedIS, π, θ, τ::BanditExperience)
    ρ = exp(logpdf(π, θ, τ.action) - τ.blogp)
    return ρ * τ.reward, ρ
end

function estimate_return_withentropy(::UnweightedIS, π, θ, τ::BanditExperience, α)
    logp = logpdf(π, θ, τ.action)
    return exp(logp - τ.blogp) * (τ.reward - α * logp)
end

function estimate_return_withentropy(::WeightedIS, π, θ, τ::BanditExperience, α)
    logp = logpdf(π, θ, τ.action)
    ρ = exp(logp - τ.blogp)
    return ρ * (τ.reward - α * logp), ρ
end
