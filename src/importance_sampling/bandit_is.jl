function estimate_return(::UnweightedIS, logpdf_fn, τ::BanditExperience)
    return exp(logpdf_fn(τ.action) - τ.blogp) * τ.reward
end

function estimate_return(::WeightedIS, logpdf_fn, τ::BanditExperience)
    ρ = exp(logpdf_fn(τ.action) - τ.blogp)
    return ρ * τ.reward, ρ
end

function estimate_return_withentropy(::UnweightedIS, logpdf_fn, τ::BanditExperience, α)
    logp = logpdf_fn(τ.action)
    return exp(logp - τ.blogp) * (τ.reward - α * logp)
end

function estimate_return_withentropy(::WeightedIS, logpdf_fn, τ::BanditExperience, α)
    logp = logpdf_fn(τ.action)
    ρ = exp(logpdf_fn(τ.action) - τ.blogp)
    return ρ * (τ.reward - α * logp), ρ
end
