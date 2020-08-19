function make_logpdf(π::AbstractStatelessPolicy, θ)
    return (a)->logpdf(π, θ, a)
end

function make_logpdf(π::AbstractPolicy, θ)
    return (s,a)->logpdf(π, θ, s, a)
end

function ope(π, stat, ism::AbstractImportanceSampling; α=0.0)

    if α > 0.0
        J = (D, θ)->stat(estimate_returns_withentropy(ism, π, θ, D, α))
    else
        J = (D, θ)->stat(estimate_returns(ism, π, θ, D))
    end
    
    return J
end
