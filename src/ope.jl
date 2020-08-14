function make_logpdf(π::AbstractStatelessPolicy, θ)
    return (a)->logpdf(π, θ, a)
end

function make_logpdf(π::AbstractPolicy, θ)
    return (s,a)->logpdf(π, θ, s, a)
end

function ope(D, π, θ, stat, ism::AbstractImportanceSampling; α=0.0)

    if α > 0.0
        f = (logpdf_fn)->estimate_returns_withentropy(ism, logpdf_fn, D, α)
    else
        f = (logpdf_fn)->estimate_returns(ism, logpdf_fn, D)
    end

    J = (θ)->stat(f(make_logpdf(π, θ)))
    return J
end
