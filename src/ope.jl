function make_logpdf(π::AbstractStatelessPolicy)
    return (a)->logpdf(π, a)
end

function make_logpdf(π::AbstractPolicy)
    return (s,a)->logpdf(π, s, a)
end

function ope(π, stat, ism::AbstractImportanceSampling; α=0.0)

    if α > 0.0
        J = (D, π)->stat(estimate_returns_withentropy(ism, π, D, α))
    else
        J = (D, π)->stat(estimate_returns(ism, π, D))
    end
    
    return J
end
