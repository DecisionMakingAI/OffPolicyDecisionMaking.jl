function tinterval(x, δ, op)
    μ = mean(x)
    σ = std(x, mean=μ)
    t = 0.0
    n = length(x)
    Zygote.ignore() do
        t = quantile(TDist(n-1), δ)
    end
    s = t * σ / √n
    return op(μ,s)
end

function tdist_ci(;δ=0.05, tail=:both)
    if tail == :right
        return x->tinterval(x,δ,+)
    elseif tail == :left
        return x->tinterval(x,δ,-)
    else
        return x->tinterval(x,δ/2,(μ,s)->(μ-s,μ+s))
    end
end

function andersons_ucb(x, b, δ)
    n = length(x)
    ϵ = √(log(1/δ)/(2n))
    r = floor(Int, n*ϵ)
    upper = (1/n) * ((r+1)*x[r+1] + sum(view(x, r+1:n))) + ϵ * (b - x[r+1])
    return upper
end

function andersons_lcb(x, a, δ)
    n = length(x)
    ϵ = √(log(1/δ)/(2n))
    s = floor(Int, n*ϵ)
    lower = (1/n) * (sum(view(x, 1:n-s-1)) + (s+1)*x[n-s]) - ϵ * (x[n-s] - a)
    return lower
end

function andersons_ci(a, b; δ=0.05, tail=:both)
    if tail == :both
        function f(x)
            z = sort(x)
            upper = andersons_ucb(z, b, δ/2)
            lower = andersons_lcb(z, a, δ/2)
            return lower, upper
        end
        return f
    elseif tail == :left
        return x->andersons_lcb(sort(x), a, δ)
    elseif tail == :right
        return x->andersons_ucb(sort(x), b, δ)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end
