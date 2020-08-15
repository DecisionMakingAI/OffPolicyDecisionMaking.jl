abstract type AbstractSplitMethod end

#unbaised method for splitting
struct SplitLastK{T} <: AbstractSplitMethod where {T}
    p::T
end

# biased method for splitting
struct SplitLastKKeepTest{T} <: AbstractSplitMethod where {T}
    p::T
end


function collect_and_split!(D, train_idxs, test_idxs, π, θ, n, sample_fn!, split_method::SplitLastK)
    L = length(D)
    sample_fn!(D, π, θ, n)
    idxs = randperm(n)
    k = floor(Int, split_method.p*n)
    empty!(train_idxs)
    empty!(test_idxs)
    append!(train_idxs, 1:L)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])
end

function collect_and_split!(D, train_idxs, test_idxs, π, θ, n, sample_fn!, split_method::SplitLastKKeepTest)
    L = length(D)
    sample_fn!(D, π, θ, n)
    idxs = randperm(n)
    k = floor(Int, split_method.p*n)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])

end



struct HICOPI{TC,TJ,TO,TG}
    collect_fn!::TC
    J::TJ
    optimizer::TO
    g::TG
end


function (a::HICOPI)(D, train_idxs, safety_idxs, θ, θsafe; behavior=:safe)
    if behavior == :safe
        a.collect_fn!(D, train_idxs, safety_idxs, θsafe)
    else
        a.collect_fn!(D, train_idxs, safety_idxs, θ)
    end
    Dtrain = @view D[train_idxs]
    Dsafety = @view D[safety_idxs]
    # f = (D,θ)->a.J(D, θ)
    g = (Dsafety, θ)->a.g(Dsafety, θ, θsafe)
    p = SeldonianProblem(a.J, g)
    result = solve(p, a.optimizer, θ, Dtrain, Dsafety)
    return result
end
