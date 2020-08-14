abstract type AbstractSplitMethod end

#unbaised method for splitting
struct SplitLastK{T} <: AbstractSplitMethod where {T}
    p::T
end

# biased method for splitting
struct SplitLastKKeepTest{T} <: AbstractSplitMethod where {T}
    p::T
end


function collect_and_split!(D, train_idxs, test_idxs, π, N, sample_fn!, split_method::SplitLastK)
    L = length(D)
    sample_fn!(D, π, N)
    idxs = randperm(N)
    k = floor(Int, split_method.p*N)
    empty!(train_idxs)
    empty!(test_idxs)
    append!(train_idxs, 1:L)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])
end

function collect_and_split!(D, train_idxs, test_idxs, π, N, sample_fn!, split_method::SplitLastKKeepTest)
    L = length(D)
    sample_fn!(D, π, N)
    idxs = randperm(N)
    k = floor(Int, split_method.p*N)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])

end



"""
    safety_test

This function performs a high confidence safety test of policy π
in reference to a policy πsafe. The evaluation function f computes
the confidence interval for a policy and level δ. The either
returns a symbol indicating if π :safe or :uncertain. :uncertain
means that it cannot be gauranteed that π is better the πsafe.
"""
function safety_test(f, π, πsafe, δ)
    pilow = f(π, δ/2.0, :left)
    safehigh = f(πsafe, δ/2.0, :right)
    if pilow > safehigh
        return :safe
    else
        return :uncertain
    end
end


function hicopi_step!(oparams, π, D, train_idxs, test_idxs, optimize_fn!, confidence_bound, πsafe, δ)
    optimize_fn!(oparams, π, D, train_idxs)
    result = HICOPI_safety_test((π, δ, tail) -> confidence_bound(D, test_idxs, π, δ, tail), π, πsafe, δ)
    if result == :uncertain
        return :NSF
    else
        return π
    end
end


"""
    hicopi!(D, sample_fn!, optimize_fn, confidence_test, π, πsafe, τ, δ)

TODO update this description
This is a high confidence off-policy policy improvement function that
performs a single iteration of collecting τ data samples with policy π,
finding a canditate policy, πc, and then checking to see if πc is
better than the safe policy, πsafe, with confidence δ. If πc is better
return πc otherwise return :NSF no solution found.

This funciton is generic and takes as input:
D, previous data, which is possibly empty
sample_fn!, a function to sample new data points using policy π
optimize_fn, a funciton to find a new policy on data D
confidence_test, a function that computes a high confidence upper or lower bound on a policies performance
π, a policy to collect data with and initialize the optimization search with
πsafe, a policy that is consider a safe baseline that can always be trusted
τ, number of samples to collect data for. τ could represent the number of episodes or draws from a bandit.
δ, confidence level to use to ensure that safe policies or :NSF is return with probability at least 1-δ.
"""
function hicopi!(oparams, π, D, train_idxs, test_idxs, sample_fn!, optimize_fn!, confidence_bound, πsafe, τ, δ, split_method, num_iterations, warmup_steps)
    πbehavior = πsafe

    collect_and_split!(D, train_idxs, test_idxs, πbehavior, warmup_steps, sample_fn!, split_method)

    for i in 1:num_iterations
        n = length(D)
        collect_and_split!(D, train_idxs, test_idxs, πbehavior, τ, sample_fn!, split_method)
        result = HICOPI_step!(oparams, π, D, train_idxs, test_idxs, optimize_fn!, confidence_bound, πsafe, δ)
        if result == :NSF
            πbehavior = πsafe
        else
            πbehavior = π
        end
    end
end
