# OffPolicy

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DecisionMakingAI.github.io/OffPolicy.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DecisionMakingAI.github.io/OffPolicy.jl/dev)
[![Build Status](https://github.com/DecisionMakingAI/OffPolicy.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/OffPolicy.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/OffPolicy.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/OffPolicy.jl)

This package provides functionality useful off-policy reinforcement learning. To demonstrate some of the functionality we use the following example of performing high confidence off-policy improvement. For this problem we have some reference policy ``πsafe`` that we deem safe to execute. We want the probability of deploying a policy ``π`` that is worse than ``πsafe`` to be less than ``δ``.

First we create a bandit environment and functions to sample from that environment.

```julia
using DecisionMakingEnvironments
using Policies
using OffPolicy

using Optim
using Zygote

numA = 10
env = create_simple_discrete_bandit(numA; noise=√numA)
```

Now we create a softmax policy and and weights to optimize ``θ`` and the weights ``θsafe`` representing the safety policy.

```julia
π = StatelessSoftmax()
θ = initparams(π, numA)
θsafe = similar(θ)
θsafe .= sin.(collect(1:numA) ./ √numA) .+ randn(numA)
```

No we create an experience buffer to store the interaction history along with list for storing the indicies for the training and safety data.


```julia
D = Array{BanditExperience{Float64, Int}, 1}()
train_idxs = Array{Int,1}()
safety_idxs = Array{Int,1}()
```

We then create a method to sample from the environment and split the data into train and test sets. We use the ``SplitLastK(p)`` type, which assigns all data previous data to be in the training data set and splits up the new samples such that ``100p%`` of samples go into the training set and rest goes into the safety set. There is also the ``SplitLastKKeepTest(p)`` type which keeps the datasets separated and reuses old data for testing, but this method is biased and cannot be used to guarantee the safety condition.

```julia
function sample(env::Bandit, a)
    reward = env.r(a)
end

function sample!(H, env::Bandit, π, θ, n::Int)
    d = π(θ)
    for i in 1:n
        a = rand(d)
        logp = logpdf(d, a)
        r = sample(env, a)
        push!(H, BanditExperience(a, logp, r))
    end
end

sm = SplitLastK(0.1)
# sm = SplitLastKKeepTest(0.1)
num_samples=100  # number of samples per iteration of policy optimization

function collect_fn!(D, train_idxs, safety_idxs, θ)
    sfn = (D, π, θ, n)->sample!(D, env, π, θ, n)
    collect_and_split!(D, train_idxs, safety_idxs, π, θ, num_samples, sfn, sm)
end
```

Next we create an objective function for the policy optimization that uses weighted importance sampling and a entropy bonus of ``α``. We use Optim.jl to optimize the the policy, but any optimization software should be compatiable.

```julia
J = ope(π, mean, WIS(); α=0.1) # J is a function of some data set D and θ that produces an estimate of return using the mean

function off_policy_optimizer(f, D, θ)
    results = optimize(θ->-f(D,θ), θ, LBFGS())
    return Optim.minimizer(results)
end
```

The safety check needs to compute the upper bound on the safety policies performance and lower bound on the candidate policy performance. We compute confidence intervals of the return by using the t-Distribution. The constraint function ``g`` should return a scalar value such that when the safety test is passed it is less than or equal to zero.

```julia
δ = 0.05  # probability of failing the safety test
lower_bound = tdist_ci(δ=δ/2, tail=:left)  # creates a function that returns the lower confidence interval of a vector of samples.
upper_bound = tdist_ci(δ=δ/2, tail=:right)
lcb = ope(π, lower_bound, IS(); α=0.0)  # creates a function to compute the lower bound on return.
ucb = ope(π, upper_bound, IS(); α=0.0)
g = (D,θ,θsafe)->(ucb(D, θsafe) - lcb(D, θ))
```

Lastly we create a HICOPI (high confidence off-policy improvement) type to store the functions created above. Then ten iterations of collecting data, optimizing the candidate policy and performing the safety check are performed. If the candidate policy fails the safety check ``:NSF`` is returned and the next data collection is performed using the safe policy. If the safey check is passed the new candidate policy parameters are returned and it will be used to collect new samples of performance.

```julia
alg = HICOPI(collect_fn!, J, off_policy_optimizer,g)
behavior = :safe
d = π(θsafe)
p = sum(d.p .* collect(1:numA))  # computes the actual performance of the safety policy as a reference.
println("True safety policy performance: $p")

for i in 1:10
    result = alg(D, train_idxs, safety_idxs, θ, θsafe; behavior=behavior)
    if result == :NSF
        println("Iteration: $i - No Solution Found")
        behavior = :safe
    else
        println("Iteration $i - candidate passed safety check")
        behavior = :candidate
        θ .= result
        d = π(θ)
        p = sum(d.p .* collect(1:numA))  # computes the actual performance of the safety policy as a reference.
        println("Candidate policy performance: $p")
    end
end
```
