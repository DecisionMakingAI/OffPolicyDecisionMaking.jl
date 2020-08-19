using OffPolicyDecisionMaking
using Documenter

makedocs(;
    modules=[OffPolicyDecisionMaking],
    authors="Scott Jordan",
    repo="https://github.com/DecisionMakingAI/OffPolicyDecisionMaking.jl/blob/{commit}{path}#L{line}",
    sitename="OffPolicyDecisionMaking.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DecisionMakingAI.github.io/OffPolicyDecisionMaking.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DecisionMakingAI/OffPolicyDecisionMaking.jl",
)
