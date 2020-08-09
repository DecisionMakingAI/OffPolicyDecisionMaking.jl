using OffPolicy
using Documenter

makedocs(;
    modules=[OffPolicy],
    authors="Scott Jordan",
    repo="https://github.com/DecisionMakingAI/OffPolicy.jl/blob/{commit}{path}#L{line}",
    sitename="OffPolicy.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DecisionMakingAI.github.io/OffPolicy.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DecisionMakingAI/OffPolicy.jl",
)
