abstract type AbstractSeldonianProblem end

struct SeldonianProblem{TF,TG} <: AbstractSeldonianProblem where {TF,TG}
    f::TF
    g::TG
end

function solve(prob::SeldonianProblem, candidate_search, initialsolution, Dtrain, Dsafety)
    candidate = candidate_search(prob.f, Dtrain, initialsolution)

    safetycheck = safetytest(prob.g, Dsafety, candidate)
    if safetycheck
        return candidate
    else
        return :NSF
    end
end

function safetytest(g, Dsafety, candidate)
    return g(Dsafety, candidate) ≤ 0
end

function safetytest(g::AbstractArray, Dsafety, candidate)
    return all(safetytest.(g, (Dsafety,), (candidate,)))
end
