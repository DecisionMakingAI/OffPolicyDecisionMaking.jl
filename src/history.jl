
struct BanditExperience{T,TA} <: Any where {T,TA}
    action::TA
    blogp::T
    reward::T
end

struct Trajectory{T,TS,TA} <: Any where {T,TS,TA}
    states::Array{TS,1}
    actions::Array{TA,1}
    blogps::Array{T,1}
    rewards::Array{T,1}

    function Trajectory(::Type{T}, ::Type{TS}, ::Type{TA}) where {T,TS,TA}
        new{T,TS,TA}(Array{TS,1}(), Array{TA,1}(), Array{T,1}(), Array{T,1}())
    end
end

function length(τ::Trajectory)
    return length(τ.rewards)
end

function push!(τ::Trajectory{T,TS,TA}, state::TS, action::TA, blogp::T, reward::T) where {T,TS,TA}
    push!(τ.states, state)
    push!(τ.actions, action)
    push!(τ.blogps, blogp)
    push!(τ.rewards, reward)
end
