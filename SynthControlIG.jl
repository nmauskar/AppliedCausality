using CodecBzip2
using RData
using Distributions
using Statistics
using SpecialFunctions
using Plots
using ProgressMeter
using CSV
using DataFrames
using LinearAlgebra
using Random 
using StatsBase
using ProgressMeter
using StatsPlots
using SynthControl
using Dates
using LogExpFunctions

function SoftmaxLSE(x::Array{Float64})
    c = maximum(x)
    LSE = c + log(sum(exp.(x .- c)))
    return exp.(x .- LSE)
end

mutable struct SynthControlModel
    I::Int64
    T::Int64
    I_T::Array{Float64, 2}
    σ::Float64
    α::Array{Float64}
    β::Float64
    θ::Float64
    D::Array{Float64}
    C::Float64
    Y::Array{Float64}
    X::Array{Float64, 2}
end

function initGibbs(I::Int64, T::Int64, θ::Float64, β::Float64, α::Array{Float64})
    Y = zeros(T)
    C = 0.0
    σ = 0.0
    D = zeros(I)
    X = zeros(I, T)
    I_T = diagm(ones(T))
    return SynthControlModel(I, T, I_T, σ, α, β, θ, D, C, Y, X)
end

function sampleD!(gs::SynthControlModel)
    gs.D = rand(Dirichlet(gs.α))
    return gs.D
end

function sampleC!(gs::SynthControlModel)
    gs.C = rand(Categorical(gs.D))
    return gs.C
end

function sampleσ!(gs::SynthControlModel)
    gs.σ = rand(InverseGamma(gs.θ, gs.β))
    return gs.σ
end

function sampleY!(gs::SynthControlModel)
    gs.Y = rand(MvNormal(gs.X[Int64(gs.C), :], gs.σ .* gs.I_T))
    return gs.Y
end

function sampleY(gs::SynthControlModel, X::Array{Float64, 2}, T::Int64)
    return rand(MvNormal(X[Int64(gs.C), :], gs.σ .* diagm(ones(T))))
end

function samplePrior!(gs::SynthControlModel)
    sampleD!(gs)
    sampleC!(gs)
    sampleσ!(gs)
    sampleY!(gs)
end

function updateD!(gs::SynthControlModel)
    param = deepcopy(gs.α)
    param[Int64(gs.C)] += 1.0
    gs.D = rand(Dirichlet(param))
    return gs.D
end

function updateC!(gs::SynthControlModel)
    probs = zeros(gs.I)
    for i ∈ 1:gs.I
        N = 0.0
        for t ∈ 1:gs.T
            N += -0.5 * log(2*π) - 0.5*log(gs.σ) - 0.5 * 1.0/gs.σ * (gs.Y[t] - gs.X[i, t])^2
        end
        probs[i] = N - log(gs.D[i] + 1e-8)
    end
    newProbs = SoftmaxLSE(probs)
    gs.C = rand(Categorical(newProbs ./sum(newProbs)))
    return gs.C
end

function updateσ!(gs::SynthControlModel)
    shape = gs.θ + 0.5*gs.T
    rate = gs.β
    for t ∈ 1:gs.T
        rate += ((gs.Y[t] - gs.X[Int64(gs.C), t])^2) / 2.0
    end
    gs.σ = rand(InverseGamma(shape, rate))
    return gs.σ
end

function updateGibbs!(gs::SynthControlModel)
    updateC!(gs)
    updateD!(gs)
    updateσ!(gs)
    return gs
end
