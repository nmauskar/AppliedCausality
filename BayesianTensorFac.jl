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

mutable struct tensorFacModel
    A::Int64
    I::Int64
    T::Int64
    K::Int64
    I_K::Array{Float64, 2}
    σU::Float64
    σV::Float64
    σW::Float64
    σY::Float64
    Y::Array{Float64, 3}
    U::Array{Float64, 2}
    V::Array{Float64, 2}
    W::Array{Float64, 2}
end

function initGibbs(A::Int64, I::Int64, T::Int64, K::Int64, σU::Float64, σV::Float64, σW::Float64, σY::Float64)
    Y = zeros(A, I, T)
    U = zeros(I, K)
    V = zeros(T, K)
    W = zeros(A, K)
    I_K = diagm(ones(K))
    return tensorFacModel(A, I, T, K, I_K, σU, σV, σW, σY, Y, U, V, W)
end

function sampleU!(gs::tensorFacModel)
    gs.U = rand(Normal(0.0, gs.σU), gs.I, gs.K)
    return gs.U
end

function sampleV!(gs::tensorFacModel)
    gs.V = rand(Normal(0.0, gs.σV), gs.T, gs.K)
    return gs.V
end

function sampleW!(gs::tensorFacModel)
    gs.W = rand(Normal(0.0, gs.σW), gs.A, gs.K)
    return gs.W
end

function sampleY!(gs::tensorFacModel)
    for t ∈ 1:gs.T
        for i ∈ 1:gs.I
            for a ∈ 1:gs.A
                gs.Y[a, i, t] = rand(Normal(sum(gs.U[i, :] .* gs.V[t, :] .* gs.W[a, :]), gs.σY))
            end
        end
    end
    return gs.Y
end

function samplePrior!(gs::tensorFacModel)
    sampleU!(gs)
    sampleV!(gs)
    sampleW!(gs)
    sampleY!(gs)
end

function updateU!(gs::tensorFacModel)
    S = zeros(gs.K, gs.K)
    b = zeros(gs.K)
    
    for t ∈ 1:gs.T
        for a ∈ 1:gs.A
            S += (gs.V[t, :] .* gs.W[a, :]) * (gs.V[t, :] .* gs.W[a, :])'
        end
    end
    
    for i ∈ 1:gs.I
        b .= 0.0
        for t ∈ 1:gs.T
            for a ∈ 1:gs.A
                b += (gs.V[t, :] .* gs.W[a, :]) .* gs.Y[a,i,t]
            end
        end
        MVN = MvNormalCanon(b/gs.σY, 1.0/gs.σU .* gs.I_K + S./gs.σY)
        gs.U[i, :] = rand(MVN)
    end
    return gs.U
end

function updateV!(gs::tensorFacModel)
    S = zeros(gs.K, gs.K)
    b = zeros(gs.K)
    
    for i ∈ 1:gs.I
        for a ∈ 1:gs.A
            S += (gs.U[i, :] .* gs.W[a, :]) * (gs.U[i, :] .* gs.W[a, :])'
        end
    end
    
    for t ∈ 1:gs.T
        b .= 0.0
        for i ∈ 1:gs.I
            for a ∈ 1:gs.A
                b += (gs.U[i, :] .* gs.W[a, :]) .* gs.Y[a,i,t]
            end
        end
        MVN = MvNormalCanon(b/gs.σY, 1.0/gs.σV .* gs.I_K + S./gs.σY)
        gs.V[t, :] = rand(MVN)
    end
    return gs.V
end

function updateW!(gs::tensorFacModel)
    S = zeros(gs.K, gs.K)
    b = zeros(gs.K)
    
    for i ∈ 1:gs.I
        for t ∈ 1:gs.T
            S += (gs.U[i, :] .* gs.V[t, :]) * (gs.U[i, :] .* gs.V[t, :])'
        end
    end
    
    for a ∈ 1:gs.A
        b .= 0.0
        for i ∈ 1:gs.I
            for t ∈ 1:gs.T
                b += (gs.U[i, :] .* gs.V[t, :]) .* gs.Y[a,i,t]
            end
        end
        MVN = MvNormalCanon(b/gs.σY, 1.0/gs.σW .* gs.I_K + S./gs.σY)
        gs.W[a, :] = rand(MVN)
    end
    return gs.W
end

function updateY!(gs::tensorFacModel, mask::Set{Tuple{Int64, Int64, Int64}})
    for i ∈ mask
        gs.Y[i[1], i[2], i[3]] = rand(Normal(sum(gs.U[i[2], :] .* gs.V[i[3], :] .* gs.W[i[1], :]), gs.σY))
    end
    return gs.Y
end

function updateGibbs!(gs::tensorFacModel, mask::Set{Tuple{Int64, Int64, Int64}})
    updateU!(gs)
    updateW!(gs)
    updateV!(gs)
    updateY!(gs, mask)
    return gs
end