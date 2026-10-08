using LinearAlgebra
using Plots, DifferentialEquations
using LinearAlgebra
using NLopt
using Interpolations
using Distributions
using Roots
using CSV
using DataFrames
using LaTeXStrings
using Accessors
using Measures
using Dierckx 
using Random
using StatsPlots
using LinearAlgebra
using JLD2
using MAT
# Get the path of the current script
path = dirname(@__FILE__)



# squared Euclidean distance between two row-vectors (views) without sqrt
@inline function sqeuclid(a::AbstractVector, b::AbstractVector)
    d = a .- b
    return dot(d, d) 
end


function info_imbalance(X, Y)
    N = size(X, 1)

    rY = Vector{Int}(undef, N)
    D = Matrix{Float64}(undef, 2, N)
    for i in 1:N
        xi = view(X, i, :)

        # ---- 1) find nearest neighbor in X ----
        bestj = 0
        bestd = Inf
        for j in 1:N
            j == i && continue
            d = sqeuclid(xi, view(X, j, :))
            if d < bestd
                bestd = d
                bestj = j
            end
        end
        D[1,i] = bestd
        # ---- 2) rank that neighbor in Y ----
        yi = view(Y, i, :)
        d_target = sqeuclid(yi, view(Y, bestj, :))

        rank = 1
        for j in 1:N
            j == i && continue
            d = sqeuclid(yi, view(Y, j, :))
            if d < d_target
                rank += 1
            end
        end
        D[2,i] = d_target
        rY[i] = rank
    end

    mean_rY = sum(rY) / N
    Δ = (2 / N) * mean_rY
    return Δ
end

function average_imb(X,Y)
    Δ = 0.0
    for i in 1:1
        Δ += info_imbalance(X, Y)
    end
    return Δ / 1 
end


function posterior_covariance_equal(part_vals)
    X = part_vals
    N, p = size(X)

    # Step 1: mean
    μ = vec(mean(X, dims=1))    # p-vector

    # Step 2: center
    Xc = X .- μ'

    # Step 3: covariance
    Σ = (Xc' * Xc) / (N - 1)    # p×p


    # Step 4: eigen-decomposition
    E = eigen(Symmetric(inv(Σ)))    # or Symmetric(Σ)
    evals = E.values
    evecs = E.vectors
    for j in 1:p
        evecs[:, j] .= evecs[:, j] ./ maximum(abs.(evecs[:, j]))
    end
    # Sort sloppiest first
    idx = sortperm(evals; rev=true)
    evals = evals[idx]
    evecs = evecs[:, idx]

    slop_ratio = evals[1] / evals[end]

    return (μ=μ, Σ=Σ,evals=evals, evecs=-evecs, sloppiness_ratio=slop_ratio)
end


function run_info_imb()
    # Storage for Δ
    Δ_fit = zeros(21)
    Δ_pred = zeros(21)
    data = matread("$path\\SMC_sample_allModels.mat")
    part_vals = data["Original"]["posterior_sample"]
    θ = log.(part_vals[:, 1:20])

    # Y space predictions
    Y_fit = data["Original"]["posterior_prediction"]
    Y_mat = matread("$path\\Y_prediction_O.mat")
    Y_pred = Y_mat["Yo"]  
    # θ → fit
    Δ_fit[end] = average_imb(θ, Y_fit)  
    # θ → pred
    Δ_pred[end] = average_imb(θ, Y_pred)  
    # Sloppness analysis
    res = posterior_covariance_equal(θ)
    # compute eigenparameter
    eigen_para =  θ * res.evecs
    num_directions = length(res.evals)
    for i = 1:num_directions
        # 1_st-i_th eigenparameter → y(1:10)
        Δ_fit[i] = average_imb(eigen_para[:,1:i], Y_fit) 
        # 1_st-i_th eigenparameter → y(11:20)
        Δ_pred[i] = average_imb(eigen_para[:,1:i], Y_pred)    
    end 
    Δ = vcat(Δ_fit, Δ_pred)
    return Δ
end

Δ =  run_info_imb()

@save "$path\\Δ.jld2" Δ
