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
using ForwardDiff
# Get the path of the current script
path = dirname(@__FILE__)


function model(Θ,t_fit,t_pred)
    b11, b12, b21, b22 = Θ[1], Θ[2], Θ[3], Θ[4]
    A = [0; 0]
    B = [b11 b12; b21 b22]
    y = zeros(2, t_pred)
    y[:,1]  .= [-0.9; -1.8]
    for t in 2:t_pred
        y[:,t] = A + B*y[:,t-1]
    end
    y_vec_fit = vcat(y[1, 1:t_fit], y[2, 1:t_fit])
    y_vec_pred = vcat(y[1, t_fit+1:t_pred], y[2, t_fit+1:t_pred]) 

    return y_vec_fit,y_vec_pred
end

function predict_matrix(Θ,t_fit,t_pred)
    (r,c) = size(Θ)
    Y_fit = zeros(r, t_fit*2)
    Y_pred = zeros(r, (t_pred-t_fit)*2)
    for i in 1:r
        y_vec_fit,y_vec_pred = model(Θ[i, :],t_fit,t_pred)
        Y_fit[i, :] .+= y_vec_fit
        Y_pred[i, :] .+= y_vec_pred
    end
    return Y_fit,Y_pred
end

# squared Euclidean distance between two row-vectors (views) without sqrt
@inline function sqeuclid(a::AbstractVector, b::AbstractVector)
    d = a .- b
    return dot(d, d)   # = sum(d.^2)
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
    evecs = evecs / norm(evecs)
    # Sort sloppiest first
    idx = sortperm(evals; rev=true)
    evals = evals[idx]
    evecs = evecs[:, idx]

    slop_ratio = evals[1] / evals[end]

    return (μ=μ, Σ=Σ,evals=evals, evecs=evecs, sloppiness_ratio=slop_ratio)
end


function run_info_imb(t_fit, t_pred)
    @load "$path\\Data.jld2" Data
    # Storage for Δ
    Δ_fit = zeros(9)
    Δ_pred = zeros(9)
    @load "$path\\part_vals.jld2" part_vals
    θ = part_vals[1][:, 1:4]
    # Y space predictions
    Y_fit,Y_pred = predict_matrix(θ,t_fit,t_pred)
    # θ → y(1:34)
    Δ_fit[end] = average_imb(θ, Y_fit)  
    # θ → y(34:68)
    Δ_pred[end] = average_imb(θ, Y_pred)  
    # Sloppness analysis
    res = posterior_covariance_equal(θ)
    # compute eigenparameter
    eigen_para =  θ * res.evecs

    num_directions = length(res.evals)
    for i = 1:num_directions
        # i_th eigenparameter → y(0:34)
        Δ_fit[i] = average_imb(eigen_para[:,i], Y_fit)
        # i_th eigenparameter → y(34:68)
        Δ_pred[i] = average_imb(eigen_para[:,i], Y_pred)
        # 1_st-i_th eigenparameter → y(0:34)
        Δ_fit[i+num_directions] = average_imb(eigen_para[:,1:i], Y_fit)
        # 1_st-i_th eigenparameter → y(34:68)
        Δ_pred[i+num_directions] = average_imb(eigen_para[:,1:i], Y_pred)  
        
    end 
    return vcat(Δ_fit, Δ_pred)
end



t_fit = 34
t_pred = 68

Δ =  run_info_imb(t_fit, t_pred)

@save "$path\\Δ.jld2" Δ


