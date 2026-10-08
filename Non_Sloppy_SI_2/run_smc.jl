using JLD2
using LinearAlgebra
using Plots 
using DifferentialEquations
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


# Get the path of the current script
path = dirname(@__FILE__)


include("smc_generic.jl")


function loglik_kf(y, A, B, Q, H, α0)
    k, Tplus1 = size(y)          # includes t=0

    α_prev = α0
    P_prev = zeros(k,k)
    ll = -0.5*(k*log(2π) +
           logdet(Symmetric(H)) +
           (y[:,1] .- α0)' * (H \ (y[:,1] .- α0)))


    for i = 2:Tplus1
        St = B*P_prev*B' .+ Q .+ H
        vt = y[:,i] .- (A .+ B*α_prev)
        ll += -0.5*(k*log(2π) +
            logdet(Symmetric(St)) +
            vt' * (St \ vt))
        # UPDATE 
        Ik = Matrix{eltype(y)}(I, k, k)
        Kt = (B*P_prev*B' .+ Q) *  (St \ Ik)
        α_prev = (A .+ B*α_prev) .+ Kt*vt
        P_prev = (B * P_prev * B' .+ Q) .- Kt * St * Kt'
    end
    return ll
end


function loglike(theta, extra_args) #function to evaluate the loglikelihood for the data given parameters a

    data = extra_args["data"]
    A = [0, 0]
    B = [theta[1] theta[2];
         theta[3] theta[4]]
    Q = 0
    H = [theta[5] 0; 0 theta[6]]
    α0 = [-0.9; -1.8]
    e = loglik_kf(data, A, B, Q, H, α0)
    return e
end



# Function to simulate the prior
function prior_sim(extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    λ    = extra_args["lambda"]
    num_params = extra_args["num_params"]
    theta = zeros(num_params)
    theta[1:num_params-2] = rand(num_params-2) .* (upper .- lower) .+ lower
    theta[num_params-1:num_params] = rand(Exponential(λ), 2)
    return theta

end

# Transformation functions
function trans_f(theta, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    num_params = extra_args["num_params"]

    theta_trans = zeros(num_params)
    theta_trans[1:num_params-2] = log.((theta[1:num_params-2] .- lower)./(upper .- theta[1:num_params-2]))
    theta_trans[num_params-1:num_params] = log.(theta[num_params-1:num_params])
    return(theta_trans)
end

function trans_finv(theta_trans, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    num_params = extra_args["num_params"]

    theta = zeros(num_params)
    theta[1:num_params-2] = (lower .+ upper.*exp.(theta_trans[1:num_params-2]))./(1 .+ exp.(theta_trans[1:num_params-2])) 
    theta[num_params-1:num_params] = exp.(theta_trans[num_params-1:num_params])
    return(theta)
end

# The log prior density must be defined on the transformed space
function logprior_func(theta_trans, extra_args)
    λ = extra_args["lambda"]
    num_params = extra_args["num_params"]
    logp= sum(theta_trans[1:num_params-2] .- 2*log.(1 .+ exp.(theta_trans[1:num_params-2])))
    logp += log(λ) - λ * exp(theta_trans[num_params-1]) + theta_trans[num_params-1]
    logp += log(λ) - λ * exp(theta_trans[num_params]) + theta_trans[num_params]
    return logp
end



function run_smc(Data)
    # Tuning parameters of SMC
    N = 1000  # Particles
    E = N / 2  # ESS resampling threshold
    c = 0.01   # Probability all particles move at least once during MCMC is 1-c
    # Setting up extra_args and prior_funcs (functions in a dictionary)
    lower = [-1, -1, -1, -1]
    upper = [1, 1, 1, 1]
    λ = 0.5

    
    part_vals = Vector{Matrix{Float64}}()
    for j = 1:length(Data)
        # Run the smc_generic algorithm for jth sample simulated with Σ[i]
        extra_args = Dict("num_params" => 6, "data" => Data[j], "lower" => lower, "upper" => upper, "lambda" => λ)   # "num_params" MUST be specified
        prior_funcs = Dict("prior_sim" => prior_sim, "trans_f" => trans_f, "trans_finv" => trans_finv, "logprior_func" => logprior_func)
        part_vals_i = smc_generic(N, E, c, loglike, prior_funcs, extra_args)
        push!(part_vals, part_vals_i)
    end
    @save "$path\\part_vals.jld2" part_vals
    
end
@load "$path\\Data.jld2" Data

run_smc(Data)

@load "$path\\part_vals.jld2" part_vals
