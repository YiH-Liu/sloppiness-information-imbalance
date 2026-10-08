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


function model(θ,t)
    (θ1, θ2, θ3, σ) = θ
    y = θ1 .+ θ2.*t .+ θ3 .* t.^2
    return y
end


function loglike(theta, extra_args) #function to evaluate the loglikelihood for the data given parameters a
    t = range(0, 4, length=10)
    θ = (theta[1], theta[2], theta[3], theta[4])
    data = extra_args["data"]
    y = model(θ, t)
    dist = Normal(0.0, theta[4])
    residuals = data .- y
    e=loglikelihood(dist,residuals)
    return e
end



# Function to simulate the prior
function prior_sim(extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    λ    = extra_args["lambda"]
    num_params = extra_args["num_params"]
    theta = zeros(num_params)
    theta[1:num_params-1] = rand(num_params-1) .* (upper .- lower) .+ lower
    theta[num_params] = rand(Exponential(λ))
    return theta

end

# Transformation functions
function trans_f(theta, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    num_params = extra_args["num_params"]

    theta_trans = zeros(num_params)
    theta_trans[1:num_params-1] = log.((theta[1:num_params-1] .- lower)./(upper .- theta[1:num_params-1]))
    theta_trans[num_params] = log(theta[num_params])
    return(theta_trans)
end

function trans_finv(theta_trans, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]
    num_params = extra_args["num_params"]

    theta = zeros(num_params)
    theta[1:num_params-1] = (lower .+ upper.*exp.(theta_trans[1:num_params-1]))./(1 .+ exp.(theta_trans[1:num_params-1])) 
    theta[num_params] = exp(theta_trans[num_params])
    return(theta)
end

# The log prior density must be defined on the transformed space
function logprior_func(theta_trans, extra_args)
    λ = extra_args["lambda"]
    num_params = extra_args["num_params"]
    logp= sum(theta_trans[1:num_params-1] .- 2*log.(1 .+ exp.(theta_trans[1:num_params-1])))
    logp += log(λ) - λ * exp(theta_trans[num_params]) + theta_trans[num_params]
    return logp
end



function run_smc(Data)
    # Tuning parameters of SMC
    N = 1000  # Particles
    E = N / 2  # ESS resampling threshold
    c = 0.01   # Probability all particles move at least once during MCMC is 1-c
    # Setting up extra_args and prior_funcs (functions in a dictionary)
    lower = [-10, -10, -10]
    upper = [10, 10, 10]
    λ = 0.5

    
   
    # Run the smc_generic algorithm for jth sample simulated with Σ[i]
    extra_args = Dict("num_params" => 4, "data" => Data, "lower" => lower, "upper" => upper, "lambda" => λ)   # "num_params" MUST be specified
    prior_funcs = Dict("prior_sim" => prior_sim, "trans_f" => trans_f, "trans_finv" => trans_finv, "logprior_func" => logprior_func)
    part_vals = smc_generic(N, E, c, loglike, prior_funcs, extra_args)
    @save "$path\\part_vals.jld2" part_vals

end
@load "$path\\Data.jld2" Data
run_smc(Data)

