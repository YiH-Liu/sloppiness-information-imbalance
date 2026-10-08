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

# Get the path of the current script
path = dirname(@__FILE__)

include("smc_generic.jl")


# Tuning parameters of SMC
N = 1000  # Particles
E = N / 2  # ESS resampling threshold
c = 0.01   # Probability all particles move at least once during MCMC is 1-c
T = 100
# Read the CSV file: data
datas = DataFrame(CSV.File("$path\\data.csv")); 



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
        P_prev = (B * P_prev * B' + Q) .- Kt * St * Kt'
    end
    return ll
end


function loglike_Mar(theta, extra_args) #function to evaluate the loglikelihood for the data given parameters a

    data11 = extra_args["data11"]
    data12 = extra_args["data12"]
    A = [theta[1],theta[2]]
    B = [theta[3] theta[4];
         theta[5] theta[6]]
    Q = Matrix{Float64}(I,2,2).*theta[7]
    H = Matrix{Float64}(I,2,2).*theta[8]
    α0 = [10.0,10.0]
    e = loglik_kf(y, A, B, Q, H, α0)
    return e
end


# Function to simulate the prior
function prior_sim(extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]

    num_params = extra_args["num_params"]

    return rand(num_params).*(upper .- lower) .+ lower

end

# Transformation functions
function trans_f(theta, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]

    theta_trans = log.((theta .- lower)./(upper .- theta))
    return(theta_trans)
end

function trans_finv(theta_trans, extra_args)
    lower = extra_args["lower"]
    upper = extra_args["upper"]

    theta = (lower .+ upper.*exp.(theta_trans))./(1 .+ exp.(theta_trans)) 
    return(theta)
end

# The log prior density must be defined on the transformed space
function logprior_func(theta_trans, extra_args)
    return sum(theta_trans .- 2*log.(1 .+ exp.(theta_trans)))
end

# Setting up extra_args and prior_funcs (functions in a dictionary)
lower = [0  ,0  ,-1 ,-1 ,-1 ,-1 ,0 ,0]
upper = [10 ,10 ,1  ,1  ,1  ,1  ,0.1 ,0.1]

extra_args = Dict("num_params" => 8, "data11" => data11, "data12" => data12, "lower" => lower, "upper" => upper)   # "num_params" MUST be specified
prior_funcs = Dict("prior_sim" => prior_sim, "trans_f" => trans_f, "trans_finv" => trans_finv, "logprior_func" => logprior_func)

# Run the smc_generic algorithm
part_vals = smc_generic(N, E, c, loglike_Mar, prior_funcs, extra_args)


# plot posteriors and compare to true parameters

A1 = 5; A2 = 5; B11 = 0.6; B12 = 0.2; B21 = 0 ; B22 = 0.7;  q= 0.05; h=0.05
theta_true = [A1, A2, B11, B12, B21, B22, q, h]

# Parameter names for labeling
param_names = ["A1", "A2", "B11", "B12", "B21", "B22", "q", "h"]

# Create list of plots
plots = []

for i in 1:8
    p = density(part_vals[:, i], legend=false, linewidth=2, color=:blue)
    vline!([theta_true[i]], linestyle=:dash, color=:red, label="")
    title!(param_names[i])
    push!(plots, p)
end
# Combine into one layout
final_plot = plot(plots..., layout = (3, 3), size = (1000, 800))

# Save to PDF
savefig(final_plot, "posterior_marginals.pdf")

df = DataFrame(part_vals, :auto)   # auto-generate column names x1, x2, ...
CSV.write("$path\\matrix.csv", df)

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

    # Sort sloppiest first
    idx = sortperm(evals; rev=true)
    evals = evals[idx]
    evecs = evecs[:, idx]

    slop_ratio = evals[1] / evals[end]

    return (μ=μ, Σ=Σ,evals=evals, evecs=evecs, sloppiness_ratio=slop_ratio)
end


sloppy_res = posterior_covariance_equal(part_vals[:,1:6])

function eig_table_lambda_params(evals, evecs)
    param_names = ["A1", "A2", "B11", "B12", "B21", "B22"]

    p = length(evals)
    @assert size(evecs, 1) == p && size(evecs, 2) == p
    @assert length(param_names) == p "Parameter name count must match p"

    # Create DataFrame with lambda column
    df = DataFrame(lambda = evals)

    # Each row i: [lambda_i, evecs[1,i], evecs[2,i], ..., evecs[p,i]]
    for j in 1:p
        df[!, Symbol(param_names[j])] = evecs[j, :]
    end

    return df
end

evals = sloppy_res.evals./ sloppy_res.evals[1]

evals = round.(evals, digits=5)
evecs = round.(sloppy_res.evecs, digits=1)
df = eig_table_lambda_params(evals, evecs)#=  =#





