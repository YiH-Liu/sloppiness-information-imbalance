using Distributions
using CSV
using DataFrames
using Plots
using JLD2

# Get the path of the current script
path = dirname(@__FILE__)

# ───────────────────────────────
#  Simulate two-var MAR(1) + measurement noise
# ───────────────────────────────
function model(theta,α0,T)
    A = [theta[1]; theta[2]]
    B = [theta[3] theta[4]; theta[5] theta[6]]
    # Observation error
    H = [theta[7] 0; 0 theta[8]]
    α = zeros(2, T+1)
    y = zeros(2, T+1)
    α[:,1] .= α0
    y[:,1]  .= α[:,1] + rand(MvNormal([0.0,0.0], Symmetric(H)))
    for t in 2:T+1
        α[:,t] = A + B*α[:,t-1]
        y[:,t] = α[:,t] + rand(MvNormal([0.0,0.0], Symmetric(H)))
    end
    return α, y
end

function simulate_data(theta,α0,T)
    (α, y) = model(theta,α0,T)
    return y
end




function Save_data(Num_of_Sample,α0,theta,T)
    Data = Vector{Matrix{Float64}}()
    for i = 1:Num_of_Sample
        y = simulate_data(theta,α0,T)
        push!(Data, y)
    end
    @save "$path\\Data.jld2" Data
    return Data
end


Num_of_Sample = 1
# Time series length
T = 34 
# Number of species
p = 2
# Initial condition
α0 = [-0.9; -1.8]
# Growth rates
A = [0 ; 0]
# Interaction terms
B = [ 0.771 -0.233;   # species 2 positively affects species 1
    0.215 0.6601]    # species 1 negatively affects species 2
# Process error
Var_1 = 0.01
Var_2 = 0.01

Θ = [A[1];A[2]; B[1,1]; B[1,2]; B[2,1]; B[2,2]; Var_1; Var_2]
Data = Save_data(Num_of_Sample,α0,Θ,T)
p = plot(0:1:T, Data[1][1,:],linestyle=:dot, label="Species 1", xlabel="Time", ylabel="Abundance", title="Simulated Time Series")
plot!(0:1:T, Data[1][2,:], linestyle=:dot, label="Species 2",ylim=(-3,3))
