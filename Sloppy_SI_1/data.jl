using Distributions
using CSV
using DataFrames
using Plots
using JLD2
# Get the path of the current script
path = dirname(@__FILE__)


function model(theta,T)
    t = 1:1:T
    y = theta[1] .+ theta[2] .+ theta[3].*t
    return y,t
end

function simulate_data(T, theta)
    y_clean,t = model(theta,T)
    noise = rand(Normal(0.0, theta[4]), length(t))
    return y_clean .+ noise, t
end


function run_date(σ)
    T = 10
    A = Matrix{Float64}(undef, T, 0) 
    for i in σ
       theta = [0.3,0.5,0.4,i]
       data,t = simulate_data(T, theta)
       A = hcat(A, data)
    end
    return A
end

function Save_data(Num_of_Sample,σ)
    Data = Vector{Matrix{Float64}}()
    for i = 1:Num_of_Sample
        A = run_date(σ)
        push!(Data, A)
    end
    @save "$path\\Data.jld2" Data
end
σ = [0.5]
Num_of_Sample = 1
Save_data(Num_of_Sample,σ)
