using Distributions
using CSV
using DataFrames
using Plots
using JLD2
# Get the path of the current script
path = dirname(@__FILE__)



function model(theta,t)
    (K, r, y0, σ) = theta
    y = K ./ (1 .+ ((K - y0) / y0) .* exp.(-r .* t))
    return y
end

function simulate_data(t, theta)
    (K, r, y0, σ) = theta
    y_clean = model(theta, t)
    noise = rand(Normal(0.0, σ), length(t))
    return y_clean .+ noise
end




function Save_data(theta,t)
    Data = simulate_data(t, theta)
    display(scatter(t,Data))
    @save "$path\\Data.jld2" Data
end
σ = 1
theta = [500.0, 0.5, 2.0, σ]
t = range(0, 4, length=20)
Save_data(theta,t)
