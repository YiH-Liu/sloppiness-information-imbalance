using Distributions
using CSV
using DataFrames
using Plots
using JLD2
# Get the path of the current script
path = dirname(@__FILE__)



function model(theta,t)
    (θ1,θ2,θ3) = theta
    y = θ1 .* θ2 .*exp.(-θ3 .* t)
    return y
end

function simulate_data(t, theta)
    (θ1,θ2,θ3,σ) = theta
    y_clean = model(theta, t)
    noise = rand(Normal(0.0, σ), length(t))
    return y_clean .+ noise
end




function Save_data(theta,t)
    Data = simulate_data(t, theta)
    display(scatter(t,Data))
    @save "$path\\Data.jld2" Data
end
σ = 0.5
theta = [2.0, 3.0, 0.4, σ ]  
t = range(0, 10, length=20)
Save_data(theta,t)
