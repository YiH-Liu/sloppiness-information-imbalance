using Distributions
using CSV
using DataFrames
using Plots
using JLD2
using KernelDensity
using LaTeXStrings
using LinearAlgebra
using StatsPlots
path = dirname(@__FILE__)


include("Print_Sloppy_Analysis_Table.jl")

function model(theta,t)
    T = maximum(t)
    A = [0; 0]
    B = [theta[1] theta[2]; theta[3] theta[4]]
    # Observation error
    H = [theta[5] 0; 0 theta[6]]
    α = zeros(2, T+1)
    y = zeros(2, T+1)
    α[:,1] .= [-0.9; -1.8]
    y[:,1]  .= α[:,1] + rand(MvNormal([0.0,0.0], Symmetric(H)))
    for t in 2:T+1
        α[:,t] = A + B*α[:,t-1]
        y[:,t] = α[:,t] + rand(MvNormal([0.0,0.0], Symmetric(H)))
    end
    return y
end


function reflected_kde(samples; a=0.0, b=1.0, ngrid=5000, h=nothing)
    x = samples[(samples .>= a) .& (samples .<= b)]
    N = length(x)
    N == 0 && error("No samples inside [a,b].")

    if isnothing(h)
        s = std(x)
        h = max(1.06 * s * N^(-1/5), 1e-3)
    end

    xgrid = range(a, b, length=ngrid)
    f = zeros(ngrid)

    for (k, xx) in enumerate(xgrid)
        f[k] = sum(
            pdf.(Normal(), (xx .- x) ./ h) .+
            pdf.(Normal(), (xx .- (2a .- x)) ./ h) .+
            pdf.(Normal(), (xx .- (2b .- x)) ./ h)
        ) / (N * h)
    end

    dx = step(xgrid)
    f ./= sum(f) * dx   # normalize density on [a,b]

    return xgrid, f
end

#Plot posterior
function Plot_posterior()

    @load "$path\\part_vals.jld2" part_vals
    part_vals = part_vals[1]

    # true parameter values
    B = [ 0.771 -0.233;   # species 2 positively affects species 1
        0.215 0.6601]   # species 1 negatively affects species 2
    # Process error
    Var_1 = 0.01; Var_2 = 0.01  
    theta_true = [ B[1,1]; B[1,2]; B[2,1]; B[2,2]; Var_1; Var_2]   
    param_names = [L"b_{11}", L"b_{12}",
                   L"b_{21}", L"b_{22}",
                   L"\sigma_1^2", L"\sigma_2^2"]

    # create a 2x3 grid of subplots
    plots = [plot(legend=false, grid=false,tickfontsize=10,
             labelfontsize=10,framestyle=:box) for i in 1:6]

    # boundaries for KDE
    lower = [-1, -1, -1, -1, 0, 0]
    upper = [1, 1, 1, 1, 10, 10]

    # boundaries for x-axis ticks in each subplot
    lower_fig =  [0.3, -0.4, -0.1, 0.4, 0, 0]
    upper_fig = [1, 0, 0.5, 1, 0.1, 0.1]
 
    # Plot KDE for each parameter 
    ll_paranam = [L"p(b_{11} \mid y)", L"p(b_{12} \mid y)",
                  L"p(b_{21} \mid y)", L"p(b_{22} \mid y)", 
                  L"p(\sigma_1^2 \mid y)", L"p(\sigma_2^2 \mid y)"]
    for i in 1:6
        a = lower[i]; b = upper[i]
        c = lower_fig[i]; d = upper_fig[i]
        samples = part_vals[:,i]
        x, f = reflected_kde(samples; a, b)
       
        plot!(plots[i], x, f, lw = 2, 
              xlabel = param_names[i], 
              ylabel = ll_paranam[i], color=:blue)

        # x-axis ticks
        xt = range(c, d, length=3)
        xt_latex = [latexstring(round(v, digits=2)) for v in xt]

        # y-axis ticks based on KDE density
        yt = range(minimum(f), maximum(f), length=3)
        yt_latex = [latexstring(round(v, digits=2)) for v in yt]

        plot!(plots[i], xlim=(c, d), xticks=(xt, xt_latex), 
              yticks=false, color=:blue)

        if i==1
            # annotate subplot (a) with label
            x_percentage = 0.45; y_percentage = 0.07
            x_cord = lower_fig[1] - (upper_fig[1] - lower_fig[1])*x_percentage
            y_cord = maximum(f).* (1 + y_percentage)
            #= annotate!(plots[1], x_cord, y_cord, text(L"(a)", :left, 12)) =#
        end
    end

    # plot vertical lines for true parameter values and prior distributions
    for i in 1:6
        if i > 4
            d = Exponential(0.5)
            x = 0:0.001:10
            s = 10
        else
            d = Uniform(lower[i], upper[i])
            x = lower[i]:0.1:upper[i]
            s = 10
        end

        vline!(plots[i], [theta_true[i]], linestyle=:dot, 
               color=:red, linewidth=2)
    
        plot!(plots[i], x, pdf.(d, x).*s,lw=2,
              label="prior", color=:black)
    end

   # combine subplots into a single figure with shared legend
    final_plot = plot(plots..., layout=(2,3), size=(550,250),
                      leftmargin=10mm,topmargin=1mm,bottommargin=3mm)

    display(final_plot)

    # create a separate legend plot
    legend_plot = plot([NaN], [NaN], label=L"\mathrm{Prior}", 
                       linewidth=2, legendfontsize=6, 
                       color=:black, legend=:top, framestyle=:none,
                       grid=false, axis=false, size=(80,50),
                       margin = 0Plots.mm,
                       left_margin =-500Plots.mm,
                       right_margin = -500Plots.mm,
                       top_margin = -8Plots.mm,
                       bottom_margin = 10Plots.mm)
    plot!(legend_plot, [NaN], [NaN], label=L"\mathrm{Posterior}",
          linewidth=2, color=:blue)
    plot!(legend_plot, [NaN], [NaN], label=L"\mathrm{True \ value}",
          linewidth=2, color=:red, linestyle=:dot)
    
    display(legend_plot)

    # Save to PDF
    savefig(final_plot, "$path\\posterior_marginals.pdf")
    savefig(legend_plot, "$path\\legend_posterior.pdf")
end

# Plot Data fitting
function fitting(t)
    @load "$path\\Data.jld2" Data
    @load "$path\\part_vals.jld2" part_vals
    Data = Data[1]
    part_vals = part_vals[1]

    # define the plot
    p = plot(xlabel = L"t", ylabel = L"y_{s}(t)", 
             legend=false, grid = false, 
             tickfontsize = 10, labelfontsize = 10, 
             size=(200,170))

    # observed data
    scatter!(p, t, Data[1, :], 
             label = L"\mathrm{Observed\ data}", 
             ms = 2, color = :red, markerstrokecolor = :red)
    scatter!(p, t, Data[2, :], 
             label = L"\mathrm{Observed\ data}", 
             ms = 2, color = :green, markerstrokecolor = :green)

    #  posterior predictive samples
    N = size(part_vals, 1)
    nt = length(t)
    Y1 = zeros(N, nt)
    Y2 = zeros(N, nt)
    for i in 1:N
        θ = part_vals[i, :]
        y = model(θ, t)
        Y1[i, :] = y[1, :]
        Y2[i, :] = y[2, :]  
    end

    # compute median and 95% credible interval at each t
    y1_median = [median(Y1[:, j]) for j in 1:nt]
    y1_lower  = [quantile(Y1[:, j], 0.025) for j in 1:nt]
    y1_upper  = [quantile(Y1[:, j], 0.975) for j in 1:nt]

    y2_median = [median(Y2[:, j]) for j in 1:nt]
    y2_lower  = [quantile(Y2[:, j], 0.025) for j in 1:nt]
    y2_upper  = [quantile(Y2[:, j], 0.975) for j in 1:nt]

    # plot median and credible interval
    plot!(p, t, y1_median,
    ribbon = (y1_median .- y1_lower, y1_upper .- y1_median),
    linewidth = 2, fillalpha = 0.25, color = :red)

    plot!(p, t, y2_median,
    ribbon = (y2_median .- y2_lower, y2_upper .- y2_median),
    linewidth = 2, fillalpha = 0.25, color = :green)

    # x and y axis ticks
    xt = [0,8,16,24,34]
    xt_latex = [latexstring("$x") for x in xt]

    ymin = minimum(vcat(vec(Data),  vec(Y1), vec(Y2)))
    ymax = maximum(vcat(vec(Data), vec(Y1), vec(Y2)))
    yt = [-2,-1,0,1]
    yt_latex = [latexstring("$v") for v in yt]

    # plot with custom ticks
    xlmin = minimum(t) - 0.5; xlmax = maximum(t) + 0.5
    ylmin = ymin - 0.1; ylmax = 1
    plot!(p, xticks = (xt, xt_latex), xlims = (xlmin, xlmax),
          yticks = (yt, yt_latex), ylims = (ylmin, ylmax),
          top_margin = 7mm)

    # annotate plot with label (b)
    x_percentage = 0.37; y_percentage = 0.1
    x_cord = xlmin - (xlmax - xlmin)*x_percentage
    y_cord = ylmax + (ylmax - ylmin)*y_percentage
    annotate!(p,x_cord, y_cord, text(L"(b)", :left, 12))
    display(p)

    # plot legend separately
    legend_plot_fitting = plot([NaN], [NaN], 
                               label=L"y_1^{\mathrm{o}}", seriestype = :scatter,
                               marker = :circle, color=:red,legend = :top,
                               markersize=1, markerstrokecolor=:red, legendfontsize=10, 
                               framestyle=:none, grid=false, axis=false, size=(200,170), 
                               margin = 0Plots.mm,
                               left_margin =-10Plots.mm,
                               right_margin = -0Plots.mm,
                               top_margin = -0Plots.mm,
                               bottom_margin = 0Plots.mm)
    
    plot!(legend_plot_fitting, [NaN], [NaN],seriestype = :scatter,
          marker = :circle, color=:green, 
          markersize=1, markerstrokecolor=:green,
          label=L"y_2^{\mathrm{o}}")

    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 1, color = :red,
          label = L"\tilde{f}_{1}(t)")

    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 1, color = :green,
          label = L"\tilde{f}_{2}(t)")

    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 8, color = :red, alpha = 0.25,
          label = L"C_{1,0.95}")

    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 8, color = :green, alpha = 0.25,
          label = L"C_{2,0.95}")
    display(legend_plot_fitting)

    # Save to PDF
    savefig(p, "$path\\Fitting.pdf")
    savefig(legend_plot_fitting, "$path\\legend_plot_fitting.pdf")

end

function Predicting(t)
    @load "$path\\part_vals.jld2" part_vals
    part_vals = part_vals[1]

    # define the plot
    p = plot(xlabel = L"t", ylabel = L"y_{s}(t)", 
             legend=false, grid = false, 
             tickfontsize = 10, labelfontsize = 10, 
             size=(200,170))


    #  posterior predictive samples
    N = size(part_vals, 1)
    nt = length(t)
    Y1 = zeros(N, nt)
    Y2 = zeros(N, nt)
    for i in 1:N
        θ = part_vals[i, :]
        y = model(θ, t)
        Y1[i, :] = y[1, :]
        Y2[i, :] = y[2, :]  
    end

    # compute median and 95% credible interval at each t
    y1_median = [median(Y1[:, j]) for j in 1:nt]
    y1_lower  = [quantile(Y1[:, j], 0.025) for j in 1:nt]
    y1_upper  = [quantile(Y1[:, j], 0.975) for j in 1:nt]

    y2_median = [median(Y2[:, j]) for j in 1:nt]
    y2_lower  = [quantile(Y2[:, j], 0.025) for j in 1:nt]
    y2_upper  = [quantile(Y2[:, j], 0.975) for j in 1:nt]

    # plot median and credible interval
    plot!(p, t, y1_median,
    ribbon = (y1_median .- y1_lower, y1_upper .- y1_median),
    linewidth = 2, fillalpha = 0.25, color = :red)

    plot!(p, t, y2_median,
    ribbon = (y2_median .- y2_lower, y2_upper .- y2_median),
    linewidth = 2, fillalpha = 0.25, color = :green)

    # x and y axis ticks
    xt = [34,42,51,60,68]
    xt_latex = [latexstring("$x") for x in xt]

    ymin = -2
    ymax = 2
    yt = [-2,-1,0,1]
    yt_latex = [latexstring("$v") for v in yt]

    # plot with custom ticks
    xlmin = 34 - 0.5; xlmax = 68 + 0.5
    ylmin = ymin - 0.1; ylmax = 1
    plot!(p, xticks = (xt, xt_latex), xlims = (xlmin, xlmax),
          yticks = (yt, yt_latex), ylims = (ylmin, ylmax),
          top_margin = 7mm)

    # annotate plot with label (b)
    x_percentage = 0.37; y_percentage = 0.1
    x_cord = xlmin - (xlmax - xlmin)*x_percentage
    y_cord = ylmax + (ylmax - ylmin)*y_percentage
    annotate!(p,x_cord, y_cord, text(L"(c)", :left, 12))
    display(p)

    
    # Save to PDF
    savefig(p, "$path\\Predict1.pdf")

end

function Plot_info_imb()
    @load "$path\\Δ.jld2" Δ

    values = Δ
    L_val = Int(length(values)/2)
    values_fit = values[5:L_val]
    values_pred = values[L_val+5:2*L_val]
    x_bar = collect(5:L_val) .* 2
    x_label = [L"\hat{\theta}^1", L"\hat{\theta}^2",
               L"\hat{\theta}^3", L"\hat{\theta}^4", 
               L"\theta"]
    xt = (x_bar, x_label)

    bar_colors = vcat(fill(:green, 4),fill(:red, 1))
    
    p_fit = bar(x_bar, values_fit, xticks = xt,
                yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_fit,5.5, 1.3, text(L"(d)", :left, 12))
    annotate!(p_fit, 13.35, 1.2, text(L"y_{\mathrm{obs}}", :left, 12))

    p_pred = bar(x_bar, values_pred, xticks = xt,
                 yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                 tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                 ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                 legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_pred,5.5, 1.3, text(L"(e)", :left, 12))
    annotate!(p_pred, 13.35, 1.2, text(L"y_{\mathrm{pred}}", :left, 12))
    
    p = plot(p_fit,p_pred, layout=(1,2), size=(550,200),leftmargin=4mm,topmargin=7mm)

    display(p)
    savefig(p, "$path\\Δ.pdf")
end


Plot_posterior()
t = 0:1:34
fitting(t)
Predicting(0:1:68)
Plot_info_imb()

parameter_labels = [
    L"\theta_1",
    L"\theta_2",
    L"\theta_3",
    L"\theta_4"
]

@load "$path\\part_vals.jld2" part_vals
part_vals = part_vals[1][:, 1:4]   
green_grad = cgrad(
    [RGB(0.15, 0.65, 0.30), RGB(1, 1, 1), RGB(0.15, 0.65, 0.30)],
    [0.0, 0.5, 1.0]
)

blue_grad = cgrad(
    [RGB(0.20, 0.45, 0.95), RGB(1, 1, 1), RGB(0.20, 0.45, 0.95)],
    [0.0, 0.5, 1.0]
)

p, result, cb = sloppiness_table_from_samples(
    part_vals;
    parameter_labels = parameter_labels,
    eigenvalue_digits = 4,
    eigenvector_digits = 1,
    signed_colour = true,
    main_size = (280, 190),
    cell_grad = blue_grad,
)
annotate!(p, 0, 6 - 0.15, text(L"(a)", 12, :black, :left))
display(p)
savefig(p, "$path\\sloppytable.pdf")
savefig(cb, "$path\\sloppytable_colorbar.pdf")