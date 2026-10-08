using Distributions
using CSV
using DataFrames
using Plots
using JLD2
using KernelDensity
using LaTeXStrings
using Measures
path = dirname(@__FILE__)
include("Print_Sloppy_Analysis_Table.jl")
function model(theta,t)
    (K, r, y0, σ) = theta
    y = K ./ (1 .+ ((K - y0) / y0) .* exp.(-r .* t))
    noise = rand(Normal(0.0, σ), length(t))
    return y .+ noise
end


function reflected_kde(samples; a = 0.0, b = 1.0, ngrid = 5000, h = nothing)

    # remove NaN/Inf and keep samples inside [a, b]
    x = samples[isfinite.(samples) .& (samples .>= a) .& (samples .<= b)]

    N = length(x)
    N == 0 && error("No finite samples inside [a,b].")
    b <= a && error("Need b > a.")
    ngrid < 2 && error("Need ngrid >= 2.")

    if isnothing(h)
        if N == 1
            h = 0.05 * (b - a)
        else
            s = std(x; corrected = false)
            h = max(1.06 * s * N^(-1/5), 1e-3)
        end
    end

    !isfinite(h) && error("Bandwidth h is not finite.")
    h <= 0 && error("Bandwidth h must be positive.")

    xgrid = range(a, b, length = ngrid)
    f = zeros(Float64, ngrid)

    for (k, xx) in enumerate(xgrid)
        z1 = (xx .- x) ./ h
        z2 = (xx .- (2a .- x)) ./ h
        z3 = (xx .- (2b .- x)) ./ h

        f[k] = sum(pdf.(Normal(), z1) .+
                   pdf.(Normal(), z2) .+
                   pdf.(Normal(), z3)) / (N * h)
    end

    dx = step(xgrid)
    area = sum(f) * dx

    if !isfinite(area) || area <= 0
        error("KDE normalization failed: area = $area")
    end

    f ./= area

    return xgrid, f
end

#Plot posterior
function Plot_posterior()
    @load "$path\\part_vals.jld2" part_vals

    # True parameter values
    K = 500.0; r = 0.5; y0 = 2.0; σ = 1.0

    theta_true = [K, r, y0, σ]

    param_names = [L"K", L"r",
                   L"y_0", L"\sigma"]

    # create a 2x2 grid of subplots
    plots = [plot(legend=false, grid=false,tickfontsize=10,
             labelfontsize=12,framestyle=:box) for i in 1:4]

    # boundaries for KDE
    lower = [10.0, 0.05, 0.5, 0]
    upper = [1000.0, 1.5, 20.0,10]

    # boundaries for x-axis ticks in each subplot
    lower_fig =  [10.0, 0.2, 0.5, 0]
    upper_fig = [1000.0, 0.8, 3,3]

    llparanames = [L"p( K \mid y)", L"p( r \mid y)",
                   L"p( y_0 \mid y)", L"p( \sigma \mid y)"]


    # Plot KDE for each parameter      
    for i in 1:4
        a = lower[i]; b = upper[i]
        c = lower_fig[i]; d = upper_fig[i]
        samples = part_vals[:,i]
        x, f = reflected_kde(samples; a, b)
        plot!(plots[i], x, f, lw = 2, 
              xlabel = param_names[i], 
              ylabel = llparanames[i], color=:blue)

        # x-axis ticks
        xt = range(c, d, length=5)
        xt_latex = [latexstring(round(v, digits=1)) for v in xt]

        # y-axis ticks based on KDE density
        yt = range(minimum(f), maximum(f), length=3)
        yt_latex = [latexstring(round(v, digits=2)) for v in yt]

        plot!(plots[i], xlim=(c, d), xticks=(xt, xt_latex), 
              yticks=false, color=:blue)

        if i==1
            # annotate subplot (a) with label
            x_percentage = 0.3; y_percentage = 0.025
            x_cord = lower_fig[1] - (upper_fig[1] - lower_fig[1])*x_percentage
            y_cord = maximum(f).* (1 + y_percentage)
            display([x_cord, y_cord])
            annotate!(plots[1], x_cord, y_cord, text(L"(a)", :left, 12))
        end
    end 
    
    # plot vertical lines for true parameter values and prior distributions
    for i in 1:4
        if i==4
            d = Exponential(0.5)
            x = 0:0.001:10
            s = 1
        else
            d = Uniform(lower[i], upper[i])
            x = lower[i]:0.1:upper[i]
            s = 5
            if i==1
                s = 1
            end
        end

        vline!(plots[i], [theta_true[i]], linestyle=:dot, 
               color=:red, linewidth=2)
    
        plot!(plots[i], x, pdf.(d, x).*s,lw=2,
              label="prior", color=:black)
    end

    # combine subplots into a single figure with shared legend
    final_plot = plot(plots..., layout=(2,2), size=(550,250),
                      leftmargin=13mm,topmargin=5mm,bottommargin=3mm)

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

function fitting(t)

    @load "$path\\Data.jld2" Data
    @load "$path\\part_vals.jld2" part_vals

    # define the plot
    p = plot(xlabel = L"t", ylabel = L"y(t)", 
             legend=false, grid = false, 
             tickfontsize = 10, labelfontsize = 10, 
             size=(200,170))

    # observed data
    scatter!(p, t, Data, 
             label = L"\mathrm{Observed\ data}", 
             ms = 4, color = :black, markerstrokecolor = :black)

    #  posterior predictive samples
    N = size(part_vals, 1)
    nt = length(t)
    Y = zeros(N, nt)
    for i in 1:N
        θ = part_vals[i, :]
        Y[i, :] = model(θ, t)
    end

    # compute median and 95% credible interval at each t
    y_median = [median(Y[:, j]) for j in 1:nt]
    y_lower  = [quantile(Y[:, j], 0.025) for j in 1:nt]
    y_upper  = [quantile(Y[:, j], 0.975) for j in 1:nt]

    # plot median and credible interval
    plot!(p, t, y_median,
    ribbon = (y_median .- y_lower, y_upper .- y_median),
    linewidth = 2, fillalpha = 0.25, color = :orange)

    # x and y axis ticks
    xt = range(minimum(t), maximum(t), length=5)
    xt_latex = [latexstring("$(round(x, digits = 1))") for x in xt]

    ymin = minimum(vcat(vec(Data),  vec(Y)))
    ymax = maximum(vcat(vec(Data), vec(Y)))
    yt = range(ymin, ymax, length = 4,)
    yt_latex = [latexstring("$(round(v, digits = 1))") for v in yt]

    # plot with custom ticks
    xlmin = minimum(t) - 0.5; xlmax = maximum(t) + 0.5
    ylmin = ymin - 0.5; ylmax = ymax + 0.5
    plot!(p, xticks = (xt, xt_latex), xlims = (xlmin, xlmax),
          yticks = (yt, yt_latex), ylims = (ylmin, ylmax),
          top_margin = 7mm)

    # annotate plot with label (b)
    x_percentage = 0.45; y_percentage = 0.1
    x_cord = xlmin - (xlmax - xlmin)*x_percentage
    y_cord = ylmax + (ylmax - ylmin)*y_percentage
    annotate!(p,x_cord, y_cord, text(L"(b)", :left, 12))
    display(p)

    # plot legend separately
    legend_plot_fitting = plot([NaN], [NaN], 
                               label=L"y_{\mathrm{obs}}", seriestype = :scatter,
                               marker = :circle, color=:black,legend = :top,
                               markersize=1, markerstrokecolor=:black, legendfontsize=10, 
                               framestyle=:none, grid=false, axis=false, size=(200,100), 
                               margin = 0Plots.mm,
                               left_margin =-10Plots.mm,
                               right_margin = -0Plots.mm,
                               top_margin = -0Plots.mm,
                               bottom_margin = 0Plots.mm)
    
    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 1, color = :orange,
          label = L"\tilde{f}(t)")

    plot!(legend_plot_fitting, [NaN], [NaN],
          linewidth = 8, color = :orange, alpha = 0.25,
          label = L"C_{0.95}")

    #= display(legend_plot_fitting) =#

    # Save to PDF
    savefig(p, "$path\\Fitting.pdf")
    savefig(legend_plot_fitting, "$path\\legend_plot_fitting.pdf")
end


function Plot_info_imb()
    @load "$path\\Δ.jld2" Δ

    values = Δ
    L_val = Int(length(values)/3)
    values_fit = values[4:L_val]
    values_pred1 = values[L_val+4:2*L_val]
    values_pred2 = values[2*L_val+4:3*L_val]
    x_bar = collect(4:L_val) .* 2
    x_label = [L"\hat{\theta}^1", L"\hat{\theta}^2",
               L"\hat{\theta}^3", L"\theta"]
    xt = (x_bar, x_label)


    bar_colors = vcat(fill(:green, 3),fill(:red, 1))

    p_fit = bar(x_bar, values_fit, xticks = xt,
                yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_fit,2.5, 1.3, text(L"(d)", :left, 12))
    annotate!(p_fit, 10.35, 1.2, text(L"y_{\mathrm{obs}}", :left, 12))

    p_pred1 = bar(x_bar, values_pred1, xticks = xt,
                  yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                  tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                  ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                  legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_pred1,2.5, 1.3, text(L"(e)", :left, 12))
    annotate!(p_pred1, 10.35, 1.2, text(L"y_{\mathrm{pred1}}", :left, 12))

    p_pred2 = bar(x_bar, values_pred2, xticks = xt,
                  yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                  tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                  ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                  legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_pred2,2.5, 1.3, text(L"(f)", :left, 12))
    annotate!(p_pred2, 10.35, 1.2, text(L"y_{\mathrm{pred2}}", :left, 12))

    p = plot(p_fit,p_pred1,p_pred2, layout=(1,3), size=(550,200),leftmargin=4mm,topmargin=7mm)

    display(p)
    savefig(p, "$path\\Δ.pdf")
end



Plot_posterior()
t = range(0, 4, length=20)
fitting(t)
Plot_info_imb()

parameter_labels = [
    L"K",
    L"r",
    L"y_0"
]

@load "$path\\part_vals.jld2" part_vals
part_vals = part_vals[:, 1:3]   
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
    main_size = (280, 170),
    cell_grad = blue_grad,
)
annotate!(p, 0, 5 - 0.15, text(L"(c)", 12, :black, :left))
display(p)
savefig(p, "$path\\sloppytable.pdf")
savefig(cb, "$path\\sloppytable_colorbar.pdf")