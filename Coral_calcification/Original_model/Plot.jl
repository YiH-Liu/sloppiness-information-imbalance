using Distributions
using CSV
using DataFrames
using Plots
using JLD2
using KernelDensity
using LaTeXStrings
using Measures
using MAT
path = dirname(@__FILE__)

include("Print_Sloppy_Analysis_Table.jl")


function Plot_info_imb()
    @load "$path\\Δ.jld2" Δ   
    x_label = vcat([latexstring("\\hat{\\theta}^{$i}") for i in 1:20],
                   [L"\theta"])

    values = Δ
    L_val = Int(length(values)/2)
    values_fit = values[1:L_val]
    values_pred = values[L_val+1:end]
    x_bar = collect(1:L_val) 
    xt = (x_bar, x_label)


    bar_colors = vcat(fill(:green, 20),fill(:red, 1))

    p_fit = bar(x_bar, values_fit, xticks = xt,
                yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_fit,-4, 1.3, text(L"(a)", :left, 12))
    annotate!(p_fit, 10.35, 1.2, text(L"y_{\mathrm{obs}}", :left, 12))

    p_pred = bar(x_bar, values_pred, xticks = xt,
                 yticks = (0:0.2:1.0,[L"0.0", L"0.2", L"0.4", L"0.6", L"0.8", L"1.0"]),
                 tickfontsize = 10, labelfontsize = 12, titlefontsize = 12,
                 ylabel = L"\Delta", xrotation = 0, ylims = (0, 1.1), grid = false,
                 legend = false, color = bar_colors, bottom_margin = 8mm)
    annotate!(p_pred,-4, 1.3, text(L"(b)", :left, 12))
    annotate!(p_pred, 10.35, 1.2, text(L"y_{\mathrm{pred}}", :left, 12))
    
    p = plot(p_fit, p_pred, layout=(2,1), size=(500,600),leftmargin=4mm,topmargin=17mm)

    display(p)
    savefig(p, "$path\\Coral_Original.pdf")
end



Plot_info_imb()

data = matread("$path\\SMC_sample_allModels.mat")
part_vals = data["Original"]["posterior_sample"]
part_vals = log.(part_vals[:, 1:20])


parameter_labels = [
    L"k_{CO2}", L"k_{pp}", L"s",
    L"\alpha", L"\beta", L"v_{H_c}", L"E_{0_c}",
    L"k_{1fc}", L"k_{2fc}", L"k_{3fc}",
    L"k_{1bc}", L"k_{2bc}", L"k_{3bc}",
    L"E_{0_h}",
    L"k_{1fc}", L"k_{2fc}", L"k_{3fc}",
    L"k_{1bc}", L"k_{2bc}", L"k_{3bc}"
]

group_sizes = [3, 10, 7]

group_labels = [
    [
        L"3 \times \mathrm{Diffusion}",
        L"\mathrm{Mechanisms}"
    ],
    L"\mathrm{Ca\!-\!ATPase\ Pump\ Mechanism}",
    L"\mathrm{BAT\ Pump\ Mechanism}"
]

eigen_labels = [latexstring(string(i)) for i in 1:20]

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
    group_sizes = group_sizes,
    group_labels = group_labels,
    eigen_labels = eigen_labels,
    eigenvalue_digits = 4,
    eigenvector_digits = 1,
    signed_colour = true,
    main_size = (900, 620),
    cell_grad = blue_grad,
)
#= annotate!(p, 0, 23 - 0.15, text(L"(a)", 12, :black, :left)) =#
display(p)
savefig(p, "$path\\sloppytable.pdf")
savefig(cb, "$path\\sloppytable_colorbar.pdf")