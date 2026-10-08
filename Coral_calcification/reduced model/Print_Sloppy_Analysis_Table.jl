using Printf
using LaTeXStrings
using Plots
using PlotUtils
using Measures
using Colors
using LinearAlgebra



function posterior_covariance_equal(part_vals)
    X = part_vals
    N, p = size(X)

    μ = vec(mean(X, dims=1))
    Xc = X .- μ'
    Σ = (Xc' * Xc) / (N - 1)

    E = eigen(Symmetric(inv(Σ)))
    evals = E.values
    evecs = E.vectors

    idx = sortperm(evals; rev=true)
    evals = evals[idx]
    evecs = evecs[:, idx]

    slop_ratio = evals[1] / evals[end]

    return (μ=μ, Σ=Σ, evals=evals, evecs=evecs, sloppiness_ratio=slop_ratio)
end

function sloppiness_table_from_samples(part_vals; parameter_labels,
    group_sizes = nothing,
    group_labels = nothing,
    eigen_labels = nothing,
    eigenvalue_digits = 2,
    eigenvector_digits = 2,
    signed_colour = true,
    main_size = (900, 620),
    cell_grad = cgrad(
        [RGB(0.20, 0.45, 0.95), RGB(1, 1, 1), RGB(0.20, 0.45, 0.95)],
        [0.0, 0.5, 1.0]
    )
)

    result = posterior_covariance_equal(part_vals)

    evals = result.evals
    evecs = result.evecs

    E = evecs'

    # Normalise each eigenvector by its maximum absolute element
    E_norm = E ./ maximum(abs.(E), dims = 2)

    # Fix sign convention:
    # make the largest absolute component in each eigenvector positive
    for i in 1:size(E_norm, 1)
        idx = argmax(abs.(E_norm[i, :]))
        if E_norm[i, idx] < 0
            E_norm[i, :] .*= -1
        end
    end

    nrow, nparam = size(E_norm)

    if isnothing(eigen_labels)
        eigen_labels = [latexstring(string(i)) for i in 1:nrow]
    end

    if isnothing(group_sizes)
        group_sizes = fill(1, nparam)
    end

    if length(parameter_labels) != nparam
        error("length(parameter_labels) must equal number of columns in part_vals")
    end

    if sum(group_sizes) != nparam
        error("sum(group_sizes) must equal number of parameter columns")
    end

    if !isnothing(group_labels) && length(group_labels) != length(group_sizes)
        error("length(group_labels) must equal length(group_sizes)")
    end

    use_group_header = !isnothing(group_labels)

    rel_evals = evals ./ evals[1]

    label_col = zeros(nrow)

    # -------------------------------------------------
    # Colour values
    # Second column eigenvalues are forced to white.
    # H_body controls colour only.
    # T_body below controls printed values.
    # -------------------------------------------------
    if signed_colour
        H_body = hcat(label_col, zeros(nrow), E_norm)
        heatmap_clim = (-1.0, 1.0)
    else
        H_body = hcat(label_col, zeros(nrow), abs.(E_norm))
        heatmap_clim = (0.0, 1.0)
    end

    nbody, ncol = size(H_body)

    # -------------------------------------------------
    # Column widths
    # -------------------------------------------------
    col_widths = ones(ncol)
    col_widths[2] = 1.6      # make only second column wider

    x_edges = cumsum(vcat(0.5, col_widths))
    xpos = x_edges[1:end-1] .+ col_widths ./ 2

    # -------------------------------------------------
    # Equal row positions
    # -------------------------------------------------
    if use_group_header
        total_rows = nbody + 2
        param_header_y = nbody + 1
        group_header_y = nbody + 2
    else
        total_rows = nbody + 1
        param_header_y = nbody + 1
    end

    y_top = total_rows + 0.5

    # -------------------------------------------------
    # Labels
    # -------------------------------------------------
    col_labels = vcat(
        [L"\hat{\theta}_{i}", L"\lambda_{i}/\lambda_1"],
        parameter_labels
    )

    # Reverse body rows so eigenparameter 1 appears at the top
    H_body_plot = reverse(H_body, dims = 1)

    # Add header rows
    if use_group_header
        Hplot = vcat(H_body_plot, zeros(2, ncol))
    else
        Hplot = vcat(H_body_plot, zeros(1, ncol))
    end

    # Values printed in cells
    # This still contains real eigenvalues in column 2
    T_body = hcat(rel_evals, E_norm)
    Tplot = reverse(T_body, dims = 1)

    eigen_labels_plot = reverse(eigen_labels)

    # -------------------------------------------------
    # Main table drawn manually as rectangles
    # -------------------------------------------------
    p = plot(
        xlims = (0.5, x_edges[end]),
        ylims = (0.5, y_top),
        legend = false,
        xticks = false,
        yticks = false,
        framestyle = :box,
        grid = false,
        size = main_size,
        left_margin = 6mm,
        right_margin = 1mm,
        bottom_margin = 2mm,
        top_margin = 4mm
    )

    clo, chi = heatmap_clim

    for i in 1:size(Hplot, 1)
        for j in 1:ncol

            val = Hplot[i, j]

            t = (val - clo) / (chi - clo)
            t = clamp(t, 0.0, 1.0)

            cell_color = get(cell_grad, t)

            x1 = x_edges[j]
            x2 = x_edges[j + 1]
            y1 = i - 0.5
            y2 = i + 0.5

            plot!(
                p,
                Shape([x1, x2, x2, x1], [y1, y1, y2, y2]),
                fillcolor = cell_color,
                linecolor = :transparent,
                label = false
            )
        end
    end

    # -------------------------------------------------
    # Group names inside top header row
    # group_labels[g] can be:
    # 1. one LaTeXString
    # 2. a vector of LaTeXStrings for multiple lines
    # -------------------------------------------------
    if use_group_header
        start_col = 3

        for (g, gsize) in enumerate(group_sizes)
            stop_col = start_col + gsize - 1

            group_x = (x_edges[start_col] + x_edges[stop_col + 1]) / 2

            if group_labels[g] isa AbstractVector
                nlines = length(group_labels[g])

                for k in 1:nlines
                    y_shift = 0.22 * (nlines + 1 - 2k)

                    annotate!(
                        p,
                        group_x,
                        group_header_y + y_shift,
                        text(group_labels[g][k], 10, :black, :center)
                    )
                end
            else
                annotate!(
                    p,
                    group_x,
                    group_header_y,
                    text(group_labels[g], 10, :black, :center)
                )
            end

            start_col = stop_col + 1
        end
    end

    # -------------------------------------------------
    # Parameter header labels
    # -------------------------------------------------
    for j in 1:ncol
        annotate!(
            p,
            xpos[j], param_header_y,
            text(col_labels[j], 10, :black, :center)
        )
    end

    # -------------------------------------------------
    # Body rows
    # -------------------------------------------------
    for i in 1:nbody
        y = i

        annotate!(
            p,
            xpos[1], y,
            text(eigen_labels_plot[i], 10, :black, :center)
        )

        for j in 2:ncol
            val = Tplot[i, j - 1]

            if j == 2
                rounded_val = round(val, digits = eigenvalue_digits)

                # avoid printing -0.0000
                if rounded_val == 0
                    rounded_val = 0.0
                end

                numstr = @sprintf("%.*f", eigenvalue_digits, rounded_val)
            else
                rounded_val = round(val, digits = eigenvector_digits)

                # avoid printing -0.0
                if rounded_val == 0
                    rounded_val = 0.0
                end

                numstr = @sprintf("%.*f", eigenvector_digits, rounded_val)
            end

            annotate!(
                p,
                xpos[j], y,
                text(latexstring(numstr), 10, :black, :center)
            )
        end
    end

    # -------------------------------------------------
    # Vertical separators
    # -------------------------------------------------
    vline!(p, [x_edges[2]], color = :black, lw = 1.0, label = false)
    vline!(p, [x_edges[3]], color = :black, lw = 1.0, label = false)

    # Group separators
    start_col = 3

    for gsize in group_sizes
        stop_col = start_col + gsize - 1
        vline!(p, [x_edges[stop_col + 1]], color = :black, lw = 0.9, label = false)
        start_col = stop_col + 1
    end

    # -------------------------------------------------
    # Horizontal lines
    # -------------------------------------------------
    for y in 1.5:1:(nbody - 0.5)
        hline!(p, [y], color = RGBA(0, 0, 0, 0.15), lw = 0.7, label = false)
    end

    # line between body and parameter header
    hline!(p, [nbody + 0.5], color = :black, lw = 1.0, label = false)

    # line between parameter header and group header
    if use_group_header
        hline!(p, [nbody + 1.5], color = :black, lw = 1.0, label = false)
    end

    # top border
    hline!(p, [y_top], color = :black, lw = 1.0, label = false)

    # -------------------------------------------------
    # Colour bar
    # -------------------------------------------------
    z = reshape(range(-1, 1, length = 200), 200, 1)

    cb = heatmap(
        [1],
        range(-1, 1, length = 200),
        z,
        c = cell_grad,
        clim = (-1, 1),
        colorbar = false,
        legend = false,
        xticks = false,
        yticks = false,
        ymirror = true,
        framestyle = :box,
        grid = false,
        xlims = (0.5, 1.5),
        ylims = (-1, 1),
        size = (60, 150),
        left_margin = 1mm,
        right_margin = 8mm,
        top_margin = 2mm,
        bottom_margin = 2mm
    )

    annotate!(cb, 2, -1, text(L"-1", 10, :black, :left))
    annotate!(cb, 3,  0, text(L"0", 10, :black, :left))
    annotate!(cb, 3,  1, text(L"1", 10, :black, :left))

    return p, result, cb
end