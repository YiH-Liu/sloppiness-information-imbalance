using Random
using Distributions
using LinearAlgebra
using Roots
using StatsBase



function calc_ess(newgamma, oldgamma, part_w, part_loglike)
    log_part_w = log.(part_w) .+ (newgamma - oldgamma) .* part_loglike

    # Numerically stabilize before exponentiating
    log_part_w .-= maximum(log_part_w)
    part_w_scale = exp.(log_part_w)

    # Normalize weights
    part_w_scale = part_w_scale./sum(part_w_scale)

    # Compute effective sample size
    return 1.0 / sum(part_w_scale .^ 2)
end



function smc_generic(N, E, c, loglike_func, prior_funcs, extra_args)
    num_params = extra_args["num_params"]
    S = 2  # initial number of MCMC trial iterations

    # Simulate particles from uniform priors and evaluate loglikelihood
    part_vals = zeros(N, num_params)
    part_loglike = zeros(N)
    for i in 1:N
        part_vals[i, :] .= prior_funcs["prior_sim"](extra_args)
        part_loglike[i] = loglike_func(part_vals[i, :], extra_args)
    end

    # Transform parameters
    for i in 1:N
        part_vals[i, :] .= prior_funcs["trans_f"](part_vals[i, :], extra_args)
    end

    # Initialise all our weights to equal
    part_w = ones(N) / N

    # Loop over temperature schedule
    gamma_t = 0.0
    temp_hist = [gamma_t]
    while gamma_t < 1.0
        # SMC code here

        # Determine gamma_{t+1}
        ess1 = calc_ess(1.0, gamma_t, part_w, part_loglike)
        if ess1 > E
            newgamma = 1.0
        else
            fun(newgamma) = calc_ess(newgamma, gamma_t, part_w, part_loglike) - E
            newgamma = find_zero(fun, (gamma_t, 1.0), Roots.Bisection())
        end

        println("Next temperature is $newgamma")
        push!(temp_hist, newgamma)

        # Perform reweighting step
        log_part_w = log.(part_w) .+ (newgamma - gamma_t) .* part_loglike

        # Numerically stabilize before exponentiating
        log_part_w .-= maximum(log_part_w)
        part_w .= exp.(log_part_w)

        # Normalize weights
        part_w ./= sum(part_w)

        # Compute effective sample size
        ess = 1.0 / sum(part_w .^ 2)
        println("ESS is $ess")

        println("**** Doing a resample-move ******")

        # Resampling multinomially
        ind = sample(1:N, Weights(part_w), N)
        part_vals .= part_vals[ind, :]
        part_loglike .= part_loglike[ind]

        # Re-set weights
        part_w .= ones(N) / N

        # Covariance for MCMC
        cov_rw = cov(part_vals)

        # Perform S MCMC iterations
        count = 0
        alpha = zeros(N)
        for i in 1:N
            for k in 1:S
                # Proposal
                part_vals_prop = rand(MvNormal(part_vals[i, :], cov_rw), 1)

                # Compute loglikelihood at proposal
                part_loglike_prop = loglike_func(prior_funcs["trans_finv"](part_vals_prop,extra_args), extra_args)

                log_prior_curr = prior_funcs["logprior_func"](part_vals[i, :],extra_args)
                log_prior_prop = prior_funcs["logprior_func"](part_vals_prop,extra_args)

                alpha[i] = exp(newgamma * (part_loglike_prop - part_loglike[i]) +
                               log_prior_prop - log_prior_curr)

                if alpha[i] > rand()
                    # Then accept proposal
                    part_vals[i, :] .= part_vals_prop
                    part_loglike[i] = part_loglike_prop
                    count += 1
                end
            end
        end

        # Estimate MCMC acceptance probability
        p = count / (S * N)

        R = ceil(Int, log(c) / log(1 - p))
        println("Adapted R: $R")

        # Run the remaining R-S MCMC iterations
        for i in 1:N
            for k in 1:(R - S)
                # Proposal
                part_vals_prop = rand(MvNormal(part_vals[i, :], cov_rw), 1)

                # Compute loglikelihood at proposal
                part_loglike_prop = loglike_func(prior_funcs["trans_finv"](part_vals_prop,extra_args), extra_args)

                log_prior_curr = prior_funcs["logprior_func"](part_vals[i, :],extra_args)
                log_prior_prop = prior_funcs["logprior_func"](part_vals_prop,extra_args)

                alpha[i] = exp(newgamma * (part_loglike_prop - part_loglike[i]) +
                               log_prior_prop - log_prior_curr)

                if alpha[i] > rand()
                    # Then accept proposal
                    part_vals[i, :] .= part_vals_prop
                    part_loglike[i] = part_loglike_prop
                    count += 1
                end
            end
        end

        println("The number of unique particles after resample-move is $(length(unique(part_vals[:, 1])))")

        S = floor(Int, 0.5 * R)  # update number of trial MCMC iterations to use next
        gamma_t = newgamma
    end

    # Compute inverse transform to original parameterisation
    for i in 1:N
        part_vals[i, :] .= prior_funcs["trans_finv"](part_vals[i, :], extra_args)
    end

    return part_vals
end
