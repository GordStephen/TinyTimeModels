using TinyTimeModels
using FactCheck

srand(10)

rtol = 0.07

facts("Parameter estimation") do

    n, m = 1000, 5
    σ = 1
    η = .1
    β = randn(m)

    X = randn(n, m)

    context("Local level innovation and model error variances") do

        y = cumsum(η*randn(n)) + σ*randn(n)
        llfit = fit(y)

        @fact llfit.locallevel_innovation_std --> roughly(η, rtol=rtol)
        @fact llfit.residual_std --> roughly(σ, rtol=rtol)

    end

    context("Variances and regression coefficients") do

        y = X*β + cumsum(η*randn(n)) + σ*randn(n)
        llfit = fit(y, X)

        @fact llfit.regression_coefs --> roughly(β, rtol=rtol)
        @fact llfit.locallevel_innovation_std --> roughly(η, rtol=rtol)
        @fact llfit.residual_std --> roughly(σ, rtol=rtol)

    end

    context("Regression with parametrized covariate transformation") do
        function thresholdify(p::Vector, X::Matrix)
            above_p = X .>= p'
            X_above = X .- p'
            X_below = -X_above
            X_above[!above_p] = 0
            X_below[above_p] = 0
            return [X_above X_below]
        end #thresholdify

        β_new = randn(2)
        p = 0.1randn(1)
        X_new = thresholdify(p, X[:,1:1])

        y = X_new*β_new + cumsum(η*randn(n)) + σ*randn(n)
        llfit = fit(y, thresholdify, 0.1randn(1), X[:, 1:1])

        @fact llfit.regression_coefs --> roughly(β_new, rtol=rtol)
        @fact llfit.transform_params --> roughly(p, rtol=rtol)
        @fact llfit.locallevel_innovation_std --> roughly(η, rtol=rtol)
        @fact llfit.residual_std --> roughly(σ, rtol=rtol)

    end

end
