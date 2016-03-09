immutable LocalLevelFit{T}
    residual_std::T
    locallevel_innovation_std::T
    regression_coefs::Vector{T}
    transform_params::Vector{T}
    y::Vector{T}
    y_residuals::Vector{T}
    locallevel::Vector{T}
    locallevel_std::Vector{T}
    locallevel_innovations::Vector{T}
end #LocalLevelFit

#TODO
#function Base.show(llf::LocalLevelFit)
#end #show

function fit{T}(y::Vector{T}, X::Matrix{T}=zeros(length(y),0))

    n   = length(y) # Number of observations
    @assert n == size(X, 1)
    notmissing = !isnan(y)

    X   = X'
    m   = size(X,1) # Number of regressors

    nₓ  = 1 + m
    Q = zeros(nₓ, nₓ)
    C = [ones(1, n); X] .* notmissing'

    yclean = y .* notmissing

    function ll(params::Vector{T})
        Q[1] = exp(2params[2])
        return likelihood(yclean, C, exp(2params[1]), Q)
    end #ll

    params = optimize(ll, [0., 0.]).minimum

    #println(exp(params))
    σϵ², σμ² = exp(2params)
    y_residuals, x, Pₓ, x_residuals = smooth(yclean, C, σϵ², σμ²)

    return LocalLevelFit(sqrt(σϵ²), sqrt(σμ²), x[2:end,end], zeros(0), y, y_residuals, vec(x[1,:]), vec(Pₓ[1,1,:]), vec(x_residuals[1,:]))

end #fit

function fit{T}(y::Vector{T}, f::Function, p₀::Vector{T}, X::Matrix{T})

    n   = length(y) # Number of observations
    m   = size(f(p₀, X), 2) # Number of regressors
    @assert n == size(X, 1)

    notmissing = !isnan(y)

    nₓ  = 1 + m
    Q = zeros(nₓ, nₓ)
    yclean = y .* notmissing
    c = ones(1, n)

    function ll(params::Vector{T})
        C = [c; f(params[3:end], X)'] .* notmissing'
        Q[1] = exp(2params[2])
        return likelihood(yclean, C, exp(2params[1]), Q)
    end #ll

    params = optimize(ll, [0.; 0.; p₀]).minimum

    σϵ², σμ² = exp(2params[1:2])
    p = params[3:end]
    fX = f(p, X)'
    C = [ones(1, n); fX] .* notmissing'
    y_residuals, x, Pₓ, x_residuals = smooth(yclean, C, σϵ², σμ²)

    return LocalLevelFit(sqrt(σϵ²), sqrt(σμ²), x[2:end,end], p, y, y_residuals, vec(x[1,:]), vec(Pₓ[1,1,:]), vec(x_residuals[1,:]))

end #fit
