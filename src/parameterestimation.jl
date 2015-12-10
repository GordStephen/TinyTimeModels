function fit{T}(y::Vector{T}, X::Matrix{T}=zeros(length(y),0))

    n   = length(y) # Number of observations
    @assert n == size(X, 1)

    X   = X'
    m   = size(X,1) # Number of regressors

    nₓ  = 1 + m
    A = eye(nₓ) 
    Q = zeros(nₓ, nₓ)
    C = [1; zeros(m)]

    function ll(params::Vector{T})
        Q[1] = exp(2params[2])
        return likelihood(y, X, A, C, exp(2params[1]), Q)
    end #ll

    params = optimize(ll, [0., 0.]).minimum

    return exp(params)

    #σϵ, σμ = exp(params)
    #Q[1] = σμ 
    #results = smooth(y, A, C, σϵ, Q)
    #return LocalLevelFit(...)

end #fit
