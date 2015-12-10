symmetrize(P::Matrix) = (P+P')/2 

function likelihood{T}(y::Vector{T}, X::Matrix{T}, A::Matrix{T}, C::Vector{T}, σϵ²::T, Q::Matrix{T})

    n   = length(y)
    nₓ  = length(C)

    x  = zeros(nₓ)
    P∞ = eye(nₓ)
    P = zeros(nₓ, nₓ)

    ll = -n*log(2π)

    d = 0
    diffuseP = true

    while diffuseP

        d += 1
        C[2:end] = X[:, d]
        ν⁰ = y[d] - dot(C, x)

        F∞  = dot(C, P∞ * C)
        F   = dot(C, P * C) + σϵ²
        F¹  = 1/F∞
        F²  = - F¹ * F * F¹

        M∞  = P∞ * C
        M   = P * C

        K⁰ = A * M∞ * F¹
        K¹ = A * (M * F¹ + M∞ * F²)

        L⁰ = A - K⁰ * C'
        L¹ = -K¹ * C'

        # Calculate loglikelihood contribution
        ll -= log(F∞)

        # Calculate next values
        x   = A * x + K⁰ * ν⁰
        P = A * (P∞ * L¹' + P * L⁰') + Q
        P∞ = A * P∞ * L⁰'

        diffuseP = findfirst(round(P∞,12)) > 0

    end #while

    for t in d+1:n

        C[2:end] = X[:, t]

        νₜ = y[t] - dot(C, x)
        Fₜ = dot(C, P*C) + σϵ²
        Kₜ = A * P * C / Fₜ

        # Calculate loglikelihood contribution
        ll -= log(Fₜ) + νₜ*νₜ/Fₜ

        # Calculate next values
        x = A * x + Kₜ * νₜ
        P = A * P * (A - Kₜ * C')' + Q

    end #for

    return -ll/2

end #likelihood 


function statesmooth()

    #=
    n   = length(y)
    nₓ  = length(C)

    K   = Array{T}(nₓ, n)
    F⁻¹ = Array{T}(n)
    ν   = Array{T}(n)

    x  = zeros(nₓ)
    P∞ = eye(nₓ)
    P = zeros(nₓ, nₓ)

    ll = -n*log(2π)

    d = 0
    diffuseP = true

    while diffuseP

        d += 1
        C[2:end] = X[:, d]
        ν⁰ = y[d] - dot(C, x)

        F∞  = dot(C, P∞ * C)
        F   = dot(C, P * C) + σϵ²
        F¹  = 1/F∞
        F²  = - F¹ * F * F¹

        M∞  = P∞ * C
        M   = P * C

        K⁰ = A * M∞ * F¹
        K¹ = A * (M * F¹ + M∞ * F²)

        L⁰ = A - K⁰ * C'
        L¹ = -K¹ * C'

        ν[d]    = ν⁰
        F⁻¹[d]  = F¹
        K[:, d] = K⁰

        # Calculate loglikelihood contribution
        ll -= log(F∞)

        # Calculate next values
        x   = A * x + K⁰ * ν⁰
        P = A * (P∞ * L¹' + P * L⁰') + Q
        P∞ = A * P∞ * L⁰'

        diffuseP = findfirst(round(P∞,12)) > 0

    end #while

    for t in d+1:n

        C[2:end] = X[:, t]

        νₜ = y[t] - dot(C, x)
        Fₜ = dot(C, P*C) + σϵ²
        Kₜ = A * P * C / Fₜ

        ν[t]    = νₜ
        F⁻¹[t]  = 1/Fₜ
        K[:,t]  = Kₜ

        # Calculate loglikelihood contribution
        ll -= log(Fₜ) + νₜ*νₜ/Fₜ

        # Calculate next values
        x = A * x + Kₜ * νₜ
        P = A * P * (A - Kₜ * C')' + Q

    end #for

    rₜ, Nₜ = zeros(nₓ), zeros(nₓ, nₓ)

    for t in n:-1:d+1

        C[2:end] = Float64[] # for eventual regression

        F⁻¹ₜ  = F⁻¹[t]
        νₜ    = ν[t]
        Kₜ    = K[:,t]

        uₜ = F⁻¹ₜ * νₜ - dot(Kₜ, rₜ)
        Dₜ = F⁻¹ₜ + dot(Kₜ, Nₜ * Kₜ)

        #Calculate next values
        rₜ = C * uₜ + A' * rₜ
        Θ = C * Kₜ' * Nₜ * A
        Nₜ = C * Dₜ * C' + A' * Nₜ * A - Θ - Θ' 

    end #for

    r⁰ₜ, N⁰ₜ = rₜ, Nₜ

    for t in d:-1:1
        
        C[2:end]  = Float64[] # for eventual regression
        K⁰ₜ       = K[:,t]
        uₜ        = -dot(K⁰ₜ, r⁰ₜ)
        Dₜ        = dot(K⁰ₜ, N⁰ₜ * K⁰ₜ)

        #Calculate next values
        r⁰ₜ = C * uₜ + A' * r⁰ₜ
        Θ = C * K⁰ₜ' * N⁰ₜ * A
        N⁰ₜ = C * Dₜ * C' + A' * N⁰ₜ * A - Θ - Θ' 

    end #for
    =#

end #statesmooth
