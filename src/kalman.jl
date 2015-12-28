symmetrize(P::Matrix) = (P+P')/2 #+ 1e-15eye(P) #TODO: Hacky. Square root filter?

function likelihood{T}(y::Vector{T}, C::Matrix{T}, σϵ²::T, Q::Matrix{T})

    n   = length(y)
    nₓ  = size(C, 1)

    x  = zeros(nₓ)
    P∞ = eye(nₓ)
    P = zeros(nₓ, nₓ)

    ll = -n*log(2π)

    d = 0
    diffuseP = true

    while diffuseP

        d += 1
        Cₜ = C[:,d]

        ν⁰ = y[d] - dot(Cₜ, x)

        F∞  = dot(Cₜ, P∞*Cₜ)
        F   = dot(Cₜ, P*Cₜ) + σϵ²
        F¹  = 1/F∞
        F²  = - F¹ * F * F¹

        M∞  = P∞ * Cₜ
        M   = P * Cₜ

        K⁰  = F∞ == 0 ? M/F : M∞*F¹
        K¹  = M * F¹ + M∞ * F²

        L⁰  = I - K⁰ * Cₜ'
        L¹  = -K¹ * Cₜ'

        # Calculate loglikelihood contribution
        #ll-= F∞ == 0 ? log(F) + ν⁰*ν⁰/F : log(F∞) 
        ll-= F∞ == 0 ? 0 : log(F∞) 

        # Calculate next values
        x   = x + K⁰*ν⁰
        P   = F∞ == 0 ? symmetrize(P*L⁰' + Q) : symmetrize(P∞*L¹' + P*L⁰' + Q)
        P∞  = F∞ == 0 ? P∞        : symmetrize(P∞*L⁰')

        diffuseP = findfirst(round(P∞,12)) > 0

    end #while
    #println(d)

    for t in d+1:n-1

        Cₜ = C[:,t]

        νₜ = y[t] - dot(Cₜ, x)
        Fₜ = dot(Cₜ, P*Cₜ) + σϵ²
        Kₜ = P * Cₜ / Fₜ

        # Calculate loglikelihood contribution
        ll -= Cₜ[1] == 0 ? 0 : log(Fₜ) + νₜ*νₜ/Fₜ

        # Predict next values
        x += Kₜ * νₜ
        P = P * (I - Cₜ * Kₜ') + Q

    end #for

    Cₜ = C[:,n]
    νₜ = y[n] - dot(Cₜ, x)
    Fₜ = dot(Cₜ, P*Cₜ) + σϵ²
    ll -= log(Fₜ) + νₜ*νₜ/Fₜ

    println(ll)
    return -ll/2

end #likelihood 


function smooth{T}(y::Vector{T}, C::Matrix{T}, σϵ²::T, σμ²::T)

    n   = length(y)
    nₓ  = size(C, 1)
    Q = zeros(nₓ, nₓ)
    Q[1] = σμ²

    K   = Array{T}(nₓ, n)
    F⁻¹ = Array{T}(n)
    ν   = Array{T}(n)

    P∞  = Matrix{T}[]
    L⁰  = Matrix{T}[]
    L¹  = Matrix{T}[]
    F∞  = T[]
    F²  = T[]

    x   = Array{T}(nₓ, n)
    P   = Array{T}(nₓ, nₓ, n)
    ϵ   = Array{T}(n)
    η   = Array{T}(nₓ, n)

    xₜ  = zeros(nₓ)
    P∞ₜ = eye(nₓ)
    Pₜ  = zeros(nₓ, nₓ)

    ll = -n*log(2π)

    d           = 0
    x[:,1]      = xₜ
    P[:,:,1]    = Pₜ
    push!(P∞, P∞ₜ)
    diffuseP    = true

    while diffuseP

        d += 1
        Cₜ = C[:,d]
        ν⁰ = y[d] - dot(Cₜ, xₜ)

        F∞ₜ = dot(Cₜ, P∞ₜ * Cₜ)
        Fₜ  = dot(Cₜ, Pₜ * Cₜ) + σϵ²
        F¹ₜ = 1/F∞ₜ
        F²ₜ = - F¹ₜ * Fₜ * F¹ₜ

        M∞  = P∞ₜ * Cₜ
        M   = Pₜ * Cₜ

        K⁰  = F∞ₜ == 0 ? M/Fₜ : M∞*F¹ₜ
        K¹ = M * F¹ₜ + M∞ * F²ₜ

        L⁰ₜ = I - K⁰ * Cₜ'
        L¹ₜ = -K¹ * Cₜ'

        ν[d]    = ν⁰
        F⁻¹[d]  = 1/Fₜ
        K[:, d] = K⁰
        push!(F∞, F∞ₜ)
        push!(F², F²ₜ)
        push!(L⁰, L⁰ₜ)
        push!(L¹, L¹ₜ)

        # Calculate next values
        xₜ  = xₜ + K⁰ * ν⁰
        Pₜ  = F∞ₜ == 0 ? Pₜ*L⁰ₜ' + Q : P∞ₜ*L¹ₜ' + Pₜ*L⁰ₜ' + Q
        P∞ₜ = F∞ₜ == 0 ? P∞ₜ         : P∞ₜ*L⁰ₜ'

        x[:,d+1]    = xₜ
        P[:,:,d+1]  = Pₜ
        push!(P∞, P∞ₜ)

        diffuseP = findfirst(round(P∞ₜ,12)) > 0

    end #while

    for t in d+1:n-1

        Cₜ = C[:,t]
        νₜ = y[t] - dot(Cₜ, xₜ)
        Fₜ = dot(Cₜ, Pₜ*Cₜ) + σϵ²
        Kₜ = Pₜ * Cₜ / Fₜ

        ν[t]    = νₜ
        F⁻¹[t]  = 1/Fₜ
        K[:,t]  = Kₜ

        # Predict next values
        xₜ += Kₜ * νₜ
        Pₜ = Pₜ * (I - Cₜ * Kₜ') + Q

        x[:,t+1]   = xₜ
        P[:,:,t+1] = Pₜ

    end #for

    Cₜ = C[:,n]
    νₜ = y[n] - dot(Cₜ, xₜ)
    Fₜ = dot(Cₜ, Pₜ*Cₜ) + σϵ²
    Kₜ = Pₜ * Cₜ / Fₜ

    ν[n]    = νₜ
    F⁻¹[n]  = 1/Fₜ
    K[:,n]  = Kₜ

    rₜ, Nₜ = zeros(nₓ), zeros(nₓ, nₓ)
    uₜ = νₜ/Fₜ - dot(Kₜ, rₜ)
    Dₜ = 1/Fₜ + dot(Kₜ, Nₜ * Kₜ)

    # Calculate smoothed disturbances
    ϵ[n] = σϵ² * uₜ
    η[:, n] = Q * rₜ

    # Calculate helper recursions 
    rₜ = Cₜ * uₜ + rₜ
    Θ = Cₜ * Kₜ' * Nₜ
    Nₜ = Cₜ * Dₜ * Cₜ' + Nₜ - Θ - Θ' 

    for t in n-1:-1:d+1

        Cₜ    = C[:,t]
        F⁻¹ₜ  = F⁻¹[t]
        νₜ    = ν[t]
        Kₜ    = K[:,t]
        Pₜ    = P[:,:,t]

        uₜ = F⁻¹ₜ * νₜ - dot(Kₜ, rₜ)
        Dₜ = F⁻¹ₜ + dot(Kₜ, Nₜ * Kₜ)

        # Calculate smoothed disturbances
        ϵ[t] = σϵ² * uₜ
        η[:, t] = Q * rₜ

        # Calculate helper recursions 
        rₜ = Cₜ * uₜ + rₜ
        Θ = Cₜ * Kₜ' * Nₜ
        Nₜ = Cₜ * Dₜ * Cₜ' + Nₜ - Θ - Θ' 

        # Calculate smoothed states
        x[:, t] += Pₜ * rₜ
        P[:,:,t] -= Pₜ * Nₜ * Pₜ

    end #for

    r⁰ₜ, r¹ₜ      = rₜ, zeros(rₜ) 
    N⁰ₜ, N¹ₜ, N²ₜ = Nₜ, zeros(Nₜ), zeros(Nₜ)

    for t in d:-1:1
        
        Cₜ        = C[:,t]
        F∞ₜ       = F∞[t]
        #F¹ₜ       = F⁻¹[t]
        F²ₜ       = F²[t]
        F⁻¹ₜ      = F∞ₜ == 0 ? F⁻¹[t] : 0
        ν⁰ₜ       = ν[t]
        K⁰ₜ       = K[:,t]
        L⁰ₜ, L¹ₜ  = L⁰[t], L¹[t]
        Pₜ, P∞ₜ   = P[:,:,t], P∞[t]

        uₜ        = F⁻¹ₜ*ν⁰ₜ - dot(K⁰ₜ, r⁰ₜ)
        Dₜ        = F⁻¹ₜ + dot(K⁰ₜ, N⁰ₜ * K⁰ₜ)

        # Calculate smoothed disturbances
        ϵ[t] = σϵ² * uₜ
        η[:, t] = Q * rₜ

        # Calculate helper recursions
        r¹ₜ = Cₜ*ν⁰ₜ/F∞ₜ + L⁰ₜ'*r¹ₜ + L¹ₜ'*r⁰ₜ
        r⁰ₜ += Cₜ * uₜ
        Δ = L⁰ₜ' * N¹ₜ * L¹ₜ
        N²ₜ = Cₜ*F²ₜ*Cₜ' + L⁰ₜ'*N²ₜ*L⁰ₜ + Δ + Δ' + L¹ₜ'*N⁰ₜ*L¹ₜ 
        N¹ₜ = Cₜ*Cₜ'/F∞ₜ + L⁰ₜ'*N¹ₜ*L⁰ₜ + L¹ₜ'*N⁰ₜ*L⁰ₜ
        Θ = Cₜ*K⁰ₜ'*N⁰ₜ
        N⁰ₜ += Cₜ*Dₜ*Cₜ' - Θ - Θ'

        # Calculate smoothed states
        Γ = P∞ₜ * N¹ₜ * Pₜ
        x[:,t] += Pₜ*r⁰ₜ + P∞ₜ*r¹ₜ
        P[:,:,t] -= Pₜ*N⁰ₜ*Pₜ + P∞ₜ*N²ₜ*P∞ₜ + Γ + Γ'

    end #for

    return ϵ, x, P, η

end #statesmooth
