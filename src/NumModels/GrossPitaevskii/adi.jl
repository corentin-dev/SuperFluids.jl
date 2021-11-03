function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField2D}
    # references
    ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    x, y = n.f.g.x, n.f.g.y
    ξx, ξy = n.plan.ξx, n.plan.ξy
    plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
    # perform FFT
    mul!(ϕhat_x, plan_x, ϕ)
    # compute the laplacian and rotation in the Fourier space (x)
    @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
    # backward FFT
    ldiv!(ϕ, plan_x, ϕhat_x)
    # perform FFT
    mul!(ϕhat_y, plan_y, ϕ)
    # compute the laplacian and rotation in the Fourier space (y)
    @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
    # backward FFT
    ldiv!(ϕ, plan_y, ϕhat_y)
    return nothing
 end

 function solveLapRot!(n::AbstractNumModel{F,P}, Δtl) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters}
    # references
    ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
    ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
    plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
    # perform FFT
    mul!(ϕhat_x, plan_x, ϕ)
    # compute the laplacian and rotation in the Fourier space (x)
    @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
    # backward FFT
    ldiv!(ϕ, plan_x, ϕhat_x)
    # perform FFT
    mul!(ϕhat_y, plan_y, ϕ)
    # compute the laplacian and rotation in the Fourier space (y)
    @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
    # backward FFT
    ldiv!(ϕ, plan_y, ϕhat_y)
    # perform FFT
    mul!(ϕhat_z, plan_z, ϕ)
    # compute the laplacian and rotation in the Fourier space (y)
    @. ϕhat_z = exp(im*(coeffΔ*ξz^2)*Δtl) * ϕhat_z
    # backward FFT
    ldiv!(ϕ, plan_z, ϕhat_z)
    return nothing
 end

 function solveNL!(n::AbstractNumModel{F,P}, Δtl) where {F<:AbstractField, P<:GrossPitaevskiiParameters}
    # ϕ ↦ exp ( -i ( V + ∥ϕ∥² ) Δt ) ϕ
    n.f.ϕ .= exp.(-1im * ( non_linear(n.f, n.param) ) * Δtl) .* n.f.ϕ
 end