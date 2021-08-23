mutable struct NumModelExternalVelocity{F,P} <: AbstractNumModel{F,P}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat
   potential :: P
   writers :: AbstractWriterCollection{F}
end

function NumModelExternalVelocity(f, p,
      coeffΔ::Real, β::Real, Ω::Real, Δt::Real, niter::Integer, freqbckp::Integer)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   return NumModelExternalVelocity{typeof(f),typeof(p)}(f,
         Δt, niter, freqbckp,
         coeffΔ, β, Ω, plan, ϕ_hat, p,
         writers
      )
end

Base.show(io::IO, n::NumModelExternalVelocity) = print(io,
         "ARGLE scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField2D}

Performs a single time step for stationnary field approximating a velocity field.
"""
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   Δt, coeffΔ, Ω, β = n.Δt, n.coeffΔ, n.Ω, n.β
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan, plan_x, plan_y = n.plan.plan, n.plan.plan_x, n.plan.plan_y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V, uadvx, uadvy = n.potential.V, n.potential.uadvx, n.potential.uadvy
   α = 1.
   γ = 1.
   # create working vectors
   ψ₁ = copy(ϕ)
   ψ₁ .= 0.
   ϕw = similar(ϕ)
   # compute gradients
   mul!(ϕhat_x, plan_x, ϕ)
   ∇ϕ_x = plan_x \ (im .* ξx .* ϕhat_x) # we get grad x
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_y = plan_y \ (im .* ξy .* ϕhat_y) # we get grad y
   # compute ψ₁
   @. ψ₁ = ϕ + γ * Δt * (
                - β * real(ϕ * conj(ϕ)) * ϕ
                + β * ϕ
                -  (uadvx^2+uadvy^2)/(-4*coeffΔ) * ϕ
                - im * uadvx * ∇ϕ_x - im * uadvy * ∇ϕ_y
               )
   # all to frequency domain
   ψ₁hat = plan * ψ₁
   mul!(ϕhat, plan, ϕ)
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
   ldiv!(ϕ, plan, ψ₁hat)
end

"""
    timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}

Performs a single time step for stationnary field approximating a velocity field.
"""
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z, ϕhat = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   Δt, coeffΔ, Ω, β = n.Δt, n.coeffΔ, n.Ω, n.β
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan, plan_x, plan_y, plan_z = n.plan.plan, n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V, uadvx, uadvy, uadvz = n.potential.V, n.potential.uadvx, n.potential.uadvy, n.potential.uadvz
   α = 1.
   γ = 1.
   # create working vectors
   ψ₁ = copy(ϕ)
   ψ₁ .= 0.
   ϕw = similar(ϕ)
   # compute gradients
   mul!(ϕhat_x, plan_x, ϕ)
   ∇ϕ_x = plan_x \ (im .* ξx .* ϕhat_x) # we get grad x
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_y = plan_y \ (im .* ξy .* ϕhat_y) # we get grad y
   mul!(ϕhat_z, plan_z, ϕ)
   ∇ϕ_z = plan_z \ (im .* ξz .* ϕhat_z) # we get grad z
   # compute ψ₁
   @. ψ₁ = ϕ + γ * Δt * (
                - β * real(ϕ * conj(ϕ)) * ϕ
                + β * ϕ
                -  (uadvx^2+uadvy^2+uadvz^2)/(-4*coeffΔ) * ϕ
                - im * uadvx * ∇ϕ_x - im * uadvy * ∇ϕ_y - im * uadvz * ∇ϕ_z
               )
   # all to frequency domain
   ψ₁hat = plan * ψ₁
   mul!(ϕhat, plan, ϕ)
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2+ξz^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2+ξz^2) / 2 )
   ldiv!(ϕ, plan, ψ₁hat)
end
