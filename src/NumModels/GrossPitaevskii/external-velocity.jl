mutable struct NumModelExternalVelocity{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

function NumModelExternalVelocity(f, param,
      Δt::Real, niter::Integer, freqbckp::Integer;)
   plan = Plan(f)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelExternalVelocity{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
         writers
      )
end

Base.show(io::IO, n::NumModelExternalVelocity) = print(io,
         "ARGLE scheme\n",
         "  ├───────────  model: coeff Δ : $(n.param.coeffΔ) β : $(n.param.β), Ω : $(n.param.Ω)", '\n',
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField2D}

Performs a single time step for stationnary field approximating a velocity field.
"""
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField2D}
   # references
   ϕ = n.f.ϕ
   ϕhat = n.plan.ϕ_hat
   Δt, coeffΔ, β = n.Δt, n.param.coeffΔ, n.param.β
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan = n.plan.plan
   V, uadvx, uadvy = n.param.pot.V, n.param.pot.uadvx, n.param.pot.uadvy
   α = 1.
   γ = 1.
   # create working vectors
   ψ₁ = copy(ϕ)
   ψ₁ .= 0.
   ϕw = similar(ϕ)
   computeDerivatives!(n.gf, n.plan, ϕ)
   # function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)

   # compute gradients
   @. ψ₁ = ϕ + γ * Δt * (
                - β * abs2(ϕ) * ϕ
                + β * ϕ
                -  (uadvx^2+uadvy^2)/(-4*coeffΔ) * ϕ
                - im * uadvx * n.gf.dx - im * uadvy * n.gf.dy
               )
   # all to frequency domain
   # ψ₁ ← ( ψ₁ + α Δt /2 (ddx + ddy ) ϕ ) / ()
   ψ₁hat = plan * ψ₁
   mul!(ϕhat, plan, ϕ)
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
   ldiv!(ϕ, plan, ψ₁hat)
   return 1
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
