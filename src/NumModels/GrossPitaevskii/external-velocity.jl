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
   ϕxhat = n.plan.ϕx_hat
   ψ₁hat, ϕytmphat = n.plan.ϕy_hat, n.plan.ϕytmp_hat
   Δt, coeffΔ, β = n.Δt, n.param.coeffΔ, n.param.β
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   uadvx, uadvy = n.param.pot.uadvx, n.param.pot.uadvy
   # parameters
   α = 1.
   γ = 1.
   # create working vectors
   ψ₁ = similar(ϕ)
   # compute gradients
   computeDerivatives!(n.gf, n.plan, ϕ)
   # compute ψ₁
   @. ψ₁ = ϕ + γ * Δt * (
      - β * abs2(ϕ) * ϕ
      + β * ϕ
      -  (uadvx^2+uadvy^2)/(-4*coeffΔ) * ϕ
      - im * uadvx * n.gf.dx - im * uadvy * n.gf.dy
   )
# # compute gradients
# mul!(ϕhat_x, plan_x, ϕ)
# ∇ϕ_x = plan_x \ (im .* ξx .* ϕhat_x) # we get grad x
# mul!(ϕhat_y, plan_y, ϕ)
# ∇ϕ_y = plan_y \ (im .* ξy .* ϕhat_y) # we get grad y
# # compute ψ₁
# @. ψ₁ = ϕ + γ * Δt * (
#              - β * real(ϕ * conj(ϕ)) * ϕ
#              + β * ϕ
#              -  (uadvx^2+uadvy^2)/(-4*coeffΔ) * ϕ
#              - im * uadvx * ∇ϕ_x - im * uadvy * ∇ϕ_y
#             )
# all to frequency domain
# ψ₁hat = plan * ψ₁
# mul!(ϕhat, plan, ϕ)
# @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
#          1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
# ldiv!(ϕ, plan, ψ₁hat)

   # ψ₁ ← ( ψ₁ + α Δt /2 (ddx + ddy ) ϕ ) / ()
   # ψ₁hat
   mul!(parent(ϕxhat),plan_x,parent(ψ₁))
   transpose!(ϕytmphat,ϕxhat)
   mul!(parent(ψ₁hat),plan_y,parent(ϕytmphat))
   # ϕhat
   ϕhat = similar(ψ₁hat)
   mul!(parent(ϕxhat),plan_x,parent(ϕ))
   transpose!(ϕytmphat,ϕxhat)
   mul!(parent(ϕhat), plan_y, parent(ϕytmphat))
   # compute
   gridξ = localgrid(n.plan.pen_y, (n.plan.ξx, n.plan.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
   # back to physical
   ldiv!(parent(ϕytmphat), plan_y, parent(ψ₁hat))
   transpose!(ϕxhat, ϕytmphat)
   ldiv!(parent(ϕ), plan_x, parent(ϕxhat))
   return 1
end

"""
    timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}

Performs a single time step for stationnary field approximating a velocity field.
"""
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z, ϕhat = n.f.ϕ, n.plan.ϕ_hat, n.plan.ϕ_hat, n.plan.ϕ_hat, n.plan.ϕ_hat
   Δt, coeffΔ, Ω, β = n.Δt, n.param.coeffΔ, n.param.Ω, n.param.β
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan, plan_x, plan_y, plan_z = n.plan.plan, n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V, uadvx, uadvy, uadvz = n.param.pot.V, n.param.pot.uadvx, n.param.pot.uadvy, n.param.pot.uadvz
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
   return 1
end
