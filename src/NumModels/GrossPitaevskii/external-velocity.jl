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
   ϕhat = n.plan.ϕy_hat
   ψ₁hat = similar(ϕhat)
   Δt, coeffΔ, β = n.Δt, n.param.coeffΔ, n.param.β
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
   # ψ₁hat
   mul_all!(ψ₁hat, n.plan, ψ₁)
   # ϕhat
   mul_all!(ϕhat, n.plan, ϕ)
   # compute
   x, y, ξx, ξy = grid_y(n.plan)
   # ψ₁ ← ( ψ₁ + α Δt /2 (ddx + ddy ) ϕ ) / ()
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
   # back to physical
   ldiv_all!(ϕ, n.plan, ψ₁hat)
   return 1
end

"""
    timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}

Performs a single time step for stationnary field approximating a velocity field.
"""
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField3D}
   # references
   ϕ = n.f.ϕ
   ϕxhat = n.plan.ϕx_hat
   ϕyhat, ϕytmphat = n.plan.ϕy_hat, n.plan.ϕytmp_hat
   ψ₁hat, ϕztmphat = n.plan.ϕz_hat, n.plan.ϕztmp_hat
   Δt, coeffΔ, β = n.Δt, n.param.coeffΔ, n.param.β
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   uadvx, uadvy, uadvz = n.param.pot.uadvx, n.param.pot.uadvy, n.param.pot.uadvz
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
      -  (uadvx^2+uadvy^2+uadvz^2)/(-4*coeffΔ) * ϕ
      - im * uadvx * n.gf.dx - im * uadvy * n.gf.dy - im * uadvz * n.gf.dz
   )

   # ψ₁ ← ( ψ₁ + α Δt /2 (ddx + ddy ) ϕ ) / ()
   # ψ₁hat
   mul!(parent(ϕxhat),plan_x,parent(ψ₁))
   transpose!(ϕytmphat,ϕxhat)
   mul!(parent(ϕyhat),plan_y,parent(ϕytmphat))
   transpose!(ϕztmphat,ϕyhat)
   mul!(parent(ψ₁hat),plan_z,parent(ϕztmphat))
   # ϕhat
   ϕhat = similar(ψ₁hat)
   mul!(parent(ϕxhat),plan_x,parent(ϕ))
   transpose!(ϕytmphat,ϕxhat)
   mul!(parent(ϕyhat),plan_y,parent(ϕytmphat))
   transpose!(ϕztmphat,ϕyhat)
   mul!(parent(ϕhat),plan_z,parent(ϕztmphat))
   # compute
   gridξ = localgrid(n.plan.pen_z, (n.plan.ξx, n.plan.ξy, n.plan.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2+ξz^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2+ξz^2) / 2 )
   # back to physical
   ldiv!(parent(ϕztmphat), plan_z, parent(ψ₁hat))
   transpose!(ϕyhat, ϕztmphat)
   ldiv!(parent(ϕytmphat), plan_y, parent(ϕyhat))
   transpose!(ϕxhat, ϕytmphat)
   ldiv!(parent(ϕ), plan_x, parent(ϕxhat))
   return 1
end