function timeStep!(n::NumModelBackwardEuler{F,P}) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
   # compute matrices and vectors
   # M⁻¹ = Δt⁻¹ + NL
   # NL = V + β × ∥ϕ∥² for GP
   @. n.M = 1. / ( 1 / n.Δt + n.param.pot.V + n.param.β * abs2(n.f.ϕ) )
   # b = M × ϕ × Δt⁻¹
   @. n.b = n.M * n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function prodA(n::NumModelBackwardEuler{F,P}, ϕt) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
   ϕt .- n.M .* lapRot(n,ϕt)
end

function timeStep!(n::NumModelBackwardEulerNoPrecond{F,P}) where {F <: AbstractField, P <: GrossPitaevskiiParameters}
   # compute matrices and vectors
   # Anl = V + β × ∥ϕ∥²
   @. n.Anl = n.param.pot.V + n.param.β * abs2(n.f.ϕ)
   # b = ϕ × Δt⁻¹
   @. n.b = n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function prodA(n::NumModelBackwardEulerNoPrecond, ϕt)
   ( (1. / n.Δt) .+ n.Anl ) .* ϕt .- lapRot(n,ϕt)
end