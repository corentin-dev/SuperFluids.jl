function timeStep!(n::NumModelBackwardEuler{F,P}) where {F<:AbstractField2D,P<:GrossPitaevskiiParameters}
   # compute matrices and vectors
   # M⁻¹ = Δt⁻¹ + NL
   # NL = V + β × ∥ϕ∥² for GP
   @. n.M = 1. / ( 1 / n.Δt + n.param.V(n.f.g.x,n.f.g.y) + n.param.β * real( n.f.ϕ * conj(n.f.ϕ) ) )
   # b = M × ϕ × Δt⁻¹
   @. n.b = n.M * n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function timeStep!(n::NumModelBackwardEuler{F,P}) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters}
   # compute matrices and vectors
   # M⁻¹ = Δt⁻¹ + NL
   # NL = V + β × ∥ϕ∥² for GP
   @. n.M = 1. / ( 1 / n.Δt + n.param.V(n.f.g.x,n.f.g.y,n.f.g.z) + n.param.β * real( n.f.ϕ * conj(n.f.ϕ) ) )
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

function timeStep!(n::NumModelBackwardEulerNoPrecond{F,P}) where {F <: AbstractField2D, P <: GrossPitaevskiiParameters}
   # compute matrices and vectors
   # Anl = V + β × ∥ϕ∥²
   @. n.Anl = n.param.V(n.f.g.x,n.f.g.y) + n.param.β * real( n.f.ϕ * conj(n.f.ϕ) )
   # b = ϕ × Δt⁻¹
   @. n.b = n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function timeStep!(n::NumModelBackwardEulerNoPrecond{F,P}) where {F <: AbstractField3D, P <: GrossPitaevskiiParameters}
   # compute matrices and vectors
   @. n.Anl = n.param.V(n.f.g.x,n.f.g.y,n.f.g.z) + n.param.β * real( n.f.ϕ * conj(n.f.ϕ) )
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