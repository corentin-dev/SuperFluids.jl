mutable struct NumModelBackwardEuler{F,P,W} <: AbstractNumModel{F,P,W}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   M :: Array
   b :: Array
   potential :: P
   writer :: W
end

Base.show(io::IO, n::NumModelBackwardEuler) = print(io,
         "Backward Euler\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelBackwardEuler)
   # compute matrices and vectors
   # M⁻¹ = Δt⁻¹ + V + β × ∥ϕ∥²
   @. n.M = 1. / ( 1. / n.Δt + n.potential.V + n.β * real( n.f.ϕ * conj(n.f.ϕ) ) )
   # b = M × ϕ × Δt⁻¹
   @. n.b = n.M * n.f.ϕ / n.Δt
   # solving
   krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
end

function prodA(n::NumModelBackwardEuler, ϕt)
   ϕt .- n.M .* lapRot(n,ϕt)
end

mutable struct NumModelBackwardEulerNoPrecond{F,P,W} <: AbstractNumModel{F,P,W}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   Anl :: Array
   b :: Array
   potential :: P
   writer :: W
end

Base.show(io::IO, n::NumModelBackwardEulerNoPrecond) = print(io,
         "Backward Euler without preconditionning\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelBackwardEulerNoPrecond)
   # compute matrices and vectors
   # Anl = V + β × ∥ϕ∥²
   @. n.Anl = n.potential.V + n.β * real( n.f.ϕ * conj(n.f.ϕ) )
   # b = ϕ × Δt⁻¹
   @. n.b = n.f.ϕ / n.Δt
   # solving
   krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
end

function prodA(n::NumModelBackwardEulerNoPrecond, ϕt)
   ( (1. / n.Δt) .+ n.Anl ) .* ϕt .- lapRot(n,ϕt)
end
