mutable struct NumModelBackwardEuler{F,P} <: AbstractNumModel{F,P}
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
   ϕ_hat
   M
   b
   potential :: P
   writer :: AbstractWriter{F}
   saver :: AbstractWriter{F}
end

nmodel(n::NumModelBackwardEuler) = 20

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

mutable struct NumModelBackwardEulerNoPrecond{F,P} <: AbstractNumModel{F,P}
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
   ϕ_hat
   Anl
   b
   potential :: P
   writer :: AbstractWriter{F}
   saver :: AbstractWriter{F}
end

nmodel(n::NumModelBackwardEulerNoPrecond) = 22

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

mutable struct NumModelBackwardEulerNL{F,P} <: AbstractNumModel{F,P}
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
   ϕ_hat
   M
   b
   potential :: P
   writer :: AbstractWriter{F}
   saver :: AbstractWriter{F}
end

nmodel(n::NumModelBackwardEulerNL) = 3

Base.show(io::IO, n::NumModelBackwardEulerNL) = print(io,
         "Backward Euler\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├─────────────  non-linear Ω", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelBackwardEulerNL)
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

function prodA(n::NumModelBackwardEulerNL, ϕt)
   ϕt .- n.M .* lapRot(n,ϕt)
end

function lapRot(n::NumModelBackwardEulerNL{F}, ϕt) where {F<:AbstractField2D}
   # references
   b = n.b
   ϕthat_x, ϕthat_y = n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # NL rotation
   lnl = 2
   xnl = min.(x,lnl) ./ lnl
   ynl = min.(y,lnl) ./ lnl

   # Ωnlx = Ω .* (
   #              (xnl .- 1) .* sin.(1.5 .*π .* xnl)
   #             + xnl .* x
   # )
   Ωnlx = Ω .* ( (xnl .- 1) .* x .+ xnl .* x )
   # Ωnlx = Ω .* ( (1 .- xnl).*x .+ xnl .* x )
   # Ωnly = Ω .* (
   #              (ynl .- 1) .* sin.(1.5 .*π .* ynl)
   #             .+ ynl .* y
   # )
   # Ωnly = Ω .* y
   Ωnly = Ω .* ( (ynl .- 1) .* y .+ ynl .* y )
   # Ωnly = Ω .* ( (1 .- ynl).*y .+ ynl .* y )
   # perform FFT
   mul!(ϕthat_x, plan_x, ϕt)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕthat_x = (coeffΔ*ξx^2 - Ωnly*ξx) * ϕthat_x
   # backward FFT
   ϕtx = plan_x \ ϕthat_x
   # perform FFT
   mul!(ϕthat_y, plan_y, ϕt)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕthat_y = (coeffΔ*ξy^2 + Ωnlx*ξy) * ϕthat_y
   # backward FFT
   ϕty = plan_y \ ϕthat_y
   # return
   @. ϕtx = ϕtx+ϕty
   return ϕtx
end
