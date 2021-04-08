mutable struct NumModelCrankNicolson{F,P,W} <: AbstractNumModel{F,P,W}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   M :: Array
   b :: Array
   Anl :: Array
   Anl2 :: Array
   potential :: P
   writer :: W
end

Base.show(io::IO, n::NumModelCrankNicolson) = print(io,
         "Crank-Nicolson Newton-Raphson scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelCrankNicolson)
   # create working vectors
   ψ = similar(n.f.ϕ)
   ϕ₁ = copy(n.f.ϕ)
   # for krylov
   ϕw = similar(n.f.ϕ)
   ϕw .= 0
   # newton iterations
   for itnewton = 1:n.nnewton
      # compute matrices and vectors
      # ψ = 1/2 (ϕ₁+ϕ₀)
      @. ψ = 0.5 * (ϕ₁+n.f.ϕ)
      # Anls = V + 2 × β × ∥ψ∥²
      @. n.Anl = n.potential.V + 2 * n.β * ψ * conj(ψ)
      # Anls2 = β × ψ²
      @. n.Anl2 = n.β * ψ * ψ
      # M⁻¹ = Δt⁻¹ + V + 3 × β × ∥ψ∥² + 0.5
      @. n.M = 1. / ( 1. / n.Δt + ( n.potential.V + 3 *  n.β * real( ψ * conj(ψ) ) ) * 0.5 )
      # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ - coeffΩ × Ω) × ψ
      n.b .= (1/n.Δt) .* (ϕ₁ .- n.f.ϕ) .+ (n.potential.V .+ n.β .* conj.(ψ).*ψ ) .* ψ .- lapRot(n,ψ)
      # solving
      krylovPreCond!(n, ϕw)
      # update solution
      ϕ₁ .= ϕ₁ .- ϕw
      # newton residual
      resNewton = sqrt(sum(real.(ϕw.*conj.(ϕw))))
      # println("norm ψw $(resNewton)")
      if resNewton < n.nnewton
         break
      end
   end
   # update field
   n.f.ϕ .= ϕ₁
   # normalize
   normalize!(n.f)
end

function prodA(n::NumModelCrankNicolson, ϕt)
   (1/n.Δt .+ n.Anl) .* ϕt .+ 0.5 .* n.Anl2 .* conj.(ϕt) .- 0.5 .* lapRot(n,ϕt)
end
