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
   ϕ_hat
   M
   b
   Anl
   Anl2
   potential :: P
   writer :: W
end

nmodel(n::NumModelCrankNicolson) = 21

Base.show(io::IO, n::NumModelCrankNicolson) = print(io,
         "Crank-Nicolson Newton-Raphson scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelCrankNicolson)

Performs a single time step for the stationnary Crank-Nicolson scheme.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ + 1/2 Aₙₗ₂ conj(ϕ) - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 2 × β × ∥ψ∥²

Aₙₗ₂ = β × ψ²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
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
      # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
      n.b .= (1/n.Δt) .* (ϕ₁ .- n.f.ϕ) .+ (n.potential.V .+ n.β .* conj.(ψ).*ψ ) .* ψ .- lapRot(n,ψ)
      # solving
      krylovPreCond!(n, ϕw)
      # update solution
      ϕ₁ .= ϕ₁ .- ϕw
      # newton residual
      resNewton = sqrt(sum(real.(ϕw.*conj.(ϕw))))
      if resNewton < n.tolnewton
         break
      end
   end
   # update field
   n.f.ϕ .= ϕ₁
   # normalize
   normalize!(n.f)
end

"""
    prodA(n::NumModelCrankNicolson, ϕt)

Matrix-vector product for the stationnary Crank-Nicolson Newton-Raphson method :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ + 1/2 Aₙₗ₂ conj(ϕ) - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolson, ϕt)
   (1/n.Δt .+ n.Anl) .* ϕt .+ 0.5 .* n.Anl2 .* conj.(ϕt) .- 0.5 .* lapRot(n,ϕt)
end

mutable struct NumModelCrankNicolsonQuasiNewton{F,P,W} <: AbstractNumModel{F,P,W}
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
   ϕ_hat
   M
   b
   Anl
   potential :: P
   writer :: W
end

nmodel(n::NumModelCrankNicolsonQuasiNewton) = 25

Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewton) = print(io,
         "Crank-Nicolson Quasi-Newton scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelCrankNicolsonQuasiNewton)

Performs a single time step for the stationnary Crank-Nicolson scheme.

# Detail

A quasi-Newton loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 3 × β × ∥ψ∥²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function timeStep!(n::NumModelCrankNicolsonQuasiNewton)
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
      # Anls = V + 3 × β × ∥ψ∥²
      @. n.Anl = n.potential.V + 3 * n.β * ψ * conj(ψ)
      # M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )
      @. n.M = 1. / ( 1. / n.Δt + ( n.potential.V + 3 *  n.β * real( ψ * conj(ψ) ) ) * 0.5 )
      # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
      n.b .= (1/n.Δt) .* (ϕ₁ .- n.f.ϕ) .+ (n.potential.V .+ n.β .* conj.(ψ).*ψ ) .* ψ .- lapRot(n,ψ)
      # solving
      krylovPreCond!(n, ϕw)
      # update solution
      ϕ₁ .= ϕ₁ .- ϕw
      # newton residual
      resNewton = sqrt(sum(real.(ϕw.*conj.(ϕw))))
      if resNewton < n.tolnewton
         break
      end
   end
   # update field
   n.f.ϕ .= ϕ₁
   # normalize
   normalize!(n.f)
end

"""
    prodA(n::NumModelCrankNicolsonQuasiNewton, ϕt)

Matrix-vector product for the stationnary Crank-Nicolson Quasi-Newton method :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonQuasiNewton, ϕt)
   (1/n.Δt .+ n.Anl .* 0.5) .* ϕt .- 0.5 .* lapRot(n,ϕt)
end

mutable struct NumModelCrankNicolsonT{F,P,W} <: AbstractNumModel{F,P,W}
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
   ϕ_hat
   M
   b
   Anl
   Anl2
   potential :: P
   writer :: W
end

nmodel(n::NumModelCrankNicolsonT) = 45

Base.show(io::IO, n::NumModelCrankNicolsonT) = print(io,
         "Time dependant Crank-Nicolson Newton-Raphson scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelCrankNicolsonT)

Performs a single time step for the unstationnary Crank-Nicolson scheme.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ - 1/2 Aₙₗ₂ conj(ϕ) + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ - 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 2 × β × ∥ψ∥²

Aₙₗ₂ = β × ψ²

b = i Δt⁻¹ × (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function timeStep!(n::NumModelCrankNicolsonT)
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
      # M⁻¹ = i Δt⁻¹ - 1/2 ( V + 3 × β × ∥ψ∥²)
      @. n.M = 1. / ( 1. * im / n.Δt - ( n.potential.V + 3 *  n.β * real( ψ * conj(ψ) ) ) * 0.5 )
      # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
      n.b .= (1. * im / n.Δt) .* (ϕ₁ .- n.f.ϕ) .- (n.potential.V .+ n.β .* conj.(ψ).*ψ ) .* ψ .+ lapRot(n,ψ)
      # solving
      krylovPreCond!(n, ϕw)
      # update solution
      ϕ₁ .= ϕ₁ .- ϕw
      # newton residual
      resNewton = sqrt(sum(real.(ϕw.*conj.(ϕw))))
      if resNewton < n.tolnewton
         break
      end
   end
   # update field
   n.f.ϕ .= ϕ₁
end

"""
    prodA(n::NumModelCrankNicolsonT, ϕt)

Matrix-vector product for the unstationnary Crank-Nicolson Newton-Raphson method :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ - 1/2 Aₙₗ₂ conj(ϕ) + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonT, ϕt)
   ( (1. * im /n.Δt ) .- 0.5 .* n.Anl) .* ϕt .- 0.5 .* n.Anl2 .* conj.(ϕt) .+ 0.5 .* lapRot(n,ϕt)
end

mutable struct NumModelCrankNicolsonQuasiNewtonT{F,P,W} <: AbstractNumModel{F,P,W}
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
   ϕ_hat
   M
   b
   Anl
   potential :: P
   writer :: W
end

nmodel(n::NumModelCrankNicolsonQuasiNewtonT) = 46

Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewtonT) = print(io,
         "Time dependant Crank-Nicolson Quasi-Newton scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelCrankNicolsonQuasiNewton)

Performs a single time step for the stationnary Crank-Nicolson scheme.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 3 × β × ∥ψ∥²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function timeStep!(n::NumModelCrankNicolsonQuasiNewtonT)
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
      # Anls = V + 3 × β × ∥ψ∥²
      @. n.Anl = n.potential.V + 3 * n.β * ψ * conj(ψ)
      # M⁻¹ = i Δt⁻¹ - 1/2 × ( V + 3 × β × ∥ψ∥² )
      @. n.M = 1. / ( 1. * im / n.Δt - ( n.potential.V + 3 *  n.β * real( ψ * conj(ψ) ) ) * 0.5 )
      # b = i Δt⁻¹ * (ϕ₁ - ϕ₀) - (V + β × ∥ψ∥²) × ψ + (coeffΔ Δ + Ω Lz) × ψ + coeffΔ Δϕ - Ω Lz Φ
      n.b .= (1. * im / n.Δt) .* (ϕ₁ .- n.f.ϕ) .- (n.potential.V .+ n.β .* conj.(ψ).*ψ ) .* ψ .+ lapRot(n,ψ)
      # solving
      krylovPreCond!(n, ϕw)
      # update solution
      ϕ₁ .= ϕ₁ .- ϕw
      # newton residual
      resNewton = sqrt(sum(real.(ϕw.*conj.(ϕw))))
      if resNewton < n.tolnewton
         break
      end
   end
   # update field
   n.f.ϕ .= ϕ₁
end

"""
    prodA(n::NumModelCrankNicolsonQuasiNewtonT, ϕt)

Matrix-vector product for the unstationnary Crank-Nicolson Quasi-Newton method :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonQuasiNewtonT, ϕt)
   ( (1. * im / n.Δt) .- n.Anl .* 0.5) .* ϕt .+ 0.5 .* lapRot(n,ϕt)
end
