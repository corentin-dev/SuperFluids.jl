mutable struct NumModelADI1{F,P} <: AbstractNumModel{F,P}
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

Base.show(io::IO, n::NumModelADI1) = print(io,
         "Splitting Order 1\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI1)
   solveLapRot!(n,n.Δt)
   solveNL!(n,n.Δt)
end

mutable struct NumModelADI2{F,P} <: AbstractNumModel{F,P}
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

Base.show(io::IO, n::NumModelADI2) = print(io,
         "Splitting Order 2\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI2)
   solveLapRot!(n,n.Δt*0.5)
   solveNL!(n,n.Δt)
   solveLapRot!(n,n.Δt*0.5)
end

function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
   # backward FFT
   ldiv!(ϕ, plan_x, ϕhat_x)
   # perform FFT
   mul!(ϕhat_y, plan_y, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
   # backward FFT
   ldiv!(ϕ, plan_y, ϕhat_y)
   return nothing
end

function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
   # backward FFT
   ldiv!(ϕ, plan_x, ϕhat_x)
   # perform FFT
   mul!(ϕhat_y, plan_y, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
   # backward FFT
   ldiv!(ϕ, plan_y, ϕhat_y)
   # perform FFT
   mul!(ϕhat_z, plan_z, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_z = exp(im*(coeffΔ*ξz^2)*Δtl) * ϕhat_z
   # backward FFT
   ldiv!(ϕ, plan_z, ϕhat_z)
   return nothing
end

function solveNL!(n::AbstractNumModel, Δtl)
   # references
   ϕ = n.f.ϕ
   β = n.β
   V = n.potential.V
   # computation
   @. ϕ = exp(-1im * ( V + β*ϕ*conj(ϕ) ) * Δtl) * ϕ
end
