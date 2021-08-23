"Abstract supertype for numerical models."
abstract type AbstractNumModel{F,P} end

export NumModelADI1, NumModelADI2
export NumModelBackwardEuler, NumModelBackwardEulerNoPrecond, NumModelBackwardEulerNL
export NumModelCrankNicolson, NumModelCrankNicolsonQuasiNewton, NumModelCrankNicolsonT, NumModelCrankNicolsonQuasiNewtonT
export NumModelExternalVelocity

include("adi.jl")
include("backward-euler.jl")
include("crank-nicolson.jl")
include("external-velocity.jl")

include("krylov.jl")

function lapRot(n::AbstractNumModel{F}, ϕt) where {F<:AbstractField2D}
   # references
   b = n.b
   ϕthat_x, ϕthat_y = n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # perform FFT
   mul!(ϕthat_x, plan_x, ϕt)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕthat_x = (coeffΔ*ξx^2 - Ω*y*ξx) * ϕthat_x
   # backward FFT
   ϕtx = plan_x \ ϕthat_x
   # perform FFT
   mul!(ϕthat_y, plan_y, ϕt)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕthat_y = (coeffΔ*ξy^2 + Ω*x*ξy) * ϕthat_y
   # backward FFT
   ϕty = plan_y \ ϕthat_y
   # return
   @. ϕtx = ϕtx+ϕty
   return ϕtx
end

function lapRot(n::AbstractNumModel{F}, ϕt) where {F<:AbstractField3D}
   # references
   ϕthat_x, ϕthat_y, ϕthat_z = n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   # perform FFT
   mul!(ϕthat_x, plan_x, ϕt)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕthat_x = (coeffΔ*ξx^2 - Ω*y*ξx) * ϕthat_x
   # backward FFT
   ϕtx = plan_x \ ϕthat_x
   # perform FFT
   mul!(ϕthat_y, plan_y, ϕt)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕthat_y = (coeffΔ*ξy^2 + Ω*x*ξy) * ϕthat_y
   # backward FFT
   ϕty = plan_y \ ϕthat_y
   # perform FFT
   mul!(ϕthat_z, plan_z, ϕt)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕthat_z = coeffΔ*ξz^2 * ϕthat_z
   # backward FFT
   ϕtz = plan_z \ ϕthat_z
   # return
   @. ϕtx = (ϕtx+ϕty+ϕtz)
end

function solve!(n::AbstractNumModel;istart=1,plot=false)
   if plot
      @eval using SuperFluids.Plots
      p = Plot(n.f)
      createPlot!(p)
   end
   write!(n.writers,prefix="res",icpu=0,istep=istart,Δt=n.Δt)
   for it = istart:istart+n.niter
      println("iteration $(it)")
      energy(n,true)
      timeStep!(n)
      if it % n.freqbckp == 0
         write!(n.writers,prefix="res",icpu=0,istep=it,Δt=n.Δt)
      end
      if plot
         updatePlot!(p)
      end
   end
   energy(n,true)
   if plot
      return p
   else
      return nothing
   end
end

function energy(n::AbstractNumModel{F}, showEnergy=false) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω, β = n.coeffΔ, n.Ω, n.β
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   p = n.potential
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute grad x
   ∇ϕ_hat = im .* ξx .* ϕhat_x
   ∇ϕ_x = plan_x \ ∇ϕ_hat # we get grad x
   abs∇ϕ_x = sum(real.(∇ϕ_x.*conj(∇ϕ_x)))
   # compute rotx
   ∇ϕ_hat .= im .*  Ω .* y .* ξx .* ϕhat_x
   ldiv!(∇ϕ_x, plan_x, ∇ϕ_hat) # we get rot x
   # compute grad y
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_hat = im .* ξy .* ϕhat_y
   ∇ϕ_y = plan_y \ ∇ϕ_hat # we get grad y
   abs∇ϕ_y = sum(real.(∇ϕ_y.*conj(∇ϕ_y)))
   # compute roty
   ∇ϕ_hat = im .* -Ω .* x .* ξy .* ϕhat_y
   ldiv!(∇ϕ_y, plan_y, ∇ϕ_hat) # we get rot y
   # computing energies (locally)
   EΩ = sum(real.(im.*conj.(ϕ).*(∇ϕ_x.+∇ϕ_y))) * Δx * Δy
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(real.(p.(x,y)).*real.(ϕ.*conj.(ϕ)))) * Δx * Δy
   Eβ = sum( 0.5*β*(real.(ϕ.*conj.(ϕ)).^2)) * Δx * Δy
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println("Angular Momentum Energy : $(EΩ)")
      println("Kinetic + Potential Energy : $(EΔ)")
      println("Interaction Energy : $(Eβ)")
      println("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end

function energy(n::AbstractNumModel{F}, showEnergy=false) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω, β = n.coeffΔ, n.Ω, n.β
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   p = n.potential
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute grad x
   ∇ϕ_hat = im .* ξx .* ϕhat_x
   ∇ϕ_x = plan_x \ ∇ϕ_hat # we get grad x
   abs∇ϕ_x = sum(real.(∇ϕ_x.*conj(∇ϕ_x)))
   # compute rotx
   ∇ϕ_hat .= im .*  Ω .* y .* ξx .* ϕhat_x
   ldiv!(∇ϕ_x, plan_x, ∇ϕ_hat) # we get rot x
   # compute grad y
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_hat = im .* ξy .* ϕhat_y
   ∇ϕ_y = plan_y \ ∇ϕ_hat # we get grad y
   abs∇ϕ_y = sum(real.(∇ϕ_y.*conj(∇ϕ_y)))
   # compute roty
   ∇ϕ_hat = im .* -Ω .* x .* ξy .* ϕhat_y
   ldiv!(∇ϕ_y, plan_y, ∇ϕ_hat) # we get rot y
   # compute grad z 
   mul!(ϕhat_z, plan_z, ϕ)
   ∇ϕ_hat = im .* ξz .* ϕhat_z
   ∇ϕ_z = plan_z \ ∇ϕ_hat # we get grad x
   abs∇ϕ_z = sum(real.(∇ϕ_z.*conj(∇ϕ_z)))
   # computing energies (locally)
   EΩ = sum(real.(im.*conj.(ϕ).*(∇ϕ_x.+∇ϕ_y+∇ϕ_z))) * Δx * Δy *Δz
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y.+abs∇ϕ_z) + sum(p.(x,y,z).*real.(ϕ.*conj.(ϕ)))) * Δx * Δy * Δz
   Eβ = sum( 0.5*β*(real.(ϕ.*conj.(ϕ)).^2)) * Δx * Δy * Δz
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println("Angular Momentum Energy : $(EΩ)")
      println("Kinetic + Potential Energy : $(EΔ)")
      println("Interaction Energy : $(Eβ)")
      println("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end
