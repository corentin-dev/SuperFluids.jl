function NumModel(f::F, p::P, conf::ConfParse) where F where P
   nummodel = nothing
   nmodel = parse(Int64,retrieve(conf, "solver", "model"))
   plan = Plan(f)
   writer = Writer(f)
   ϕ_hat = similar(f.ϕ)
   Δt =  parse(Float64,retrieve(conf, "time", "deltat"))
   niter = parse(Float64,retrieve(conf, "time", "itermax"))
   restart = ( parse(Int64,retrieve(conf, "time", "restart")) == 1 )
   freqbckp = parse(Int64,retrieve(conf, "io", "freqbckp"))
   coeffΔ = parse(Float64,retrieve(conf, "model", "delta"))
   β = parse(Float64,retrieve(conf, "model", "beta"))
   Ω = parse(Float64,retrieve(conf, "model", "omega"))
   if nmodel == 1
      println("Sobolev, not yet implemented")
      return nothing
   elseif nmodel == 2
      nkrylov = parse(Int64,retrieve(conf, "solver", "iterkrylov"))
      tolkrylov = parse(Float64,retrieve(conf, "solver", "tolkrylov"))
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      nummodel = NumModelBackwardEuler{typeof(f),typeof(p),typeof(writer)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, p,
                       writer
                      )
   elseif nmodel == 3
      # should be Crank-Nicolson
      nkrylov = parse(Int64,retrieve(conf, "solver", "iterkrylov"))
      tolkrylov = parse(Float64,retrieve(conf, "solver", "tolkrylov"))
      M = similar(f.ϕ)
      Anl = similar(f.ϕ)
      Anl2 = similar(f.ϕ)
      nummodel = NumModelCrankNicolson{typeof(f),typeof(p),typeof(writer)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, Anl, Anl2, p,
                       writer
                      )
   elseif nmodel == 41
      # should be ADI1
      nummodel = NumModelADI2{typeof(f),typeof(p),typeof(writer)}(f,
                       Δt, niter, freqbckp,
                       coeffΔ, β, Ω, plan, ϕ_hat, p,
                       writer
                      )
   elseif nmodel == 42
      nummodel = NumModelADI2{typeof(f),typeof(p),typeof(writer)}(f,
                       Δt, niter, freqbckp,
                       coeffΔ, β, Ω, plan, ϕ_hat, p,
                       writer
                      )
   end
   return nummodel
end

mutable struct NumModelADI2{F,P,W} <: AbstractNumModel{F,P,W}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   potential :: P
   writer :: W
end

function solveLapRot!(n::NumModelADI2{F}, Δtl) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
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

function solveLapRot!(n::NumModelADI2{F}, Δtl) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.f.g.ξx, n.f.g.ξy, n.f.g.ξz
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

function solveNL!(n::NumModelADI2, Δtl)
   # references
   ϕ = n.f.ϕ
   β = n.β
   V = n.potential.V
   # computation
   @. ϕ = exp(-1im * ( V + β*ϕ*conj(ϕ) ) * Δtl) * ϕ
end

function timeStep!(n::NumModelADI2)
   solveLapRot!(n,n.Δt*0.5)
   solveNL!(n,n.Δt)
   solveLapRot!(n,n.Δt*0.5)
end

Base.show(io::IO, n::NumModelADI2) = print(io,
         "Splitting Order 2\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

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

function lapRot(n::AbstractNumModel{F}, ϕt) where {F<:AbstractField2D}
   # references
   M, b = n.M, n.b
   ϕthat_x, ϕthat_y = n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
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
end

function lapRot(n::AbstractNumModel{F}, ϕt) where {F<:AbstractField3D}
   # references
   ϕthat_x, ϕthat_y, ϕthat_z = n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy, ξz = n.f.g.ξx, n.f.g.ξy, n.f.g.ξz
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

function prodA(n::NumModelBackwardEuler, ϕt)
   ϕt .- n.M .* lapRot(n,ϕt)
end

mutable struct NumModelCrankNicolson{F,P,W} <: AbstractNumModel{F,P,W}
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
   Anl :: Array
   Anl2 :: Array
   potential :: P
   writer :: W
end

Base.show(io::IO, n::NumModelCrankNicolson) = print(io,
         "Crank-Nicolson Newton-Raphson scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

#println("Crank Nicolson, not yet implemented")
# model_NR(-2, deltat_t, true)
#   psi1 = phi0
#   solver_newton(-2, deltat_t)
#     do it < itmax_newton
#       solver_krylov_init(-2, 1, 1, coeff_deltas, delta_tt)
#         phi_2 = (psi_1+phi_0)*0.5
#         tmpr = phi_2*conj(phi_2)
#         A_nls = V_xm + 2*beta*tmpr
#         A_nls2 = beta*phi_2
#         M = 1 / (1/delta_t + (V_xm + beta*3*tmpr) + 0.5)
#         phi_tilde = 0
#       solver_krylov(-2, 1, 1, coeff_deltas, 1, delta_tt, false, true)
#         solver_matprecond : vec <- M*vec
#           vec2 <- A_nls*vec1-rmb
#         solver_matvecprod(1, -2, 1, 1, coeff_deltas, phi_tilde, residu, 1/delta_t)
#           comput_lap_rot(vec1)
#             rmb <- invfft( (-coeff_Omega*rotx+coeff_Delta*Lapx)*fft(vec1) )
#           vec2 <- 1/delta_t * vec1 + A_nls*0.5*vec1 + A_nls2*0.5*conj(vec1) - rmb*0.5
#         pscalaire(1, 1, residu, residu2, rho, true)
#           rho <- sum(residu.*residu2)
#       if ||phi_tilde|| < tol_newton
#         stop
#   phi0 = psi1
function timeStep!(n::NumModelCrankNicolson)
   # create working vectors
   ψ = similar(n.f.ϕ)
   ϕ₁ = copy(n.f.ϕ)
   # for krylov
   ϕw = similar(n.f.ϕ)
   ϕw .= 0
   # newton iterations
   for itnewton = 1:10
      # compute matrices and vectors
      # ψ = 1/2 (ϕ₁+ϕ₀)
      @. ψ = 0.5 * (ϕ₁+n.f.ϕ)
      # Anls = V + 2 × β × ∥ψ∥²
      @. n.Anl = n.potential.V + 2*n.β*ψ*conj(ψ)
      # Anls2 = β × ψ
      @. n.Anl2 = n.β * ψ
      # M⁻¹ = Δt⁻¹ + V + 3 × β × ∥ψ∥² + 0.5
      @. n.M = 1. / ( 1. / n.Δt + n.potential.V +3 *  n.β * real( ψ * conj(ψ) ) + 0.5 )
         # bb(i,j,k,n)=(delta_t1*(psi_1(i,j,k,n)-phi_0(i,j,k,n))+(V_xm(i,j,k,n)+&
      # &              beta*(conjg(phi_2(i,j,k,n))*phi_2(i,j,k,n)))*&
      # &              phi_2(i,j,k,n))-rMb(i,j,k) 
      ψhat = plan * ψ
      @. n.b = 1/n.Δt * (ϕ₁ - n.f.ϕ) + (n.potential.V + β * (conj(ψ)+ψ) ) * ψ - rmb 
      # solving
      krylov!(n, ϕw)
   end
   # normalize
   normalize!(n.f)
end

function prodA(n::NumModelCrankNicolson{F}, ϕt) where {F<:AbstractField2D}
   # references
   M, Anl, Anl2, b = n.M, n.Anl, n.Anl2, n.b
   ϕthat_x, ϕthat_y = n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
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
   # vec2 <- 1/delta_t * vec1 + A_nls*0.5*vec1 + A_nls2*0.5*conj(vec1) - rmb*0.5
   @. ϕtx = 1 / n.Δt * ϕt + 0.5 * n.Anl * ϕt + 0.5 * n.Anl2 * conj(ϕt) - 0.5 * (ϕtx+ϕty)
end

function prodA(n::NumModelCrankNicolson{F}, ϕ̃) where {F<:AbstractField3D}
   # references
   M, b = n.M, n.b
   ϕ̃hat_x, ϕ̃hat_y, ϕ̃hat_z = n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy, ξz = n.f.g.ξx, n.f.g.ξy, n.f.g.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   # perform FFT
   mul!(ϕ̃hat_x, plan_x, ϕ̃)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕ̃hat_x = (coeffΔ*ξx^2 - Ω*y*ξx) * ϕ̃hat_x
   # backward FFT
   ϕ̃x = plan_x \ ϕ̃hat_x
   # perform FFT
   mul!(ϕ̃hat_y, plan_y, ϕ̃)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕ̃hat_y = (coeffΔ*ξy^2 + Ω*x*ξy) * ϕ̃hat_y
   # backward FFT
   ϕ̃y = plan_y \ ϕ̃hat_y
   # perform FFT
   mul!(ϕ̃hat_z, plan_z, ϕ̃)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕ̃hat_z = coeffΔ*ξz^2 * ϕ̃hat_z
   # backward FFT
   ϕ̃z = plan_z \ ϕ̃hat_z
   # return
   # ( I - M A ) ϕ̃
   #
   @. ϕ̃x = ϕ̃ - M * (ϕ̃x+ϕ̃y+ϕ̃z)
end

function solve!(n::AbstractNumModel)
   for it = 1:n.niter
      energy(n,true)
      println("iteration $(it)")
      timeStep!(n)
      if it % n.freqbckp == 0
         addFile!(n.writer,"res",0,it,n.Δt)
      end
   end
   energy(n,true)
   finishWriter!(n.writer)
   return nothing
end

function krylov!(n::AbstractNumModel, ϕ)
   normb=sqrt(real(sum(n.b.*conj.(n.b))))
   r = n.b - prodA(n,ϕ)
   r₂ = copy(r)
   p = copy(r)
   for i = 1:n.nkrylov
      Ap = prodA(n,p)
      α = real(sum(r.*conj.(r₂))/sum(Ap.*conj.(r₂)))
      s = r - α * Ap
      As = prodA(n,s)
      ω = sum(As.*conj.(s))/sum(As.*conj.(As))
      @. ϕ = ϕ + α*p + ω*s
      rr₂ = sum(r.*conj.(r₂))
      r = s - ω*As
      if sqrt(real(sum(r.*conj.(r))))/normb < n.tolkrylov
         break
      end
      β = sum(r.*conj.(r₂)) / rr₂ * α / ω
      p = r + β * (p - ω*Ap)
   end
   return nothing
end

function krylovPreCond!(n::AbstractNumModel, ϕ)
   normb=sqrt(real(sum(n.b.*conj.(n.b))))
   r = n.b - prodA(n,ϕ)
   r₂ = copy(r)
   p = copy(r)
   for i = 1:n.nkrylov
      Ap = prodA(n,p)
      α = real(sum(r.*conj.(r₂))/sum(Ap.*conj.(r₂)))
      s = r - α * Ap
      As = prodA(n,s)
      ω = sum(As.*conj.(s))/sum(As.*conj.(As))
      @. ϕ = ϕ + α*p + ω*s
      rr₂ = sum(r.*conj.(r₂))
      r = s - ω*As
      if sqrt(real(sum(r.*conj.(r))))/normb < n.tolkrylov
         break
      end
      β = sum(r.*conj.(r₂)) / rr₂ * α / ω
      p = r + β * (p - ω*Ap)
   end
   return nothing
end

function energy(n::AbstractNumModel{F}, showEnergy=false) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω, β = n.coeffΔ, n.Ω, n.β
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V = n.potential.V
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
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(V.*real.(ϕ.*conj.(ϕ)))) * Δx * Δy
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
   ξx, ξy, ξz = n.f.g.ξx, n.f.g.ξy, n.f.g.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V = n.potential.V
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
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y.+abs∇ϕ_z) + sum(V.*real.(ϕ.*conj.(ϕ)))) * Δx * Δy * Δz
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
