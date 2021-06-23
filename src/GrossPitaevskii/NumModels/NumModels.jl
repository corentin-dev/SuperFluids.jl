function NumModel(f::F, p::P, conf::AbstractConfig) where F where P
   nummodel = nothing
   nmodel = retrieve(conf, "solver", "model", Int64)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   Δt =  retrieve(conf, "time", "deltat", Float64)
   niter = retrieve(conf, "time", "itermax", Float64)
   restart = ( retrieve(conf, "time", "restart", Int64) == 1 )
   freqbckp = retrieve(conf, "io", "freqbckp", Int64)
   coeffΔ = retrieve(conf, "model", "delta", Float64)
   β = retrieve(conf, "model", "beta", Float64)
   Ω = retrieve(conf, "model", "omega", Float64)
   if nmodel == 1
      println("Sobolev, not yet implemented")
      return nothing
   elseif nmodel == 2 || nmodel == 20
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      nummodel = NumModelBackwardEuler{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, p,
                       writers
                      )
   elseif nmodel == 3
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      nummodel = NumModelBackwardEulerNL{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, p,
                       writers
                      )
   elseif nmodel == 21
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      Anl = similar(f.ϕ)
      Anl2 = similar(f.ϕ)
      nummodel = NumModelCrankNicolson{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, Anl, Anl2, p,
                       writers
                      )
   elseif nmodel == 22
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      Anl = similar(f.ϕ)
      b = similar(f.ϕ)
      nummodel = NumModelBackwardEulerNoPrecond{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov,
                       coeffΔ, β, Ω, plan, ϕ_hat, Anl, b, p,
                       writers
                      )
   elseif nmodel == 25
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      Anl = similar(f.ϕ)
      nummodel = NumModelCrankNicolsonQuasiNewton{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, Anl, p,
                       writers
                      )
   elseif nmodel == 27
   # case (27)
   #    call model_imaginary_time(7) ! semi-implicit backward Euler with external velocity, no renormalization
      println("Warning : should be Backward Euler with external velocity")
   elseif nmodel == 28
   # case (28)
   #    call model_imaginary_time(8) ! full-implicit Crank-Nicolson with external velocity, no renormalization
      println("Warning : should be without Crank-Nicolson with external velocity")
   elseif nmodel == 29
      nummodel = NumModelExternalVelocity{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp,
                       coeffΔ, β, Ω, plan, ϕ_hat, p,
                       writers
                      )
   elseif nmodel == 41
      nummodel = NumModelADI1{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp,
                       coeffΔ, β, Ω, plan, ϕ_hat, p,
                       writers
                      )
   elseif nmodel == 42
      nummodel = NumModelADI2{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp,
                       coeffΔ, β, Ω, plan, ϕ_hat, p,
                       writers
                      )
   elseif nmodel == 43
   # case (43)
   #    call model_REL
      println("Warning : should be ReSP (?)")
   elseif nmodel == 45
      println("")
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      Anl = similar(f.ϕ)
      nummodel = NumModelCrankNicolsonQuasiNewtonT{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, Anl, p,
                       writers
                      )
   elseif nmodel == 46
      println("")
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      nkrylov = retrieve(conf, "solver", "iterkrylov", Int64)
      tolkrylov = retrieve(conf, "solver", "tolkrylov", Float64)
      nnewton = retrieve(conf, "solver", "iternewton", Int64)
      tolnewton = retrieve(conf, "solver", "tolnewton", Float64)
      M = similar(f.ϕ)
      b = similar(f.ϕ)
      Anl = similar(f.ϕ)
      Anl2 = similar(f.ϕ)
      nummodel = NumModelCrankNicolsonT{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
                       coeffΔ, β, Ω, plan, ϕ_hat, M, b, Anl, Anl2, p,
                       writers
                      )
   end
   return nummodel
end
   # select case(model)
   # case (0)
   #    call model_approximate
   # case (1)
   #    call model_one_component_sobolev
   # case (26)
   #    call model_imaginary_time(6) ! full-implicit C-N : Newton by "real vector"
   # case(299)
   #    call model_imaginary_time(99) ! toy model FL

# !
# ! multi component
   # case (3,30)
   #    call model_imaginary_time_two(1) ! With B-E T-F approx (for strong interaction only)
   # case (31)
   #    call model_multi_component_sobolev ! With G-S (for weak interaction only)
   # case (32)
   #    call model_imaginary_time_two(2) ! With G-S (for weak interaction only)
# ! unstationary
   # case (44) ! Newton with real value
   #    call model_NR(4,delta_t,flag_cutoff) ! Full implicit C-N
   # ! unstationary multi component
   # case (51)
   #    call model_TS1_ADI_two
   # case (52)
   #    call model_TS2_ADI_two
   # case (53)
   #    call model_REL_two
   # case (56) ! Crank-Nicolson Newton with complex nonlinear GCR
   #    call model_NR_two(2,delta_t,flag_cutoff)
   # case (55) ! Crank-Nicolson quasi-Newton
   #    call model_NR_two(3,delta_t,flag_cutoff)
   # case (54) ! Crank-Nicolson Newton with real value
   #    call model_NR_two(4,delta_t,flag_cutoff)
   # case (57) ! backward-Euler Newton with real value
   #    call model_NR_two(1,delta_t,flag_cutoff)

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
   return ϕtx
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
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(real.(V).*real.(ϕ.*conj.(ϕ)))) * Δx * Δy
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
