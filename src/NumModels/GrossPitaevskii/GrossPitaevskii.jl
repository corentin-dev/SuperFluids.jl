mutable struct NumModelCrankNicolson{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: Plan
   M :: AbstractArray
   b :: AbstractArray
   Anl :: AbstractArray
   Anl2 :: AbstractArray
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolson(f, param,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   Anl2 = similar(f.ϕ)
   return NumModelCrankNicolson{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, M, b, Anl, Anl2,
         writers
      )
end

Base.show(io::IO, n::NumModelCrankNicolson) = print(io,
         "Crank-Nicolson Newton-Raphson scheme\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

mutable struct NumModelCrankNicolsonQuasiNewton{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: Plan
   M :: AbstractArray
   b :: AbstractArray
   Anl :: AbstractArray
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonQuasiNewton(f, param,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   return NumModelCrankNicolsonQuasiNewton{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, M, b, Anl,
         writers
      )
end

Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewton) = print(io,
         "Crank-Nicolson Quasi-Newton scheme\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

mutable struct NumModelCrankNicolsonT{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: Plan
   M :: AbstractArray
   b :: AbstractArray
   Anl :: AbstractArray
   Anl2 :: AbstractArray
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonT(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   Anl2 = similar(f.ϕ)
   return NumModelCrankNicolsonT{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, M, b, Anl, Anl2,
         writers
      )
end

Base.show(io::IO, n::NumModelCrankNicolsonT) = print(io,
         "Time dependant Crank-Nicolson Newton-Raphson scheme\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

mutable struct NumModelCrankNicolsonQuasiNewtonT{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: Plan
   M :: AbstractArray
   b :: AbstractArray
   Anl :: AbstractArray
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonQuasiNewtonT(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   return NumModelCrankNicolsonQuasiNewtonT{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, M, b, Anl,
         writers
      )
end

Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewtonT) = print(io,
         "Time dependant Crank-Nicolson Quasi-Newton scheme\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   return -coeffΔ .* ( n.gf.ddx .+ n.gf.ddy) + Ω .* im .* (n.gf.rx+n.gf.ry)
end

function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField3D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   return -coeffΔ .* ( n.gf.ddx .+ n.gf.ddy .+ n.gf.ddz) + Ω .* im .* (n.gf.rx+n.gf.ry)
end

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField2D,P<:GrossPitaevskiiParameters,Plan}
   # references
   ϕ = n.f.ϕ
   coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
   x, y = n.f.g.x, n.f.g.y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V = n.param.pot.V

   computeDerivatives!(n.gf, n.plan, ϕ)

   abs∇ϕ_x = sum( abs2.( n.gf.dx ) )
   abs∇ϕ_y = sum( abs2.( n.gf.dy ) )

   # compute energies
   EΩ = sum( real.( im.*conj.(ϕ).*(
      Ω.*( n.gf.rx .+  n.gf.ry )
      ) ) ) * Δx * Δy
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(V.*abs2.(ϕ))) * Δx * Δy
   Eβ = sum( 0.5*β*(abs2.(ϕ).^2)) * Δx * Δy
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

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters,Plan<:AbstractFFTPlan}
   # references
   ϕ = n.f.ϕ
   coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V = n.param.pot.V

   computeDerivatives!(n.gf, n.plan, ϕ)

   abs∇ϕ_x = sum( abs2.( n.gf.dx ) )
   abs∇ϕ_y = sum( abs2.( n.gf.dy ) )
   abs∇ϕ_z = sum( abs2.( n.gf.dz ) )

   # compute energies
   EΩ = sum( real.( im.*conj.(ϕ).*(
      Ω.*( n.gf.rx .+  n.gf.ry )
      ) ) ) * Δx * Δy * Δz
   EΔ = (-coeffΔ * (abs∇ϕ_x.+abs∇ϕ_y.+abs∇ϕ_z) + sum(V.*abs2.(ϕ))) * Δx * Δy * Δz
   Eβ = sum( 0.5*β*(abs2.(ϕ).^2)) * Δx * Δy * Δz
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

include("backward-euler.jl")
include("crank-nicolson.jl")
include("adi.jl")