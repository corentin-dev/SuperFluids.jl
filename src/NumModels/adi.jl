mutable struct NumModelADI1{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: AbstractPlan{F}
   ϕ_hat
   writers :: AbstractWriterCollection{F}
end

function NumModelADI1(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   return NumModelADI1{typeof(f),typeof(param),typeof(plan)}(f,param,
         Δt, niter, freqbckp,
         plan, ϕ_hat,
         writers
      )
end

Base.show(io::IO, n::NumModelADI1) = print(io,
         "Splitting Order 1\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI1)
   solveLapRot!(n,n.Δt)
   solveNL!(n,n.Δt)
   return 1
end

mutable struct NumModelADI2{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: AbstractPlan{F}
   ϕ_hat
   writers :: AbstractWriterCollection{F}
end

function NumModelADI2(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   return NumModelADI2{typeof(f),typeof(param),typeof(plan)}(f,param,
         Δt, niter, freqbckp,
         plan, ϕ_hat,
         writers
      )
end

Base.show(io::IO, n::NumModelADI2) = print(io,
         "Splitting Order 2\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI2)
   solveLapRot!(n,n.Δt*0.5)
   solveNL!(n,n.Δt)
   solveLapRot!(n,n.Δt*0.5)
   return 2
end

include("GrossPitaevskii/adi.jl")