mutable struct NumModelADI1{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

function NumModelADI1(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelADI1{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
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
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

function NumModelADI2(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelADI2{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
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