mutable struct NumModelCrankNicolson{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: AbstractArray{F}
   M :: AbstractArray{F}
   b :: AbstractArray{F}
   Anl :: AbstractArray{F}
   Anl2 :: AbstractArray{F}
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolson(f, param,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   Anl2 = similar(f.ϕ)
   return NumModelCrankNicolson{typeof(f),typeof(param),typeof(plan)}(f,param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, ϕ_hat, M, b, Anl, Anl2,
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
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: AbstractArray{F}
   M :: AbstractArray{F}
   b :: AbstractArray{F}
   Anl :: AbstractArray{F}
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonQuasiNewton(f, param,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   return NumModelCrankNicolsonQuasiNewton{typeof(f),typeof(param),typeof(plan)}(f,param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, ϕ_hat, M, b, Anl,
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
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: AbstractArray{F}
   M :: AbstractArray{F}
   b :: AbstractArray{F}
   Anl :: AbstractArray{F}
   Anl2 :: AbstractArray{F}
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonT(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   Anl2 = similar(f.ϕ)
   return NumModelCrankNicolsonT{typeof(f),typeof(param),typeof(plan)}(f, param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, ϕ_hat, M, b, Anl, Anl2,
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
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   nnewton :: Integer
   tolnewton :: Real
   plan :: AbstractPlan{F}
   ϕ_hat
   M
   b
   Anl
   writers :: AbstractWriterCollection{F}
end

function NumModelCrankNicolsonQuasiNewtonT(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      nnewton::Integer = 15, tolnewton::Real = 1e-6)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   Anl = similar(f.ϕ)
   return NumModelCrankNicolsonQuasiNewtonT{typeof(f),typeof(param),typeof(plan)}(f,param,
         Δt, niter, freqbckp, nkrylov, tolkrylov, nnewton, tolnewton,
         plan, ϕ_hat, M, b, Anl,
         writers
      )
end

Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewtonT) = print(io,
         "Time dependant Crank-Nicolson Quasi-Newton scheme\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

include("GrossPitaevskii/crank-nicolson.jl")
