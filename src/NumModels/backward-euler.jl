mutable struct NumModelBackwardEuler{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   plan :: Plan
   ϕ_hat
   M
   b
   writers :: AbstractWriterCollection{F}
end

"""
    NumModelBackwardEuler(f::AbstractField, param::AbstractParameters,
          Δt::Real, niter::Integer, freqbckp::Integer;
          nkrylov::Integer = 70, tolkrylov::Real = 1e-8)

Returns a Backward Euler numerical model.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> param = GrossPitaevskiiParameters();
julia> nummodel = NumModelBackwardEuler(field, param, 0.01, 1000, 100)
Backward Euler
  ├──────────  krylov: n iterations 70, tolerance 1.0e-8
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelBackwardEuler(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   return NumModelBackwardEuler{typeof(f),typeof(param),typeof(plan)}(f, param,
         Δt, niter, freqbckp,
         nkrylov, tolkrylov,
         plan, ϕ_hat, M, b,
         writers
      )
end

Base.show(io::IO, n::NumModelBackwardEuler) = print(io,
         "Backward Euler\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

mutable struct NumModelBackwardEulerNoPrecond{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   plan :: AbstractPlan{F}
   ϕ_hat
   Anl
   b
   writers :: AbstractWriterCollection{F}
end

function NumModelBackwardEulerNoPrecond(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   Anl = similar(f.ϕ)
   b = similar(f.ϕ)
   return NumModelBackwardEulerNoPrecond{typeof(f),typeof(param),typeof(plan)}(f, param,
         Δt, niter, freqbckp,
         nkrylov, tolkrylov,
         plan, ϕ_hat, Anl, b,
         writers
      )
end

Base.show(io::IO, n::NumModelBackwardEulerNoPrecond) = print(io,
         "Backward Euler without preconditionning\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

include("GrossPitaevskii/backward-euler.jl")
