mutable struct NumModelRK2{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   plan :: Plan
   ϕ_hat
   writers :: AbstractWriterCollection{F}
end

"""
    NumModelRK2(f::AbstractField, param::AbstractParameters,
          Δt::Real, niter::Integer, freqbckp::Integer)

Returns a Runge-Kutta 2 numerical model.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> param = GrossPitaevskiiParameters();
julia> nummodel = NumModelRK2(field, param, 0.01, 1000, 100)
Backward Euler
  ├──────────  krylov: n iterations 70, tolerance 1.0e-8
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelRK2(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   return NumModelRK2{typeof(f),typeof(param),typeof(plan)}(f, param,
         Δt, niter, freqbckp,
         plan, ϕ_hat,
         writers
      )
end

Base.show(io::IO, n::NumModelRK2) = print(io,
         "Backward Euler\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelRK2{F,P}) where {F<:AbstractField3D}
end
