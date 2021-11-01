mutable struct NumModelForwardEuler{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   ϕ_hat
   writers :: AbstractWriterCollection{F}
end

"""
    NumModelForwardEuler(f::AbstractField, param::AbstractParameters,
          Δt::Real, niter::Integer, freqbckp::Integer)

Returns a Runge-Kutta 2 numerical model.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> param = GrossPitaevskiiParameters();
julia> nummodel = NumModelForwardEuler(field, param, 0.01, 1000, 100)
Backward Euler
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelForwardEuler(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   ϕ_hat = similar(f.ϕ)
   return NumModelForwardEuler{typeof(f),typeof(param),typeof(plan)}(f, param,
         Δt, niter, freqbckp,
         plan, ϕ_hat,
         writers
      )
end

Base.show(io::IO, n::NumModelForwardEuler) = print(io,
         "Backward Euler\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelForwardEuler{F}) where {F<:AbstractField3D}
   n.ϕ_hat .+= rhs(n) * n.Δt
   ldiv!(n.f.ϕ, n.plan.plan, n.ϕ_hat)
   return 1
end
