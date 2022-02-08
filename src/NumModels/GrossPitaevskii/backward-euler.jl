mutable struct NumModelBackwardEuler{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   plan :: Plan
   M :: AbstractArray
   b :: AbstractArray
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
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      plantype::PlanType = FFTPlan())
   gf = GradientField(f,rotation=true)
   plan = Plan(f,t=plantype)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   M = similar(f.ϕ)
   b = similar(f.ϕ)
   return NumModelBackwardEuler{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         nkrylov, tolkrylov,
         plan, M, b,
         writers
      )
end

Base.show(io::IO, n::NumModelBackwardEuler) =
   print(io,
      "Backward Euler\n",
      "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
      "  ├───────  time step: $(n.Δt)\n",
      "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

mutable struct NumModelBackwardEulerNoPrecond{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   nkrylov :: Integer
   tolkrylov :: Real
   plan :: Plan
   Anl :: AbstractArray
   b :: AbstractArray
   writers :: AbstractWriterCollection{F}
end

function NumModelBackwardEulerNoPrecond(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer;
      nkrylov::Integer = 70, tolkrylov::Real = 1e-8,
      plantype::PlanType = FFTPlan())
   gf = GradientField(f,rotation=true)
   plan = Plan(f,t=plantype)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   Anl = similar(f.ϕ)
   b = similar(f.ϕ)
   return NumModelBackwardEulerNoPrecond{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         nkrylov, tolkrylov,
         plan, Anl, b,
         writers
      )
end

Base.show(io::IO, n::NumModelBackwardEulerNoPrecond) =
      print(io,
         "Backward Euler without preconditionning\n",
         "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelBackwardEuler{F,P}) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
   # compute matrices and vectors
   # M⁻¹ = Δt⁻¹ + NL
   # NL = V + β × ∥ϕ∥² for GP
   @. n.M = 1. / ( 1 / n.Δt + n.param.pot.V + n.param.β * abs2(n.f.ϕ) )
   # b = M × ϕ × Δt⁻¹
   @. n.b = n.M * n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function prodA(n::NumModelBackwardEuler{F,P}, ϕt) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
   ϕt .- n.M .* lapRot(n,ϕt)
end

function timeStep!(n::NumModelBackwardEulerNoPrecond{F,P}) where {F <: AbstractField, P <: GrossPitaevskiiParameters}
   # compute matrices and vectors
   # Anl = V + β × ∥ϕ∥²
   @. n.Anl = n.param.pot.V + n.param.β * abs2(n.f.ϕ)
   # b = ϕ × Δt⁻¹
   @. n.b = n.f.ϕ / n.Δt
   # solving
   nkrylov = krylov!(n, n.f.ϕ)
   # normalize
   normalize!(n.f)
   return nkrylov
end

function prodA(n::NumModelBackwardEulerNoPrecond, ϕt)
   ( (1. / n.Δt) .+ n.Anl ) .* ϕt .- lapRot(n,ϕt)
end