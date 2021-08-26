"Abstract supertype for grids."
abstract type AbstractParameters end

"""
    GrossPitaevskiiParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.
"""
struct GrossPitaevskiiParameters <: AbstractParameters
   "coefficient in front of Δ term"
   coeffΔ :: Real
   "interaction coefficient"
   β :: Real
   "potential"
   V :: Function
   "rotation coefficient"
   Ω :: Real
end

"""
    GrossPitaevskiiParameters(;
      coeffΔ = -0.5 :: Real,
      β = 1. :: Real,
      Ω = 0. :: Real)

Returns a GrossPitaevskiiParameters.

Example
=======
```
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8)
```
"""
function GrossPitaevskiiParameters(field<:AbstractField2D;
      coeffΔ :: Real = -0.5,
      β :: Real = 1.,
      V :: Function = PotentialZero(field),
      Ω :: Real = 0.)
   return GrossPitaevskiiParameters(coeffΔ, β, Ω)
end

Base.show(io::IO, param::GrossPitaevskiiParameters) =
     print(io, "GrossPitaevskiiParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── idϕ/dt = $(param.coeffΔ) Δϕ + $(param.β) |ϕ|²ϕ  + V(x)ϕ - i Lz $(param.Ω) ϕ \n",
         "  └──────────────── coeffΔ: $(param.coeffΔ), β: $(param.β), Ω: $(param.Ω)\n")
