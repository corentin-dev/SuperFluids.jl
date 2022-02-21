"Abstract supertype for parameters."
abstract type AbstractParameters end

export GrossPitaevskiiParameters
export NavierStokesParameters

"""
    GrossPitaevskiiParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.

It contains the following informations:

- `coeffΔ`: coefficient in front of Δ term,
- `β`: interaction coefficient,
- `pot`: potential,
- `Ω`: rotation coefficient.
"""
struct GrossPitaevskiiParameters <: AbstractParameters
    coeffΔ :: Real
    β :: Real
    pot :: AbstractPotential
    Ω :: Real
end

"""
    GrossPitaevskiiParameters(;
      coeffΔ = -0.5 :: Real,
      β = 1. :: Real,
      pot :: AbstractPotential,
      Ω = 0. :: Real)

Returns a `GrossPitaevskiiParameters``.

Parameters are:
- `coeffΔ`: coefficient in front of the ``Δ`` operator,
- `β`: interaction coefficient,
- `pot`: potential (of class `AbstractPotential`),
- `Ω`: rotation along ``z`` axis.

Example
=======
```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8)
GrossPitaevskiiParameters
  ├──────  equation solved:
  ├─────── idϕ/dt = -0.5 Δϕ + 1000 |ϕ|²ϕ  + V(x)ϕ - i 0.8 Lz ϕ
  └──────────────── coeffΔ: -0.5, β: 1000, Ω: 0.8
```
"""
function GrossPitaevskiiParameters(;
      coeffΔ :: Real = -0.5,
      β :: Real = 1.,
      pot :: AbstractPotential,
      Ω :: Real = 0.)
   return GrossPitaevskiiParameters(coeffΔ, β, pot, Ω)
end

Base.show(io::IO, param::GrossPitaevskiiParameters) =
     print(io, "GrossPitaevskiiParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── idϕ/dt = $(param.coeffΔ) Δϕ + $(param.β) |ϕ|²ϕ  + V(x)ϕ - i $(param.Ω) Lz ϕ \n",
         "  └──────────────── coeffΔ: $(param.coeffΔ), β: $(param.β), Ω: $(param.Ω)")

"""
    NavierStokesParameters <: AbstractParameters

Parameters to compute Navier-Stokes equations.
"""
struct NavierStokesParameters <: AbstractParameters
   "viscosity"
   ν :: Real
   "volumic mass"
   ρ :: Real
end

"""
    NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1.)

Returns a NavierStokesParameters.

Example
=======
```jldoctest
julia> param = NavierStokesParameters()
NavierStokesParameters
  ├──────  equation solved:
  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu
  ├─────── ∇⋅u = 0
  └─────── ν: 0.001
```
"""
function NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1.)
   return NavierStokesParameters(ν, ρ)
end

Base.show(io::IO, param::NavierStokesParameters) = print(
    io, "NavierStokesParameters\n",
    "  ├──────  equation solved:\n",
    "  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu\n",
    "  ├─────── ∇⋅u = 0\n",
    "  └─────── ν: $(param.ν) ρ: $(param.ρ)"
    )