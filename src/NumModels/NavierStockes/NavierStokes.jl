export NavierStokesParameters

"""
    NavierStokesParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.
"""
struct NavierStokesParameters <: AbstractParameters
   "viscosity"
   ν :: Real
   "volumic mass"
   ρ :: Real
   "forcing term"
   f :: Function
end

"""
    NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1,
      f :: Function)

Returns a NavierStokesParameters.

Example
=======
```jldoctest
julia> param = NavierStokesParameters()
NavierStokesParameters
  ├──────  equation solved:
  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu + f
  ├─────── ∇⋅u = 0
  └─────── ν: 0.001
```
"""
function NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1,
      f :: Function)
   return NavierStokesParameters(coeffΔ, β, V, Ω)
end

Base.show(io::IO, param::NavierStokesParameters) =
     print(io, "NavierStokesParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu + f\n",
         "  ├─────── ∇⋅u = 0\n",
         "  └─────── ν: $(param.ν) ρ: $(param.ρ)")

function non_linear(field<:AbstractField3D, param<:NavierStokesParameters)
   return 0
end

function rhs(field<:AbstractField3D, param<:NavierStokesParameters)
   return 0
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField2D,P<:NavierStokesParameters}

   if(showEnergy)
      println("for the moment no energy is computed")
   end

   return nothing
end

function energy(n::AbstractNumModel{F}, showEnergy=false) where {F<:AbstractField3D}

   if(showEnergy)
      println("for the moment no energy is computed")
   end

   return nothing
end
