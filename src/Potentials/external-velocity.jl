"""
    PotentialTaylorGreen{F} <: AbstractPotential{F}

Represents a potential used for external velocities ``u_\\text{adv}``. It can be used for [`GrossPitaevskiiParameters`](@ref).
It has the following form:

The potential has the following informations:

- `f`: the field on which the potential acts,
- `V`: a potential field (real),
- `uadvx`: a vector field,
- `uadvy`: a vector field,
- `uadvz`: a vector field (for 3D only).

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> β = 40.; coeffΔ = -0.05;
julia> PotentialTaylorGreen(field, α=-4*coeffΔ*β)
TaylorGreen Potential
```
"""
struct PotentialTaylorGreen{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
   uadvx :: AbstractArray
   uadvy :: AbstractArray
   uadvz :: AbstractArray

   function PotentialTaylorGreen(f::F; α :: Real = 1) where {F<:AbstractField2D}
      V = real.(similar(f.ϕ))
      uadvx = similar(V)
      uadvy = similar(V)
      uadvz = []
      # velocity field
      @. uadvx =  sin(f.x)*cos(f.y)
      @. uadvy = -cos(f.x)*sin(f.y)
      # potential
      @. V = ( uadvx^2 + uadvy^2 ) / α
      new{F}(f, V, uadvx, uadvy, uadvz)
   end

   function PotentialTaylorGreen(f::F; α :: Real = 1) where {F<:AbstractField3D}
      V = real.(similar(f.ϕ))
      uadvx = similar(V)
      uadvy = similar(V)
      uadvz = similar(V)
      # velocity field
      @. uadvx =  sin(f.x)*cos(f.y)*cos(f.z)
      @. uadvy = -cos(f.x)*sin(f.y)*cos(f.z)
      @. uadvz = 0.
      # potential
      @. V = ( uadvx^2 + uadvy^2 + uadvz^2 ) / α
      new{F}(f, V, uadvx, uadvy, uadvz)
   end
end

Base.show(io::IO, p::PotentialTaylorGreen) = print(io, "TaylorGreen Potential")