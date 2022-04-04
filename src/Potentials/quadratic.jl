"""
    PotentialQuadratic{F} <: AbstractPotential{F}

Represents a quadratic potential. It can be used for [`GrossPitaevskiiParameters`](@ref).
It has the following form:
```math
\\dfrac{1}{2}(1-α)\\left(
   γ_x x² + γ_y y² + γ_z z²
\\right)
```

The potential has the following informations:

- `f`: the field on which the potential acts,
- `V`: a potential field (real),
- `α`: a scalar,
- `γx`: a scalar,
- `γy`: a scalar,
- `γz`: a scalar (only for 3D).

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> PotentialQuadratic(field, γx = 0.5, γy = 0.5)
Quadratic Potential for 2D fields
   ├──────  parameters: α 0 γx 0.5 γy 0.5
   └──────────────  V = (1-α)/2 × (γx x² + γy y²)
```
"""
struct PotentialQuadratic{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real

   function PotentialQuadratic(f::F; α::Real = 0, γx::Real = 1, γy::Real = 1, γz::Real = 1) where {F<:AbstractField{1,FT,FFT,A}} where {FT,FFT,A}
      V = PencilArray(f.ϕ.pencil, A{FT}(undef, size_local(f.ϕ)))

      p = new{F}(f, V, α, γx, γy, γz)
      compute!(p)
      return p
   end
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField2D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.x^2+p.γy*p.f.y^2)
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField3D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.x^2+p.γy*p.f.y^2+p.γz*p.f.z^2)
end

Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField2D} =
     print(io, "Quadratic Potential for 2D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField3D} =
     print(io, "Quadratic Potential for 3D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy) γz $(p.γz)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")