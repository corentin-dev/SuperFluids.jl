"""
    PotentialQuarticQuadratic{F} <: AbstractPotential{F}

Represents a quadratic potential. It can be used for [`GrossPitaevskiiParameters`](@ref).
It has the following form:
```math
\\dfrac{1}{2}(1-α)\\left(
   γ_x x² + γ_y y² + γ_z z² + κ_4 r⁴
\\right)
```
with ``r=\\sqrt{x²+y²+z²}`` and ``z=0`` in ``2D``.

The potential has the following informations:

- `f`: the field on which the potential acts,
- `V`: a potential field (real),
- `α`: a scalar,
- `γx`: a scalar,
- `γy`: a scalar,
- `γz`: a scalar (only for 3D),
- `κ4`: a scalar.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> PotentialQuarticQuadratic(field, γx = 0.5, γy = 0.5, κ4 = 1.)
Quartic-Quadratic Potential for 2D fields
  ├──────  parameters: α 0 γx 0.5 γy 0.5  κ₄ 1.0
  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴
```
"""
struct PotentialQuarticQuadratic{F} <: AbstractPotential{F}
    f::F
    V::AbstractArray
    α::Real
    γx::Real
    γy::Real
    γz::Real
    κ4::Real

    function PotentialQuarticQuadratic(f::F; α::Real=0, γx::Real=1, γy::Real=1, γz::Real=1,
                                       κ4::Real=1) where {F<:AbstractField{1,FT,FFT,A}} where {FT,
                                                                                               FFT,
                                                                                               A}
        V = PencilArray(f.ϕ.pencil, A{FT}(undef, size_local(f.ϕ)))

        p = new{F}(f, V, α, γx, γy, γz, κ4)
        compute!(p)
        return p
    end
end

function compute!(p::PotentialQuarticQuadratic{F}) where {F<:AbstractField2D}
    @. p.V = 0.5 * (1 - p.α) * (p.γx * p.f.x^2 + p.γy * p.f.y^2) +
             0.5 * p.κ4 * (p.f.x^2 + p.f.y^2)^2
end

function compute!(p::PotentialQuarticQuadratic{F}) where {F<:AbstractField3D}
    @. p.V = 0.5 * (1 - p.α) * (p.γx * p.f.x^2 + p.γy * p.f.y^2 + p.γz * p.f.z^2) +
             0.5 * p.κ4 * (p.f.x^2 + p.f.y^2)^2
end

function Base.show(io::IO, p::PotentialQuarticQuadratic{F}) where {F<:AbstractField2D}
    return print(io, "Quartic-Quadratic Potential for 2D fields\n",
                 "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
                 "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")
end

function Base.show(io::IO, p::PotentialQuarticQuadratic{F}) where {F<:AbstractField3D}
    return print(io, "Quartic-Quadratic Potential for 3D fields\n",
                 "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
                 "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")
end
