"""
$(TYPEDEF)

Represents a quadratic potential. It can be used for [`GrossPitaevskiiParameters`](@ref).
It has the following form:
```math
\\dfrac{1}{2}(1-α)\\left(
   γ_x x² + γ_y y² + γ_z z²
\\right)
```

The potential has the following informations:

$(TYPEDFIELDS)

Constructor:

$(METHODLIST)

# Example

```jldoctest
julia> PotentialQuadratic(field, γx = 0.5, γy = 0.5)
Quadratic Potential for 2D fields
    ├──────  parameters: α 0 γx 0.5 γy 0.5
    └──────────────  V = (1-α)/2 × (γx x² + γy y²)
```
"""
struct PotentialQuadratic{F} <: AbstractPotential{F}
    "field on which the potential acts."
    f::F
    "potential field (real)."
    V::AbstractArray
    "scaling scalar"
    α::Real
    "x quadratic coefficient."
    γx::Real
    "y quadratic coefficient."
    γy::Real
    "z quadratic coefficient (for 3D)."
    γz::Real

    """
    $(TYPEDSIGNATURES)

    Returns a `PotentialQuarticQuadratic` potential.
    """
    function PotentialQuadratic(f::F; α::Real=0, γx::Real=1, γy::Real=1,
                                γz::Real=1) where {F<:AbstractField{N,1,FT,FFT,A}} where {N,
                                                                                          FT,
                                                                                          FFT,
                                                                                          A}
        V = PencilArray(f.ϕ.pencil, A{FT}(undef, size_local(f.ϕ)))

        p = new{F}(f, V, α, γx, γy, γz)
        compute!(p)
        return p
    end
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField2D}
    @. p.V = 0.5 * (1 - p.α) * (p.γx * p.f.x^2 + p.γy * p.f.y^2)
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField3D}
    @. p.V = 0.5 * (1 - p.α) * (p.γx * p.f.x^2 + p.γy * p.f.y^2 + p.γz * p.f.z^2)
end

function Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField2D}
    return print(io, "Quadratic Potential for 2D fields\n",
                 "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
                 "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")
end

function Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField3D}
    return print(io, "Quadratic Potential for 3D fields\n",
                 "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy) γz $(p.γz)\n",
                 "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")
end
