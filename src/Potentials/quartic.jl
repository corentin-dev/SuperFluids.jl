struct PotentialQuarticQuadratic{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
   κ4 :: Real
end

function PotentialQuarticQuadratic(f::F; α::Real = 0, γx::Real = 1, γy::Real = 1, κ4::Real = 1) where {F<:AbstractField}
   V = real.(similar(f.ϕ))
   p = PotentialQuarticQuadratic{F}(f, V, α, γx, γy, γz, κ4)
   compute!(p)
   return p
end

function compute!(p::PotentialQuarticQuadratic{F}) where {F<:AbstractField2D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.g.x^2+γy*p.f.g.y^2) + 0.5*p.κ4 * (p.f.g.x^2 + p.f.g.y^2)^2
end

function compute!(p::PotentialQuarticQuadratic{F}) where {F<:AbstractField3D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.g.x^2+γy*p.f.g.y^2+γz*p.f.g.z^2) + 0.5*p.κ4 * (p.f.g.x^2 + p.f.g.y^2)^2
end

Base.show(io::IO, p::PotentialQuarticQuadratic{F}) where {F<:AbstractField2D} =
     print(io, "Quartic-Quadratic Potential for 2D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")

Base.show(io::IO, p::PotentialQuarticQuadratic{F}) where {F<:AbstractField3D} =
     print(io, "Quartic-Quadratic Potential for 3D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")