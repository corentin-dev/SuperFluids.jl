struct PotentialQuadratic2D <: AbstractPotential2D
   V
   α :: Real
   γx :: Real
   γy :: Real
end

struct PotentialQuadratic3D <: AbstractPotential3D
   V
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

function PotentialQuadratic(f::AbstractField2D, α::Real, γx::Real, γy::Real)
   V = real(similar(f.ϕ))
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)
   return PotentialQuadratic2D(V, α, γx, γy)
end

function PotentialQuadratic(f::AbstractField3D, α::Real, γx::Real, γy::Real, γz::Real)
   V = real(similar(f.ϕ))
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)
   return PotentialQuadratic3D(V, α, γx, γy, γz)
end

Base.show(io::IO, p::PotentialQuadratic2D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

Base.show(io::IO, p::PotentialQuadratic3D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")
